"""FastAPI application: REST API + SSE + LLM proxy + static dashboard.

Single-process, portable (Replicas workspace first, deployable to Cloud Run /
any container host later). The proxy is mounted at /proxy/v1 — experiment
containers reach it via 127.0.0.1 because upstream runs containers with host
networking.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import paths
from .config import ServiceConfig, model_catalog, ExperimentConfig
from .llm import ProxyLLMClient
from .proxy import create_proxy_router, PLATFORM_TOKEN
from .runner import ExperimentRunner
from .shell import ShellAgent
from .store import Store, EventBus


def create_app(service: ServiceConfig | None = None) -> FastAPI:
    service = service or ServiceConfig.from_env()
    Path(service.home).mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="HyperAgents Control Platform", version="0.1.0")
    store = Store(paths.db_path(Path(service.home)))
    bus = EventBus(store)
    runner = ExperimentRunner(store, bus, service)
    shell_client = ProxyLLMClient(
        service.proxy_v1_base, PLATFORM_TOKEN, service.shell_model
    )
    shell = ShellAgent(shell_client, store, Path(service.home),
                       max_tool_calls=service.shell_max_tool_calls)

    app.state.service = service
    app.state.store = store
    app.state.bus = bus
    app.state.runner = runner
    app.state.shell = shell

    @app.on_event("startup")
    async def _startup():
        bus.attach_loop(asyncio.get_running_loop())
        store.seed_slots(service.initial_slots)

    @app.on_event("shutdown")
    async def _shutdown():
        runner.shutdown()
        store.close()

    # ------------------------------------------------------------------ basic
    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "version": "0.1.0",
            "upstream": service.upstream,
            "experiments": len(store.list_experiments()),
            "slots": store.slot_count(),
        }

    @app.get("/api/models")
    def models():
        return {"models": model_catalog()}

    # ------------------------------------------------------------------ slots
    @app.get("/api/slots")
    def get_slots():
        return {"slots": store.list_slots()}

    @app.post("/api/slots/expand")
    def expand_slots(count: int = 1):
        return {"slots": store.expand_slots(max(1, min(count, 20)))}

    # ------------------------------------------------------------------ experiments
    @app.get("/api/experiments")
    def list_experiments():
        return {"experiments": store.list_experiments()}

    @app.post("/api/experiments")
    async def create_experiment(request: Request):
        body = await request.json()
        slot = body.pop("slot", None)
        try:
            exp = runner.create(body, slot=slot)
        except Exception as e:
            raise HTTPException(400, f"invalid experiment config: {e}")
        return exp

    @app.get("/api/experiments/{exp_id}")
    def get_experiment(exp_id: str):
        exp = store.get_experiment(exp_id)
        if not exp:
            raise HTTPException(404, "experiment not found")
        ws = paths.workspace_dir(exp_id, Path(service.home))
        status = {"workspace_exists": ws.exists()}
        if ws.exists():
            from .workspace import workspace_status

            wstat = workspace_status(exp_id, Path(service.home))
            status.update(
                instrumented=wstat["instrumented"],
                initial_evals_ready=wstat["initial_evals_ready"],
                missing_initial_evals=wstat["missing_initial_evals"],
            )
        exp["workspace_status"] = status
        return exp

    @app.post("/api/experiments/{exp_id}/start")
    def start_experiment(exp_id: str):
        try:
            return runner.start(exp_id)
        except Exception as e:
            raise HTTPException(409, str(e))

    @app.post("/api/experiments/{exp_id}/resume")
    def resume_experiment(exp_id: str):
        try:
            return runner.resume(exp_id)
        except Exception as e:
            raise HTTPException(409, str(e))

    @app.post("/api/experiments/{exp_id}/pause")
    def pause_experiment(exp_id: str):
        try:
            return runner.pause(exp_id)
        except Exception as e:
            raise HTTPException(409, str(e))

    @app.post("/api/experiments/{exp_id}/stop")
    def stop_experiment(exp_id: str):
        try:
            return runner.stop(exp_id, new_status="stopped", reason="user stop")
        except Exception as e:
            raise HTTPException(409, str(e))

    @app.post("/api/experiments/{exp_id}/seed")
    def seed_experiment(exp_id: str):
        try:
            return runner.seed(exp_id)
        except Exception as e:
            raise HTTPException(409, str(e))

    @app.patch("/api/experiments/{exp_id}")
    async def update_experiment(exp_id: str, request: Request):
        """Update experiment config (budgets, models, generation caps...).
        Only allowed while the experiment is not running — e.g. raise a budget
        after the kill-switch paused the experiment."""
        exp = store.get_experiment(exp_id)
        if not exp:
            raise HTTPException(404, "experiment not found")
        if exp["status"] in {"running", "seeding"}:
            raise HTTPException(409, "cannot update config while experiment is running/seeding")
        body = await request.json()
        merged = {**exp["config"], **body}
        if merged.get("loop") != exp["config"].get("loop"):
            raise HTTPException(400, "loop mode cannot be changed after creation")
        try:
            ExperimentConfig(**merged)
        except Exception as e:
            raise HTTPException(400, f"invalid experiment config: {e}")
        # Propagate model changes into an existing workspace's platform config
        # (the instrumentation patch reads it at call time, so no re-copy needed)
        ws = paths.workspace_dir(exp_id, Path(service.home))
        cfg_path = ws / paths.platform_config_relpath()
        if cfg_path.exists():
            pcfg = json.loads(cfg_path.read_text())
            pcfg["meta_model"] = merged["meta_model"]
            pcfg["task_model"] = merged["task_model"]
            cfg_path.write_text(json.dumps(pcfg, indent=2))
        store.set_config(exp_id, merged)
        return store.get_experiment(exp_id)

    @app.get("/api/experiments/{exp_id}/usage")
    def experiment_usage(exp_id: str):
        if not store.get_experiment(exp_id):
            raise HTTPException(404, "experiment not found")
        return {
            "totals_proxy": store.usage_totals(exp_id, source="proxy"),
            "totals_local": store.usage_totals(exp_id, source="local"),
            "breakdown": store.usage_breakdown(exp_id),
        }

    # ------------------------------------------------------------------ SSE
    @app.get("/api/experiments/{exp_id}/events")
    async def experiment_events(exp_id: str, replay: int = 100):
        if not store.get_experiment(exp_id):
            raise HTTPException(404, "experiment not found")
        q = bus.subscribe(exp_id)

        async def gen():
            try:
                for ev in store.list_events(exp_id, limit=replay):
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
                while True:
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
            finally:
                bus.unsubscribe(exp_id, q)

        return StreamingResponse(gen(), media_type="text/event-stream")

    # ------------------------------------------------------------------ comparison
    @app.get("/api/comparison")
    def comparison(ids: str):
        exp_ids = [i.strip() for i in ids.split(",") if i.strip()]
        if not exp_ids:
            raise HTTPException(400, "provide ids=a,b,c")
        return {"comparison": store.comparison(exp_ids)}

    # ------------------------------------------------------------------ shell
    @app.post("/api/shell/chat")
    async def shell_chat(request: Request):
        body = await request.json()
        message = body.get("message", "").strip()
        history = body.get("history") or []
        if not message:
            raise HTTPException(400, "message required")
        shell_obj = request.app.state.shell
        result = await asyncio.to_thread(shell_obj.run, message, history)
        return result

    # ------------------------------------------------------------------ proxy (LLM gateway)
    app.include_router(create_proxy_router(store, bus, service), prefix="/proxy")

    # ------------------------------------------------------------------ dashboard
    web_dir = paths.platform_dir() / "web"
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/")
        def index():
            return FileResponse(str(web_dir / "index.html"))

    return app
