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
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import paths
from .config import ServiceConfig, model_catalog, ExperimentConfig
from .llm import ProxyLLMClient
from .watcher import DOMAIN_SCORE_KEYS


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

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
            cfg = ExperimentConfig(**merged)
        except Exception as e:
            raise HTTPException(400, f"invalid experiment config: {e}")
        # store the *validated* form so model aliases are canonicalized
        merged = cfg.model_dump()
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

    # ------------------------------------------------------------------ lineage & history
    def _output_dir(exp_id: str) -> Optional[Path]:
        exp = store.get_experiment(exp_id)
        if not exp or not exp.get("output_dir"):
            return None
        p = Path(exp["output_dir"])
        return p if p.exists() else None

    def _read_scores_for(gen_dir: Path) -> dict:
        scores = {}
        for eval_dir in sorted(gen_dir.iterdir()):
            if not eval_dir.is_dir():
                continue
            name = eval_dir.name
            if name.endswith("_eval"):
                domain, split = name[: -len("_eval")], "train"
            elif name.endswith("_eval_val"):
                domain, split = name[: -len("_eval_val")], "val"
            elif name.endswith("_eval_test"):
                domain, split = name[: -len("_eval_test")], "test"
            else:
                continue
            report = _read_json(eval_dir / "report.json")
            if not report:
                continue
            key = DOMAIN_SCORE_KEYS.get(domain, "overall_accuracy")
            if report.get(key) is not None:
                scores.setdefault(domain, {})[split] = report[key]
        return scores

    @app.get("/api/experiments/{exp_id}/lineage")
    def experiment_lineage(exp_id: str):
        """Evolutionary lineage tree reconstructed from upstream's own state:
        archive.jsonl (node list) + gen_*/metadata.json (parent links, validity)
        + report.json (scores)."""
        if not store.get_experiment(exp_id):
            raise HTTPException(404, "experiment not found")
        out = _output_dir(exp_id)
        if not out:
            return {"nodes": [], "roots": [], "note": "no output directory yet"}
        archive = None
        archive_file = out / "archive.jsonl"
        if archive_file.exists():
            try:
                lines = [ln for ln in archive_file.read_text().splitlines() if ln.strip()]
                archive = json.loads(lines[-1]).get("archive") if lines else None
            except Exception:
                archive = None
        nodes = []
        by_gen = {}
        prev_mtime = None
        def _gen_sort_key(p: Path):
            try:
                return (p / "metadata.json").stat().st_mtime
            except OSError:
                return 0.0
        for gen_dir in sorted(out.glob("gen_*"), key=_gen_sort_key):
            meta = _read_json(gen_dir / "metadata.json")
            if meta is None:
                continue
            gen = str(meta.get("current_genid", gen_dir.name.replace("gen_", "")))
            try:
                mtime = (gen_dir / "metadata.json").stat().st_mtime
            except OSError:
                mtime = None
            node = {
                "gen": gen,
                "parent": meta.get("parent_genid"),
                "valid_parent": meta.get("valid_parent", True) if gen != "initial" else True,
                "parent_agent_success": meta.get("parent_agent_success"),
                "scores": _read_scores_for(gen_dir),
                "completed_ts": mtime,
                "duration_s": round(mtime - prev_mtime, 2) if (mtime and prev_mtime) else None,
            }
            prev_mtime = mtime or prev_mtime
            nodes.append(node)
            by_gen[gen] = node
        # children edges
        for n in nodes:
            if n["parent"] is not None:
                p = by_gen.get(str(n["parent"]))
                if p:
                    p.setdefault("children", []).append(n["gen"])
        roots = [n["gen"] for n in nodes if n["parent"] is None]
        return {
            "nodes": nodes,
            "roots": roots,
            "archive": archive,
            "total_generations": len(nodes),
        }

    @app.get("/api/experiments/{exp_id}/generations/{gen}")
    def generation_detail(exp_id: str, gen: str):
        """One generation's history + memory: metadata, scores, the meta-agent's
        diff and chat-history tail."""
        if not store.get_experiment(exp_id):
            raise HTTPException(404, "experiment not found")
        out = _output_dir(exp_id)
        if not out:
            raise HTTPException(404, "no output directory yet")
        gen_dir = out / f"gen_{gen}"
        if not gen_dir.exists():
            raise HTTPException(404, f"generation {gen} not found")
        meta = _read_json(gen_dir / "metadata.json") or {}
        detail = {
            "gen": gen,
            "metadata": meta,
            "scores": _read_scores_for(gen_dir),
            "files": sorted(p.name for p in gen_dir.rglob("*") if p.is_file()),
        }
        patch = gen_dir / "agent_output" / "model_patch.diff"
        chat = gen_dir / "agent_output" / "meta_agent_chat_history.md"
        if patch.exists():
            detail["model_patch"] = patch.read_text()[:6000]
        if chat.exists():
            detail["meta_chat_history_tail"] = chat.read_text()[-6000:]
        return detail

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
        rows = store.comparison(exp_ids)
        # enrich with latest generation summaries: scores, count, speed
        for col in rows:
            gens = store.latest_generations(col["id"])
            if gens:
                best_train: dict[str, float] = {}
                durations = [g.get("duration_s") for g in gens if g.get("duration_s")]
                for g in gens:
                    for domain, splits in (g.get("scores") or {}).items():
                        tr = splits.get("train")
                        if tr is not None:
                            best_train[domain] = max(best_train.get(domain, float("-inf")), tr)
                col["generation_stats"] = {
                    "count": len(gens),
                    "best_train_scores": best_train,
                    "avg_generation_seconds": (
                        round(sum(durations) / len(durations), 2) if durations else None
                    ),
                }
            else:
                col["generation_stats"] = {"count": 0, "best_train_scores": {}, "avg_generation_seconds": None}
        return {"comparison": rows}

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
