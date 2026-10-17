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

import httpx
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
            pcfg["meta_reasoning"] = merged.get("meta_reasoning_effort")
            pcfg["task_reasoning"] = merged.get("task_reasoning_effort")
            pcfg["meta_completion"] = merged.get("meta_completion_cap")
            pcfg["task_completion"] = merged.get("task_completion_cap")
            cfg_path.write_text(json.dumps(pcfg, indent=2))
        store.set_config(exp_id, merged)
        return store.get_experiment(exp_id)

    @app.get("/api/experiments/{exp_id}/usage")
    def experiment_usage(exp_id: str):
        if not store.get_experiment(exp_id):
            raise HTTPException(404, "experiment not found")
        from .budget import RESERVATIONS

        reserved_tokens, reserved_cost = RESERVATIONS.totals(exp_id)
        return {
            "totals_proxy": store.usage_totals(exp_id, source="proxy"),
            "totals_local": store.usage_totals(exp_id, source="local"),
            "breakdown": store.usage_breakdown(exp_id),
            "in_flight": {
                "reserved_tokens": reserved_tokens,
                "reserved_cost_usd": round(reserved_cost, 6),
                "active_requests": RESERVATIONS.active_count(exp_id),
            },
        }

    @app.get("/api/experiments/{exp_id}/cost_report")
    def experiment_cost_report(exp_id: str):
        """Accurate per-phase / per-generation cost attribution.

        Live request-time tagging is best-effort (the in-flight generation's
        number isn't knowable until it completes), so attribution here uses
        each generation's completion timestamp window — the same boundary
        semantics as upstream's own generation checkpointing.
        """
        exp = store.get_experiment(exp_id)
        if not exp:
            raise HTTPException(404, "experiment not found")
        from .budget import RESERVATIONS

        out = _output_dir(exp_id)
        rows = store.list_usage(exp_id, source="proxy")
        phases: list[dict] = []

        def _aggregate(name: str, lo: float | None, hi: float | None) -> dict:
            models: dict[str, dict] = {}
            roles: dict[str, dict] = {}
            tot = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                   "total_tokens": 0, "cost_usd": 0.0, "errors": 0, "blocked": 0,
                   "latency_sum": 0.0, "latency_n": 0}
            for r in rows:
                if lo is not None and r["ts"] <= lo:
                    continue
                if hi is not None and r["ts"] > hi:
                    continue
                st = r["status"]
                if st == "blocked":
                    tot["blocked"] += 1
                    continue
                if st == "error":
                    tot["errors"] += 1
                    continue
                m = models.setdefault(r["model"], {"model": r["model"], "requests": 0,
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0})
                rl = roles.setdefault(r["role"], {"role": r["role"], "requests": 0,
                    "avg_latency_s": None, "_lat_sum": 0.0, "_lat_n": 0})
                m["requests"] += 1
                m["prompt_tokens"] += r["prompt_tokens"]
                m["completion_tokens"] += r["completion_tokens"]
                m["total_tokens"] += r["total_tokens"]
                m["cost_usd"] = round(m["cost_usd"] + r["cost_usd"], 6)
                tot["requests"] += 1
                tot["prompt_tokens"] += r["prompt_tokens"]
                tot["completion_tokens"] += r["completion_tokens"]
                tot["total_tokens"] += r["total_tokens"]
                tot["cost_usd"] = round(tot["cost_usd"] + r["cost_usd"], 6)
                if r["latency_s"] is not None:
                    tot["latency_sum"] += r["latency_s"]
                    tot["latency_n"] += 1
                    rl["_lat_sum"] += r["latency_s"]
                    rl["_lat_n"] += 1
                rl["requests"] += 1
            for rl in roles.values():
                lat_sum = rl.pop("_lat_sum")
                lat_n = rl.pop("_lat_n")
                rl["avg_latency_s"] = round(lat_sum / lat_n, 4) if lat_n else None
            lat_sum = tot.pop("latency_sum")
            lat_n = tot.pop("latency_n")
            tot["avg_latency_s"] = round(lat_sum / lat_n, 4) if lat_n else None
            return {"phase": name, "models": list(models.values()), "roles": list(roles.values()), "totals": tot}

        if out is None or not rows:
            phases.append(_aggregate("seed", None, None))
        else:
            # generation boundaries = metadata mtime per gen dir (chronological)
            def _mtime(p: Path) -> float:
                try:
                    return (p / "metadata.json").stat().st_mtime
                except OSError:
                    try:
                        return p.stat().st_mtime
                    except OSError:
                        return 0.0

            gens = sorted(out.glob("gen_*"), key=_mtime)
            boundaries = [(g.name.replace("gen_", "") or "initial", _mtime(g)) for g in gens if _mtime(g) > 0]
            prev = None
            if boundaries:
                phases.append(_aggregate("seed", None, boundaries[0][1]))
                for i, (gname, mt) in enumerate(boundaries):
                    hi = boundaries[i + 1][1] if i + 1 < len(boundaries) else None
                    phases.append(_aggregate(f"gen {gname}", mt, hi))
            else:
                phases.append(_aggregate("seed", None, None))

        reserved_tokens, reserved_cost = RESERVATIONS.totals(exp_id)
        grand = {
            "requests": sum(p["totals"]["requests"] for p in phases),
            "prompt_tokens": sum(p["totals"]["prompt_tokens"] for p in phases),
            "completion_tokens": sum(p["totals"]["completion_tokens"] for p in phases),
            "total_tokens": sum(p["totals"]["total_tokens"] for p in phases),
            "cost_usd": round(sum(p["totals"]["cost_usd"] for p in phases), 6),
            "errors": sum(p["totals"]["errors"] for p in phases),
            "blocked": sum(p["totals"]["blocked"] for p in phases),
            "in_flight_reserved_tokens": reserved_tokens,
            "in_flight_reserved_cost_usd": round(reserved_cost, 6),
        }
        return {"phases": phases, "grand_totals": grand, "budget_state": exp["budget_state"]}

    # ------------------------------------------------------------------ provider verification
    @app.post("/api/admin/verify-provider")
    async def verify_provider():
        """NON-BILLABLE provider connectivity check.

        Calls only OpenRouter's free endpoints (GET /key, GET /models) — zero
        completion tokens, zero cost. Verifies both configured models exist as
        OpenRouter slugs and cross-checks live pricing against the platform
        registry. The API key is never included in the response.
        """
        if service.upstream != "openrouter":
            return {
                "ok": False,
                "upstream": service.upstream,
                "hint": "start the platform with HACP_UPSTREAM=openrouter (and OPENROUTER_API_KEY set) to verify the real provider",
            }
        from .proxy import _read_openrouter_key, OpenRouterUpstream

        key = _read_openrouter_key()
        if not key:
            return {"ok": False, "error": "missing_key",
                    "hint": "set OPENROUTER_API_KEY in the platform environment or platform-home/secrets.env"}
        from .config import openrouter_slug

        client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={"Authorization": f"Bearer {key}", "HTTP-Referer": "https://hyperagents.local"},
            timeout=30.0,
        )
        try:
            key_resp = await client.get("/key")
            key_info = key_resp.json() if key_resp.status_code == 200 else {"http_status": key_resp.status_code}
            models_resp = await client.get("/models")
            if models_resp.status_code != 200:
                return {"ok": False, "error": "models_endpoint", "http_status": models_resp.status_code}
            models = {m["id"]: m for m in models_resp.json().get("data", [])}
        finally:
            await client.aclose()

        wanted = {}
        for exp in store.list_experiments():
            cfg = exp.get("config", {})
            for role in ("meta_model", "task_model"):
                mid = cfg.get(role)
                if mid:
                    wanted.setdefault(mid, role)
        wanted.setdefault("anthropic/claude-sonnet-4-5-20250929", "meta_model (default)")
        wanted.setdefault("openai/gpt-4o-mini", "task_model (default)")

        model_report = []
        for canonical, role in wanted.items():
            slug = openrouter_slug(canonical)
            m = models.get(slug)
            if not m:
                model_report.append({"canonical": canonical, "slug": slug, "role": role, "available": False})
                continue
            live_in = float(m.get("pricing", {}).get("prompt", 0) or 0) * 1e6
            live_out = float(m.get("pricing", {}).get("completion", 0) or 0) * 1e6
            from .config import resolve_model

            reg = resolve_model(canonical)
            model_report.append({
                "canonical": canonical, "slug": slug, "role": role, "available": True,
                "context_length": m.get("context_length"),
                "live_input_usd_per_mtok": round(live_in, 4),
                "live_output_usd_per_mtok": round(live_out, 4),
                "registry_input_usd_per_mtok": reg.input_usd_per_mtok,
                "registry_output_usd_per_mtok": reg.output_usd_per_mtok,
                "pricing_matches_registry": abs(live_in - reg.input_usd_per_mtok) < 0.01 and abs(live_out - reg.output_usd_per_mtok) < 0.01,
            })
        return {
            "ok": key_resp.status_code == 200,
            "key_valid": key_resp.status_code == 200,
            "key_label": (key_info.get("data", {}) or {}).get("label") if isinstance(key_info, dict) else None,
            "usage": (key_info.get("data", {}) or {}).get("usage") if isinstance(key_info, dict) else None,
            "limit": (key_info.get("data", {}) or {}).get("limit") if isinstance(key_info, dict) else None,
            "models_checked": model_report,
            "all_available": all(r["available"] for r in model_report),
            "note": "verified with zero-cost endpoints (/key, /models); no completion tokens were spent",
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
            # gen_initial has no metadata.json by design (upstream writes only
            # the initial evals there) — still include it as a root node
            has_content = any(gen_dir.iterdir())
            if meta is None and not has_content:
                continue
            gen = str((meta or {}).get("current_genid", gen_dir.name.replace("gen_", "")))
            try:
                mtime = (gen_dir / "metadata.json").stat().st_mtime
            except OSError:
                try:
                    mtime = gen_dir.stat().st_mtime
                except OSError:
                    mtime = None
            node = {
                "gen": gen,
                "parent": (meta or {}).get("parent_genid"),
                "valid_parent": (meta or {}).get("valid_parent", True) if gen != "initial" else True,
                "parent_agent_success": (meta or {}).get("parent_agent_success"),
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

    @app.get("/api/experiments/{exp_id}/generations")
    def experiment_generations(exp_id: str):
        """Latest per-generation summaries (REST; the SSE 'generations' event is
        the live updater — the SSE replay window can be dominated by llm_call
        bursts, so the dashboard bootstraps from this endpoint)."""
        if not store.get_experiment(exp_id):
            raise HTTPException(404, "experiment not found")
        return {"generations": store.latest_generations(exp_id)}

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
