"""End-to-end runner lifecycle with the stub loop against a LIVE server
(the stub loop is a subprocess making real HTTP calls to the platform proxy,
with the offline stub upstream — zero API cost).

Covers: start, live monitoring, stop/pause/resume, budget kill-switch,
history preservation, and pristine-repo safety.
"""
import json
import time
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _wait_status(client, eid, statuses, timeout=30):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"/api/experiments/{eid}").json()
        if last["status"] in statuses:
            return last
        time.sleep(0.4)
    raise AssertionError(f"timeout waiting for {statuses}; last={last}")


def _make(client, name, **kw):
    r = client.post("/api/experiments", json={"name": name, "loop": "stub", **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_full_lifecycle_start_stop_resume_finish(live_server, service):
    client = httpx.Client(base_url=live_server, timeout=30)
    # --- stop mid-run (stub loop runs ~1.2s/gen; use many generations so it
    #     is still running when we stop it)
    exp = _make(client, "offline-lifecycle", max_generation=30)
    eid = exp["id"]

    # --- start
    r = client.post(f"/api/experiments/{eid}/start")
    assert r.status_code == 200, r.text
    running = _wait_status(client, eid, {"running", "finished"}, timeout=10)
    assert running["status"] in ("running", "finished")

    # --- live log stream (SSE) produces events
    with client.stream("GET", f"/api/experiments/{eid}/events?replay=10", timeout=30) as resp:
        got_types = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                got_types.append(ev["type"])
            if len(got_types) >= 6:
                break
    assert "started" in got_types

    # --- stop mid-run (stub loop runs ~1.2s/gen)
    if client.get(f"/api/experiments/{eid}").json()["status"] == "running":
        stopped = client.post(f"/api/experiments/{eid}/stop").json()
        assert stopped["status"] == "stopped"
        # history preserved: output dir + archive intact
        ws = Path(service.home) / "experiments" / eid / "workspace"
        out_dirs = list((ws / "outputs").glob("generate_*"))
        assert out_dirs, "output history missing after stop"
        archive = (out_dirs[0] / "archive.jsonl").read_text()
        assert "archive" in archive

    # --- resume: continues from the last completed generation
    r = client.post(f"/api/experiments/{eid}/resume")
    assert r.status_code == 200, r.text
    exp_final = _wait_status(client, eid, {"finished", "failed"}, timeout=60)
    assert exp_final["status"] == "finished", exp_final

    # --- accounting: stub generations made real proxy calls
    usage = client.get(f"/api/experiments/{eid}/usage").json()
    assert usage["totals_proxy"]["requests"] >= 2
    assert usage["totals_proxy"]["total_tokens"] > 0
    # cost was estimated with the configured models' real prices
    assert usage["totals_proxy"]["cost_usd"] > 0

    # --- generations + scores recorded
    gens = [
        ev["data"]["generations"]
        for ev in _all_events(client, eid)
        if ev["type"] == "generations"
    ]
    assert gens, "no generation summaries recorded"
    last_gens = gens[-1]
    assert any(g["scores"]["paper_review"]["train"] for g in last_gens)
    # execution speed: per-generation durations recorded
    durations = [g["duration_s"] for g in last_gens if g.get("duration_s") is not None]
    assert durations, "no generation durations recorded"

    # --- evolutionary lineage endpoint
    lin = client.get(f"/api/experiments/{eid}/lineage").json()
    assert lin["total_generations"] >= 4
    assert "initial" in lin["roots"]
    gens_by_name = {n["gen"]: n for n in lin["nodes"]}
    assert gens_by_name["1"]["parent"] == "initial"
    assert str(gens_by_name["1"]["parent"]) == "initial"
    assert gens_by_name["initial"].get("children") and "1" in gens_by_name["initial"]["children"]
    assert gens_by_name["1"]["scores"]["paper_review"]["train"] > 0.52

    # --- generation detail: history + memory (meta-agent patch & chat)
    detail = client.get(f"/api/experiments/{eid}/generations/1").json()
    assert detail["metadata"]["parent_genid"] == "initial"
    assert "model_patch" in detail and "stub patch for generation 1" in detail["model_patch"]
    assert "meta_chat_history_tail" in detail
    assert "Generation 1" in detail["meta_chat_history_tail"]
    missing = client.get(f"/api/experiments/{eid}/generations/999")
    assert missing.status_code == 404

    # --- comparison includes generation stats (scores + speed)
    exp2_id = client.post("/api/experiments", json={"name": "cmp-2", "loop": "stub", "max_generation": 1}).json()["id"]
    client.post(f"/api/experiments/{exp2_id}/start")
    _wait_status(client, exp2_id, {"finished", "failed"}, timeout=30)
    comp = client.get(f"/api/comparison?ids={eid},{exp2_id}").json()["comparison"]
    for col in comp:
        assert col["generation_stats"]["count"] >= 2
        assert col["generation_stats"]["best_train_scores"]["paper_review"] > 0
        assert col["generation_stats"]["avg_generation_seconds"] is not None

    # --- pristine repo untouched
    pristine = (REPO_ROOT / "agent" / "llm.py").read_text()
    assert "HACP-PLATFORM-INSTRUMENTATION" not in pristine

    # --- workspace instrumented and isolated
    ws = Path(service.home) / "experiments" / eid / "workspace"
    assert "HACP-PLATFORM-INSTRUMENTATION" in (ws / "agent" / "llm.py").read_text()
    assert (ws / "misc" / "hacp" / "platform_config.json").exists()


def _all_events(client, eid, max_wait=2.0):
    """Collect replayed SSE events without hanging: the stream is infinite,
    so read until the per-request timeout fires and return what we got."""
    events = []
    try:
        with client.stream("GET", f"/api/experiments/{eid}/events?replay=500", timeout=max_wait) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
    except httpx.HTTPError:
        pass  # expected: the SSE stream never ends on its own
    return events


def test_budget_kill_switch_pauses_experiment(live_server, service):
    client = httpx.Client(base_url=live_server, timeout=30)
    # tiny budget: the stub loop's ~50-token calls will exceed it within gen 1
    exp = _make(client, "budget-test", max_generation=10,
                budget={"max_tokens": 40}, meta_model="stub/tiny", task_model="stub/tiny")
    eid = exp["id"]
    r = client.post(f"/api/experiments/{eid}/start")
    assert r.status_code == 200, r.text

    final = _wait_status(client, eid, {"paused_budget", "failed"}, timeout=30)
    assert final["status"] == "paused_budget", final

    events = _all_events(client, eid)
    types = [e["type"] for e in events]
    assert "budget_exceeded" in types or "budget_blocked" in types

    # raise the budget via PATCH, then resume
    r = client.patch(f"/api/experiments/{eid}", json={"budget": {"max_tokens": 10_000_000}})
    assert r.status_code == 200
    r = client.post(f"/api/experiments/{eid}/resume")
    assert r.status_code == 200, r.text
    resumed = _wait_status(client, eid, {"running", "finished"}, timeout=15)
    assert resumed["status"] in ("running", "finished")


def test_cannot_start_real_loop_without_seeding(live_server, service):
    client = httpx.Client(base_url=live_server, timeout=30)
    exp = _make(client, "real-loop-guard", loop="generate", max_generation=2)
    eid = exp["id"]
    r = client.post(f"/api/experiments/{eid}/start")
    assert r.status_code == 409
    assert "seed" in r.json()["detail"]


def test_seed_refused_in_stub_mode(live_server, service):
    client = httpx.Client(base_url=live_server, timeout=30)
    exp = _make(client, "seed-guard", loop="stub")
    r = client.post(f"/api/experiments/{exp['id']}/seed")
    assert r.status_code == 409
