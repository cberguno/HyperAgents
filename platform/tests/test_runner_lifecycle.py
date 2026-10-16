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
