#!/usr/bin/env python3
"""Offline end-to-end demo of the M1 control plane. ZERO API cost.

Runs the full platform in stub mode against a live local server:
  1. boots the API + proxy + dashboard (ephemeral port)
  2. expands slots (5 -> 6, the '+' button backend)
  3. creates an offline stub experiment with a budget
  4. starts it; the stub loop drives generations through the real proxy
  5. streams live events, then stops mid-run and resumes
  6. prints accounting (tokens/cost per role/model), generations, comparison
  7. exercises the conversational shell with a scripted (offline) LLM

Usage: platform/.venv/bin/python platform/scripts/demo_offline.py
"""
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from hacp.api import create_app  # noqa: E402
from hacp.config import ServiceConfig  # noqa: E402
from hacp.llm import FakeLLMClient  # noqa: E402
from hacp.shell import ShellAgent  # noqa: E402


def main():
    import tempfile

    tmp = tempfile.mkdtemp(prefix="hacp-demo-")
    service = ServiceConfig(
        host="127.0.0.1", port=0, home=tmp, upstream="stub",
        upstream_repo=str(Path(__file__).resolve().parent.parent.parent),
        shell_model="stub/tiny",
    )
    app = create_app(service)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    while not server.started:
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    service.proxy_public_base = base

    c = httpx.Client(base_url=base, timeout=30)
    print(f"[demo] platform live at {base} (upstream={service.upstream}, zero API cost)")

    # 2. slots: 5 initial + expand to 6
    slots = c.get("/api/slots").json()["slots"]
    print(f"[demo] initial slots: {len(slots)}")
    n = c.post("/api/slots/expand?count=1").json()["slots"]
    print(f"[demo] expanded to {n} slots")

    # 3-4. create + start an offline stub experiment with a budget
    exp = c.post("/api/experiments", json={
        "name": "offline-demo paper_review",
        "loop": "stub", "max_generation": 4,
        "meta_model": "anthropic/claude-sonnet-4.5",
        "task_model": "openai/gpt-4o-mini",
        "budget": {"max_cost_usd": 5.0, "max_tokens": 500_000},
    }).json()
    eid = exp["id"]
    print(f"[demo] created experiment {eid} (slot 0)")
    c.post(f"/api/experiments/{eid}/start")
    print("[demo] started; streaming events...")

    # 5. stream a few events live, then stop mid-run and resume
    seen = []
    deadline = time.time() + 25
    with c.stream("GET", f"/api/experiments/{eid}/events?replay=0", timeout=30) as resp:
        for line in resp.iter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                seen.append(ev["type"])
                if ev["type"] == "generations":
                    gens = ev["data"]["generations"]
                    latest = gens[-1]
                    print(f"[demo]   generation {latest['gen']} scores={latest['scores']}")
                if len(seen) >= 8 or time.time() > deadline:
                    break
    status = c.get(f"/api/experiments/{eid}").json()["status"]
    if status == "running":
        stopped = c.post(f"/api/experiments/{eid}/stop").json()
        print(f"[demo] stopped mid-run: status={stopped['status']} (history preserved)")
        resumed = c.post(f"/api/experiments/{eid}/resume").json()
        print(f"[demo] resumed: status={resumed['status']} (continues from last generation)")
    for _ in range(60):
        e = c.get(f"/api/experiments/{eid}").json()
        if e["status"] in {"finished", "failed", "paused_budget"}:
            break
        time.sleep(0.5)
    print(f"[demo] final status: {e['status']}")

    # 6. accounting + generations + comparison
    usage = c.get(f"/api/experiments/{eid}/usage").json()
    print("[demo] usage (authoritative proxy ledger):")
    print(f"        {json.dumps(usage['totals_proxy'])}")
    print("[demo] breakdown by role/model:")
    for b in usage["breakdown"]:
        print(f"        {b['role']:5s} {b['model']:45s} tokens={b['total_tokens']:>8} cost=${b['cost_usd']:.6f}")
    gens_ev = [x for x in _replay(c, eid) if x["type"] == "generations"]
    if gens_ev:
        for g in gens_ev[-1]["data"]["generations"]:
            print(f"[demo] gen {g['gen']}: scores={g['scores']}")

    # second experiment for comparison
    exp2 = c.post("/api/experiments", json={
        "name": "offline-demo-2", "loop": "stub", "max_generation": 2,
        "meta_model": "openai/gpt-4o-mini", "task_model": "openai/gpt-4o-mini",
    }).json()
    c.post(f"/api/experiments/{exp2['id']}/start")
    for _ in range(60):
        e2 = c.get(f"/api/experiments/{exp2['id']}").json()
        if e2["status"] in {"finished", "failed"}:
            break
        time.sleep(0.5)
    comp = c.get(f"/api/comparison?ids={eid},{exp2['id']}").json()["comparison"]
    print("[demo] side-by-side comparison:")
    for col in comp:
        print(f"        {col['name']:25s} status={col['status']:12s} "
              f"tokens={col['usage']['total_tokens']:>7} cost=${col['usage']['cost_usd']:.6f}")

    # 7. conversational shell (scripted offline LLM; tools run against the real store)
    app.state.shell = ShellAgent(
        FakeLLMClient([
            '<json>{"tool_name": "list_experiments", "tool_input": {}}</json>',
            "You have 2 offline experiments: 'offline-demo paper_review' (finished) and 'offline-demo-2'.",
        ]),
        app.state.store, service.home,
    )
    shell_out = c.post("/api/shell/chat", json={"message": "what experiments exist?"}).json()
    print(f"[demo] shell reply: {shell_out['reply']}")
    print(f"[demo] shell tools used: {[t['tool_name'] for t in shell_out['tool_calls']]}")

    print("[demo] DONE — full M1 stack exercised offline with zero API cost.")
    print(f"[demo] artifacts in {tmp}")
    server.should_exit = True
    t.join(timeout=10)
    return 0


def _replay(c, eid):
    """Read replayed SSE events; the stream never ends on its own, so tolerate
    the read timeout and return what was collected."""
    events = []
    try:
        with c.stream("GET", f"/api/experiments/{eid}/events?replay=500", timeout=2.0) as resp:
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
    except httpx.HTTPError:
        pass
    return events


if __name__ == "__main__":
    raise SystemExit(main())
