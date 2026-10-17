"""Integration: the instrumented upstream agent/llm.py end-to-end.

This is the core M1 correctness proof, entirely offline:

  upstream agent/llm.py (copied + patched) in a temp workspace
    -> get_response_from_llm() with role detection (meta vs task)
    -> model overridden per-role from platform_config.json
    -> real litellm HTTP call -> platform proxy (live server) -> stub upstream
    -> usage recorded in the proxy ledger AND the local usage.jsonl
    -> zero API cost, no provider keys.
"""
import json
import os
import shutil
import subprocess
import sys

import httpx
import pytest

from hacp.instrumentation import apply_instrumentation

META_DRIVER = """
import sys
from agent.llm import get_response_from_llm
text, hist, info = get_response_from_llm("hello meta agent", model="anthropic/claude-3-haiku-20240307")
print(text)
"""

TASK_DRIVER = """
import sys
from agent.llm import get_response_from_llm
text, hist, info = get_response_from_llm("hello task agent", model="openai/gpt-4o")
print(text)
"""


@pytest.fixture
def instrumented_workspace(tmp_path, upstream_repo):
    (tmp_path / "agent").mkdir()
    shutil.copy(upstream_repo / "agent" / "llm.py", tmp_path / "agent" / "llm.py")
    apply_instrumentation(tmp_path)
    (tmp_path / "misc" / "hacp").mkdir(parents=True)
    (tmp_path / "misc" / "hacp" / "usage.jsonl").touch()
    (tmp_path / "run_meta_agent_sim.py").write_text(META_DRIVER)
    (tmp_path / "harness_sim.py").write_text(TASK_DRIVER)
    return tmp_path


def _run_driver(ws, script, timeout=180):
    env = {k: v for k, v in os.environ.items()
           if k not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"}}
    env["OPENAI_API_KEY"] = "hacp-test-not-a-real-key"  # litellm requires presence; proxy holds the real key
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(
        [sys.executable, script],
        cwd=str(ws),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_instrumented_llm_full_path(instrumented_workspace, live_server):
    client = httpx.Client(base_url=live_server, timeout=30)
    exp = client.post("/api/experiments", json={
        "name": "instrumented",
        "meta_model": "anthropic/claude-sonnet-4.5",   # alias -> canonical
        "task_model": "openai/gpt-4o-mini",
    }).json()
    eid = exp["id"]

    cfg = {
        "experiment_id": eid,
        "meta_model": "anthropic/claude-sonnet-4.5",
        "task_model": "openai/gpt-4o-mini",
        "llm_api_base": f"{live_server}/proxy/v1",
        "meta_token": f"hacp-{eid}-meta",
        "task_token": f"hacp-{eid}-task",
        "usage_log": "misc/hacp/usage.jsonl",
    }
    (instrumented_workspace / "misc" / "hacp" / "platform_config.json").write_text(
        json.dumps(cfg, indent=2)
    )

    # --- meta role: default claude-3-haiku overridden -> sonnet 4.5 via proxy
    r = _run_driver(instrumented_workspace, "run_meta_agent_sim.py")
    assert r.returncode == 0, r.stderr
    assert "[stub:anthropic/claude-sonnet-4.5]" in r.stdout

    # --- task role: default gpt-4o overridden -> gpt-4o-mini via proxy
    r = _run_driver(instrumented_workspace, "harness_sim.py")
    assert r.returncode == 0, r.stderr
    assert "[stub:openai/gpt-4o-mini]" in r.stdout

    # --- proxy ledger: authoritative, role-aware, priced
    usage = client.get(f"/api/experiments/{eid}/usage").json()
    assert usage["totals_proxy"]["requests"] == 2
    assert usage["totals_proxy"]["total_tokens"] > 0
    assert usage["totals_proxy"]["cost_usd"] > 0
    models = {(b["role"], b["model"]) for b in usage["breakdown"]}
    assert ("meta", "anthropic/claude-sonnet-4-5-20250929") in models  # alias normalized
    assert ("task", "openai/gpt-4o-mini") in models

    # --- local usage.jsonl written by the instrumentation (reconciliation path)
    lines = (instrumented_workspace / "misc" / "hacp" / "usage.jsonl").read_text().splitlines()
    assert len(lines) == 2
    local = [json.loads(ln) for ln in lines]
    assert {e["role"] for e in local} == {"meta", "task"}
    assert {e["model"] for e in local} == {"openai/anthropic/claude-sonnet-4.5", "openai/openai/gpt-4o-mini"}


def test_instrumented_llm_inactive_without_config(instrumented_workspace, monkeypatch):
    """Without platform_config.json the patched code must behave exactly like
    upstream: same model passed through, no proxy routing."""
    (instrumented_workspace / "misc" / "hacp" / "platform_config.json").unlink(missing_ok=True)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hacp_test_agent_llm_inert", instrumented_workspace / "agent" / "llm.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod._hacp_load_config() == {}
    override, extra = mod._hacp_before_completion("openai/gpt-4o")
    assert override is None and extra == {}

    # full call path with a stubbed litellm: model unchanged, no api_base
    recorded = {}

    def fake_completion(**kwargs):
        recorded.update(kwargs)
        resp = {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
        return resp

    monkeypatch.setattr(mod.litellm, "completion", fake_completion)
    text, hist, info = mod.get_response_from_llm("hi", model="openai/gpt-4o")
    assert text == "ok"
    assert recorded["model"] == "openai/gpt-4o"
    assert "api_base" not in recorded
    assert "api_key" not in recorded
    # no config -> nothing written to any usage log
    assert (instrumented_workspace / "misc" / "hacp" / "usage.jsonl").read_text() == ""
