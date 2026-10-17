"""Option C — env-var routing: instrumentation config from the process
environment cannot be removed by file operations.

Run e6e3168d3b3ca gen 2 deleted platform_config.json + misc/hacp/ from
inside its container. These tests prove env-based routing survives exactly
that attack, and that without env vars and files the patch is inert
(upstream behavior).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from hacp.instrumentation import apply_instrumentation, verify_instrumentation

META_DRIVER = """
import sys
sys.path.insert(0, ".")
from agent.llm import get_response_from_llm, _hacp_routing_active
print("ROUTING-ACTIVE:", _hacp_routing_active())
text, hist, info = get_response_from_llm("hello env routing", model="anthropic/claude-3-haiku-20240307")
print(text)
"""

META_DRIVER_NAME = "run_meta_agent_envsim.py"


@pytest.fixture
def env_ws(tmp_path, upstream_repo):
    ws = tmp_path / "workspace"
    (ws / "agent").mkdir(parents=True)
    shutil.copy(upstream_repo / "agent" / "llm.py", ws / "agent" / "llm.py")
    apply_instrumentation(ws)
    assert verify_instrumentation(ws)
    (ws / META_DRIVER_NAME).write_text(META_DRIVER)
    return ws


def _base_env():
    from hacp.runner import ExperimentRunner  # noqa: F401

    class _D:
        pass

    return ExperimentRunner._subprocess_env(_D())


def test_env_routing_survives_agent_deleting_all_config_files(env_ws, live_server):
    """The exact gen-2 attack: agent deletes every platform file. Env vars
    injected by the platform keep routing + accounting intact."""
    client = httpx.Client(base_url=live_server, timeout=30)
    exp = client.post(
        "/api/experiments",
        json={"name": "env-attack", "loop": "stub", "meta_model": "stub/tiny",
              "task_model": "stub/tiny"},
    ).json()

    # the 'attacker' removes every config artifact from its workspace
    (env_ws / "platform_config.json").unlink(missing_ok=True)
    shutil.rmtree(env_ws / "misc", ignore_errors=True)

    # platform env (as injected by the Docker socket proxy + runner):
    env = _base_env()
    env.update({
        "HACP_EXPERIMENT_ID": exp["id"],
        "HACP_LLMBASE": f"{live_server}/proxy/v1",
        "HACP_META_MODEL": "stub/tiny",
        "HACP_TASK_MODEL": "stub/tiny",
        "HACP_META_TOKEN": f"hacp-{exp['id']}-meta",
        "HACP_TASK_TOKEN": f"hacp-{exp['id']}-task",
        "HACP_USAGE_LOG": "misc/hacp/usage.jsonl",
        "HACP_ROUTING_ACTIVE": "1",
    })
    r = subprocess.run(
        [sys.executable, META_DRIVER_NAME], cwd=str(env_ws), env=env,
        capture_output=True, text=True, timeout=180,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ROUTING-ACTIVE: True" in r.stdout
    assert "[stub:stub/tiny]" in r.stdout  # routed through the platform proxy
    # and the call was accounted in the authoritative ledger
    usage = client.get(f"/api/experiments/{exp['id']}/usage").json()
    assert usage["totals_proxy"]["requests"] == 1
    assert usage["totals_proxy"]["total_tokens"] > 0


def test_without_env_and_files_patch_is_inert(env_ws):
    """No env vars, no config files: patched code behaves like upstream
    (no proxy routing, model passed through unchanged)."""
    r = subprocess.run(
        [sys.executable, META_DRIVER_NAME], cwd=str(env_ws),
        env={k: v for k, v in os.environ.items()
             if not k.startswith("HACP_")},
        capture_output=True, text=True, timeout=60,
    )
    assert "ROUTING-ACTIVE: False" in r.stdout
    # upstream direct litellm call: no provider key in env -> fails (proving
    # no proxy routing), unlike the stub path which never fails
    assert r.returncode != 0


def test_env_overrides_stale_files(env_ws, live_server):
    """Env wins over file values (e.g. agent tampered platform_config.json to
    point elsewhere; platform env still routes correctly)."""
    client = httpx.Client(base_url=live_server, timeout=30)
    exp = client.post("/api/experiments", json={"name": "env-wins", "loop": "stub"}).json()
    # tampered file: attacker endpoint
    (env_ws / "platform_config.json").write_text(json.dumps({
        "llm_api_base": "http://127.0.0.1:1/v1",
        "meta_model": "stub/tiny",
        "meta_token": "x",
    }))
    env = _base_env()
    env.update({
        "HACP_EXPERIMENT_ID": exp["id"],
        "HACP_LLMBASE": f"{live_server}/proxy/v1",
        "HACP_META_MODEL": "stub/tiny",
        "HACP_META_TOKEN": f"hacp-{exp['id']}-meta",
        "HACP_USAGE_LOG": "misc/hacp/usage.jsonl",
    })
    r = subprocess.run(
        [sys.executable, META_DRIVER_NAME], cwd=str(env_ws), env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[stub:stub/tiny]" in r.stdout  # env routing won over the tampered file


def test_reasoning_effort_forwarded(tmp_path, upstream_repo, monkeypatch):
    """Configured reasoning effort (e.g. GPT-6 Luna at 'max') reaches the
    completion kwargs via env config."""
    ws = tmp_path / "workspace"
    (ws / "agent").mkdir(parents=True)
    shutil.copy(upstream_repo / "agent" / "llm.py", ws / "agent" / "llm.py")
    apply_instrumentation(ws)
    (ws / META_DRIVER_NAME).write_text(META_DRIVER)

    import importlib.util

    spec = importlib.util.spec_from_file_location("llm_env_effort", ws / "agent" / "llm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.setenv("HACP_LLMBASE", "http://127.0.0.1:9/proxy/v1")
    monkeypatch.setenv("HACP_META_MODEL", "openai/gpt-6-luna")
    monkeypatch.setenv("HACP_META_TOKEN", "hacp-e-test-meta")
    monkeypatch.setenv("HACP_META_REASONING", "max")
    monkeypatch.setattr(mod._hacp_sys, "argv", ["run_meta_agent_sim.py"])
    mod._HACP_CFG = None
    mod._HACP_CFG_LOADED = False

    override, extra = mod._hacp_before_completion("openai/gpt-4o")
    assert override == "openai/openai/gpt-6-luna"
    assert extra["api_base"] == "http://127.0.0.1:9/proxy/v1"
    assert extra["api_key"] == "hacp-e-test-meta"
    assert extra["reasoning_effort"] == "max"

    # role override: task role uses its own model + effort
    monkeypatch.setenv("HACP_TASK_MODEL", "deepseek/deepseek-v4.1-flash")
    monkeypatch.setenv("HACP_TASK_REASONING", "")
    mod._HACP_CFG = None
    mod._HACP_CFG_LOADED = False
    monkeypatch.setattr(mod._hacp_sys, "argv", ["domains.harness"])
    override2, extra2 = mod._hacp_before_completion("openai/gpt-4o")
    assert override2 == "openai/deepseek/deepseek-v4.1-flash"
    assert "reasoning_effort" not in extra2  # empty effort -> not forwarded


def test_runner_container_env_payload(service, upstream_repo):
    """The exact env payload the runner injects into containers via the proxy."""
    from hacp.config import ExperimentConfig
    from hacp.runner import ExperimentRunner

    store = None  # not needed for the helper
    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.service = service
    cfg = ExperimentConfig(
        name="payload", meta_model="openai/gpt-6-luna",
        task_model="deepseek/deepseek-v4.1-flash",
        meta_reasoning_effort="max",
    )
    payload = runner._container_env("e0000000test", cfg)
    assert payload["HACP_LLMBASE"] == service.proxy_v1_base
    assert payload["HACP_META_MODEL"] == "openai/gpt-6-luna"
    assert payload["HACP_TASK_MODEL"] == "deepseek/deepseek-v4.1-flash"
    assert payload["HACP_META_TOKEN"] == "hacp-e0000000test-meta"
    assert payload["HACP_META_REASONING"] == "max"
    assert payload["HACP_TASK_REASONING"] == ""
    assert payload["HACP_ROUTING_ACTIVE"] == "1"
    assert payload["HACP_EXPERIMENT_ID"] == "e0000000test"
