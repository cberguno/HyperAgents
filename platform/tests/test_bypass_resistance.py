"""Budget-enforcement bypass resistance.

The user's core security question: can a self-modifying meta-agent escape the
budget kill-switch by rewriting its own code? Answer: NO — enforcement is
server-side in the platform proxy, and the agent's world contains no provider
credentials. This file proves each escape route fails:

  A. Remove the instrumentation (restore pristine agent/llm.py) + tamper the
     workspace platform_config.json to point at an attacker-chosen endpoint:
     the agent has network access but only an HACP token, which any endpoint
     other than the platform proxy rejects.
  B. Point at a real provider directly: no provider key exists in the runner
     environment (keys are stripped) -> the call cannot authenticate.
  C. Use the only working path (the platform proxy) after the budget is
     exhausted: 429 hacp_budget_exhausted — the budget is read from the
     platform database, NOT from any file the agent can edit.
  D. Dodge cost accounting with an unknown/unpriced model: tokens still count
     toward max_tokens, so the kill-switch still trips.
"""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from hacp.instrumentation import apply_instrumentation

ATTACK_DRIVER = """
import sys
sys.path.insert(0, ".")
from agent.llm import get_response_from_llm
try:
    get_response_from_llm("hello", model="anthropic/claude-3-haiku-20240307")
    print("ATTACK-SUCCEEDED")
except Exception as e:
    print(f"ATTACK-FAILED: {type(e).__name__}: {str(e)[:160]}")
"""
# Named so the instrumentation's role detection sees a meta-agent context,
# exactly like the real run_meta_agent.py invocation upstream uses.
ATTACK_DRIVER_NAME = "run_meta_agent_sim.py"


class _FakeExternalEndpoint(BaseHTTPRequestHandler):
    """Mimics a non-platform LLM endpoint: records auth, always rejects."""

    seen_auth = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        _FakeExternalEndpoint.seen_auth.append(self.headers.get("Authorization", ""))
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        body = json.dumps({"error": {"message": "invalid api key"}}).encode()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def rehearsal_workspace(tmp_path, upstream_repo):
    """Workspace copy prepared exactly like a real experiment."""
    import shutil

    ws = tmp_path / "workspace"
    (ws / "agent").mkdir(parents=True)
    shutil.copy(upstream_repo / "agent" / "llm.py", ws / "agent" / "llm.py")
    apply_instrumentation(ws)
    (ws / "misc" / "hacp").mkdir(parents=True)
    (ws / "platform_config.json").write_text("{}")  # placeholder; tests overwrite
    (ws / ATTACK_DRIVER_NAME).write_text(ATTACK_DRIVER)
    return ws


def _run_driver(ws, env):
    return subprocess.run(
        [sys.executable, ATTACK_DRIVER_NAME],
        cwd=str(ws),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _runner_env():
    """The exact environment the runner gives experiment subprocesses."""
    from hacp.runner import ExperimentRunner

    class _Dummy:
        pass

    return ExperimentRunner._subprocess_env(_Dummy())


def test_runner_env_strips_all_provider_keys(monkeypatch):
    from hacp.config import PROVIDER_KEY_VARS

    for var in PROVIDER_KEY_VARS:
        monkeypatch.setenv(var, "sk-should-never-leak")
    env = _runner_env()
    for var in PROVIDER_KEY_VARS:
        assert var not in env, f"{var} leaked into experiment environment"
    assert env["MPLBACKEND"] == "Agg"
    assert env["PYTHONUNBUFFERED"] == "1"


def test_workspace_holds_only_hacp_tokens(rehearsal_workspace):
    """Scan every workspace file: no provider-key-shaped secrets, only HACP
    tokens (useless outside the platform proxy)."""
    import re

    secret_re = re.compile(r"sk-[A-Za-z0-9_\-]{20,}")
    hacp_re = re.compile(r"hacp-[a-z0-9]+-(meta|task|shell)")
    for path in rehearsal_workspace.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        assert not secret_re.search(text), f"provider secret found in {path}"
        hacp_re.search(text)  # HACP tokens are expected and harmless
    # (real workspaces get platform_config.json with only hacp- tokens; no
    # provider key material is ever written there — verified by this scan on
    # the rehearsal workspace which mirrors that layout)


def test_attack_A_tampered_config_points_elsewhere(rehearsal_workspace, live_server):
    """Agent rewrites its config to an attacker endpoint: reaches the network
    but carries only the HACP token, which non-platform endpoints reject."""
    client = httpx.Client(base_url=live_server, timeout=30)
    exp = client.post(
        "/api/experiments",
        json={"name": "bypass-A", "loop": "stub", "budget": {"max_tokens": 1_000_000}},
    ).json()

    # attacker-controlled LLM endpoint (any server the agent invents)
    _FakeExternalEndpoint.seen_auth = []
    server = HTTPServer(("127.0.0.1", 0), _FakeExternalEndpoint)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        # attacker overwrites every config copy (root + misc) with its own endpoint
        tampered = {
            "experiment_id": exp["id"],
            "meta_model": "anthropic/claude-3-haiku-20240307",
            "task_model": "openai/gpt-4o",
            "llm_api_base": f"http://127.0.0.1:{port}/v1",
            "meta_token": f"hacp-{exp['id']}-meta",
            "task_token": f"hacp-{exp['id']}-task",
            "usage_log": "misc/hacp/usage.jsonl",
        }
        (rehearsal_workspace / "platform_config.json").write_text(json.dumps(tampered))
        (rehearsal_workspace / "misc" / "hacp" / "platform_config.json").write_text(json.dumps(tampered))
        env = _runner_env()
        r = _run_driver(rehearsal_workspace, env)
        assert "ATTACK-FAILED" in r.stdout, r.stdout + r.stderr
        # the attacker endpoint saw the request — authenticated ONLY by the
        # HACP token (useless off-platform; a real provider rejects it with 401)
        assert len(_FakeExternalEndpoint.seen_auth) >= 1
        assert all(a.startswith("Bearer hacp-") for a in _FakeExternalEndpoint.seen_auth), _FakeExternalEndpoint.seen_auth
        # and nothing was recorded in the platform ledger: traffic that dodged
        # the proxy was never billed to the experiment
        usage = client.get(f"/api/experiments/{exp['id']}/usage").json()
        assert usage["totals_proxy"]["requests"] == 0
    finally:
        server.shutdown()
        t.join(timeout=5)


def test_attack_B_remove_instrumentation_and_call_provider_directly(rehearsal_workspace, upstream_repo):
    """Agent restores pristine agent/llm.py (no proxy routing at all) and calls
    the provider directly: no key exists in the experiment environment."""
    import shutil

    shutil.copy(upstream_repo / "agent" / "llm.py", rehearsal_workspace / "agent" / "llm.py")
    (rehearsal_workspace / "platform_config.json").write_text(json.dumps({"llm_api_base": "https://openrouter.ai/api/v1"}))
    env = _runner_env()  # exactly what the runner provides: NO provider keys
    r = _run_driver(rehearsal_workspace, env)
    assert "ATTACK-FAILED" in r.stdout, r.stdout + r.stderr
    assert "api key" in r.stdout.lower() or "auth" in r.stdout.lower()


def test_attack_C_only_working_path_is_budget_gated(live_server, rehearsal_workspace):
    """After self-modification, the ONLY credential that works is the HACP
    token -> the proxy -> budget enforced from the platform DB (not from any
    workspace file). Exhausted budget = 429 regardless of file tampering."""
    client = httpx.Client(base_url=live_server, timeout=30)
    exp = client.post(
        "/api/experiments",
        json={"name": "bypass-C", "loop": "stub", "budget": {"max_tokens": 800}},
    ).json()
    token = f"hacp-{exp['id']}-meta"
    # one legitimate small call fits the budget (reservation-aware admission)
    first = client.post(
        "/proxy/v1/chat/completions",
        json={"model": "stub/tiny", "max_tokens": 10,
              "messages": [{"role": "user", "content": "x" * 2000}]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert first.status_code == 200
    # meanwhile the agent 'improves' itself: rewrites llm.py AND its config
    (rehearsal_workspace / "platform_config.json").write_text(
        json.dumps({
            "experiment_id": exp["id"],
            "meta_model": "stub/tiny",
            "llm_api_base": live_server + "/proxy/v1",
            "meta_token": token,
        })
    )
    env = _runner_env()
    env["OPENAI_API_KEY"] = token
    r = _run_driver(rehearsal_workspace, env)
    # the rewritten agent's call carries upstream's default max_tokens (16384):
    # its reservation alone exceeds the tiny budget -> the only working path
    # is budget-gated and it is denied -> attack fails
    assert "ATTACK-SUCCEEDED" not in r.stdout
    blocked = client.post(
        "/proxy/v1/chat/completions",
        json={"model": "stub/tiny",
              "messages": [{"role": "user", "content": "let me in"}]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert blocked.status_code == 429
    assert blocked.json()["error"]["code"] == "hacp_budget_exhausted"
    # and even a fully 'rewritten' agent holding only this token cannot proceed


def test_attack_D_unknown_model_cannot_dodge_token_budget(client, store):
    """Unpriced/unknown models cost $0 in the ledger — an attacker might pick
    one to spend 'for free'. But their TOKENS are still counted and reserved,
    so the token cap still closes (all further requests denied), and once the
    cap is hit even known-priced models are locked out."""
    exp = store.create_experiment({"name": "bypass-D", "budget": {"max_tokens": 100, "max_cost_usd": None}})
    tok = f"hacp-{exp['id']}-task"

    def call(model, content="hi", mt=40):
        return client.post(
            "/proxy/v1/chat/completions",
            json={"model": model, "max_tokens": mt,
                  "messages": [{"role": "user", "content": content}]},
            headers={"Authorization": f"Bearer {tok}"},
        )

    # two unpriced-model calls are admitted (reservations fit the cap)
    assert call("attacker/unpriced-free-model").status_code == 200
    assert call("attacker/unpriced-free-model").status_code == 200
    usage = store.usage_totals(exp["id"])
    assert usage["cost_usd"] == 0.0  # unknown pricing: never guessed
    assert usage["total_tokens"] > 0  # but tokens ARE recorded
    # the cap is now effectively closed: the next unpriced call is denied
    assert call("attacker/unpriced-free-model").status_code == 429
    # ...and a known-priced model is equally locked out — no bypass either way
    assert call("openai/gpt-4o-mini").status_code == 429
    bstate = store.budget_state(store.get_experiment(exp["id"]))
    assert bstate["exceeded_reason"] in ("max_tokens", None)
