"""Provider verification endpoint: nonbillable OpenRouter checks (mocked HTTP)."""
import json

import httpx
import pytest

from hacp.config import ServiceConfig


def _make(service):
    from fastapi.testclient import TestClient

    from hacp.api import create_app

    app = create_app(service)
    return TestClient(app), app


def test_verify_provider_stub_mode_returns_hint(tmp_path):
    service = ServiceConfig(host="127.0.0.1", port=1, home=str(tmp_path),
                            upstream="stub", upstream_repo=".",
                            proxy_public_base="http://x")
    c, _ = _make(service)
    r = c.post("/api/admin/verify-provider")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False and body["upstream"] == "stub"
    assert "HACP_UPSTREAM=openrouter" in body["hint"]


def test_verify_provider_missing_key(tmp_path, monkeypatch):
    service = ServiceConfig(host="127.0.0.1", port=1, home=str(tmp_path),
                            upstream="openrouter", upstream_repo=".",
                            proxy_public_base="http://x")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # isolate from any real platform-home/secrets.env on this machine
    monkeypatch.setenv("HACP_HOME", str(tmp_path))
    monkeypatch.setattr("hacp.paths.platform_home", lambda home=None: tmp_path)
    c, _ = _make(service)
    r = c.post("/api/admin/verify-provider")
    body = r.json()
    assert body["ok"] is False and body["error"] == "missing_key"
    assert "secrets.env" in body["hint"]


def test_verify_provider_full_check(tmp_path, monkeypatch):
    service = ServiceConfig(host="127.0.0.1", port=1, home=str(tmp_path),
                            upstream="openrouter", upstream_repo=".",
                            proxy_public_base="http://x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-not-real")
    c, app = _make(service)

    # Create an experiment so its models are verified too
    c.post("/api/experiments", json={
        "name": "verify-exp",
        "meta_model": "anthropic/claude-sonnet-4-5-20250929",
        "task_model": "openai/gpt-4o-mini",
    })

    async def fake_get(self, url, **kw):
        if url == "/key":
            return httpx.Response(200, json={"data": {"label": "test-key", "usage": 0, "limit": None}})
        if url == "/models":
            data = [
                {"id": "anthropic/claude-sonnet-4.5",
                 "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                 "context_length": 200000},
                {"id": "openai/gpt-4o-mini",
                 "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
                 "context_length": 128000},
            ]
            return httpx.Response(200, json={"data": data})
        return httpx.Response(404)

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    r = c.post("/api/admin/verify-provider")
    body = r.json()
    assert body["ok"] is True and body["key_valid"] is True
    assert body["key_label"] == "test-key"
    assert body["all_available"] is True
    by_slug = {m["slug"]: m for m in body["models_checked"]}
    sonnet = by_slug["anthropic/claude-sonnet-4.5"]
    assert sonnet["available"] is True
    assert sonnet["live_input_usd_per_mtok"] == 3.0
    assert sonnet["pricing_matches_registry"] is True
    assert sonnet["context_length"] == 200000
    mini = by_slug["openai/gpt-4o-mini"]
    assert mini["live_output_usd_per_mtok"] == 0.6
    assert mini["pricing_matches_registry"] is True
    # the API key never appears in the response
    assert "sk-or-test-not-real" not in json.dumps(body)


def test_verify_provider_flags_missing_model_and_price_drift(tmp_path, monkeypatch):
    service = ServiceConfig(host="127.0.0.1", port=1, home=str(tmp_path),
                            upstream="openrouter", upstream_repo=".",
                            proxy_public_base="http://x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-not-real")
    c, _ = _make(service)

    async def fake_get(self, url, **kw):
        if url == "/key":
            return httpx.Response(200, json={"data": {"label": "k"}})
        if url == "/models":
            # gpt-4o-mini missing; sonnet price doubled -> drift flagged
            return httpx.Response(200, json={"data": [
                {"id": "anthropic/claude-sonnet-4.5",
                 "pricing": {"prompt": "0.000006", "completion": "0.00003"}},
            ]})
        return httpx.Response(404)

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    body = c.post("/api/admin/verify-provider").json()
    assert body["all_available"] is False
    by_slug = {m["slug"]: m for m in body["models_checked"]}
    assert by_slug["openai/gpt-4o-mini"]["available"] is False
    assert by_slug["anthropic/claude-sonnet-4.5"]["pricing_matches_registry"] is False
    assert by_slug["anthropic/claude-sonnet-4.5"]["live_input_usd_per_mtok"] == 6.0


def test_openrouter_key_read_from_secrets_file(tmp_path, monkeypatch):
    """The key can also arrive via platform-home/secrets.env (gitignored)."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    (home / "secrets.env").write_text('OPENROUTER_API_KEY="sk-or-file-key"\n')
    monkeypatch.setenv("HACP_HOME", str(home))
    from hacp.proxy import _read_openrouter_key

    assert _read_openrouter_key() == "sk-or-file-key"
    # env var still wins
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-key")
    assert _read_openrouter_key() == "sk-or-env-key"
