"""Proxy: token auth, stub upstream, usage accounting, budget kill-switch (429)."""
import pytest

from hacp.store import Store


def _post_chat(client, token, model="stub/tiny", messages=None):
    return client.post(
        "/proxy/v1/chat/completions",
        json={"model": model, "messages": messages or [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {token}"},
    )


def test_missing_token_401(client):
    r = client.post("/proxy/v1/chat/completions", json={"model": "stub/tiny", "messages": []})
    assert r.status_code == 401


def test_invalid_token_401(client):
    r = _post_chat(client, "not-a-token")
    assert r.status_code == 401


def test_unknown_experiment_token_401(client, store):
    r = _post_chat(client, "hacp-e00000000000-meta")
    assert r.status_code == 401


def test_stub_completion_records_usage(client, store):
    exp = store.create_experiment({"name": "x", "domains": ["paper_review"]})
    r = _post_chat(client, f"hacp-{exp['id']}-meta", model="anthropic/claude-sonnet-4-5-20250929")
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"].startswith("[stub:")
    usage = body["usage"]
    assert usage["total_tokens"] > 0
    totals = store.usage_totals(exp["id"])
    assert totals["requests"] == 1
    assert totals["prompt_tokens"] == usage["prompt_tokens"]
    # role recorded from the token (meta) with real model pricing
    breakdown = store.usage_breakdown(exp["id"])
    assert breakdown[0]["role"] == "meta"
    assert breakdown[0]["model"] == "anthropic/claude-sonnet-4-5-20250929"
    assert breakdown[0]["cost_usd"] > 0  # priced at Sonnet 4.5 rates


def test_shell_token_is_platform_level(client, store):
    r = _post_chat(client, "hacp-platform-shell")
    assert r.status_code == 200
    assert store.usage_totals(None)["requests"] == 1


def test_budget_kill_switch_429(client, store):
    exp = store.create_experiment(
        {"name": "x", "domains": ["paper_review"], "budget": {"max_tokens": 10}}
    )
    store.add_usage(exp["id"], "task", "stub/tiny", 100, 0)  # already over
    r = _post_chat(client, f"hacp-{exp['id']}-task")
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "hacp_budget_exhausted"
    events = store.list_events(exp["id"])
    assert any(e["type"] == "budget_blocked" for e in events)
    # blocked attempts are recorded with zero tokens, not charged
    totals = store.usage_totals(exp["id"])
    assert totals["requests"] == 1


def test_model_enforcement_optional(client, store, service):
    import hacp.api as api  # noqa: F401

    # service.enforce_models defaults False -> any model passes
    exp = store.create_experiment(
        {"name": "x", "domains": ["paper_review"],
         "meta_model": "openai/gpt-4o-mini", "task_model": "openai/gpt-4o"}
    )
    r = _post_chat(client, f"hacp-{exp['id']}-task", model="totally/wrong-model")
    assert r.status_code == 200
    assert store.usage_breakdown(exp["id"])[0]["model"] == "totally/wrong-model"


def test_streaming_rejected_by_stub(client, store):
    exp = store.create_experiment({"name": "x", "domains": ["paper_review"]})
    r = client.post(
        "/proxy/v1/chat/completions",
        json={"model": "stub/tiny", "messages": [], "stream": True},
        headers={"Authorization": f"Bearer hacp-{exp['id']}-task"},
    )
    assert r.status_code == 400
