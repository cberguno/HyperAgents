"""Budget protection under concurrency + reservation semantics.

The core race: N concurrent requests all pass `committed < limit` before any
completes, then all succeed -> overshoot. Reservations close it: admission
checks committed + reserved, so concurrent bursts admit only what fits.
"""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from hacp.budget import RESERVATIONS, Reservations, combined_budget_state


def _post(client, url, token, model="stub/tiny", content="hello", max_tokens=None):
    payload = {"model": model, "messages": [{"role": "user", "content": content}]}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    return client.post(url, json=payload, headers={"Authorization": f"Bearer {token}"})


def test_concurrent_burst_cannot_overshoot_token_budget(live_server, service):
    """10 concurrent requests with a budget for ~2: exactly ~2 succeed, the
    rest are 429 — the overshoot race is closed by reservations."""
    client = httpx.Client(base_url=live_server, timeout=60)
    # each stub call reports ~sum(len/4 + 8) tokens; with reservations the
    # admission uses prompt-est*1.5 + max_tokens. Use max_tokens to control it.
    exp = client.post("/api/experiments", json={
        "name": "race-test", "loop": "stub",
        "budget": {"max_tokens": 100, "max_cost_usd": None, "max_wallclock_hours": None},
    }).json()
    token = f"hacp-{exp['id']}-task"
    # reservation per request = 1.5*(4+8) + 20 = ~38; budget 100 -> ~2 admitted
    def fire(i):
        c = httpx.Client(base_url=live_server, timeout=60)
        r = _post(c, "/proxy/v1/chat/completions", token, content="abcd", max_tokens=20)
        c.close()
        return r.status_code

    with ThreadPoolExecutor(max_workers=10) as ex:
        statuses = list(ex.map(fire, range(10)))
    ok = sum(1 for s in statuses if s == 200)
    denied = sum(1 for s in statuses if s == 429)
    assert ok + denied == 10
    # deterministic bound: each reservation is ~33 tokens vs a 100-token cap,
    # so at most 3 calls can ever be admitted regardless of timing
    assert 2 <= ok <= 3, f"overshoot: {ok} succeeded against a 100-token budget: {statuses}"
    assert denied >= 7, f"expected most requests denied, got {statuses}"
    # committed tokens stay within the cap (actual usage <= reservation each)
    totals = client.get(f"/api/experiments/{exp['id']}/usage").json()["totals_proxy"]
    assert totals["total_tokens"] <= 100, totals


def test_reservations_released_after_completion(live_server, service):
    """After requests complete, no phantom reservations block later calls."""
    client = httpx.Client(base_url=live_server, timeout=60)
    exp = client.post("/api/experiments", json={
        "name": "release-test", "loop": "stub", "budget": {"max_tokens": 100_000},
    }).json()
    token = f"hacp-{exp['id']}-task"
    for _ in range(5):
        r = _post(client, "/proxy/v1/chat/completions", token, max_tokens=50)
        assert r.status_code == 200
    assert RESERVATIONS.active_count(exp["id"]) == 0
    assert RESERVATIONS.totals(exp["id"]) == (0, 0.0)
    # usage endpoint exposes in-flight info
    inflight = client.get(f"/api/experiments/{exp['id']}/usage").json()["in_flight"]
    assert inflight["active_requests"] == 0


def test_unknown_pricing_cannot_bypass_dollar_budget(client, store):
    """A model absent from the pricing registry would accrue $0 cost — the
    dollar limit alone could be bypassed. The proxy must refuse it when a
    dollar budget is configured (tokens-only budgets stay safe: tokens count)."""
    exp = store.create_experiment(
        {"name": "unpriced-dollar", "budget": {"max_cost_usd": 5, "max_tokens": None}}
    )
    r = _post(client, "/proxy/v1/chat/completions", f"hacp-{exp['id']}-task",
              model="attacker/free-model")
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "hacp_unknown_pricing"
    # and with a tokens-only budget the same model is allowed (tokens enforce)
    exp2 = store.create_experiment({"name": "unpriced-tokens", "budget": {"max_tokens": 1_000_000}})
    r2 = _post(client, "/proxy/v1/chat/completions", f"hacp-{exp2['id']}-task",
               model="attacker/free-model")
    assert r2.status_code == 200
    assert store.usage_totals(exp2["id"])["total_tokens"] > 0


def test_combined_budget_state_includes_reservations(store):
    exp = store.create_experiment({"name": "combined", "budget": {"max_tokens": 1000}})
    store.add_usage(exp["id"], "task", "stub/tiny", 500, 0)
    RESERVATIONS.reserve(exp["id"], 600, 0.0)
    try:
        e = store.get_experiment(exp["id"])
        state = combined_budget_state(store, e)
        assert state["reserved_tokens"] == 600
        assert state["exceeded"] is True and state["exceeded_reason"] == "max_tokens"
        # plain committed-only view is not yet exceeded (500 < 1000)
        assert store.budget_state(e)["exceeded"] is False
    finally:
        RESERVATIONS.release(exp["id"], list(RESERVATIONS._per_exp.get(exp["id"], {}))[0])


def test_reservation_ttl_expires(store):
    res = Reservations(ttl_s=0.05)
    rid = res.reserve("e1", 1000, 1.0)
    assert rid
    import time
    time.sleep(0.1)
    assert res.totals("e1") == (0, 0.0)
    assert res.active_count("e1") == 0


def test_budget_pause_includes_inflight(store):
    """Runner supervisor view: with in-flight reservations counted, the pause
    triggers before completions land (no surprise overshoot after a burst)."""
    exp = store.create_experiment({"name": "pause-view", "budget": {"max_cost_usd": 2.0}})
    # committed $1.50 (10M prompt tokens of gpt-4o-mini at $0.15/M)
    store.add_usage(exp["id"], "task", "openai/gpt-4o-mini", 10_000_000, 0)
    # in-flight reservation of $0.50 -> over the $2 cap before completion
    rid = RESERVATIONS.reserve(exp["id"], 0, 0.5)
    try:
        e = store.get_experiment(exp["id"])
        state = combined_budget_state(store, e)
        assert state["reserved_cost_usd"] == 0.5
        assert state["exceeded"] is True and state["exceeded_reason"] == "max_cost_usd"
        # committed-only view is not yet exceeded ($1.50 < $2)
        assert store.budget_state(e)["exceeded"] is False
    finally:
        RESERVATIONS.release(exp["id"], rid)
