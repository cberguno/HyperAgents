"""Store: experiments, events, usage ledger, budget state, slots, comparison."""
import time

import pytest

from hacp.store import Store


@pytest.fixture
def exp(store):
    return store.create_experiment(
        {
            "name": "test-exp",
            "domains": ["paper_review"],
            "meta_model": "anthropic/claude-sonnet-4-5-20250929",
            "task_model": "openai/gpt-4o-mini",
            "max_generation": 3,
            "budget": {"max_tokens": 1000, "max_cost_usd": 5.0},
        }
    )


def test_create_and_get(exp, store):
    got = store.get_experiment(exp["id"])
    assert got["name"] == "test-exp"
    assert got["status"] == "created"
    assert got["config"]["budget"]["max_tokens"] == 1000
    assert got["usage"]["total_tokens"] == 0


def test_usage_totals_and_budget(store, exp):
    # gpt-4o-mini: 0.15/M in, 0.60/M out
    store.add_usage(exp["id"], "task", "openai/gpt-4o-mini", 1_000_000, 1_000_000)
    totals = store.usage_totals(exp["id"])
    assert totals["total_tokens"] == 2_000_000
    assert totals["cost_usd"] == pytest.approx(0.75)
    bstate = store.budget_state(store.get_experiment(exp["id"]))
    assert bstate["exceeded"] is True
    assert bstate["exceeded_reason"] == "max_tokens"


def test_budget_cost_ceiling(store, exp):
    store.set_config(exp["id"], {**exp["config"], "budget": {"max_tokens": None, "max_cost_usd": 0.01}})
    store.add_usage(exp["id"], "meta", "openai/gpt-4o", 1_000_000, 0)  # $2.50 > $0.01
    bstate = store.budget_state(store.get_experiment(exp["id"]))
    assert bstate["exceeded_reason"] == "max_cost_usd"


def test_wallclock_ceiling(store, exp):
    store.set_config(exp["id"], {**exp["config"], "budget": {"max_tokens": None, "max_wallclock_hours": 0.00001}})
    store.update_experiment(exp["id"], started_at=time.time() - 10)
    bstate = store.budget_state(store.get_experiment(exp["id"]))
    assert bstate["exceeded_reason"] == "max_wallclock_hours"


def test_local_vs_proxy_ledger_separation(store, exp):
    store.add_usage(exp["id"], "task", "stub/x", 10, 10, source="proxy")
    store.add_usage(exp["id"], "task", "stub/x", 10, 10, source="local")
    assert store.usage_totals(exp["id"], source="proxy")["requests"] == 1
    assert store.usage_totals(exp["id"], source="local")["requests"] == 1


def test_slots(store):
    store.seed_slots(5)
    assert store.slot_count() == 5
    exp = store.create_experiment({"name": "a", "domains": ["paper_review"]})
    store.assign_slot(2, exp["id"])
    slots = store.list_slots()
    assert slots[2]["experiment"]["name"] == "a"
    assert slots[0]["experiment_id"] is None
    assert store.first_free_slot() == 0
    assert store.expand_slots(2) == 7
    assert store.slot_count() == 7


def test_events(store, exp):
    store.add_event(exp["id"], "started", {"pid": 1})
    store.add_event(exp["id"], "log", {"line": "hi"})
    store.add_event(None, "docker_cleanup", {"container": "c"})
    evs = store.list_events(exp["id"])
    assert [e["type"] for e in evs] == ["started", "log"]
    assert store.list_events(None, limit=10)[-1]["type"] == "docker_cleanup"


def test_comparison(store, exp):
    other = store.create_experiment({"name": "b", "domains": ["paper_review"]})
    store.add_usage(exp["id"], "task", "openai/gpt-4o-mini", 1000, 1000)
    result = store.comparison([exp["id"], other["id"], "missing"])
    assert len(result) == 2
    assert result[0]["usage"]["cost_usd"] > 0
    assert result[1]["name"] == "b"
