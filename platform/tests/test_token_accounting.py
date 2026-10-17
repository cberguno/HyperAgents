"""Token accounting: cached vs uncached tracked separately, end to end."""
import httpx


def test_proxy_ledger_separates_cached_and_uncached(client, store):
    exp = store.create_experiment({"name": "cached", "budget": {"max_tokens": 10_000_000}})
    r = client.post(
        "/proxy/v1/chat/completions",
        json={"model": "stub/tiny",
              "messages": [{"role": "user", "content": "x" * 4000}]},
        headers={"Authorization": f"Bearer hacp-{exp['id']}-task"},
    )
    assert r.status_code == 200
    usage = r.json()["usage"]
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    assert 0 < cached <= usage["prompt_tokens"]
    totals = store.usage_totals(exp["id"])
    assert totals["cached_tokens"] == cached
    assert totals["uncached_tokens"] == totals["prompt_tokens"] - cached


def test_usage_endpoint_reports_cached_breakdown(client, store):
    exp = store.create_experiment({"name": "cached2", "budget": {"max_tokens": 10_000_000}})
    for _ in range(3):
        client.post(
            "/proxy/v1/chat/completions",
            json={"model": "stub/tiny",
                  "messages": [{"role": "user", "content": "y" * 3000}]},
            headers={"Authorization": f"Bearer hacp-{exp['id']}-task"},
        )
    usage = client.get(f"/api/experiments/{exp['id']}/usage").json()["totals_proxy"]
    assert usage["cached_tokens"] > 0
    assert usage["uncached_tokens"] == usage["prompt_tokens"] - usage["cached_tokens"]


def test_local_usage_log_carries_cached_tokens(store):
    """The workspace usage.jsonl (in-workspace instrumentation) also records
    cached tokens; the watcher ingests them for reconciliation."""
    from hacp.watcher import ExperimentWatcher

    exp = store.create_experiment({"name": "cached3"})
    ws = store.db_file.parent / "experiments" / exp["id"] / "workspace" / "misc" / "hacp"
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "usage.jsonl").write_text(
        '{"ts": 1, "experiment_id": "%s", "role": "task", "model": "stub/tiny", '
        '"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010, '
        '"cached_tokens": 700, "latency_s": 0.1}\n' % exp["id"]
    )
    w = ExperimentWatcher(exp["id"], store, store.__class__.__dict__.get("_dummy_bus", None) and None,
                          store.db_file.parent)
    # run one poll without a bus (watcher tolerates it via try/except)
    import hacp.watcher as W

    class _Bus:
        def publish(self, *a, **k):
            pass

    w.bus = _Bus()
    w.poll_once()
    assert store.usage_totals(exp["id"], source="local")["cached_tokens"] == 700
