"""Cost report: per-phase/per-generation attribution via completion windows."""
import json
import os
import time

import pytest


def _mk_gen(out_dir, gen, mtime, parent=None, accuracy=None):
    g = out_dir / f"gen_{gen}"
    g.mkdir(parents=True, exist_ok=True)
    (g / "metadata.json").write_text(json.dumps({"current_genid": gen, "parent_genid": parent}))
    if accuracy is not None:
        ev = g / "paper_review_eval"
        ev.mkdir(exist_ok=True)
        (ev / "report.json").write_text(json.dumps({"overall_accuracy": accuracy}))
    os.utime(g / "metadata.json", (mtime, mtime))
    return g


def test_cost_report_attributes_by_generation_windows(client, store, tmp_path):
    eid = client.post("/api/experiments", json={"name": "cost-attr"}).json()["id"]
    # fabricate an output dir with two generations
    out_dir = tmp_path / "outputs" / "generate_x"
    out_dir.mkdir(parents=True)
    _mk_gen(out_dir, "initial", mtime=1000.0, accuracy=0.52)
    _mk_gen(out_dir, "1", mtime=2000.0, parent="initial", accuracy=0.55)
    _mk_gen(out_dir, "2", mtime=3000.0, parent="1", accuracy=0.58)
    store.update_experiment(eid, output_dir=str(out_dir))

    base = 500.0  # ts for usage rows
    # seed: ts <= 1000 ; gen1: (1000, 2000]; gen2: (2000, 3000]; after: >3000
    store.add_usage(eid, "task", "openai/gpt-4o-mini", 1000, 100, ts=600, generation=None)
    store.add_usage(eid, "meta", "anthropic/claude-sonnet-4-5-20250929", 2000, 200, ts=1500, generation=None)
    store.add_usage(eid, "task", "openai/gpt-4o-mini", 3000, 300, ts=2500, generation=None)
    store.add_usage(eid, "task", "openai/gpt-4o-mini", 10, 10, ts=2501, status="error", cost_usd=0.0)
    store.add_usage(eid, "task", "openai/gpt-4o-mini", 0, 0, ts=2502, status="blocked", cost_usd=0.0)

    r = client.get(f"/api/experiments/{eid}/cost_report")
    assert r.status_code == 200
    rep = r.json()
    phases = {p["phase"]: p for p in rep["phases"]}
    assert set(phases) == {"seed", "gen initial", "gen 1", "gen 2"}

    seed = phases["seed"]
    assert seed["totals"]["requests"] == 1 and seed["totals"]["total_tokens"] == 1100
    assert seed["models"][0]["model"] == "openai/gpt-4o-mini"
    # the sonnet meta-agent call (ts=1500) falls in the 'gen initial' window
    gen_initial = phases["gen initial"]
    assert gen_initial["totals"]["requests"] == 1
    assert gen_initial["models"][0]["model"] == "anthropic/claude-sonnet-4-5-20250929"
    assert gen_initial["models"][0]["cost_usd"] > 0.006  # 2.2k tokens at Sonnet rates
    gen1 = phases["gen 1"]
    assert gen1["totals"]["requests"] == 1
    assert gen1["models"][0]["model"] == "openai/gpt-4o-mini"
    assert gen1["totals"]["errors"] == 1 and gen1["totals"]["blocked"] == 1
    # role-level latency present
    assert any(rl["role"] == "task" for p in rep["phases"] for rl in p["roles"])

    grand = rep["grand_totals"]
    assert grand["requests"] == 3
    assert grand["errors"] == 1 and grand["blocked"] == 1
    assert grand["total_tokens"] == 1100 + 2200 + 3300
    # cost sums across phases
    assert grand["cost_usd"] == pytest.approx(
        sum(p["totals"]["cost_usd"] for p in rep["phases"]), abs=1e-6
    )


def test_cost_report_empty_experiment(client):
    eid = client.post("/api/experiments", json={"name": "cost-empty"}).json()["id"]
    rep = client.get(f"/api/experiments/{eid}/cost_report").json()
    assert rep["phases"][0]["phase"] == "seed"
    assert rep["grand_totals"]["requests"] == 0
    assert rep["grand_totals"]["in_flight_reserved_tokens"] == 0


def test_usage_totals_expose_errors_and_blocked(client, store):
    eid = client.post("/api/experiments", json={"name": "err-count"}).json()["id"]
    store.add_usage(eid, "task", "stub/tiny", 10, 10)
    store.add_usage(eid, "task", "stub/tiny", 0, 0, status="error")
    store.add_usage(eid, "task", "stub/tiny", 0, 0, status="blocked")
    usage = client.get(f"/api/experiments/{eid}/usage").json()
    assert usage["totals_proxy"]["errors"] == 1
    assert usage["in_flight"]["active_requests"] == 0


def test_usage_migration_adds_generation_column(tmp_path):
    """Databases created before per-generation tagging get migrated in place."""
    import sqlite3

    from hacp.store import Store

    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE usage (id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id TEXT,
        ts REAL, role TEXT, model TEXT, prompt_tokens INTEGER, completion_tokens INTEGER,
        total_tokens INTEGER, cost_usd REAL, latency_s REAL, status TEXT, source TEXT)"""
    )
    conn.execute(
        "INSERT INTO usage (experiment_id, ts, role, model, prompt_tokens, completion_tokens, total_tokens, cost_usd, status, source) "
        "VALUES ('e1', 1, 'task', 'stub/tiny', 1, 1, 2, 0, 'ok', 'proxy')"
    )
    conn.commit()
    conn.close()

    s = Store(db)  # opens existing -> migration runs
    rows = s.list_usage("e1")
    assert len(rows) == 1 and rows[0]["generation"] is None
    s.add_usage("e1", "task", "stub/tiny", 1, 1, generation="1")
    rows = s.list_usage("e1")
    assert rows[-1]["generation"] == "1"
    s.close()
