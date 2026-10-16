"""Watcher: converts fabricated upstream filesystem state into events."""
import json

import pytest

from hacp.watcher import ExperimentWatcher


def _fabricate_outputs(ws):
    out = ws / "outputs" / "generate_demo"
    initial = out / "gen_initial"
    initial.mkdir(parents=True)
    (initial / "metadata.json").write_text(json.dumps({"gen_output_dir": str(initial), "run_eval": True}))
    (initial / "paper_review_eval").mkdir()
    (initial / "paper_review_eval" / "report.json").write_text(
        json.dumps({"overall_accuracy": 0.52})
    )
    (out / "archive.jsonl").write_text(
        json.dumps({"current_genid": "initial", "archive": ["initial"]}) + "\n"
    )
    with open(out / "generate.log", "a") as f:
        f.write("generation initial completed\n")
    return out


def _fabricate_gen1(out, accuracy=0.60):
    g1 = out / "gen_1"
    g1.mkdir(parents=True)
    (g1 / "metadata.json").write_text(
        json.dumps({"current_genid": 1, "parent_genid": "initial", "run_eval": True, "run_full_eval": True})
    )
    (g1 / "paper_review_eval").mkdir()
    (g1 / "paper_review_eval" / "report.json").write_text(json.dumps({"overall_accuracy": accuracy}))
    (g1 / "paper_review_eval_val").mkdir()
    (g1 / "paper_review_eval_val" / "report.json").write_text(json.dumps({"overall_accuracy": accuracy - 0.05}))
    with open(out / "archive.jsonl", "a") as f:
        f.write(json.dumps({"current_genid": 1, "archive": ["initial", 1]}) + "\n")
    with open(out / "generate.log", "a") as f:
        f.write("generation 1 completed, parent initial\n")
    return g1


def _fabricate_usage(ws, role="task", tokens=100):
    p = ws / "misc" / "hacp" / "usage.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps({
            "ts": 1700000000.0, "experiment_id": "e0", "role": role,
            "model": "stub/tiny", "prompt_tokens": tokens, "completion_tokens": 10,
            "total_tokens": tokens + 10, "latency_s": 0.01,
        }) + "\n")


def test_watcher_emits_events(store, bus, tmp_path):
    exp = store.create_experiment({"name": "w", "domains": ["paper_review"]})
    home = tmp_path
    ws = home / "experiments" / exp["id"] / "workspace"
    out = _fabricate_outputs(ws)

    w = ExperimentWatcher(exp["id"], store, bus, home)
    w.poll_once()

    types = [e["type"] for e in store.list_events(exp["id"])]
    assert "output_dir" in types
    assert "log" in types
    assert "archive" in types
    assert "generations" in types
    exp_row = store.get_experiment(exp["id"])
    assert exp_row["current_gen"] == "initial"
    assert exp_row["output_dir"] == str(out)

    # add a generation + usage -> new events on the next poll
    _fabricate_gen1(out)
    _fabricate_usage(ws)
    w.poll_once()
    gens_events = [e for e in store.list_events(exp["id"]) if e["type"] == "generations"]
    last = gens_events[-1]["data"]["generations"]
    gen1 = [g for g in last if g["gen"] == "1"][0]
    assert gen1["scores"]["paper_review"]["train"] == pytest.approx(0.60)
    assert gen1["scores"]["paper_review"]["val"] == pytest.approx(0.55)
    # local usage ingested for reconciliation (separate from proxy ledger)
    assert store.usage_totals(exp["id"], source="local")["requests"] == 1
    types = [e["type"] for e in store.list_events(exp["id"])]
    assert "usage_reconciled" in types
    exp_updated = store.get_experiment(exp["id"])
    assert exp_updated["current_gen"] == "1"


def test_watcher_survives_missing_workspace(store, bus, tmp_path):
    exp = store.create_experiment({"name": "w", "domains": ["paper_review"]})
    w = ExperimentWatcher(exp["id"], store, bus, tmp_path)
    w.poll_once()  # no exception
    assert [e["type"] for e in store.list_events(exp["id"])] == []
