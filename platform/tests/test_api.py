"""API surface: slots, experiments CRUD, models, comparison, shell injection."""
import json

from hacp.llm import FakeLLMClient
from hacp.shell import ShellAgent


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["upstream"] == "stub"
    assert body["slots"] == 5  # five initial slots


def test_expandable_slots(client):
    # "+" button foundation: 5 initial slots, expandable
    assert len(client.get("/api/slots").json()["slots"]) == 5
    r = client.post("/api/slots/expand?count=3")
    assert r.json()["slots"] == 8
    slots = client.get("/api/slots").json()["slots"]
    assert all(s["experiment_id"] is None for s in slots)


def test_create_experiment_fills_first_free_slot(client):
    r = client.post(
        "/api/experiments",
        json={"name": "exp-a", "domains": ["paper_review"],
              "meta_model": "anthropic/claude-sonnet-4-5-20250929",
              "task_model": "openai/gpt-4o-mini",
              "budget": {"max_cost_usd": 25}},
    )
    assert r.status_code == 200
    eid = r.json()["id"]
    slots = client.get("/api/slots").json()["slots"]
    assert slots[0]["experiment"]["id"] == eid
    assert slots[0]["experiment"]["budget"]["max_cost_usd"] == 25

    # second experiment -> next slot
    r = client.post("/api/experiments", json={"name": "exp-b"})
    eid2 = r.json()["id"]
    slots = client.get("/api/slots").json()["slots"]
    assert slots[1]["experiment"]["id"] == eid2


def test_invalid_config_rejected(client):
    r = client.post("/api/experiments", json={"name": "", "domains": ["paper_review"]})
    assert r.status_code == 400
    r = client.post("/api/experiments", json={"name": "x", "parent_selection": "bogus"})
    assert r.status_code == 400


def test_generate_loop_args_mirror_upstream(client):
    """The config maps 1:1 onto unmodified upstream generate_loop.py flags."""
    r = client.post(
        "/api/experiments",
        json={"name": "mirror", "max_generation": 7, "parent_selection": "score_prop",
              "eval_samples": [10], "eval_test": True},
    )
    from hacp.config import ExperimentConfig

    cfg = ExperimentConfig(**{k: v for k, v in r.json()["config"].items()
                             if k in ExperimentConfig.model_fields})
    args = cfg.generate_loop_args("run1", resume_from="/tmp/x")
    assert args[:1] == ["generate_loop.py"]
    assert args[args.index("--domains") + 1] == "paper_review"
    assert args[args.index("--max_generation") + 1] == "7"
    assert args[args.index("--eval_samples") + 1] == "10"
    assert args[args.index("--parent_selection") + 1] == "score_prop"
    assert "--eval_test" in args
    assert args[args.index("--resume_from") + 1] == "/tmp/x"
    assert "--meta_patch_files" not in args  # platform never fakes research knobs


def test_patch_endpoint_updates_budget(client):
    r = client.post("/api/experiments", json={"name": "patch-me"})
    eid = r.json()["id"]
    r = client.patch(f"/api/experiments/{eid}", json={"budget": {"max_cost_usd": 12.5}})
    assert r.status_code == 200
    assert r.json()["config"]["budget"]["max_cost_usd"] == 12.5
    # loop mode change rejected
    r = client.patch(f"/api/experiments/{eid}", json={"loop": "stub"})
    assert r.status_code == 400


def test_models_catalog(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["models"]]
    assert "openai/gpt-4o" in ids and "anthropic/claude-sonnet-4-5-20250929" in ids


def test_comparison_endpoint(client):
    a = client.post("/api/experiments", json={"name": "c1"}).json()["id"]
    b = client.post("/api/experiments", json={"name": "c2"}).json()["id"]
    r = client.get(f"/api/comparison?ids={a},{b}")
    comp = r.json()["comparison"]
    assert [c["name"] for c in comp] == ["c1", "c2"]
    # no generations recorded yet -> zeroed generation stats, no crash
    assert comp[0]["generation_stats"]["count"] == 0


def test_comparison_enriched_with_generation_stats(client):
    store = client.app.state.store
    eid = client.post("/api/experiments", json={"name": "enriched"}).json()["id"]
    store.add_event(eid, "generations", {"generations": [
        {"gen": "initial", "parent": None, "duration_s": None, "scores": {"paper_review": {"train": 0.52}}},
        {"gen": "1", "parent": "initial", "duration_s": 84.2, "scores": {"paper_review": {"train": 0.55}}},
    ]})
    comp = client.get(f"/api/comparison?ids={eid}").json()["comparison"][0]
    gs = comp["generation_stats"]
    assert gs["count"] == 2
    assert gs["best_train_scores"]["paper_review"] > 0.54
    assert gs["avg_generation_seconds"] == 84.2


def test_usage_totals_include_latency(client):
    store = client.app.state.store
    eid = client.post("/api/experiments", json={"name": "lat"}).json()["id"]
    store.add_usage(eid, "meta", "openai/gpt-4o-mini", 100, 50, latency_s=1.25)
    store.add_usage(eid, "meta", "openai/gpt-4o-mini", 100, 50, latency_s=2.75)
    usage = client.get(f"/api/experiments/{eid}/usage").json()
    assert usage["totals_proxy"]["avg_latency_s"] == 2.0


def test_shell_chat_with_injected_client(client, service):
    """Shell uses the (independently configured) shell model; tests inject a
    fake LLM client so the tool loop runs without any HTTP or API cost."""
    store = client.app.state.store
    exp = store.create_experiment({"name": "shell-target", "domains": ["paper_review"]})
    scripted = [
        '<json>{"tool_name": "list_experiments", "tool_input": {}}</json>',
        "There is 1 experiment: shell-target.",
    ]
    client.app.state.shell = ShellAgent(
        FakeLLMClient(scripted), store, service.home
    )
    r = client.post("/api/shell/chat", json={"message": "what experiments exist?"})
    assert r.status_code == 200
    body = r.json()
    assert "shell-target" in body["reply"]
    assert body["tool_calls"][0]["tool_name"] == "list_experiments"
    result_names = [e["name"] for e in body["tool_calls"][0]["result"]]
    assert "shell-target" in result_names


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "HyperAgents Control Platform" in r.text
    r = client.get("/static/app.js")
    assert r.status_code == 200
