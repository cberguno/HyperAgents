"""Conversational shell agent: tool loop, store-backed tools, read-only posture."""
import json

from hacp.llm import FakeLLMClient
from hacp.shell import ShellAgent, build_tools


def _mk_shell(store, home, responses):
    return ShellAgent(FakeLLMClient(responses), store, home)


def test_list_experiments_tool(store, tmp_path):
    store.create_experiment({"name": "alpha", "domains": ["paper_review"]})
    store.create_experiment({"name": "beta", "domains": ["paper_review"]})
    tools = {t["name"]: t for t in build_tools(store, tmp_path)}
    result = tools["list_experiments"]["fn"]()
    assert [e["name"] for e in result] == ["alpha", "beta"]


def test_usage_and_scores_tools(store, tmp_path):
    exp = store.create_experiment({"name": "alpha"})
    store.add_usage(exp["id"], "task", "openai/gpt-4o-mini", 1000, 2000)
    tools = {t["name"]: t for t in build_tools(store, tmp_path)}
    usage = tools["get_usage"]["fn"](exp["id"])
    assert usage["totals"]["total_tokens"] == 3000
    assert usage["breakdown"][0]["model"] == "openai/gpt-4o-mini"

    store.add_event(exp["id"], "generations", {"generations": [
        {"gen": "1", "scores": {"paper_review": {"train": 0.55}}}
    ]})
    scores = tools["get_scores"]["fn"](exp["id"])
    assert scores[0]["scores"]["paper_review"]["train"] == 0.55


def test_shell_tool_loop(store, tmp_path):
    store.create_experiment({"name": "gamma", "domains": ["paper_review"]})
    scripted = [
        '<json>{"tool_name": "get_experiment", "tool_input": {"experiment_id": "nope"}}</json>',
        "That experiment id does not exist.",
    ]
    shell = _mk_shell(store, tmp_path, scripted)
    out = shell.run("inspect experiment nope")
    assert out["reply"] == "That experiment id does not exist."
    assert out["tool_calls"][0]["tool_name"] == "get_experiment"
    assert "error" in out["tool_calls"][0]["result"]


def test_shell_answers_without_tools(store, tmp_path):
    shell = _mk_shell(store, tmp_path, ["You have 5 slots configured."])
    out = shell.run("how many slots?")
    assert out["tool_calls"] == []
    assert "5 slots" in out["reply"]


def test_shell_readonly_posture(store, tmp_path):
    """Control actions must not be auto-executed in M1."""
    shell = _mk_shell(store, tmp_path, [
        "I'm read-only in M1. To start experiment e123 use: POST /api/experiments/e123/start",
    ])
    out = shell.run("start experiment e123")
    assert out["tool_calls"] == []
    assert "POST /api/experiments" in out["reply"]


def test_shell_malformed_tool_json_recovers(store, tmp_path):
    scripted = [
        "<json>{bad json}</json>",
        "Recovered — final answer.",
    ]
    shell = _mk_shell(store, tmp_path, scripted)
    out = shell.run("hi")
    assert out["reply"] == "Recovered — final answer."


def test_shell_explain_failure_assembles_context(store, tmp_path):
    exp = store.create_experiment({"name": "failed-exp"})
    store.update_experiment(exp["id"], status="failed", error="boom")
    out_dir = tmp_path / "outputs" / "generate_x" / "gen_1" / "agent_output"
    out_dir.mkdir(parents=True)
    (out_dir / "meta_agent_chat_history.md").write_text("meta agent tried X and failed")
    (out_dir / "model_patch.diff").write_text("--- a/task_agent.py")
    store.update_experiment(exp["id"], output_dir=str(tmp_path / "outputs" / "generate_x"))
    tools = {t["name"]: t for t in build_tools(store, tmp_path)}
    ctx = tools["explain_failure"]["fn"](exp["id"], gen="1")
    assert "boom" in ctx["error"]
    assert "meta agent tried X" in ctx["meta_chat_history_tail"]
    assert ctx["model_patch_head"].startswith("--- a/task_agent.py")
