"""Shell context management: the platform's own coding-agent orchestration.

The shell is the only agent the platform compact/dedup/budget controls —
never the HyperAgents experimental runtime (see test_experimental_fidelity).
"""
import json

import pytest

from hacp.config import resolve_model
from hacp.llm import FakeLLMClient
from hacp.shell import ShellAgent, _est_tokens, _history_tokens, compact_shell_history, dedupe_history


def _mk_shell(store, tmp_path, responses, **kw):
    return ShellAgent(FakeLLMClient(responses), store, tmp_path, **kw)


# ------------------------------------------------------------------ compaction
def test_compact_history_bounds_tokens_preserving_essentials():
    history = [{"role": "user", "content": "IMPORTANT original task: analyze experiment alpha"}]
    for i in range(40):
        history.append({"role": "assistant", "content": f"turn {i}: " + "x" * 3000})
        history.append({"role": "user", "content": f"follow-up {i}: " + "y" * 2000})
    compacted, did = compact_shell_history(history, context_tokens=20000, recent_turns=6,
                                           incoming="latest question")
    assert did is True
    assert _history_tokens(compacted) + _est_tokens("latest question") <= 20000 + 8000
    # essentials preserved
    assert compacted[0]["content"].startswith("IMPORTANT original task")
    assert "[context note]" in compacted[0]["content"]
    assert "74 earlier messages" in compacted[0]["content"]  # explicit count, not silent loss
    assert compacted[-1]["content"].startswith("follow-up 39")  # current state kept


def test_compact_noop_when_under_budget():
    history = [{"role": "user", "content": "short"}, {"role": "assistant", "content": "ok"}]
    out, did = compact_shell_history(history, context_tokens=100000, recent_turns=6)
    assert did is False and out is history


def test_dedupe_elides_consecutive_identical_tool_results():
    r = "Tool result (JSON):\n```json\n{\"big\": \"%s\"}\n```" % ("z" * 5000)
    history = [
        {"role": "user", "content": "q1"},
        {"role": "user", "content": r},
        {"role": "user", "content": r},  # repeated identical call
        {"role": "user", "content": r},
        {"role": "user", "content": "different"},
    ]
    out = dedupe_history(history)
    assert sum(1 for m in out if m["content"] == r) == 1
    assert out[-1]["content"] == "different"


# ------------------------------------------------------- shell sessions behave
def test_shell_sends_bounded_context_on_long_conversations(store, tmp_path):
    """A 60-turn shell history stays bounded in what the LLM receives; the
    original request and recent turns survive."""
    history = [{"role": "user", "content": "Watch experiment alpha closely"}]
    for i in range(30):
        history.append({"role": "assistant", "content": f"status {i}: " + "a" * 2500})
        history.append({"role": "user", "content": f"and? {i}: " + "b" * 1500})
    shell = _mk_shell(store, tmp_path, ["current status: running"], context_tokens=15000,
                      recent_turns=4)
    out = shell.run("what is the status now?", history=history)
    sent = shell.client.calls[0]["messages"]
    body_tokens = _history_tokens(sent)
    assert body_tokens <= 15000 + 12000, body_tokens  # bounded, not 30*4000=~120k
    assert sent[1]["content"].startswith("Watch experiment alpha closely")  # original intent
    assert "[context note]" in sent[1]["content"]
    # the recent context made it through
    assert any("status 29" in m["content"] for m in sent)
    assert out["reply"] == "current status: running"


def test_shell_large_tool_results_capped_inline(store, tmp_path):
    big = [{"row_%04d" % i: "x" * 60} for i in range(400)]
    store.create_experiment({"name": "alpha", "domains": ["paper_review"]})
    scripted = [
        '<json>{"tool_name": "list_experiments", "tool_input": {}}</json>',
        "done",
    ]
    # make list_experiments return a big payload by seeding many experiments
    for i in range(60):
        store.create_experiment({"name": f"exp-{i:03d}-" + "n" * 300})
    shell = _mk_shell(store, tmp_path, scripted, context_tokens=100000)
    out = shell.run("list everything")
    tool_msg = out["history"][-2]["content"]
    assert "truncated" in tool_msg and len(tool_msg) < 7000
    # full result still delivered to the UI
    assert len(json.dumps(out["tool_calls"][0]["result"])) > 20000


def test_shell_token_budget_stops_session(store, tmp_path):
    scripted = ["answer one"]
    shell = _mk_shell(store, tmp_path, scripted, session_max_tokens=50)
    out = shell.run("hi", history=[])
    assert out.get("budget_stopped") is True
    assert "session budget" in out["reply"].lower()
    # and the NEXT call refuses without consuming another LLM call
    n_calls = len(shell.client.calls)
    out2 = shell.run("again", history=out["history"])
    assert len(shell.client.calls) == n_calls  # no further spend
    assert out2.get("budget_stopped") is True


def test_shell_cost_budget_stops_session(store, tmp_path):
    scripted = ["answer one"]
    shell = _mk_shell(store, tmp_path, scripted, session_max_cost_usd=0.000001,
                      model="openai/gpt-4o")  # priced model
    out = shell.run("hi", history=[])
    # estimation path: large system prompt guarantees >1e-6 est cost? if the
    # first call completed, the guard must at least fire on this second turn
    if not out.get("budget_stopped"):
        out2 = shell.run("again", history=out["history"])
        assert out2.get("budget_stopped") is True
    assert shell.session_cost_usd > 0


def test_shell_usage_from_client_preferred_over_estimates(store, tmp_path):
    class UsageClient:
        last_usage = {"model": "openai/gpt-4o", "prompt_tokens": 10_000,
                      "completion_tokens": 2_000, "total_tokens": 12_000, "cached_tokens": 0}

        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kw):
            self.calls += 1
            return "ok"

    shell = ShellAgent(UsageClient(), store, tmp_path)
    shell.run("hi", history=[])
    assert shell.session_tokens == 12_000
    info = resolve_model("openai/gpt-4o")
    assert shell.session_cost_usd == pytest.approx(
        10_000 / 1e6 * info.input_usd_per_mtok + 2_000 / 1e6 * info.output_usd_per_mtok
    )
