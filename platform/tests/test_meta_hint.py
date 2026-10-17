"""Option A — meta-agent guidance hint (HACP-META-HINT-v1).

Patches the *workspace copy* of meta_agent.py (never the pristine repo) to
append operator guidance to the meta-agent instruction: empty prediction
distributions are a task_agent.py parsing problem (not model capability), with
the DGM-H two-stage evaluation pipeline as the known-good structure.
"""
import hashlib
import shutil

import pytest

from hacp.instrumentation import (
    META_HINT_MARKER,
    InstrumentationError,
    apply_meta_hint,
    verify_meta_hint,
)


@pytest.fixture
def ws(tmp_path, upstream_repo):
    w = tmp_path / "workspace"
    w.mkdir()
    shutil.copy(upstream_repo / "meta_agent.py", w / "meta_agent.py")
    return w


def test_pristine_repo_has_no_hint(upstream_repo):
    src = (upstream_repo / "meta_agent.py").read_text()
    assert META_HINT_MARKER not in src
    assert "_HACP_META_HINT" not in src


def test_apply_meta_hint(ws):
    manifest = apply_meta_hint(ws)
    assert manifest["patched"] is True
    src = (ws / "meta_agent.py").read_text()
    assert META_HINT_MARKER in src
    # the guidance content (user-specified): parsing diagnosis + DGM-H path
    assert "task_agent.py" in src
    assert "parsing" in src.lower()
    assert "0.0 to 0.71" in src
    assert "two-stage evaluation pipeline" in src
    assert "accept/reject" in src
    # and the use-site: instruction gets the hint appended
    assert '`{repo_path}`." + _HACP_META_HINT' in src
    assert verify_meta_hint(ws)


def test_hint_idempotent(ws):
    apply_meta_hint(ws)
    m2 = apply_meta_hint(ws)
    assert m2["already_patched"] is True and m2["patched"] is True
    assert verify_meta_hint(ws)


def test_hint_compiles_and_agent_still_importable(ws, upstream_repo):
    apply_meta_hint(ws)
    # make the copy importable: bring the small dependency tree it needs
    shutil.copy(upstream_repo / "meta_agent.py", ws / "meta_agent.py") if False else None
    src = (ws / "meta_agent.py").read_text()
    compile(src, "meta_agent.py", "exec")


def test_hint_forward_yields_hinted_instruction(ws, upstream_repo, tmp_path, monkeypatch):
    """The patched MetaAgent.forward() includes the guidance in the message it
    sends to the LLM (captured with a fake chat_with_agent)."""
    apply_meta_hint(ws)
    # stage minimal import tree for meta_agent.py
    for rel in ("agent/base_agent.py", "agent/llm_withtools.py", "agent/llm.py", "agent/tools/__init__.py"):
        dst = ws / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(upstream_repo / rel, dst)
    (ws / "agent/__init__.py").touch()
    # upstream llm.py imports utils.thread_logger via agent.base_agent
    (ws / "utils").mkdir(exist_ok=True)
    (ws / "utils/__init__.py").touch()
    for rel in ("thread_logger.py", "constants.py", "git_utils.py", "common.py"):
        shutil.copy(upstream_repo / "utils" / rel, ws / "utils" / rel)
    # agent.llm needs utils.common etc; stub heavy deps rather than pulling all
    import sys

    sys.path.insert(0, str(ws))
    monkeypatch.setitem(sys.modules, "litellm", type(sys)("litellm"))
    captured = {}

    def fake_chat(msg, **kw):
        captured["msg"] = msg
        return [{"role": "assistant", "text": "ok"}]

    import importlib

    import agent.llm_withtools as lwt

    monkeypatch.setattr(lwt, "chat_with_agent", fake_chat)
    import meta_agent as ma

    importlib.reload(ma)
    agent = ma.MetaAgent(model="stub/tiny", chat_history_file=str(ws / "chat.md"))
    agent.forward(repo_path="/hyperagents/", eval_path="/evals")
    assert "PLATFORM GUIDANCE" in captured["msg"]
    assert "task_agent.py" in captured["msg"]
    assert "two-stage evaluation pipeline" in captured["msg"]
    sys.path.remove(str(ws))


def test_hint_refuses_on_changed_anchor(ws):
    (ws / "meta_agent.py").write_text("# anchors changed\n")
    with pytest.raises(InstrumentationError):
        apply_meta_hint(ws)


def test_upstream_repo_untouched(ws, upstream_repo):
    before = hashlib.sha256((upstream_repo / "meta_agent.py").read_bytes()).hexdigest()
    apply_meta_hint(ws)
    after = hashlib.sha256((upstream_repo / "meta_agent.py").read_bytes()).hexdigest()
    assert before == after
