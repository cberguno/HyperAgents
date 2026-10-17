"""Instrumentation patch: idempotency, anchors, pristine-repo safety."""
import hashlib

import pytest

from hacp.instrumentation import (
    ANCHOR_CALL,
    MARKER,
    InstrumentationError,
    apply_instrumentation,
    verify_instrumentation,
)


@pytest.fixture
def mini_repo(tmp_path, upstream_repo):
    """A copy of the pristine agent/llm.py in a temp workspace."""
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "llm.py").write_bytes((upstream_repo / "agent" / "llm.py").read_bytes())
    return tmp_path


def test_pristine_repo_untouched_by_fixture(upstream_repo, mini_repo):
    src = (upstream_repo / "agent" / "llm.py").read_text()
    assert MARKER not in src
    assert src.count(ANCHOR_CALL) == 1


def test_apply_patch(mini_repo):
    manifest = apply_instrumentation(mini_repo)
    assert manifest["patched"] is True
    patched = (mini_repo / "agent" / "llm.py").read_text()
    assert MARKER in patched
    assert "_hacp_before_completion" in patched
    assert ANCHOR_CALL not in patched  # original anchor replaced
    # the patched code is opt-in: config absence check present
    assert "platform_config.json" in patched


def test_patch_is_idempotent(mini_repo):
    apply_instrumentation(mini_repo)
    manifest = apply_instrumentation(mini_repo)
    assert manifest["already_patched"] is True
    assert verify_instrumentation(mini_repo) is True


def test_patch_compiles(mini_repo):
    apply_instrumentation(mini_repo)
    compile((mini_repo / "agent" / "llm.py").read_text(), "llm.py", "exec")


def test_patch_refuses_when_anchors_changed(mini_repo):
    (mini_repo / "agent" / "llm.py").write_text("# no anchors here\n")
    with pytest.raises(InstrumentationError):
        apply_instrumentation(mini_repo)


def test_upstream_repo_never_modified(upstream_repo, mini_repo):
    before = hashlib.sha256((upstream_repo / "agent" / "llm.py").read_bytes()).hexdigest()
    apply_instrumentation(mini_repo)
    after = hashlib.sha256((upstream_repo / "agent" / "llm.py").read_bytes()).hexdigest()
    assert before == after
