"""Scientific fidelity: experiment workspaces contain Meta's unmodified runtime.

The token-consumption fixes live exclusively in the platform's own agent (the
conversational shell) and in accounting. The experimental runtime — the files
that define meta-agent behavior, task-agent behavior, and the benchmark — must
be byte-identical to upstream in every experiment workspace, with the single
sanctioned exception of the routing instrumentation in agent/llm.py (gateway
routing/accounting, explicitly required to remain operational), which this
suite proves is a *reversible, purely additive* change.
"""
import shutil
from pathlib import Path

import pytest

from hacp import instrumentation as instr
from hacp.config import ExperimentConfig, ServiceConfig
from hacp.workspace import prepare_workspace


@pytest.fixture
def service(tmp_path, upstream_repo):
    return ServiceConfig(
        host="127.0.0.1", port=1, home=str(tmp_path / "home"),
        upstream="stub", upstream_repo=str(upstream_repo),
        proxy_public_base="http://127.0.0.1:1",
    )


@pytest.fixture
def ws(service, upstream_repo):
    prepare_workspace("e0000000fidel", {"name": "fidelity", "domains": ["paper_review"]}, service)
    return service and Path(service.home) / "experiments" / "e0000000fidel" / "workspace"


# files that define the experimental runtime and must be byte-identical
IDENTICAL_FILES = [
    "agent/llm_withtools.py",   # the meta/task chat loop (context behavior!)
    "agent/tools/bash.py",
    "agent/tools/edit.py",
    "agent/tools/__init__.py",
    "meta_agent.py",
    "task_agent.py",
    "domains/harness.py",
    "domains/report.py",
    "domains/paper_review/utils.py",
    "utils/gl_utils.py",
    "utils/domain_utils.py",
    "generate_loop.py",
]


@pytest.mark.parametrize("rel", IDENTICAL_FILES)
def test_experimental_file_byte_identical(ws, upstream_repo, rel):
    assert (ws / rel).read_bytes() == (upstream_repo / rel).read_bytes(), rel


def test_no_platform_tool_added_to_agent_world(ws):
    """The additive 'retrieve' tool (from the reverted context-manager work)
    must NOT be installed — it would change the agents' toolset."""
    tools = sorted(p.name for p in (ws / "agent" / "tools").glob("*.py"))
    assert tools == ["__init__.py", "bash.py", "edit.py"], tools
    # and no context-manager marker anywhere in the experimental runtime
    for f in ws.rglob("*.py"):
        if ".venv" in str(f):
            continue
        assert "HACP-CONTEXT-MANAGER" not in f.read_text(errors="replace"), f


def test_meta_agent_untouched_when_hint_off(ws, upstream_repo):
    assert (ws / "meta_agent.py").read_bytes() == (upstream_repo / "meta_agent.py").read_bytes()
    assert "HACP-META-HINT" not in (ws / "meta_agent.py").read_text()


def test_meta_hint_is_opt_in_only(service, upstream_repo):
    """With meta_hint enabled the instruction gains the operator note (an
    explicit, per-experiment operator choice); disabled (default) is pristine."""
    prepare_workspace("e0000000hint", {"name": "hint", "meta_hint": True}, service)
    ws2 = Path(service.home) / "experiments" / "e0000000hint" / "workspace"
    hinted = (ws2 / "meta_agent.py").read_text()
    assert "HACP-META-HINT" in hinted
    assert "two-stage evaluation pipeline" in hinted


def test_llm_patch_is_reversible_to_pristine(ws, upstream_repo):
    """The routing instrumentation in agent/llm.py is the ONLY sanctioned delta.
    Prove it's purely additive/reversible: strip the marked block and undo the
    two anchored replacements — the result must be byte-identical to upstream.
    """
    patched = (ws / "agent" / "llm.py").read_text()
    pristine = (upstream_repo / "agent" / "llm.py").read_text()
    assert instr.MARKER in patched
    # remove the inserted module block: everything from just after the anchor
    # line to just after the END-marker line (newline included)
    after_anchor = patched.index(instr.ANCHOR_MODULE) + len(instr.ANCHOR_MODULE)
    end_marker = "# ==== " + instr.MARKER + " END ===="
    end = patched.index(end_marker) + len(end_marker) + 1  # + trailing newline
    stripped = patched[:after_anchor] + patched[end:]
    # undo the two anchored replacements
    assert instr.PATCHED_CALL in stripped and instr.PATCHED_APPEND.rstrip() in stripped
    stripped = stripped.replace(instr.PATCHED_CALL.rstrip(), instr.ANCHOR_CALL.rstrip(), 1)
    stripped = stripped.replace(instr.PATCHED_APPEND.rstrip(), instr.ANCHOR_APPEND.rstrip(), 1)
    assert stripped == pristine, "routing patch must reconstruct upstream exactly when reversed"


def test_container_env_has_no_context_management_keys(service):
    """No HACP_CTX_* compaction/truncation knobs may reach experiment containers."""
    from hacp.runner import ExperimentRunner

    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.service = service
    cfg = ExperimentConfig(name="envcheck")
    env = runner._container_env("e0000000env", cfg)
    ctx_keys = [k for k in env if k.startswith("HACP_CTX_")]
    assert ctx_keys == [], ctx_keys
    # routing/accounting channel intact (must remain operational)
    for k in ("HACP_LLMBASE", "HACP_META_MODEL", "HACP_TASK_MODEL",
              "HACP_META_TOKEN", "HACP_TASK_TOKEN", "HACP_ROUTING_ACTIVE"):
        assert k in env, k


def test_experiment_config_has_no_session_context_fields():
    for f in ("session_context_tokens", "session_tool_output_chars",
              "session_repeat_limit", "session_max_tokens", "session_max_cost_usd"):
        assert f not in ExperimentConfig.model_fields, f
