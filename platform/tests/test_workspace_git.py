"""Regression: workspace preparation must preserve git refs.

Found by the dress rehearsal (2026-10): a blanket ignore pattern 'platform'
also matched .git/refs/heads/platform/ (the platform branch ref directory),
leaving the workspace repo with NO commits on the working branch. Upstream's
commit_repo then created a root commit and parsed '(root-commit)' as a hash,
which later crashed generate_loop's `git reset --hard (root-commit)`.
"""
import subprocess
from pathlib import Path

from hacp import paths
from hacp.workspace import prepare_workspace, workspace_exists


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60
    )


def test_workspace_preserves_git_history_and_refs(service, upstream_repo):
    eid = "e0000000gitref"
    prepare_workspace(eid, {"name": "gitref", "domains": ["paper_review"]}, service)
    ws = paths.workspace_dir(eid, Path(service.home))
    assert workspace_exists(eid, Path(service.home))

    # full history copied
    r = _git(ws, "log", "--oneline")
    assert r.returncode == 0, r.stderr
    assert len(r.stdout.strip().splitlines()) >= 3, "git history lost in workspace copy"

    # the working branch ref survives (nested under .git/refs/heads/platform/)
    r = _git(ws, "rev-parse", "HEAD")
    assert r.returncode == 0 and len(r.stdout.strip()) == 40, (
        f"branch ref missing: {r.stderr}"
    )

    # top-level platform/ code dir still excluded
    assert not (ws / "platform").exists()
    # and no venv leakage
    assert not (ws / ".venv").exists() and not (ws / "platform" / ".venv").exists()


def test_snapshot_style_commit_repo_works_after_copy(service, upstream_repo):
    """Mirror what upstream setup_initial_gen does right after the copy:
    commit_repo (git add --all + git commit) must return a REAL hash, not the
    '(root-commit)' placeholder it emits for a repository's first commit."""
    from hacp.config import ExperimentConfig

    eid = "e0000000commitok"
    cfg = ExperimentConfig(name="commitok", domains=["paper_review"], loop="generate")
    prepare_workspace(eid, cfg, service)
    ws = paths.workspace_dir(eid, Path(service.home))

    # upstream utils.commit_repo semantics
    _git(ws, "add", "--all")
    r = _git(ws, "commit", "-m", "a nonsense commit message")
    assert r.returncode == 0, r.stderr
    parts = r.stdout.strip().splitlines()[0].split()
    # 'git commit' prints '[branch <hash>] msg' — the hash must not be the
    # '(root-commit)' placeholder that only appears for a first-ever commit
    assert len(parts) >= 2
    parsed_hash = parts[1].strip("[]")
    assert parsed_hash != "(root-commit)", f"parsed placeholder: {r.stdout}"
    # and it must resolve
    assert _git(ws, "rev-parse", parsed_hash).returncode == 0
