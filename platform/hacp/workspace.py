"""Experiment workspace preparation.

Each experiment runs in its own disposable copy of the pristine HyperAgents
repository: ``<home>/experiments/<id>/workspace``. The pristine checkout in the
repo is NEVER used as a working directory and never modified.

Layout (upstream mechanics in mind):

    workspace/                     <- root_dir for generate_loop.py
      ...pristine upstream files...   (copied, .git included: upstream git-ops need it)
      agent/llm.py                    <- instrumentation patch applied here
      misc/hacp/platform_config.json   <- excluded from the evolvable agent archive
      misc/hacp/usage.jsonl            <- real-time usage log (host + in-container)
      outputs/                        <- experiment outputs live here
      gen_initial/hyperagents         <- created by upstream setup_initial_gen

``misc/`` is key: upstream ``setup_initial_gen`` excludes it when creating the
evolvable agent repo snapshot, so platform files stay out of the meta-agent's
world, while still being visible inside experiment containers (the workspace is
bind-mounted at /hyperagents by upstream ``build_container``).
"""
from __future__ import annotations

import fnmatch
import json
import os
import shutil
import subprocess
from pathlib import Path

from . import paths
from .config import ExperimentConfig, ServiceConfig
from .instrumentation import (
    apply_instrumentation,
    apply_meta_hint,
    verify_instrumentation,
    verify_meta_hint,
)

# Exclusions for copying the pristine repo into an experiment workspace.
# CRITICAL (regression-tested): these apply at the TOP LEVEL of the repo only
# — blanket name patterns would also strip nested paths like
# .git/refs/heads/platform/ (the platform's own branch ref!) and silently
# break upstream git operations. Cache artifacts are excluded everywhere.
_TOP_LEVEL_EXACT = {"platform", ".venv", "venv", "venv_nat", "outputs", "outputs_os", ".claude", ".pytest_cache", ".mypy_cache"}
_TOP_LEVEL_PATTERNS = ("outputs_os*", ".venv*", "venv*")
_EVERYWHERE = ("__pycache__", "*.pyc", ".pytest_cache")


def _make_copy_ignore(upstream_root: Path):
    upstream_root_abs = str(upstream_root.resolve())

    def ignore(src, names):
        src_abs = os.path.abspath(src)
        is_top = os.path.normpath(src_abs) == os.path.normpath(upstream_root_abs)
        ignored = []
        for name in names:
            if is_top and name in _TOP_LEVEL_EXACT:
                ignored.append(name)
            elif is_top and any(fnmatch.fnmatch(name, p) for p in _TOP_LEVEL_PATTERNS):
                ignored.append(name)
            elif any(fnmatch.fnmatch(name, p) for p in _EVERYWHERE):
                ignored.append(name)
        return ignored

    return ignore


def workspace_exists(exp_id: str, home: Path | None = None) -> bool:
    ws = paths.workspace_dir(exp_id, home)
    return (ws / "generate_loop.py").exists()


def prepare_workspace(
    exp_id: str,
    config: ExperimentConfig | dict,
    service: ServiceConfig,
    force: bool = False,
) -> dict:
    """Create (or verify) the experiment workspace. Returns a manifest."""
    if isinstance(config, dict):
        config = ExperimentConfig(**{k: v for k, v in config.items() if k in ExperimentConfig.model_fields})
    home = Path(service.home)
    upstream = Path(service.upstream_repo)
    ws = paths.workspace_dir(exp_id, home)

    if not (upstream / "generate_loop.py").exists():
        raise FileNotFoundError(
            f"upstream repo not found or invalid: {upstream} (expected generate_loop.py)"
        )

    if workspace_exists(exp_id, home) and not force:
        if not verify_instrumentation(ws):
            raise RuntimeError(
                f"workspace for {exp_id} exists but is not instrumented; "
                "use force=True to rebuild"
            )
        manifest_path = ws / "misc" / "hacp" / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text())
            if manifest_path.exists()
            else {"reused": True}
        )
        manifest["reused"] = True
        return manifest

    if ws.exists():
        shutil.rmtree(ws)
    ws.parent.mkdir(parents=True, exist_ok=True)

    # 1. Copy the pristine repo (keeping .git — upstream git operations need it,
    #    including the branch ref under .git/refs/heads/).
    shutil.copytree(upstream, ws, ignore=_make_copy_ignore(upstream), symlinks=True)

    # 2. Apply the instrumentation patch to the copy.
    patch_manifest = apply_instrumentation(ws)

    # 2b. Optional: platform guidance for the meta-agent (per experiment).
    hint_manifest = None
    if getattr(config, "meta_hint", False):
        hint_manifest = apply_meta_hint(ws)

    # 3. Platform config: per-role model overrides + proxy routing + usage log.
    #    Written BOTH at the workspace root and under misc/hacp/: upstream
    #    setup_initial_gen excludes 'misc' when creating the evolvable agent
    #    snapshot (gen_initial/hyperagents) that gets bind-mounted into
    #    experiment containers — so the ROOT copy is the one containers find
    #    (cwd/platform_config.json with cwd=/hyperagents). The misc/hacp copy
    #    keeps platform artifacts grouped and is found by host-side runs.
    hacp_dir = ws / "misc" / "hacp"
    hacp_dir.mkdir(parents=True, exist_ok=True)
    platform_config = {
        "experiment_id": exp_id,
        "meta_model": config.meta_model,
        "task_model": config.task_model,
        "llm_api_base": service.proxy_v1_base,
        "meta_token": f"hacp-{exp_id}-meta",
        "task_token": f"hacp-{exp_id}-task",
        "usage_log": paths.usage_log_relpath(),
        "meta_reasoning": getattr(config, "meta_reasoning_effort", None),
        "task_reasoning": getattr(config, "task_reasoning_effort", None),
        "meta_completion": getattr(config, "meta_completion_cap", None),
        "task_completion": getattr(config, "task_completion_cap", None),
    }
    (hacp_dir / "platform_config.json").write_text(json.dumps(platform_config, indent=2))
    (ws / "platform_config.json").write_text(json.dumps(platform_config, indent=2))
    (ws / paths.usage_log_relpath()).parent.mkdir(parents=True, exist_ok=True)
    (ws / paths.usage_log_relpath()).touch()

    # 4. Outputs dir (upstream writes here).
    (ws / "outputs").mkdir(exist_ok=True)

    # 5. Manifest (audit trail).
    upstream_commit = "?"
    try:
        upstream_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=upstream,
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception:
        pass
    manifest = {
        "experiment_id": exp_id,
        "upstream_repo": str(upstream),
        "upstream_commit": upstream_commit,
        "instrumentation": patch_manifest,
        "meta_hint": hint_manifest,
        "platform_config": platform_config,
        "created_at": __import__("time").time(),
    }
    (hacp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read_platform_config(exp_id: str, home: Path | None = None) -> dict:
    p = paths.workspace_dir(exp_id, home) / paths.platform_config_relpath()
    return json.loads(p.read_text())


def workspace_status(exp_id: str, home: Path | None = None) -> dict:
    ws = paths.workspace_dir(exp_id, home)
    status = {
        "workspace": str(ws),
        "exists": ws.exists(),
        "instrumented": verify_instrumentation(ws),
        "initial_evals_ready": True,
        "missing_initial_evals": [],
    }
    if status["exists"]:
        for d in _expected_initial_dirs(ws):
            if not (ws / "outputs" / d).exists():
                status["initial_evals_ready"] = False
                status["missing_initial_evals"].append(d)
    return status


def _expected_initial_dirs(ws: Path) -> list[str]:
    """Initial-eval output dirs generate_loop expects under outputs/."""
    # For M1 only the paper_review flow is seeded automatically; other domains
    # follow the same upstream pattern (setup_initial.sh).
    return [
        "initial_paper_review_filtered_100_train_0",
        "initial_paper_review_filtered_100_val_0",
        "initial_paper_review_filtered_100_test_0",
    ]
