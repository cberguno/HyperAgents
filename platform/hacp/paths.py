"""Filesystem layout helpers for the platform home."""
from __future__ import annotations

import os
from pathlib import Path


def platform_dir() -> Path:
    """Directory containing the hacp package (i.e. <repo>/platform)."""
    return Path(__file__).resolve().parent.parent


def platform_home() -> Path:
    """Platform home directory (experiments, data). Override with HACP_HOME."""
    env = os.environ.get("HACP_HOME")
    if env:
        return Path(env).resolve()
    return platform_dir() / "home"


def upstream_repo_default() -> Path:
    """Default location of the pristine HyperAgents repository (repo root)."""
    return platform_dir().parent


def experiments_root(home: Path | None = None) -> Path:
    return (home or platform_home()) / "experiments"


def experiment_dir(exp_id: str, home: Path | None = None) -> Path:
    return experiments_root(home) / exp_id


def workspace_dir(exp_id: str, home: Path | None = None) -> Path:
    return experiment_dir(exp_id, home) / "workspace"


def data_dir(home: Path | None = None) -> Path:
    return (home or platform_home()) / "data"


def db_path(home: Path | None = None) -> Path:
    return data_dir(home) / "hacp.db"


def usage_log_relpath() -> str:
    """Path of the in-workspace usage log, relative to the workspace root.

    'misc/' is excluded from the evolvable agent archive by upstream
    ``setup_initial_gen`` and the workspace is bind-mounted at /hyperagents
    in experiment containers, so the same relative path works on the host
    and inside containers.
    """
    return "misc/hacp/usage.jsonl"


def platform_config_relpath() -> str:
    return "misc/hacp/platform_config.json"
