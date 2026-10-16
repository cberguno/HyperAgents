"""Instrumentation patch applied to *experiment workspace copies* of the repo.

Meta's pristine repository is never modified. Each experiment runs in its own
workspace copy (``experiments/<id>/workspace``); this module applies a small,
idempotent, clearly-marked patch to that copy's ``agent/llm.py`` — the single
LLM chokepoint used by both the meta-agent and the task-agent.

The patch is strictly opt-in at runtime:

* No ``misc/hacp/platform_config.json`` in the working directory → the patched
  code is inert: behavior is identical to upstream.
* With a config present it provides:
  1. Per-role model override (meta / task) — enabling independent model
     selection per experiment, routed through the platform proxy.
  2. Usage logging to ``misc/hacp/usage.jsonl`` (relative to the repo root, so
     the host watcher sees in-container writes in real time: upstream runs
     containers with the workspace bind-mounted at /hyperagents).

The patch is part of the experiment's base commit (upstream ``setup_initial_gen``
commits the workspace before generation 0), so lineage diffs stay clean.

If an evolving agent rewrites ``agent/llm.py`` and removes the instrumentation,
platform accounting degrades gracefully (detected via reconciliation warnings);
the proxy-side kill-switch and cost accounting are unaffected.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MARKER = "HACP-PLATFORM-INSTRUMENTATION-v1"

MODULE_BLOCK = '''

# ==== ''' + MARKER + ''' BEGIN ====
# Added automatically by the HyperAgents Control Platform (platform/hacp/instrumentation.py).
# Opt-in via misc/hacp/platform_config.json; delete the config to disable.
# Provides: per-role (meta/task) model override, LLM routing through the
# platform proxy, and usage logging. Never raises into agent code paths.
import json as _hacp_json
import os as _hacp_os
import sys as _hacp_sys
import time as _hacp_time

_HACP_CFG = None
_HACP_CFG_LOADED = False


def _hacp_load_config():
    global _HACP_CFG, _HACP_CFG_LOADED
    if _HACP_CFG_LOADED:
        return _HACP_CFG
    _HACP_CFG_LOADED = True
    _HACP_CFG = {}
    candidates = []
    _env = _hacp_os.environ.get("HACP_PLATFORM_CONFIG")
    if _env:
        candidates.append(_env)
    try:
        _here = _hacp_os.path.dirname(_hacp_os.path.abspath(__file__))
        _root = _hacp_os.path.dirname(_here)
        candidates += [
            _hacp_os.path.join(_hacp_os.getcwd(), "platform_config.json"),
            _hacp_os.path.join(_root, "platform_config.json"),
            _hacp_os.path.join(_root, "misc", "hacp", "platform_config.json"),
            _hacp_os.path.join(_hacp_os.getcwd(), "misc", "hacp", "platform_config.json"),
        ]
        for _path in candidates:
            if _path and _hacp_os.path.exists(_path):
                with open(_path, "r") as _f:
                    _HACP_CFG = _hacp_json.load(_f)
                break
    except Exception:
        pass
    return _HACP_CFG


def _hacp_role():
    try:
        _argv = " ".join(_hacp_sys.argv)
        if "run_meta_agent" in _argv:
            return "meta"
        if "meta_agent" in _argv and "harness" not in _argv and "task_agent" not in _argv:
            return "meta"
        if "harness" in _argv or "task_agent" in _argv:
            return "task"
        _mods = _hacp_sys.modules
        if "meta_agent" in _mods:
            return "meta"
        if "task_agent" in _mods or "domains.harness" in _mods:
            return "task"
    except Exception:
        pass
    return "unknown"


def _hacp_before_completion(model):
    """Return (override_model, extra_completion_kwargs) or (None, {})."""
    try:
        _cfg = _hacp_load_config()
        if not _cfg or not _cfg.get("llm_api_base"):
            return None, {}
        _role = _hacp_role()
        _override = _cfg.get(f"{_role}_model")
        if not _override:
            return None, {}
        _extra = {"api_base": _cfg["llm_api_base"]}
        _token = _cfg.get(f"{_role}_token") or _cfg.get("api_token")
        if _token:
            _extra["api_key"] = _token
        return "openai/" + _override, _extra
    except Exception:
        return None, {}


def _hacp_usage_value(_usage, _key):
    try:
        _v = _usage.get(_key)
    except Exception:
        _v = None
    if _v is None:
        try:
            _v = getattr(_usage, _key, None)
        except Exception:
            _v = None
    return _v


def _hacp_log_usage(model, response, latency_s):
    try:
        _cfg = _hacp_load_config()
        if not _cfg or not _cfg.get("usage_log"):
            return
        _usage = None
        try:
            _usage = response.get("usage")
        except Exception:
            _usage = getattr(response, "usage", None)
        _entry = {
            "ts": _hacp_time.time(),
            "experiment_id": _cfg.get("experiment_id"),
            "role": _hacp_role(),
            "model": model,
            "prompt_tokens": _hacp_usage_value(_usage, "prompt_tokens"),
            "completion_tokens": _hacp_usage_value(_usage, "completion_tokens"),
            "total_tokens": _hacp_usage_value(_usage, "total_tokens"),
            "latency_s": round(float(latency_s), 4),
        }
        _path = _cfg["usage_log"]
        if not _hacp_os.path.isabs(_path):
            _path = _hacp_os.path.join(_hacp_os.getcwd(), _path)
        _hacp_os.makedirs(_hacp_os.path.dirname(_path), exist_ok=True)
        with open(_path, "a") as _f:
            _f.write(_hacp_json.dumps(_entry) + "\\n")
    except Exception as _e:
        print(f"[hacp] usage log error: {_e}", file=_hacp_sys.stderr)
# ==== ''' + MARKER + ''' END ====
'''

# Anchor 1: insert module block right after the litellm drop_params line.
ANCHOR_MODULE = "litellm.drop_params=True\n"

# Anchor 2: the completion call inside get_response_from_llm (exact upstream text).
ANCHOR_CALL = """    response = litellm.completion(**completion_kwargs)
    response_text = response['choices'][0]['message']['content']  # pyright: ignore
"""

PATCHED_CALL = """    _hacp_model, _hacp_extra = _hacp_before_completion(model)
    if _hacp_model:
        model = _hacp_model
        completion_kwargs["model"] = _hacp_model
        completion_kwargs.update(_hacp_extra)
    _hacp_t0 = _hacp_time.monotonic()
    response = litellm.completion(**completion_kwargs)
    _hacp_latency = _hacp_time.monotonic() - _hacp_t0
    response_text = response['choices'][0]['message']['content']  # pyright: ignore
    _hacp_log_usage(model, response, _hacp_latency)
"""


class InstrumentationError(RuntimeError):
    pass


def apply_instrumentation(workspace: Path) -> dict:
    """Patch <workspace>/agent/llm.py in place. Idempotent. Returns a manifest."""
    llm_path = Path(workspace) / "agent" / "llm.py"
    if not llm_path.exists():
        raise InstrumentationError(f"not a HyperAgents workspace: {llm_path} missing")
    src = llm_path.read_text()

    manifest = {
        "file": str(llm_path),
        "patched": False,
        "already_patched": False,
    }

    if MARKER in src:
        if ANCHOR_CALL in src:
            # already-patched file should NOT contain the original anchor
            raise InstrumentationError(
                "file contains both marker and original anchor; refusing to patch"
            )
        manifest["already_patched"] = True
        manifest["patched"] = True
        manifest["sha256_after"] = hashlib.sha256(src.encode()).hexdigest()
        return manifest

    if src.count(ANCHOR_MODULE) != 1:
        raise InstrumentationError(
            f"expected exactly 1 module anchor (litellm.drop_params=True), found {src.count(ANCHOR_MODULE)}. "
            "Upstream agent/llm.py changed; update platform/hacp/instrumentation.py anchors."
        )
    if src.count(ANCHOR_CALL) != 1:
        raise InstrumentationError(
            f"expected exactly 1 completion anchor, found {src.count(ANCHOR_CALL)}. "
            "Upstream agent/llm.py changed; update platform/hacp/instrumentation.py anchors."
        )

    patched = src.replace(ANCHOR_MODULE, ANCHOR_MODULE + MODULE_BLOCK, 1)
    patched = patched.replace(ANCHOR_CALL, PATCHED_CALL, 1)

    llm_path.write_text(patched)
    manifest["patched"] = True
    manifest["sha256_before"] = hashlib.sha256(src.encode()).hexdigest()
    manifest["sha256_after"] = hashlib.sha256(patched.encode()).hexdigest()
    return manifest


def verify_instrumentation(workspace: Path) -> bool:
    llm_path = Path(workspace) / "agent" / "llm.py"
    if not llm_path.exists():
        return False
    src = llm_path.read_text()
    return MARKER in src and ANCHOR_CALL not in src and "_hacp_before_completion" in src
