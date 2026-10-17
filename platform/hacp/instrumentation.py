"""Instrumentation patch applied to *experiment workspace copies* of the repo.

Meta's pristine repository is never modified. Each experiment runs in its own
workspace copy (``experiments/<id>/workspace``); this module applies small,
idempotent, clearly-marked patches to that copy:

1. ``agent/llm.py`` — the single LLM chokepoint used by both the meta-agent
   and the task-agent (HACP-PLATFORM-INSTRUMENTATION-v1).
2. ``meta_agent.py`` — optional platform guidance appended to the meta-agent's
   instruction (HACP-META-HINT-v1), enabled per experiment via
   ``ExperimentConfig.meta_hint``.

The llm.py patch is strictly opt-in at runtime and reads configuration with
ENVIRONMENT-FIRST precedence:

* Process env vars (``HACP_LLMBASE``, ``HACP_META_MODEL``, …) are injected by
  the platform into every experiment container via the Docker socket proxy
  (see ``hacp/docker_proxy.py``). A self-modifying agent cannot remove them
  with file operations — run ``e6e3168d3b3ca`` gen 2 deleted the on-disk
  ``platform_config.json`` and the usage log; env-based routing survives
  exactly that attack.
* On-disk ``platform_config.json`` remains as a fallback (host-side runs).

The patch provides:
  - per-role (meta/task) model override routed through the platform proxy
  - optional per-role reasoning effort (e.g. GPT-6 Luna at ``max``)
  - usage logging to ``misc/hacp/usage.jsonl``
  - ``_hacp_routing_active()`` used by the runner's pre-session container check

If an evolving agent rewrites ``agent/llm.py`` and removes the
instrumentation, the platform detects it: the pre-session check fails the
run (proxy-side accounting and budget enforcement are unaffected either way).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

MARKER = "HACP-PLATFORM-INSTRUMENTATION-v1"
META_HINT_MARKER = "HACP-META-HINT-v1"

MODULE_BLOCK = '''

# ==== ''' + MARKER + ''' BEGIN ====
# Added automatically by the HyperAgents Control Platform (platform/hacp/instrumentation.py).
# Opt-in via HACP_* environment variables or platform_config.json; with neither
# present, behavior is identical to upstream. Provides: per-role (meta/task)
# model override with LLM routing through the platform proxy, optional
# reasoning-effort, usage logging, and _hacp_routing_active() for the
# platform's pre-session checks. Never raises into agent code paths.
import json as _hacp_json
import os as _hacp_os
import sys as _hacp_sys
import threading as _hacp_threading
import time as _hacp_time

_HACP_CFG = None
_HACP_CFG_LOADED = False
_HACP_CFG_LOCK = _hacp_threading.Lock()
_hacp_state = _hacp_threading.local()

# Environment variables injected by the platform into experiment containers
# (via the Docker socket proxy). File operations cannot remove them, so this
# channel survives an agent deleting platform_config.json (observed in run
# e6e3168d3b3ca, gen 2). Files remain as fallback for host-side runs.
_HACP_ENV_KEYS = {
    "experiment_id": "HACP_EXPERIMENT_ID",
    "llm_api_base": "HACP_LLMBASE",
    "meta_model": "HACP_META_MODEL",
    "task_model": "HACP_TASK_MODEL",
    "meta_token": "HACP_META_TOKEN",
    "task_token": "HACP_TASK_TOKEN",
    "usage_log": "HACP_USAGE_LOG",
    "meta_reasoning": "HACP_META_REASONING",
    "task_reasoning": "HACP_TASK_REASONING",
    "meta_completion": "HACP_META_COMPLETION",
    "task_completion": "HACP_TASK_COMPLETION",
}


def _hacp_load_config():
    # Thread-safe (upstream harnesses evaluate via worker pools) and
    # env-first: env values override any file values, and the loaded-flag is
    # only set AFTER the config is fully populated so racing threads can never
    # observe an empty config (which would silently drop proxy routing).
    global _HACP_CFG, _HACP_CFG_LOADED
    with _HACP_CFG_LOCK:
        if _HACP_CFG_LOADED:
            return _HACP_CFG
        _cfg = {}
        try:
            _paths = []
            _env = _hacp_os.environ.get("HACP_PLATFORM_CONFIG")
            if _env:
                _paths.append(_env)
            _here = _hacp_os.path.dirname(_hacp_os.path.abspath(__file__))
            _root = _hacp_os.path.dirname(_here)
            _paths += [
                _hacp_os.path.join(_hacp_os.getcwd(), "platform_config.json"),
                _hacp_os.path.join(_root, "platform_config.json"),
                _hacp_os.path.join(_root, "misc", "hacp", "platform_config.json"),
                _hacp_os.path.join(_hacp_os.getcwd(), "misc", "hacp", "platform_config.json"),
            ]
            for _path in _paths:
                if _path and _hacp_os.path.exists(_path):
                    with open(_path, "r") as _f:
                        _cfg.update(_hacp_json.load(_f) or {})
                    break
        except Exception:
            pass
        for _k, _envname in _HACP_ENV_KEYS.items():
            _v = _hacp_os.environ.get(_envname)
            if _v:
                _cfg[_k] = _v
        _HACP_CFG = _cfg
        _HACP_CFG_LOADED = True
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


def _hacp_routing_active():
    """True when this process would route LLM calls through the platform proxy.
    Used by the runner's pre-session container check."""
    try:
        _cfg = _hacp_load_config()
        if not _cfg or not _cfg.get("llm_api_base"):
            return False
        return bool(_cfg.get("%s_model" % _hacp_role()))
    except Exception:
        return False


def _hacp_before_completion(model):
    """Return (override_model, extra_completion_kwargs) or (None, {})."""
    try:
        _cfg = _hacp_load_config()
        if not _cfg or not _cfg.get("llm_api_base"):
            return None, {}
        _role = _hacp_role()
        _override = _cfg.get("%s_model" % _role)
        if not _override:
            return None, {}
        _extra = {"api_base": _cfg["llm_api_base"]}
        _token = _cfg.get("%s_token" % _role)
        if _token:
            _extra["api_key"] = _token
        _effort = _cfg.get("%s_reasoning" % _role)
        if _effort:
            _extra["reasoning_effort"] = _effort
        _cap = _cfg.get("%s_completion" % _role)
        try:
            _cap = int(_cap) if _cap not in (None, "", 0) else None
        except Exception:
            _cap = None
        if _cap and _cap > 0:
            # raise the completion cap for reasoning models that would
            # otherwise burn upstream's default 16384 on reasoning and return
            # content=None (truncated)
            _extra["max_tokens"] = _cap
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


def _hacp_cached_tokens(_usage):
    try:
        _d = _usage.get("prompt_tokens_details") or {}
        _v = _d.get("cached_tokens")
        if _v is None:
            _v = _usage.get("cache_read_input_tokens")
        return int(_v or 0)
    except Exception:
        return 0


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
            "cached_tokens": _hacp_cached_tokens(_usage),
            "latency_s": round(float(latency_s), 4),
        }
        try:
            _hacp_state.last_usage = {
                "model": model,
                "prompt_tokens": _entry["prompt_tokens"],
                "completion_tokens": _entry["completion_tokens"],
                "total_tokens": _entry["total_tokens"],
                "cached_tokens": _entry["cached_tokens"],
            }
        except Exception:
            pass
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

# Anchor 2b: upstream appends the RAW message content (not response_text) to
# the history — a content=None truncation would leak None into chat histories
# and crash downstream JSON extraction (observed in run-2 seeding).
ANCHOR_APPEND = """    new_msg_history.append({"role": "assistant", "content": response['choices'][0]['message']['content']})
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
    if response_text is None:
        # ==== HACP-PLATFORM-INSTRUMENTATION-v1: reasoning models (e.g.
        # DeepSeek v4.1 Flash) return content=None when the completion cap
        # truncates reasoning; upstream would crash on it. Treat as empty.
        response_text = ""
    _hacp_log_usage(model, response, _hacp_latency)
"""

PATCHED_APPEND = """    new_msg_history.append({"role": "assistant", "content": response_text})  # ==== HACP-PLATFORM-INSTRUMENTATION-v1 sanitized (content=None truncations stay empty) ==== 
"""


# ---------------------------------------------------------------------------
# Meta-agent guidance hint (optional, per experiment)
# ---------------------------------------------------------------------------

META_HINT_DEF_ANCHOR = "from agent.llm_withtools import chat_with_agent\n"

# Module-level constant block, inserted right after the imports (HACP-META-HINT).
# It must NOT be defined inside the method body, so it lands at module scope.
META_HINT_DEF_BLOCK = '''\n\n# ==== ''' + META_HINT_MARKER + ''' BEGIN ====
# Platform-injected operator guidance (HyperAgents Control Platform) for this
# experiment; not part of upstream. Appended to the meta-agent instruction.
_HACP_META_HINT = (
    "\\n\\nPLATFORM GUIDANCE (from the experiment operator): "
    "When prior evaluation reports show accuracy 0.0 with EMPTY prediction "
    "distributions, the failure is almost always output-parsing or "
    "output-format handling in task_agent.py — the model answers, but its "
    "response never extracts into an 'accept'/'reject' prediction. That is a "
    "parsing problem, NOT a lack of model capability, so fix prediction "
    "extraction in task_agent.py first, before touching prompts or models. "
    "The strongest known path from 0.0 to 0.71 on paper_review (DGM-H) came "
    "from a structured two-stage evaluation pipeline: (1) weakness "
    "identification from prior evaluation results, then (2) explicit "
    "accept/reject decision rules applied to each paper. Consider that "
    "structure when improving the task agent."
)
# ==== ''' + META_HINT_MARKER + ''' END ====
'''

META_HINT_USE_ANCHOR = '        instruction = f"Modify any part of the codebase at `{repo_path}`."'

META_HINT_USE_PATCHED = META_HINT_USE_ANCHOR + ' + _HACP_META_HINT'


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
        if ANCHOR_CALL in src or ANCHOR_APPEND in src:
            # already-patched file should NOT contain the original anchors
            raise InstrumentationError(
                "file contains both marker and original anchor(s); refusing to patch"
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
    if src.count(ANCHOR_APPEND) != 1:
        raise InstrumentationError(
            f"expected exactly 1 history-append anchor, found {src.count(ANCHOR_APPEND)}. "
            "Upstream agent/llm.py changed; update platform/hacp/instrumentation.py anchors."
        )

    patched = src.replace(ANCHOR_MODULE, ANCHOR_MODULE + MODULE_BLOCK, 1)
    patched = patched.replace(ANCHOR_CALL, PATCHED_CALL, 1)
    patched = patched.replace(ANCHOR_APPEND, PATCHED_APPEND.rstrip() + "\n", 1)

    llm_path.write_text(patched)
    manifest["patched"] = True
    manifest["sha256_before"] = hashlib.sha256(src.encode()).hexdigest()
    manifest["sha256_after"] = hashlib.sha256(patched.encode()).hexdigest()
    return manifest


def apply_meta_hint(workspace: Path) -> dict:
    """Optionally patch <workspace>/meta_agent.py to append operator guidance
    to the meta-agent instruction. Idempotent; anchored like the llm patch."""
    meta_path = Path(workspace) / "meta_agent.py"
    if not meta_path.exists():
        raise InstrumentationError(f"not a HyperAgents workspace: {meta_path} missing")
    src = meta_path.read_text()

    manifest = {"file": str(meta_path), "patched": False, "already_patched": False}

    if META_HINT_MARKER in src:
        manifest["already_patched"] = True
        manifest["patched"] = True
        manifest["sha256_after"] = hashlib.sha256(src.encode()).hexdigest()
        return manifest

    if src.count(META_HINT_DEF_ANCHOR) != 1:
        raise InstrumentationError(
            f"expected exactly 1 meta_agent import anchor, found {src.count(META_HINT_DEF_ANCHOR)}. "
            "Upstream meta_agent.py changed; update platform/hacp/instrumentation.py anchors."
        )
    if src.count(META_HINT_USE_ANCHOR) != 1:
        raise InstrumentationError(
            f"expected exactly 1 meta_agent instruction anchor, found {src.count(META_HINT_USE_ANCHOR)}. "
            "Upstream meta_agent.py changed; update platform/hacp/instrumentation.py anchors."
        )

    patched = src.replace(META_HINT_DEF_ANCHOR, META_HINT_DEF_ANCHOR + META_HINT_DEF_BLOCK, 1)
    patched = patched.replace(META_HINT_USE_ANCHOR, META_HINT_USE_PATCHED, 1)
    meta_path.write_text(patched)
    manifest["patched"] = True
    manifest["sha256_before"] = hashlib.sha256(src.encode()).hexdigest()
    manifest["sha256_after"] = hashlib.sha256(patched.encode()).hexdigest()
    return manifest


def verify_instrumentation(workspace: Path) -> bool:
    llm_path = Path(workspace) / "agent" / "llm.py"
    if not llm_path.exists():
        return False
    src = llm_path.read_text()
    return (
        MARKER in src
        and ANCHOR_CALL not in src
        and ANCHOR_APPEND not in src
        and "_hacp_before_completion" in src
        and "_hacp_routing_active" in src
        and "_HACP_ENV_KEYS" in src
        and "response_text is None" in src  # content=None sanitization present
    )


def verify_meta_hint(workspace: Path) -> bool:
    meta_path = Path(workspace) / "meta_agent.py"
    if not meta_path.exists():
        return False
    src = meta_path.read_text()
    return (
        META_HINT_MARKER in src
        and "_HACP_META_HINT" in src
        and "+ _HACP_META_HINT" in src
        and "PLATFORM GUIDANCE" in src
    )
