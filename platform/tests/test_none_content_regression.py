"""Regression: reasoning models that hit the completion cap return content=None,
which crashed upstream chat_with_agent with TypeError (found in run-2 seeding:
DeepSeek v4.1 Flash burned the 16384 cap on 3 papers and upstream's
check_for_tool_uses(None) raised).

The patched call-site sanitizes None -> "" (treated as no tool use / no
prediction) instead of failing the whole run; and per-role completion caps
(meta/task) raise the ceiling so reasoning models can finish.
"""
import importlib.util
import shutil

from hacp.instrumentation import apply_instrumentation


def _load_patched_module(tmp_path, upstream_repo):
    ws = tmp_path
    (ws / "agent").mkdir(parents=True, exist_ok=True)
    shutil.copy(upstream_repo / "agent" / "llm.py", ws / "agent" / "llm.py")
    apply_instrumentation(ws)
    spec = importlib.util.spec_from_file_location("llm_none_test", ws / "agent" / "llm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_resp(content=None, prompt_tokens=10, completion_tokens=16384):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def test_none_content_sanitized_instead_of_crash(tmp_path, upstream_repo, monkeypatch):
    mod = _load_patched_module(tmp_path / "a", upstream_repo)
    monkeypatch.setattr(mod, "_HACP_CFG_LOADED", True)
    monkeypatch.setattr(mod, "_HACP_CFG", {})  # no routing: upstream behavior path

    monkeypatch.setattr(
        mod.litellm, "completion",
        lambda **kwargs: _fake_resp(content=None),
    )
    text, history, info = mod.get_response_from_llm(
        "hello", model="deepseek/deepseek-v4.1-flash"
    )
    assert text == ""  # sanitized: no TypeError, no None leaking into history
    assert history[-1]["text"] == ""


def test_completion_cap_forwarded_per_role(tmp_path, upstream_repo, monkeypatch):
    mod = _load_patched_module(tmp_path / "b", upstream_repo)
    envs = {
        "HACP_LLMBASE": "http://127.0.0.1:9/proxy/v1",
        "HACP_META_MODEL": "openai/gpt-6-luna",
        "HACP_META_REASONING": "max",
        "HACP_META_COMPLETION": "49152",
        "HACP_TASK_MODEL": "deepseek/deepseek-v4.1-flash",
        "HACP_TASK_COMPLETION": "32768",
    }
    for k, v in envs.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(mod._hacp_sys, "argv", ["run_meta_agent_sim.py"])
    mod._HACP_CFG = None
    mod._HACP_CFG_LOADED = False

    override, extra = mod._hacp_before_completion("openai/gpt-4o")
    assert override == "openai/openai/gpt-6-luna"
    assert extra["reasoning_effort"] == "max"
    assert extra["max_tokens"] == 49152  # meta cap raised past upstream 16384

    # task role: own cap, no effort
    monkeypatch.setattr(mod._hacp_sys, "argv", ["domains.harness"])
    mod._HACP_CFG = None
    mod._HACP_CFG_LOADED = False
    override_t, extra_t = mod._hacp_before_completion("openai/gpt-4o")
    assert override_t == "openai/deepseek/deepseek-v4.1-flash"
    assert extra_t["max_tokens"] == 32768
    assert "reasoning_effort" not in extra_t


def test_completion_cap_full_flow_reaches_litellm(tmp_path, upstream_repo, monkeypatch):
    """End to end: get_response_from_llm forwards the raised cap + effort."""
    mod = _load_patched_module(tmp_path / "c", upstream_repo)
    for k, v in {
        "HACP_LLMBASE": "http://127.0.0.1:9/proxy/v1",
        "HACP_META_MODEL": "openai/gpt-6-luna",
        "HACP_META_REASONING": "max",
        "HACP_META_COMPLETION": "49152",
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(mod._hacp_sys, "argv", ["run_meta_agent_sim.py"])
    mod._HACP_CFG = None
    mod._HACP_CFG_LOADED = False

    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_resp(content="ok", prompt_tokens=5, completion_tokens=5)

    monkeypatch.setattr(mod.litellm, "completion", fake_completion)
    text, _, _ = mod.get_response_from_llm("hello", model="openai/gpt-4o")
    assert text == "ok"
    assert captured["model"] == "openai/openai/gpt-6-luna"
    assert captured["reasoning_effort"] == "max"
    assert captured["max_tokens"] == 49152
    assert captured["api_base"] == "http://127.0.0.1:9/proxy/v1"