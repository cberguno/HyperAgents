"""Regression: the instrumentation config loader must be thread-safe.

Found by the dress rehearsal (2026-10): upstream harnesses evaluate with a
ThreadPoolExecutor; the original loader set its loaded-flag BEFORE populating
the config, so a racing worker thread could observe loaded=True with an empty
config -> proxy routing silently dropped -> 'Missing credentials' direct
provider call. This test fails on the old implementation's race window.
"""
import importlib.util
import shutil
import threading

from hacp.instrumentation import apply_instrumentation


def test_config_loader_thread_safe(tmp_path, upstream_repo):
    ws = tmp_path / "workspace"
    (ws / "agent").mkdir(parents=True)
    shutil.copy(upstream_repo / "agent" / "llm.py", ws / "agent" / "llm.py")
    apply_instrumentation(ws)
    (ws / "platform_config.json").write_text(
        '{"experiment_id": "e1", "meta_model": "openai/gpt-4o-mini", '
        '"task_model": "openai/gpt-4o-mini", "llm_api_base": "http://127.0.0.1:9/proxy/v1", '
        '"meta_token": "t1", "task_token": "t2", "usage_log": "misc/hacp/usage.jsonl"}'
    )

    spec = importlib.util.spec_from_file_location("hacp_test_thread_llm", ws / "agent" / "llm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # make the race window as wide as possible: N threads call simultaneously
    results = []
    barrier = threading.Barrier(12)

    def load():
        barrier.wait()
        results.append(mod._hacp_load_config())

    threads = [threading.Thread(target=load) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 12
    for cfg in results:
        assert cfg.get("llm_api_base") == "http://127.0.0.1:9/proxy/v1", (
            "a racing thread saw an empty config -> proxy routing would be dropped"
        )
    # role override works from any thread (pure argv/module inspection);
    # simulate the meta-agent process context explicitly
    old_argv = list(mod._hacp_sys.argv)
    try:
        mod._hacp_sys.argv = ["run_meta_agent.py", "--repo_path", "./"]
        override, extra = mod._hacp_before_completion("openai/gpt-4o")
    finally:
        mod._hacp_sys.argv = old_argv
    assert override == "openai/openai/gpt-4o-mini"
    assert extra["api_base"] == "http://127.0.0.1:9/proxy/v1"
    assert extra["api_key"] == "t1"  # meta token from the config
