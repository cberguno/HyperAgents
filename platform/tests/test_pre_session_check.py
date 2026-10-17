"""Option C — pre-session routing check: probe command + guard decisions."""
import pytest

from hacp.runner import (
    ROUTING_PROBE,
    build_probe_command,
    evaluate_routing_probe,
)


def test_probe_command_structure():
    cmd = build_probe_command()
    assert cmd[0] == "python" and cmd[1] == "-c"
    src = cmd[2]
    # the probe checks: instrumented llm module, routing active, env config,
    # and live reachability of the platform proxy — inside /hyperagents
    assert "/hyperagents" in src
    assert "_hacp_routing_active" in src
    assert "HACP_LLMBASE" in src
    assert "/api/health" in src


def test_probe_fails_cleanly_without_instrumentation():
    # no _hacp_before_completion -> HACP-NO-PATCH -> exit 3 (run must fail)
    import subprocess
    import sys

    code = (
        "import sys\n"
        "class FakeLlm: pass\n"
        "sys.path.insert(0, '/tmp')\n"
        "f = None\n"
        "print('HACP-NO-PATCH'); sys.exit(3)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    ok, detail = evaluate_routing_probe(r.returncode, r.stdout)
    assert not ok and "HACP-NO-PATCH" in detail


def test_evaluate_probe_matrix():
    ok, detail = evaluate_routing_probe(0, "HACP-OK\n")
    assert ok and "HACP-OK" in detail
    ok, _ = evaluate_routing_probe(3, "HACP-INACTIVE\n")
    assert not ok
    ok, _ = evaluate_routing_probe(3, "HACP-NO-ENV\n")
    assert not ok
    ok, _ = evaluate_routing_probe(3, "HACP-UNREACHABLE [Errno 111] refused\n")
    assert not ok
    ok, _ = evaluate_routing_probe(1, "Traceback ...\nboom")
    assert not ok
    # bytes output (docker exec returns bytes)
    ok, detail = evaluate_routing_probe(0, b"HACP-OK\n")
    assert ok and "HACP-OK" in detail
    # zero output with exit 0 is still a failure (no marker seen)
    ok, _ = evaluate_routing_probe(0, "")
    assert not ok


def test_guard_once_fails_run_on_inactive(monkeypatch):
    """_guard_once publishes routing_check_failed and reports not-ok when the
    probe fails inside the container — the run must fail (runner-level
    integration: we call the guard on a fake container)."""
    from hacp.runner import ExperimentRunner

    class FakeContainer:
        name = "hyperagents-gl-container-runx"

        def exec_run(self, cmd, workdir=None):
            class R:
                exit_code = 3
                output = b"HACP-INACTIVE\n"

            return R()

    events = []

    class FakeBus:
        def publish(self, eid, etype, data):
            events.append((etype, data))

    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.bus = FakeBus()
    ok, detail = runner._guard_once("e1", FakeContainer())
    assert not ok
    assert "HACP-INACTIVE" in detail
    assert events[0][0] == "routing_check_failed"
    assert events[0][1]["container"] == "hyperagents-gl-container-runx"


def test_guard_once_passes_on_active(monkeypatch):
    from hacp.runner import ExperimentRunner

    class FakeContainer:
        name = "hyperagents-gl-container-runy"

        def exec_run(self, cmd, workdir=None):
            class R:
                exit_code = 0
                output = b"HACP-OK\n"

            return R()

    events = []

    class FakeBus:
        def publish(self, eid, etype, data):
            events.append((etype, data))

    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.bus = FakeBus()
    ok, detail = runner._guard_once("e1", FakeContainer())
    assert ok
    assert events[0][0] == "routing_check"
    assert events[0][1]["ok"] is True


def test_guard_container_loop_fails_the_run(monkeypatch):
    """A full guard iteration: a matching container with a failing probe causes
    stop(new_status='failed') with a pre-session-check reason."""
    import threading

    from hacp.runner import ExperimentRunner

    stopped = []

    class FakeContainer:
        def __init__(self):
            self.name = "hyperagents-gl-container-runZ"
            self.id = "cid1"
            self.status = "running"

        def exec_run(self, cmd, workdir=None):
            class R:
                exit_code = 3
                output = b"HACP-NO-PATCH\n"

            return R()

    class FakeContainersAPI:
        def list(self, all=True):
            return [FakeContainer()]

    class FakeClient:
        def __init__(self):
            self.containers = FakeContainersAPI()

    class FakeProc:
        alive = threading.Event()

        def poll(self):
            return None if not self.alive.is_set() else 0

    events = []

    class FakeBus:
        def publish(self, eid, etype, data):
            events.append(etype)

    runner = ExperimentRunner.__new__(ExperimentRunner)
    runner.bus = FakeBus()
    runner._lock = threading.Lock()
    runner._procs = {}

    def fake_stop(exp_id, new_status="stopped", reason=None):
        stopped.append((exp_id, new_status, reason))
        FakeProc.alive.set()

    monkeypatch.setattr(runner, "stop", fake_stop)
    import docker as real_docker

    monkeypatch.setattr(real_docker, "DockerClient", lambda: FakeClient())

    proc = FakeProc()
    # run one poll iteration with a tiny sleep by calling the loop body logic
    # indirectly: guard polls; to keep the test bounded we stop it after fail
    t = threading.Thread(
        target=runner._guard_containers,
        args=("e1", "runZ", proc, 0.2),
        daemon=True,
    )
    t.start()
    t.join(timeout=15)  # returns after stop() is triggered
    assert not t.is_alive()
    assert stopped, "guard must fail the run on routing failure"
    assert stopped[0][1] == "failed"
    assert "pre-session routing check failed" in stopped[0][2]
    assert stopped[0][2].startswith("pre-session routing check failed in container hyperagents-gl-container-runZ")
    assert "routing_check_failed" in events
