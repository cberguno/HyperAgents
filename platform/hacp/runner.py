"""Experiment runner: supervises the (unmodified) upstream loop as a subprocess.

Design decisions (documented honestly):

* **No upstream modification**: the runner executes ``generate_loop.py`` (or the
  offline ``stub_loop`` for tests/demos) as a plain subprocess inside the
  experiment's workspace copy.
* **Stop/pause**: upstream has no cooperative pause hook. Stop = SIGINT
  (KeyboardInterrupt), which lets upstream's ``finally`` blocks clean up Docker
  containers; the runner then force-removes any leftover containers named for
  this run (best-effort) and marks the experiment ``stopped``/``paused``.
  An in-flight generation is lost — upstream ``--resume_from`` re-runs it from
  the last completed generation boundary. This matches upstream's own
  checkpointing semantics.
* **Resume**: launches ``generate_loop.py --resume_from <output_dir>``.
* **Budget kill-switch**: a supervisor thread watches the authoritative proxy
  ledger; on breach it SIGINTs the loop and marks ``paused_budget``. The proxy
  has already refused further LLM calls at that point (server-side).
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from . import paths
from .config import ExperimentConfig, ServiceConfig
from .store import Store, EXPERIMENT_STATUSES

STOP_STATUSES = {"stopped", "paused", "paused_budget", "finished", "failed"}
# 'finished' is startable: resuming it continues the same lineage (upstream
# --resume_from semantics) — e.g. after raising max_generation.
STARTABLE = {"created", "paused", "stopped", "paused_budget", "failed", "finished"}

# Pre-session routing probe executed inside every generation container before
# the generation's work is trusted (see _guard_containers). Exit codes:
#   0 -> routing active (instrumented + env-based config + proxy reachable)
#   3 -> routing NOT active (agent de-instrumented its lineage, or no env)
ROUTING_PROBE = '''
import os, sys, urllib.request
sys.path.insert(0, "/hyperagents")
try:
    from agent import llm as L
except Exception as e:
    print("HACP-NO-LLM %s" % e); sys.exit(3)
f = getattr(L, "_hacp_before_completion", None)
if f is None:
    print("HACP-NO-PATCH"); sys.exit(3)
# The probe runs as `python -c <src>` so argv-based role detection cannot
# identify the caller; the guard sets HACP_PROBE_ROLE=meta explicitly and the
# loader honors it (role detection is for the real agent processes, whose
# argv/module context is unambiguous).
_role = os.environ.get("HACP_PROBE_ROLE", "meta")
if not L._hacp_routing_active():
    print("HACP-INACTIVE (routing_active false)"); sys.exit(3)
_override, _extra = f("openai/gpt-4o")
if not _override or not _extra.get("api_base"):
    print("HACP-INACTIVE (no meta override)"); sys.exit(3)
if not _override.endswith(os.environ.get("HACP_PROBE_EXPECT_MODEL", _override.rsplit("/", 1)[0])):
    pass  # informational only
base = os.environ.get("HACP_LLMBASE", "").rsplit("/proxy/v1", 1)[0]
if not base:
    print("HACP-NO-ENV"); sys.exit(3)
try:
    urllib.request.urlopen(base + "/api/health", timeout=8).read()
except Exception as e:
    print("HACP-UNREACHABLE %s" % e); sys.exit(3)
print("HACP-OK role=%s model=%s" % (_role, _override)); sys.exit(0)
'''


def build_probe_command() -> list:
    # NOTE: HACP_PROBE_ROLE is set by the guard's container exec env; the
    # loader reads it so routing checks work in probe processes (whose argv
    # cannot indicate a role).
    return ["python", "-c", ROUTING_PROBE]


def build_probe_command() -> list:
    return ["python", "-c", ROUTING_PROBE]


def evaluate_routing_probe(exit_code, output="") -> tuple:
    """(ok, detail) for a probe result."""
    text = (output or "").decode("utf-8", "replace") if isinstance(output, bytes) else (output or "")
    ok = exit_code == 0 and "HACP-OK" in text
    detail = (text.strip().splitlines() or [""])[-1] if text.strip() else f"exit {exit_code}"
    return ok, f"exit={exit_code} {detail[:160]}"


class RunnerError(RuntimeError):
    pass


class ExperimentRunner:
    def __init__(self, store: Store, bus, service: ServiceConfig):
        self.store = store
        self.bus = bus
        self.service = service
        self.home = Path(service.home)
        self._procs: dict[str, subprocess.Popen] = {}
        self._watchers: dict[str, "object"] = {}
        self._supervisors: dict[str, threading.Thread] = {}
        self._stop_targets: dict[str, str] = {}
        self._docker_proxies: dict[str, "object"] = {}
        self._guards: dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ helpers
    def _subprocess_env(self) -> dict:
        """Environment for experiment subprocesses.

        Security: ALL provider API keys are stripped — experiment processes
        (and the containers they spawn) never see provider credentials; the
        only LLM path is the platform proxy with per-experiment HACP tokens.
        """
        from .config import PROVIDER_KEY_VARS

        env = {k: v for k, v in os.environ.items() if k not in PROVIDER_KEY_VARS}
        env["PYTHONUNBUFFERED"] = "1"
        env["MPLBACKEND"] = "Agg"  # headless matplotlib for upstream analysis plots
        return env

    def _container_env(self, exp_id: str, cfg: ExperimentConfig) -> dict:
        """HACP_* environment injected into every container the run creates
        (via the Docker socket proxy) and also set for host-side subprocesses.

        Env vars cannot be removed by file operations inside the container —
        this is the channel that survives an agent deleting
        platform_config.json (run e6e3168d3b3ca gen 2 did exactly that).
        """
        return {
            "HACP_EXPERIMENT_ID": exp_id,
            "HACP_LLMBASE": self.service.proxy_v1_base,
            "HACP_META_MODEL": cfg.meta_model,
            "HACP_TASK_MODEL": cfg.task_model,
            "HACP_META_TOKEN": f"hacp-{exp_id}-meta",
            "HACP_TASK_TOKEN": f"hacp-{exp_id}-task",
            "HACP_USAGE_LOG": paths.usage_log_relpath(),
            "HACP_META_REASONING": cfg.meta_reasoning_effort or "",
            "HACP_TASK_REASONING": cfg.task_reasoning_effort or "",
            "HACP_META_COMPLETION": str(cfg.meta_completion_cap or ""),
            "HACP_TASK_COMPLETION": str(cfg.task_completion_cap or ""),
            "HACP_ROUTING_ACTIVE": "1",
            # context manager (per-task budgets; empty = platform defaults)
        }

    def _workspace(self, exp_id: str) -> Path:
        return paths.workspace_dir(exp_id, self.home)

    def _exp_dir(self, exp_id: str) -> Path:
        return paths.experiment_dir(exp_id, self.home)

    def _exp(self, exp_id: str) -> dict:
        exp = self.store.get_experiment(exp_id)
        if exp is None:
            raise RunnerError(f"experiment {exp_id} not found")
        return exp

    def _stub_loop_path(self) -> Path:
        return paths.platform_dir() / "tests" / "stub_loop.py"

    def _new_run_id(self) -> str:
        from datetime import datetime

        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # ------------------------------------------------------------------ lifecycle
    def create(self, config: dict, slot: Optional[int] = None) -> dict:
        cfg = ExperimentConfig(**{k: v for k, v in config.items() if k in ExperimentConfig.model_fields})
        exp = self.store.create_experiment(cfg.model_dump())
        if slot is None:
            slot = self.store.first_free_slot()
            if slot is None:
                slot = self.store.expand_slots(1) - 1
        self.store.assign_slot(slot, exp["id"])
        self.bus.publish(exp["id"], "created", {"name": cfg.name, "slot": slot})
        return exp

    def start(self, exp_id: str, resume: bool = False) -> dict:
        exp = self._exp(exp_id)
        cfg = ExperimentConfig(**{k: v for k, v in exp["config"].items() if k in ExperimentConfig.model_fields})
        live_proc = self._procs.get(exp_id)
        still_alive = live_proc is not None and live_proc.poll() is None
        if still_alive:
            raise RunnerError(f"experiment {exp_id} is already running")
        if live_proc is not None:
            with self._lock:  # dead entry left by an exited run; drop it
                self._procs.pop(exp_id, None)
        if exp["status"] not in STARTABLE:
            raise RunnerError(f"cannot start experiment in status '{exp_id}' state '{exp['status']}'")

        # 1. Workspace (copy of pristine repo + instrumentation + platform config)
        from .workspace import prepare_workspace

        ws = self._workspace(exp_id)
        if not ws.exists():
            manifest = prepare_workspace(exp_id, cfg, self.service)
            self.bus.publish(exp_id, "workspace_prepared", {"upstream_commit": manifest.get("upstream_commit")})
        else:
            from .workspace import verify_instrumentation
            if not verify_instrumentation(ws):
                raise RunnerError("workspace exists but is not instrumented")

        # 1b. Ensure the meta-agent guidance hint matches the config
        # (idempotent; no-op when already applied)
        if cfg.meta_hint:
            from .instrumentation import apply_meta_hint

            apply_meta_hint(ws)

        # 2. Resume bookkeeping
        run_id = exp.get("run_id") or self._new_run_id()
        resume_from = None
        if resume:
            out_root = ws / "outputs"
            gen_dirs = sorted(
                [p for p in out_root.glob("generate_*") if p.is_dir()],
                key=lambda p: p.stat().st_mtime,
            )
            if not gen_dirs:
                raise RunnerError("nothing to resume: no previous output directory")
            resume_from = str(gen_dirs[-1])
            self.bus.publish(exp_id, "resuming", {"resume_from": resume_from})
        else:
            run_id = self._new_run_id()

        # 3. Build the subprocess command (unmodified upstream entry point)
        env = self._subprocess_env()
        env["HACP_PLATFORM_CONFIG"] = str(ws / paths.platform_config_relpath())
        # Belt and braces: env-based routing for host-side subprocesses too
        # (immune to the agent deleting config files in the workspace).
        hacp_env = self._container_env(exp_id, cfg)
        env.update(hacp_env)
        if cfg.seed is not None:
            env.setdefault("PYTHONHASHSEED", str(cfg.seed))

        if cfg.loop == "stub":
            if not self._stub_loop_path().exists():
                raise RunnerError(f"stub loop not found at {self._stub_loop_path()}")
            cmd = [
                sys.executable, str(self._stub_loop_path()),
                "--workspace", str(ws),
                "--run-id", run_id,
                "--max-generation", str(cfg.max_generation),
                "--sleep", "1.2",
            ]
            if resume_from:
                cmd += ["--resume-from", resume_from]
        else:
            # Real upstream loop: needs Docker + seeded initial evaluations.
            from .workspace import workspace_status

            status = workspace_status(exp_id, self.home)
            if not status["initial_evals_ready"]:
                raise RunnerError(
                    "initial evaluations missing — run the seed phase first "
                    f"(POST /api/experiments/{exp_id}/seed). Missing: "
                    f"{status['missing_initial_evals']}"
                )
            args = cfg.generate_loop_args(run_id, resume_from)
            cmd = [sys.executable] + args  # e.g. generate_loop.py --domains ...
            self.bus.publish(exp_id, "command", {"cmd": " ".join(shlex.quote(c) for c in cmd)})

            # Option C: Docker socket proxy + sitecustomize shim — every
            # container upstream creates for this run (gl-/nextparent-/ens-)
            # gets the HACP_* env injected, which no file operation inside the
            # container can remove. Delivery: docker-py's zero-argument
            # DockerClient() ignores DOCKER_HOST, so the shim (on PYTHONPATH)
            # redirects the no-arg construction to the proxy socket.
            from .docker_proxy import DockerEnvProxy

            # NOTE: AF_UNIX paths are capped at ~108 chars; the experiment
            # directory is too deep — use a short /tmp path (per-experiment name)
            proxy = DockerEnvProxy(
                env=hacp_env,
                socket_dir="/tmp",
            )
            sock_path = proxy.start()
            with self._lock:
                self._docker_proxies[exp_id] = proxy
            shim_dir = str(Path(__file__).resolve().parent / "docker_shim")
            env["PYTHONPATH"] = (
                shim_dir + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else shim_dir
            )
            env["HACP_DOCKER_PROXY_URL"] = f"unix://{sock_path}"
            env["DOCKER_HOST"] = f"unix://{sock_path}"  # for any from_env() caller
            self.bus.publish(
                exp_id, "docker_proxy_started",
                {"socket": sock_path, "injected_keys": sorted(hacp_env)},
            )

        # 4. Launch
        exp_dir = self._exp_dir(exp_id)
        exp_dir.mkdir(parents=True, exist_ok=True)
        attempt = (exp.get("attempt") or 0) + 1
        runner_log = exp_dir / f"runner_{attempt}.log"
        log_f = open(runner_log, "ab")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ws),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        with self._lock:
            self._procs[exp_id] = proc
        self.store.update_experiment(
            exp_id,
            status="running",
            run_id=run_id,
            started_at=time.time(),
            finished_at=None,
            error=None,
            attempt=attempt,
            current_gen=None,
        )
        self.bus.publish(
            exp_id, "started",
            {"run_id": run_id, "pid": proc.pid, "resume": bool(resume_from),
             "command": " ".join(shlex.quote(c) for c in cmd)},
        )

        # 5. Watcher + supervisor
        from .watcher import ExperimentWatcher

        watcher = ExperimentWatcher(exp_id, self.store, self.bus, self.home, runner_log=runner_log)
        watcher.start()
        with self._lock:
            self._watchers[exp_id] = watcher

        sup = threading.Thread(
            target=self._supervise, args=(exp_id, proc, cfg, watcher, run_id), daemon=True
        )
        sup.start()
        with self._lock:
            self._supervisors[exp_id] = sup

        # Option C: pre-session container guard — verify proxy routing is
        # active inside every container this run creates; fail the run if not.
        if cfg.loop != "stub":
            guard = threading.Thread(
                target=self._guard_containers,
                args=(exp_id, run_id, proc), daemon=True,
                name=f"hacp-guard-{exp_id}",
            )
            guard.start()
            with self._lock:
                self._guards[exp_id] = guard
        return self.store.get_experiment(exp_id)

    def stop(self, exp_id: str, new_status: str = "stopped", reason: Optional[str] = None) -> dict:
        if new_status not in {"stopped", "paused", "paused_budget", "failed"}:
            raise RunnerError(f"invalid stop status '{new_status}'")
        exp = self._exp(exp_id)
        proc = self._procs.get(exp_id)
        data = {"reason": reason or new_status}
        if proc is not None and proc.poll() is None:
            # register intent so the supervisor doesn't misclassify the exit
            self._stop_targets[exp_id] = new_status
            self.bus.publish(exp_id, "stopping", data)
            try:
                os.kill(proc.pid, signal.SIGINT)  # graceful: upstream finally-blocks run
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)
                data["force_killed"] = True
            self._cleanup_docker(exp.get("run_id"))
        self._stop_targets.pop(exp_id, None)
        self._finalize(exp_id, new_status, reason=reason)
        self.bus.publish(exp_id, new_status, data)
        return self.store.get_experiment(exp_id)

    def pause(self, exp_id: str) -> dict:
        return self.stop(exp_id, new_status="paused", reason="user pause")

    def resume(self, exp_id: str) -> dict:
        self._exp(exp_id)
        return self.start(exp_id, resume=True)

    # ------------------------------------------------------------------ seeding
    def seed(self, exp_id: str) -> dict:
        """Run the upstream initial-evaluation steps (host-side, LLM via proxy).

        Equivalent to upstream setup_initial.sh for the paper_review domain:
        curate subsets -> harness on train/val/test -> report. Requires an
        approved upstream (openrouter) and is therefore never invoked by tests
        or the offline demo.
        """
        exp = self._exp(exp_id)
        cfg = ExperimentConfig(**{k: v for k, v in exp["config"].items() if k in ExperimentConfig.model_fields})
        if exp["status"] == "running":
            raise RunnerError("cannot seed a running experiment")
        if cfg.loop == "stub":
            # stub loop fabricates initial evals itself
            raise RunnerError("stub experiments do not need seeding")
        if cfg.domains != ["paper_review"]:
            raise RunnerError(
                f"automatic seeding is implemented for paper_review only (M1); "
                f"requested: {cfg.domains}. Follow upstream setup_initial.sh for other domains."
            )
        if self.service.upstream != "openrouter":
            raise RunnerError(
                "seeding makes real LLM calls — requires HACP_UPSTREAM=openrouter "
                "and explicit user approval"
            )

        from .workspace import prepare_workspace, workspace_status

        ws = self._workspace(exp_id)
        if not ws.exists():
            prepare_workspace(exp_id, cfg, self.service)
        if workspace_status(exp_id, self.home)["initial_evals_ready"]:
            return {"seeded": True, "already": True}

        env = self._subprocess_env()
        env["HACP_PLATFORM_CONFIG"] = str(ws / paths.platform_config_relpath())
        # env-based routing for host-side seed subprocesses as well (immune
        # to any file tampering in the workspace)
        exp_cfg = ExperimentConfig(**{k: v for k, v in exp["config"].items() if k in ExperimentConfig.model_fields})
        env.update(self._container_env(exp_id, exp_cfg))

        steps = [
            [sys.executable, "-m", "domains.paper_review.curate_subsets"],
            [sys.executable, "-m", "domains.harness", "--domain", "paper_review",
             "--run_id", "initial_paper_review_filtered_100_train_0",
             "--subset", "_filtered_100_train", "--num_samples", "10"],
            [sys.executable, "-m", "domains.harness", "--domain", "paper_review",
             "--run_id", "initial_paper_review_filtered_100_val_0",
             "--subset", "_filtered_100_val", "--num_samples", "10"],
            [sys.executable, "-m", "domains.harness", "--domain", "paper_review",
             "--run_id", "initial_paper_review_filtered_100_test_0",
             "--subset", "_filtered_100_test", "--num_samples", "10"],
            [sys.executable, "-m", "domains.report", "--domain", "paper_review",
             "--dname", "./outputs/initial_paper_review_filtered_100_train_0"],
            [sys.executable, "-m", "domains.report", "--domain", "paper_review",
             "--dname", "./outputs/initial_paper_review_filtered_100_val_0"],
            [sys.executable, "-m", "domains.report", "--domain", "paper_review",
             "--dname", "./outputs/initial_paper_review_filtered_100_test_0"],
        ]
        self.store.update_experiment(exp_id, status="seeding")
        self.bus.publish(exp_id, "seeding_started", {"steps": len(steps)})
        seed_log = self._exp_dir(exp_id) / "seed.log"
        seed_log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(seed_log, "ab") as lf:
                for i, step in enumerate(steps):
                    self.bus.publish(exp_id, "seed_step", {"index": i, "cmd": " ".join(step)})
                    rc = subprocess.call(step, cwd=str(ws), stdout=lf, stderr=subprocess.STDOUT, env=env)
                    if rc != 0:
                        raise RunnerError(f"seed step {i} failed with rc={rc} (see {seed_log})")
        except Exception as e:
            self._finalize(exp_id, "failed", reason=str(e))
            raise
        self._finalize(exp_id, "created")
        self.bus.publish(exp_id, "seeding_finished", {"ok": True})
        return {"seeded": True}

    # ------------------------------------------------------------------ internals
    def _supervise(self, exp_id: str, proc: subprocess.Popen, cfg: ExperimentConfig, watcher, run_id: str | None = None):
        """Watch one subprocess: budget, wallclock, exit status.

        Guards against resume races: this supervisor only finalizes the
        experiment while *its own* process is still the registered one. If a
        newer run (resume) has replaced it, this thread exits without
        touching the new run's state.
        """
        try:
            while proc.poll() is None:
                time.sleep(2.0)
                with self._lock:
                    if self._procs.get(exp_id) is not proc:
                        return  # superseded by a resume
                exp = self.store.get_experiment(exp_id)
                if exp is None:
                    return
                # Include in-flight reservations: a burst of concurrent calls
                # still trips the pause even before completions land in the ledger
                from .budget import combined_budget_state

                bstate = combined_budget_state(self.store, exp)
                if bstate["exceeded"]:
                    self.bus.publish(
                        exp_id, "budget_exceeded",
                        {"reason": bstate["exceeded_reason"], **{k: bstate[k] for k in ("total_tokens", "cost_usd", "wallclock_hours")}},
                    )
                    self.stop(exp_id, new_status="paused_budget", reason=bstate["exceeded_reason"])
                    return
        finally:
            # our watcher always stops with us
            try:
                watcher.stop()
            except Exception:
                pass
            time.sleep(0.3)
            with self._lock:
                current = self._procs.get(exp_id)
                if current is not proc:
                    # a newer run owns the experiment now; do not touch it
                    if self._watchers.get(exp_id) is watcher:
                        self._watchers.pop(exp_id, None)
                    return
                self._procs.pop(exp_id, None)
                if self._watchers.get(exp_id) is watcher:
                    self._watchers.pop(exp_id, None)
                self._supervisors.pop(exp_id, None)
                self._guards.pop(exp_id, None)
                proxy = self._docker_proxies.pop(exp_id, None)
            if proxy is not None:
                try:
                    proxy.stop()
                except Exception:
                    pass
            proc_rc = proc.poll()
            exp = self.store.get_experiment(exp_id)
            if exp is None:
                return
            if exp["status"] != "running":
                return  # stop()/budget already finalized
            from .budget import combined_budget_state

            bstate = combined_budget_state(self.store, exp)
            if bstate["exceeded"]:
                self._finalize(
                    exp_id, "paused_budget",
                    reason=f"budget exceeded ({bstate['exceeded_reason']}) at loop exit",
                )
            elif proc_rc == 2 and cfg.loop == "stub":
                # stub-loop convention: exit code 2 = the proxy refused service
                # for budget reasons (429 hacp_budget_exhausted)
                self._finalize(
                    exp_id, "paused_budget",
                    reason="budget exhausted: LLM access refused by platform proxy",
                )
            else:
                self._finalize(
                    exp_id, "finished" if proc_rc == 0 else "failed",
                    reason=f"exit code {proc_rc}",
                )
            self.bus.publish(exp_id, "loop_exited", {"returncode": proc_rc})
            # upstream's own finally-block cleanup may have been preempted by
            # an exception — always sweep containers named for this run
            self._cleanup_docker(run_id)

    def _finalize(self, exp_id: str, status: str, reason: Optional[str] = None):
        """Idempotent: only transition out of active statuses."""
        if status not in EXPERIMENT_STATUSES:
            raise RunnerError(f"invalid status {status}")
        exp = self.store.get_experiment(exp_id)
        if exp is None:
            return
        if exp["status"] not in {"running", "seeding", "created"}:
            return  # someone already finalized (e.g. stop() won the race)
        self.store.update_experiment(
            exp_id, status=status, finished_at=time.time(), error=reason
        )

    # ------------------------------------------------------------------ container guard (Option C)
    def _guard_containers(self, exp_id: str, run_id: str, proc: subprocess.Popen,
                          poll_interval: float = 5.0):
        """Watch for containers created for this run (via the REAL docker
        socket, not the proxy) and pre-session-check each one: routing must be
        active inside the container before its generation's work is trusted.
        On failure the run is failed immediately with a clear reason."""
        try:
            import docker

            client = docker.DockerClient()  # real daemon; guard must not depend on the proxy
        except Exception as e:
            self.bus.publish(exp_id, "routing_check_error", {"error": f"guard cannot reach docker: {e}"})
            return
        seen: set[str] = set()
        while proc.poll() is None:
            try:
                containers = client.containers.list(all=True)
            except Exception:
                time.sleep(poll_interval)
                continue
            for c in containers:
                name = c.name or ""
                if run_id not in name or c.id in seen:
                    continue
                seen.add(c.id)
                # wait for the container to actually run before probing
                deadline = time.time() + 30
                while time.time() < deadline and c.status != "running" and proc.poll() is None:
                    time.sleep(1.0)
                    try:
                        c.reload()
                    except Exception:
                        break
                if proc.poll() is not None:
                    return
                ok, detail = self._guard_once(exp_id, c)
                if not ok:
                    try:
                        self.stop(exp_id, new_status="failed",
                                  reason=f"pre-session routing check failed in container {name}: {detail}")
                    except Exception:
                        pass
                    return
            time.sleep(poll_interval)

    def _guard_once(self, exp_id: str, container) -> tuple:
        """Run the routing probe inside one container. Returns (ok, detail)."""
        try:
            result = container.exec_run(
                build_probe_command(),
                workdir="/hyperagents",
                environment={"HACP_PROBE_ROLE": "meta"},
            )
            ok, detail = evaluate_routing_probe(result.exit_code, result.output)
        except Exception as e:
            ok, detail = False, f"probe execution error: {e}"
        self.bus.publish(exp_id, "routing_check" if ok else "routing_check_failed",
                         {"container": container.name, "ok": ok, "detail": detail})
        return ok, detail

    def _cleanup_docker(self, run_id: Optional[str]):
        """Best-effort removal of containers upstream names for this run
        (hyperagents-gl-container-<run_id>, -nextparent-, -ens-)."""
        if not run_id:
            return
        try:
            import docker  # noqa

            client = docker.DockerClient()
            prefixes = (
                f"hyperagents-gl-container-{run_id}",
                f"hyperagents-nextparent-container-{run_id}",
                f"hyperagents-ens-container-{run_id}",
            )
            for c in client.containers.list(all=True):
                if any((c.name or "").startswith(p) for p in prefixes):
                    try:
                        c.remove(force=True)
                        self.bus.publish(None, "docker_cleanup", {"container": c.name})
                    except Exception:
                        pass
        except Exception:
            pass  # no docker daemon / SDK — nothing to clean

    # ------------------------------------------------------------------ shutdown
    def shutdown(self):
        for exp_id in list(self._procs.keys()):
            try:
                self.stop(exp_id, new_status="stopped", reason="platform shutdown")
            except Exception:
                pass
