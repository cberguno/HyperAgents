"""Experiment watcher: turns upstream's file-based state into live platform events.

Upstream HyperAgents writes all state to the filesystem (metadata.json,
archive.jsonl, report.json, generate.log, per-agent chat histories). This
watcher polls the experiment workspace and converts changes into structured
events for the store, the SSE stream and the dashboard — without any upstream
code changes. It also ingests the instrumentation usage log for
reconciliation against the proxy's authoritative ledger.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

from . import paths

# Mirrors utils/domain_utils.get_domain_score_key (kept in sync manually; only
# M1-supported domains are listed plus the rest for completeness).
DOMAIN_SCORE_KEYS = {
    "search_arena": "overall_accuracy",
    "paper_review": "overall_accuracy",
    "imo_grading": "overall_accuracy",
    "balrog_babyai": "average_progress",
    "balrog_babaisai": "average_progress",
    "balrog_minihack": "average_progress",
    "balrog_nle": "average_progress",
    "genesis_go2walking": "average_fitness",
    "genesis_go2walkback": "average_fitness",
    "genesis_go2hop": "average_fitness",
    "polyglot": "accuracy_score",
    "imo_proof": "points_percentage",
}


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


class ExperimentWatcher(threading.Thread):
    """Polls one experiment's workspace, emitting events; runs while started."""

    def __init__(self, exp_id: str, store, bus, home: Path, runner_log: Optional[Path] = None,
                 interval: float = 1.0):
        super().__init__(daemon=True, name=f"watcher-{exp_id}")
        self.exp_id = exp_id
        self.store = store
        self.bus = bus
        self.home = Path(home)
        self.runner_log = Path(runner_log) if runner_log else None
        self.interval = interval
        self._stop = threading.Event()
        # offsets for tailed files (path -> bytes read)
        self._offsets: dict[Path, int] = {}
        # last-seen snapshots (to detect change, not emit spam)
        self._archive: Optional[list] = None
        self._gen_summaries: dict[str, dict] = {}
        self._usage_lines_seen = 0
        self._proxy_rows_seen = 0

    # -- lifecycle -------------------------------------------------------------
    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.wait(self.interval):
            try:
                self.poll_once()
            except Exception as e:  # never die on transient fs states
                self.store.add_event(self.exp_id, "watcher_error", {"error": str(e)})

    # -- tailing ------------------------------------------------------------------
    def _tail(self, path: Path, max_lines: int = 50) -> list[str]:
        if not path.exists():
            return []
        size = path.stat().st_size
        off = self._offsets.get(path, 0)
        if size <= off:
            return []
        try:
            with open(path, "rb") as f:
                f.seek(off)
                data = f.read()
        except OSError:
            return []
        self._offsets[path] = size
        text = data.decode("utf-8", errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return lines[-max_lines:]

    # -- polls ---------------------------------------------------------------------
    def poll_once(self):
        ws = paths.workspace_dir(self.exp_id, self.home)
        if not ws.exists():
            return

        # 1. Runner subprocess stdout
        if self.runner_log:
            for line in self._tail(self.runner_log):
                self.bus.publish(self.exp_id, "log", {"source": "runner", "line": line[:2000]})

        # 2. Instrumentation usage log (reconciliation source)
        usage_path = ws / paths.usage_log_relpath()
        self._poll_local_usage(usage_path)

        # 3. Output directory artifacts
        out_root = ws / "outputs"
        gen_dirs = sorted(
            [p for p in out_root.glob("generate_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
        )
        if not gen_dirs:
            return
        output_dir = gen_dirs[-1]  # latest run
        if self._output_dir_changed(output_dir):
            self.bus.publish(
                self.exp_id, "output_dir", {"output_dir": str(output_dir)}
            )
            self.store.update_experiment(self.exp_id, output_dir=str(output_dir))
            self._offsets.pop(output_dir / "generate.log", None)
            self._archive = None

        # 3a. generate.log tail
        for line in self._tail(output_dir / "generate.log"):
            self.bus.publish(self.exp_id, "log", {"source": "generate", "line": line[:2000]})

        # 3b. archive
        archive_data = _read_json_last_line(output_dir / "archive.jsonl")
        if archive_data is not None and archive_data.get("archive") != self._archive:
            self._archive = archive_data.get("archive")
            self.bus.publish(self.exp_id, "archive", {"generations": self._archive})
            self.store.update_experiment(
                self.exp_id, current_gen=str(self._archive[-1]) if self._archive else None
            )

        # 3c. per-generation metadata + scores (chronological by completion
        # time so duration chains are correct: initial -> 1 -> 2 ...)
        summaries = {}
        prev_mtime = None
        def _gen_sort_key(p: Path):
            try:
                return (p / "metadata.json").stat().st_mtime
            except OSError:
                return 0.0
        for gen_dir in sorted(output_dir.glob("gen_*"), key=_gen_sort_key):
            s = self._generation_summary(gen_dir, prev_mtime)
            prev_mtime = s["completed_ts"] if s else prev_mtime
            if s:
                summaries[s["gen"]] = s
        changed = summaries != self._gen_summaries
        if changed:
            self._gen_summaries = summaries
            self.bus.publish(
                self.exp_id, "generations", {"generations": list(summaries.values())}
            )

    def _output_dir_changed(self, output_dir: Path) -> bool:
        key = "__output_dir__"
        if self._offsets.get(key) != hash(str(output_dir)):
            self._offsets[key] = hash(str(output_dir))
            return True
        return False

    def _poll_local_usage(self, usage_path: Path):
        if not usage_path.exists():
            return
        lines = None
        size = usage_path.stat().st_size
        off = self._offsets.get(usage_path, 0)
        if size <= off:
            return
        with open(usage_path, "rb") as f:
            f.seek(off)
            data = f.read()
        self._offsets[usage_path] = size
        lines = [
            json.loads(ln)
            for ln in data.decode("utf-8", errors="replace").splitlines()
            if ln.strip()
        ]
        for entry in lines:
            self.store.add_usage(
                self.exp_id,
                role=entry.get("role", "unknown"),
                model=entry.get("model", "unknown"),
                prompt_tokens=int(entry.get("prompt_tokens") or 0),
                completion_tokens=int(entry.get("completion_tokens") or 0),
                total_tokens=int(entry.get("total_tokens") or 0),
                latency_s=entry.get("latency_s"),
                status="ok",
                source="local",
                ts=entry.get("ts"),
            )
        if lines:
            local = self.store.usage_totals(self.exp_id, source="local")
            proxy = self.store.usage_totals(self.exp_id, source="proxy")
            self.bus.publish(
                self.exp_id,
                "usage_reconciled",
                {"local_requests": local["requests"], "proxy_requests": proxy["requests"]},
            )
            # local rows without a matching proxy row mean the agent found a way
            # to call a provider directly (no keys in the env, so unlikely) —
            # surface it loudly.
            if local["requests"] > proxy["requests"]:
                self.bus.publish(
                    self.exp_id,
                    "accounting_warning",
                    {
                        "message": "local usage log has more calls than the proxy ledger",
                        "local": local["requests"],
                        "proxy": proxy["requests"],
                    },
                )

    def _generation_summary(self, gen_dir: Path, prev_mtime: Optional[float] = None) -> Optional[dict]:
        metadata = _read_json(gen_dir / "metadata.json")
        if metadata is None:
            return None
        gen = str(metadata.get("current_genid", gen_dir.name.replace("gen_", "")))
        try:
            completed_ts = (gen_dir / "metadata.json").stat().st_mtime
        except OSError:
            completed_ts = None
        summary = {
            "gen": gen,
            "parent": metadata.get("parent_genid"),
            "run_eval": metadata.get("run_eval"),
            "run_full_eval": metadata.get("run_full_eval"),
            "valid_parent": metadata.get("valid_parent"),
            "parent_agent_success": metadata.get("parent_agent_success"),
            "completed_ts": completed_ts,
            "duration_s": (
                round(completed_ts - prev_mtime, 2)
                if completed_ts is not None and prev_mtime is not None
                else None
            ),
            "scores": {},
        }
        for eval_dir in sorted(gen_dir.iterdir()):
            if not eval_dir.is_dir() or not eval_dir.name.endswith("eval"):
                continue
            name = eval_dir.name
            domain = name[:-len("_eval")]
            split = "train"
            report = _read_json(eval_dir / "report.json")
            if report is None:
                continue
            key = DOMAIN_SCORE_KEYS.get(domain, "overall_accuracy")
            score = report.get(key)
            if score is None:
                continue
            summary["scores"].setdefault(domain, {})[split] = score
        # val/test live in <domain>_eval_val / _test dirs
        for suffix, split in (("_eval_val", "val"), ("_eval_test", "test")):
            for eval_dir in sorted(gen_dir.iterdir()):
                if not eval_dir.is_dir() or not eval_dir.name.endswith(suffix):
                    continue
                domain = eval_dir.name[: -len(suffix)]
                report = _read_json(eval_dir / "report.json")
                if report is None:
                    continue
                key = DOMAIN_SCORE_KEYS.get(domain, "overall_accuracy")
                score = report.get(key)
                if score is None:
                    continue
                summary["scores"].setdefault(domain, {})[split] = score
        return summary


def _read_json_last_line(path: Path):
    """Parse the last non-empty JSON line of a JSONL file (upstream archive.jsonl)."""
    if not path.exists():
        return None
    try:
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        return json.loads(lines[-1]) if lines else None
    except Exception:
        return None
