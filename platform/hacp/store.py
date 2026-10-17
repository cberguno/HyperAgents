"""SQLite-backed experiment registry, event log and usage ledger + event bus.

M1 keeps everything in one SQLite database (WAL mode) inside the platform
home, so the whole control plane is a single portable directory. The store is
thread-safe (a single connection guarded by a lock); the event bus bridges
worker threads (runner, watcher, proxy) to asyncio SSE subscribers.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .config import estimate_cost, normalize_model

EXPERIMENT_STATUSES = [
    "created",    # registered, workspace not prepared / not started
    "seeding",    # running initial evaluations (upstream setup_initial.sh steps)
    "running",    # evolution loop subprocess alive
    "paused",     # stopped by user, intends to resume
    "paused_budget",  # stopped by budget kill-switch, intends to resume w/ higher budget
    "stopped",    # stopped by user (no pending resume)
    "finished",   # loop completed all generations
    "failed",     # loop exited with an error
]


def new_experiment_id() -> str:
    return "e" + secrets.token_hex(6)


class Store:
    def __init__(self, db_file: Path):
        self.db_file = Path(db_file)
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_file), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._create_tables()

    # -- schema ------------------------------------------------------------
    def _create_tables(self):
        cur = self._conn
        cur.execute(
            """CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'created',
                run_id TEXT,
                output_dir TEXT,
                started_at REAL,
                finished_at REAL,
                current_gen TEXT,
                error TEXT,
                attempt INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                ts REAL NOT NULL,
                type TEXT NOT NULL,
                data_json TEXT NOT NULL
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                ts REAL NOT NULL,
                role TEXT,
                model TEXT,
                prompt_tokens INTEGER DEFAULT 0,
                completion_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0,
                latency_s REAL,
                status TEXT DEFAULT 'ok',
                source TEXT DEFAULT 'proxy',
                generation TEXT
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS slots (
                position INTEGER PRIMARY KEY,
                experiment_id TEXT
            )"""
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_exp ON events(experiment_id, id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_usage_exp ON usage(experiment_id, id)"
        )
        self._conn.commit()
        self._migrate()

    def _migrate(self):
        """In-place schema migrations for databases created by older versions."""
        with self._lock:
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(usage)")}
            if "generation" not in cols:
                self._conn.execute("ALTER TABLE usage ADD COLUMN generation TEXT")
            self._conn.commit()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _now() -> float:
        return time.time()

    def _touch(self, exp_id: str, **fields):
        sets = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values())
        with self._lock:
            self._conn.execute(
                f"UPDATE experiments SET updated_at=?, {sets} WHERE id=?",
                [self._now(), *vals, exp_id],
            )
            self._conn.commit()

    # -- experiments ---------------------------------------------------------
    def create_experiment(self, config: dict, exp_id: str | None = None) -> dict:
        exp_id = exp_id or new_experiment_id()
        now = self._now()
        with self._lock:
            self._conn.execute(
                """INSERT INTO experiments
                   (id, name, config_json, status, created_at, updated_at)
                   VALUES (?, ?, ?, 'created', ?, ?)""",
                (exp_id, config["name"], json.dumps(config), now, now),
            )
            self._conn.commit()
        return self.get_experiment(exp_id)

    def get_experiment(self, exp_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM experiments WHERE id=?", (exp_id,)
            ).fetchone()
        if not row:
            return None
        exp = dict(row)
        exp["config"] = json.loads(exp.pop("config_json"))
        exp["usage"] = self.usage_totals(exp_id)
        exp["budget_state"] = self.budget_state(exp)
        return exp

    def list_experiments(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM experiments ORDER BY created_at"
            ).fetchall()
        return [self.get_experiment(r["id"]) for r in rows]  # type: ignore

    def update_experiment(self, exp_id: str, **fields):
        allowed = {
            "status", "run_id", "output_dir", "started_at", "finished_at",
            "current_gen", "error", "attempt",
        }
        fields = {k: v for k, v in fields.items() if k in allowed}
        if fields:
            self._touch(exp_id, **fields)

    def set_config(self, exp_id: str, config: dict):
        with self._lock:
            self._conn.execute(
                "UPDATE experiments SET config_json=?, updated_at=? WHERE id=?",
                (json.dumps(config), self._now(), exp_id),
            )
            self._conn.commit()

    # -- events ---------------------------------------------------------------
    def add_event(self, experiment_id: Optional[str], type: str, data: dict | None = None) -> dict:
        event = {
            "experiment_id": experiment_id,
            "ts": self._now(),
            "type": type,
            "data": data or {},
        }
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (experiment_id, ts, type, data_json) VALUES (?,?,?,?)",
                (experiment_id, event["ts"], type, json.dumps(event["data"])),
            )
            self._conn.commit()
        return event

    def list_events(self, experiment_id: Optional[str], after_id: int = 0, limit: int = 500) -> list[dict]:
        q = "SELECT * FROM events WHERE id>? "
        params: list[Any] = [after_id]
        if experiment_id is not None:
            q += "AND experiment_id=? "
            params.append(experiment_id)
        q += "ORDER BY id LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "experiment_id": r["experiment_id"],
                    "ts": r["ts"],
                    "type": r["type"],
                    "data": json.loads(r["data_json"]),
                }
            )
        return out

    # -- usage ledger -----------------------------------------------------------
    def add_usage(
        self,
        experiment_id: Optional[str],
        role: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
        latency_s: float | None = None,
        status: str = "ok",
        source: str = "proxy",
        ts: float | None = None,
        generation: Optional[str] = None,
    ) -> dict:
        total = total_tokens if total_tokens is not None else (prompt_tokens + completion_tokens)
        if cost_usd is None:
            cost_usd = estimate_cost(model, prompt_tokens, completion_tokens) if status == "ok" else 0.0
        row = {
            "experiment_id": experiment_id,
            "ts": ts or self._now(),
            "role": role,
            "model": normalize_model(model),
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(total or 0),
            "cost_usd": round(float(cost_usd), 6),
            "latency_s": latency_s,
            "status": status,
            "source": source,
            "generation": generation,
        }
        with self._lock:
            self._conn.execute(
                """INSERT INTO usage
                   (experiment_id, ts, role, model, prompt_tokens, completion_tokens,
                    total_tokens, cost_usd, latency_s, status, source, generation)
                   VALUES (:experiment_id, :ts, :role, :model, :prompt_tokens,
                           :completion_tokens, :total_tokens, :cost_usd, :latency_s,
                           :status, :source, :generation)""",
                row,
            )
            self._conn.commit()
        return row

    def usage_totals(self, experiment_id: Optional[str], source: str = "proxy") -> dict:
        """Authoritative accounting comes from the proxy (server-side).
        source='local' is the in-workspace instrumentation log (reconciliation)."""
        q = (
            "SELECT COALESCE(SUM(prompt_tokens),0) p, COALESCE(SUM(completion_tokens),0) c, "
            "COALESCE(SUM(total_tokens),0) t, COALESCE(SUM(cost_usd),0) cost, COUNT(*) n, "
            "AVG(latency_s) avg_latency, "
            "SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) errors, "
            "SUM(CASE WHEN status='blocked' THEN 1 ELSE 0 END) blocked "
            "FROM usage WHERE status='ok'"
        )
        params: list[Any] = []
        if experiment_id is not None:
            q += " AND experiment_id=?"
            params.append(experiment_id)
        q += " AND source=?"
        params.append(source)
        with self._lock:
            r = self._conn.execute(q, params).fetchone()
        err_q = (
            "SELECT SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) e, "
            "SUM(CASE WHEN status='blocked' THEN 1 ELSE 0 END) b "
            "FROM usage WHERE source=?"
        )
        err_params: list[Any] = [source]
        if experiment_id is not None:
            err_q += " AND experiment_id=?"
            err_params.append(experiment_id)
        with self._lock:
            er = self._conn.execute(err_q, err_params).fetchone()
        return {
            "prompt_tokens": r["p"],
            "completion_tokens": r["c"],
            "total_tokens": r["t"],
            "cost_usd": round(r["cost"], 6),
            "requests": r["n"],
            "avg_latency_s": round(r["avg_latency"], 4) if r["avg_latency"] is not None else None,
            "errors": er["e"] or 0,
            "blocked": er["b"] or 0,
            "source": source,
        }

    def list_usage(self, experiment_id: str, source: str = "proxy") -> list[dict]:
        """Raw usage rows (for cost attribution by time windows)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, role, model, prompt_tokens, completion_tokens, "
                "total_tokens, cost_usd, latency_s, status, generation "
                "FROM usage WHERE experiment_id=? AND source=? ORDER BY ts",
                (experiment_id, source),
            ).fetchall()
        return [dict(r) for r in rows]

    def usage_breakdown(self, experiment_id: Optional[str], source: str = "proxy") -> list[dict]:
        q = (
            "SELECT role, model, COUNT(*) requests, SUM(prompt_tokens) p, "
            "SUM(completion_tokens) c, SUM(total_tokens) t, SUM(cost_usd) cost "
            "FROM usage WHERE status='ok' AND source=?"
        )
        params: list[Any] = [source]
        if experiment_id is not None:
            q += " AND experiment_id=?"
            params.append(experiment_id)
        with self._lock:
            rows = self._conn.execute(q + " GROUP BY role, model ORDER BY cost DESC", params).fetchall()
        return [
            {
                "role": r["role"],
                "model": r["model"],
                "requests": r["requests"],
                "prompt_tokens": r["p"],
                "completion_tokens": r["c"],
                "total_tokens": r["t"],
                "cost_usd": round(r["cost"], 6),
            }
            for r in rows
        ]

    def budget_state(self, exp: dict) -> dict:
        budget = exp.get("config", {}).get("budget", {})
        totals = self.usage_totals(exp["id"])
        started = exp.get("started_at")
        elapsed_h = (time.time() - started) / 3600.0 if started else 0.0
        state = {
            "limits": budget,
            "prompt_tokens": totals["prompt_tokens"],
            "completion_tokens": totals["completion_tokens"],
            "total_tokens": totals["total_tokens"],
            "cost_usd": totals["cost_usd"],
            "requests": totals["requests"],
            "wallclock_hours": round(elapsed_h, 4),
            "exceeded": False,
            "exceeded_reason": None,
        }
        max_cost = budget.get("max_cost_usd")
        max_tokens = budget.get("max_tokens")
        max_h = budget.get("max_wallclock_hours")
        if max_cost is not None and totals["cost_usd"] >= max_cost:
            state["exceeded"] = True
            state["exceeded_reason"] = "max_cost_usd"
        elif max_tokens is not None and totals["total_tokens"] >= max_tokens:
            state["exceeded"] = True
            state["exceeded_reason"] = "max_tokens"
        elif max_h is not None and elapsed_h >= max_h:
            state["exceeded"] = True
            state["exceeded_reason"] = "max_wallclock_hours"
        return state

    # -- slots -----------------------------------------------------------------
    def seed_slots(self, count: int):
        with self._lock:
            for i in range(count):
                self._conn.execute(
                    "INSERT OR IGNORE INTO slots (position) VALUES (?)", (i,)
                )
            self._conn.commit()

    def slot_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) n FROM slots").fetchone()["n"]

    def expand_slots(self, count: int = 1) -> int:
        with self._lock:
            n = self.slot_count()
            for i in range(n, n + count):
                self._conn.execute("INSERT INTO slots (position) VALUES (?)", (i,))
            self._conn.commit()
            return self.slot_count()

    def list_slots(self) -> list[dict]:
        """Slots joined with experiment summaries (the dashboard grid)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT position, experiment_id FROM slots ORDER BY position"
            ).fetchall()
        out = []
        for r in rows:
            slot = {"position": r["position"], "experiment_id": r["experiment_id"]}
            if r["experiment_id"]:
                exp = self.get_experiment(r["experiment_id"])
                if exp:
                    slot["experiment"] = _experiment_summary(exp)
            out.append(slot)
        return out

    def assign_slot(self, position: int, experiment_id: str | None):
        with self._lock:
            self._conn.execute(
                "UPDATE slots SET experiment_id=? WHERE position=?",
                (experiment_id, position),
            )
            self._conn.commit()

    def first_free_slot(self) -> Optional[int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT position FROM slots WHERE experiment_id IS NULL ORDER BY position LIMIT 1"
            ).fetchone()
        return row["position"] if row else None

    def latest_generations(self, experiment_id: str) -> list[dict]:
        """Most recent 'generations' event payload for an experiment."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT data_json FROM events WHERE experiment_id=? AND type='generations' "
                "ORDER BY id DESC LIMIT 1",
                (experiment_id,),
            ).fetchall()
        if not rows:
            return []
        try:
            return json.loads(rows[0]["data_json"]).get("generations", [])
        except Exception:
            return []

    # -- comparison --------------------------------------------------------------
    def comparison(self, exp_ids: list[str]) -> list[dict]:
        out = []
        for eid in exp_ids:
            exp = self.get_experiment(eid)
            if not exp:
                continue
            out.append(
                {
                    "id": exp["id"],
                    "name": exp["name"],
                    "status": exp["status"],
                    "domains": exp["config"].get("domains"),
                    "meta_model": exp["config"].get("meta_model"),
                    "task_model": exp["config"].get("task_model"),
                    "usage": exp["usage"],
                    "budget_state": exp["budget_state"],
                    "started_at": exp["started_at"],
                    "finished_at": exp["finished_at"],
                }
            )
        return out

    def close(self):
        with self._lock:
            self._conn.close()


def _experiment_summary(exp: dict) -> dict:
    return {
        "id": exp["id"],
        "name": exp["name"],
        "status": exp["status"],
        "domains": exp["config"].get("domains", []),
        "meta_model": exp["config"].get("meta_model"),
        "task_model": exp["config"].get("task_model"),
        "max_generation": exp["config"].get("max_generation"),
        "current_gen": exp["current_gen"],
        "total_tokens": exp["usage"]["total_tokens"],
        "cost_usd": exp["usage"]["cost_usd"],
        "budget": exp["config"].get("budget"),
        "created_at": exp["created_at"],
    }


class EventBus:
    """Bridges worker threads to async SSE subscribers. Persists every event."""

    def __init__(self, store: Store):
        self.store = store
        self._subs: dict[Optional[str], set[asyncio.Queue]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    def attach_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    def subscribe(self, experiment_id: Optional[str]) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        with self._lock:
            self._subs.setdefault(experiment_id, set()).add(q)
        return q

    def unsubscribe(self, experiment_id: Optional[str], q: asyncio.Queue):
        with self._lock:
            self._subs.get(experiment_id, set()).discard(q)

    def publish(self, experiment_id: Optional[str], type: str, data: dict | None = None):
        event = self.store.add_event(experiment_id, type, data)
        # dispatch: exact experiment subscribers + wildcard (None) subscribers
        targets: list[asyncio.Queue] = []
        with self._lock:
            targets += list(self._subs.get(experiment_id, ()))
            targets += list(self._subs.get(None, ())) if experiment_id is not None else []
        for q in targets:
            self._put(q, event)

    def _put(self, q: asyncio.Queue, event: dict):
        if self._loop and self._loop.is_running():
            try:
                self._loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:
                pass
        else:
            try:
                q.put_nowait(event)
            except Exception:
                pass
