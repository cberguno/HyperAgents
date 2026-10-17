"""Budget enforcement core: reservations against concurrent-request races.

Why this exists: without reservations, N concurrent requests (upstream
evaluates with a thread pool) can ALL pass the budget check before any of
them completes — each seeing `committed < limit` — and then all succeed,
overshooting the cap by up to N × request-cost. This module makes budget
checking atomic across in-flight requests:

  * Before forwarding, the proxy RESERVES a conservative estimate
    (prompt estimate x safety factor + requested max_tokens).
  * A request is admitted only if  committed + reserved  stays within limits.
  * On completion/failure the reservation is released and the ACTUAL usage is
    recorded (the authoritative ledger).

Worst-case overshoot is therefore bounded by (see docs in the readiness
report): the sum of live reservations at the moment the cap is crossed,
where each actual usage <= its reservation x tokenization-error factor.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Optional


class Reservations:
    """Thread-safe in-flight budget reservations per experiment."""

    def __init__(self, ttl_s: float = 600.0):
        self._lock = threading.Lock()
        self._per_exp: dict[str, dict[str, tuple[int, float, float]]] = {}
        # exp_id -> {req_id: (tokens, cost_usd, expires_ts)}
        self._ttl = ttl_s

    def reserve(self, exp_id: str, est_tokens: int, est_cost: float) -> Optional[str]:
        """Create a reservation; returns req_id or None if the registry is full
        for this experiment (safety valve, 10k in flight)."""
        req_id = uuid.uuid4().hex[:16]
        with self._lock:
            bucket = self._per_exp.setdefault(exp_id, {})
            if len(bucket) > 10_000:
                return None
            bucket[req_id] = (int(est_tokens), float(est_cost), time.time() + self._ttl)
            return req_id

    def release(self, exp_id: str, req_id: Optional[str]):
        if req_id is None:
            return
        with self._lock:
            bucket = self._per_exp.get(exp_id)
            if bucket:
                bucket.pop(req_id, None)

    def totals(self, exp_id: str) -> tuple[int, float]:
        """(reserved_tokens, reserved_cost) of unexpired reservations."""
        now = time.time()
        tokens = 0
        cost = 0.0
        with self._lock:
            bucket = self._per_exp.get(exp_id)
            if not bucket:
                return 0, 0.0
            expired = [rid for rid, (_, _, exp) in bucket.items() if exp < now]
            for rid in expired:
                bucket.pop(rid, None)
            for est_tokens, est_cost, _ in bucket.values():
                tokens += est_tokens
                cost += est_cost
        return tokens, cost

    def active_count(self, exp_id: str) -> int:
        _, _ = self.totals(exp_id)  # prune expired
        with self._lock:
            return len(self._per_exp.get(exp_id, {}))


# Process-wide singleton: shared by the proxy (creates reservations) and the
# runner supervisor (accounts for in-flight spend when deciding to pause).
RESERVATIONS = Reservations()


def combined_budget_state(store, exp: dict) -> dict:
    """Budget state including in-flight reservations — the single authoritative
    view used by the proxy's admission check, the runner's pause loop and the
    dashboard's live metrics."""
    state = store.budget_state(exp)
    if state.get("exceeded"):
        return state
    res_tokens, res_cost = RESERVATIONS.totals(exp["id"])
    state["reserved_tokens"] = res_tokens
    state["reserved_cost_usd"] = round(res_cost, 6)
    # Re-evaluate limits including reservations
    limits = state["limits"]
    max_cost = limits.get("max_cost_usd")
    max_tokens = limits.get("max_tokens")
    state["exceeded"] = False
    state["exceeded_reason"] = None
    if max_cost is not None and state["cost_usd"] + res_cost >= max_cost:
        state["exceeded"] = True
        state["exceeded_reason"] = "max_cost_usd"
    elif max_tokens is not None and state["total_tokens"] + res_tokens >= max_tokens:
        state["exceeded"] = True
        state["exceeded_reason"] = "max_tokens"
    return state
