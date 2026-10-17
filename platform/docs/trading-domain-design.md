# Trading Simulation Domain — Design Foundation (M5)

**Status: DESIGN ONLY — no implementation yet. Zero real-money capability by construction.**
Author: platform team · Scope: a new HyperAgents domain (`trading_sim`) where evolving
agents select stocks, place simulated orders, manage virtual portfolios, and compete
against market benchmarks on historical data.

## 1. Non-negotiable constraints

1. **No real brokerage accounts, no real orders, no live market feeds.** The domain is a
   deterministic offline backtest engine over a static bundled dataset (same pattern as
   `domains/paper_review/dataset.csv`).
2. **Meta's pristine repository stays untouched.** The domain ships as a platform-managed
   *domain pack* applied to experiment workspace copies only (same discipline as the
   instrumentation patch — see §7).
3. **The existing paper_review domain is fully preserved.** Cross-domain transfer
   (`--domains paper_review trading_sim`) reuses upstream's multi-domain averaging in
   `select_parent` unchanged.

## 2. What the agents do (task contract)

Following the upstream domain plugin convention (`domains/<name>/utils.py`):

```python
QUESTION_ID = "episode_id"          # one backtest episode = one "question"
GROUND_TRUTH_KEY = "benchmark_return"   # buy-and-hold benchmark return for the episode
MODEL = "openai/gpt-4o-mini"       # base task-agent model (evolvable)

def format_input_dict(row):
    # row carries: episode window start/end, tradable universe,
    # per-ticker historical OHLCV up to the episode start (the agent's "memory"),
    # starting cash, and the order protocol description.
    ...
```

The task agent responds in the existing JSON protocol (`{"response": ...}`) with a
**trading program** for the episode (see §4), which the harness feeds to the backtest
engine. No streaming decisions needed in v1 — decisions are made episode-locally from
the context window. This keeps the domain a pure function of (agent, data) → metrics,
exactly like upstream's accuracy-scored domains.

## 3. Backtest engine (deterministic, pure functions)

New module `domains/trading_sim/engine.py` (part of the domain pack):

- **Input:** agent's trading program + episode window (e.g., 60 trading days) + universe
  (start with 20–50 liquid US equities) + OHLCV daily bars.
- **Execution model (v1, daily bars, next-open fills):**
  - Orders: `{action: buy|sell|hold, ticker, weight}` (portfolio weights, long-only v1).
  - Fills at next day's **open** with configurable **slippage (bps)** and
    **commission (bps + minimum)** — defaults conservative.
  - Position accounting: fractional shares, cash + positions marked to close daily.
  - Constraints: fully-invested check, weight bounds [0, w_max], no leverage, no
    shorting (v1). Violations → clipped + recorded (a persistent penalty signal the
    meta-agent can learn from).
- **Determinism:** no randomness in fills or metrics; same inputs → same score, so
  evolution signal is clean.
- **Output per episode:** equity curve (daily), trades log, and metrics (§5).

## 4. Anti-overfitting evaluation (the core design point)

Upstream's train/val/test discipline maps directly onto **time-ordered splits**:

| Split | Period (example) | Use |
|---|---|---|
| train | 2010–2018 | meta-agent sees eval artifacts (its "memory") |
| val | 2019–2021 | parent selection (`select_parent` uses val — upstream behavior) |
| test | 2022–2023 | **reported only**, never shown to the meta-agent |

- **No shuffling, no overlap**: episodes are disjoint windows; leak checks mirror
  `curate_subsets.py` (question_id/episode overlap = 0).
- **Automatic val/test hiding**: upstream's `copy_prev_eval_to_container` prunes
  `*_eval_val`/`*_eval_test` dirs before the meta-agent sees them — our domain naming
  (`trading_sim_eval_val`) inherits this protection for free.
- **Optional walk-forward variant** (v2): rolling train/val windows for a second
  robustness axis; test stays frozen.
- **Benchmark-relative scoring** (§5) further suppresses "just hold the index" solutions.

## 5. Score key and metrics

`report.json` per episode + aggregate, mirroring upstream's `domains/report.py` shape:

| Metric | Definition |
|---|---|
| **`risk_adjusted_return` (primary score key)** | median (across episodes) of `clip(Sharpe_ep, -3, 3)` with a drawdown floor: if `max_drawdown > 50%`, score × 0.5 |
| `total_return` / `CAGR` | geometric return of the virtual portfolio |
| `max_drawdown` | peak-to-trough of equity curve |
| `win_rate_vs_benchmark` | fraction of episodes beating buy-and-hold of the same universe (SPY buy-and-hold reported alongside) |
| `turnover` | mean daily traded fraction (cost drag transparency) |
| `benchmark_return` (GROUND_TRUTH_KEY) | buy-and-hold return for the episode; used for the beat-rate metric |

Primary score rationale: raw return rewards degenerate risk-taking; Sharpe-with-
drawdown-floor rewards consistent, benchmark-beating behavior. All metrics are computed
by pure functions so the meta-agent's improvements are measurable exactly like
`overall_accuracy` today.

`get_domain_score_key("trading_sim") → "risk_adjusted_return"`;
`get_domain_splits → ["train","val","test"]` (test reported only);
`can_domain_ensembled → False` (v1: ensembling trading programs is undefined).

## 6. Data source

- **Provider (v1):** redistributable **daily** OHLCV from a permissive source (Stooq
  or Tiingo free tier; to be confirmed for redistribution terms before bundling).
- **Bundled, static, versioned:** `domains/trading_sim/dataset.csv` (~50 tickers ×
  ~3500 trading days ≈ 175k rows ≈ 30–60MB parquet/CSV) — same storage pattern as
  paper_review's 49MB dataset; committed to the domain pack, checksummed, never
  fetched at runtime. No live feeds, no API keys in the domain.
- Universe selection: liquid US large caps with continuous history 2010–2023;
  survivorship-bias caveat documented in the domain README (v1 uses today's survivors —
  acceptable for simulation research, noted honestly).

## 7. Domain pack delivery (preserving Meta's repo)

The domain requires two upstream-adjacent integration points. Solution — the **domain
pack**, applied by `platform/hacp/domain_packs.py` to workspace copies only:

1. `domains/trading_sim/` — **pure new directory** (utils.py, engine.py, harness
   integration via upstream's existing `domains/harness.py` dispatch — it imports
   `domains.{domain_folder}.{utils_prefix}utils` dynamically ✓ no edit needed).
2. **Additive, marked patches** (idempotent, anchor-checked, same discipline as
   `instrumentation.py`):
   - `utils/domain_utils.py`: extend the three mapping functions with `trading_sim`
     entries (score key, splits, ensembling flag).
   - `generate_loop.py`: extend the `--domains` argparse `choices` list with
     `trading_sim` (and only that list).
3. Seed phase (`runner.seed`): generalize the existing seed steps to trading_sim
   (curate episodes → harness on train/val/test → report) with the same proxy-gated,
   approval-required posture as paper_review.
4. Pristine-repo guard: existing tests assert repo hash equality; domain-pack tests
   assert patch idempotency + upstream files unchanged in the checkout.

## 8. Platform integration (all existing, no new platform concepts)

- Config: `domains: ["trading_sim"]` (or `["paper_review","trading_sim"]` for transfer).
- Budgets/kill-switches/proxy accounting: role-based, already model-agnostic.
- Watcher: `DOMAIN_SCORE_KEYS["trading_sim"]="risk_adjusted_return"` entry; dashboard
  renders per-split scores + (small M5 UI addition) an equity-curve/drawdown chart from
  `report.json` artifacts.
- Shell: `get_scores`/`get_lineage` work as-is; add a `get_portfolio` tool for episode
  artifacts (trades, equity curve).

## 9. Safety posture

- No real-money path exists in the code (engine is pure simulation over bundled CSV).
- No network egress for the domain (data is local; LLM calls go through the proxy).
- The meta-agent can rewrite the task agent/trading program code — it stays inside the
  same sandboxed containers as every other domain; budget kill-switch unchanged.

## 10. Completion & failure criteria (explicit)

**M5 is complete when:**
1. Domain pack applies cleanly to a workspace copy; pristine-repo hash tests pass.
2. Seeding produces train/val/test episode sets with zero overlap (leak test).
3. A stub-model offline run (zero API cost) produces `report.json` per split with all
   §5 metrics from the deterministic engine (golden-file engine test).
4. A budget-capped run with cheap real models (after your approval) evolves
   `risk_adjusted_return` above the buy-and-hold benchmark on **val**, and reports
   test-set metrics without ever exposing test artifacts to the meta-agent.
5. A cross-domain run (paper_review + trading_sim) completes with upstream's
   multi-domain parent selection intact.

**M5 fails if:** any split overlap is detected; engine metrics differ across identical
inputs (non-determinism); test artifacts leak into the meta-agent's container; or any
real-market/brokerage code path appears in review.

## 11. Cost envelope (estimates)

- Seeding: engine is local (no LLM cost); the base task-agent eval on ~30 train episodes
  ≈ 30 task calls ≈ cents with gpt-4o-mini.
- Evolution: identical per-generation cost structure to paper_review (meta-agent call +
  eval calls); budgets enforced as always. Dataset: one-time ~$0 (free/redistributable
  provider TBD).
