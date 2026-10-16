# HyperAgents Control Platform (HACP) — M1

A web-based control plane for running, monitoring and comparing multiple
independent [HyperAgents](https://arxiv.org/abs/2603.19461) experiments.
**Meta's original research code is never modified** — this directory contains
only the platform that wraps it.

## What M1 provides

| Capability | How |
|---|---|
| Experiment registry + expandable slots (5 initial, "+" to add) | `hacp/store.py`, `hacp/api.py`, `web/` |
| Independent meta-agent / task-agent / shell model selection | per-experiment `platform_config.json` read by the instrumentation patch; `HACP_SHELL_MODEL` for the shell |
| OpenRouter integration | `hacp/proxy.py` — single upstream key, all models, OpenAI-compatible endpoint |
| Token tracking + accurate cost accounting | proxy ledger (authoritative, server-side) + local `usage.jsonl` reconciliation; pricing registry in `hacp/config.py` |
| Budget kill-switches | `hacp/proxy.py` (refuses requests, 429) + `hacp/runner.py` (pauses the experiment) |
| Web dashboard foundation | `web/` (vanilla JS): slots grid, live SSE logs, metrics, generations, comparison, shell |
| History / checkpoints / stop & resume | per-experiment workspaces preserved; stop = graceful SIGINT; resume = upstream `--resume_from` |
| Conversational shell foundation | `hacp/shell.py` — read-only inspection tools, independently configurable model |
| Offline tests & demo | stub loop + stub LLM upstream — **zero API cost** |

## M2 additions

- **Experiment editing** — models, budgets, max generations via `PATCH` + Edit dialog (model ids canonicalized on write)
- **Execution speed** — per-generation durations (watcher → SSE → table), avg LLM latency, avg gen time, elapsed time
- **Evolutionary lineage** — `GET /api/experiments/{id}/lineage` reconstructs the tree from upstream `archive.jsonl` + `metadata.json`; rendered in the detail view
- **Generation memory & history** — `GET /api/experiments/{id}/generations/{gen}` returns metadata, scores, the meta-agent's `model_patch.diff` and chat-history tail; clickable rows in the dashboard
- **Richer comparison** — best train score, avg gen time, generation count, LLM calls, correct wallclock
- **Shell tools** — `get_lineage`, `get_generation` (8 read-only tools total)
- **Trading domain design** — `platform/docs/trading-domain-design.md` (M5 foundation, no implementation yet)

## Architecture

```
platform/
  hacp/
    config.py           # model registry + pricing, ExperimentConfig, ServiceConfig
    store.py            # SQLite registry + usage ledger + event bus
    proxy.py            # OpenAI-compatible LLM gateway (OpenRouter | offline stub)
    instrumentation.py  # opt-in patch applied to *experiment workspace copies* only
    workspace.py        # per-experiment disposable copy of the pristine repo
    runner.py           # subprocess supervisor: start/pause/stop/resume/seed
    watcher.py          # filesystem watcher -> live events (SSE)
    llm.py, shell.py    # platform LLM client + conversational shell
    api.py              # FastAPI: REST + SSE + proxy + dashboard
  web/                  # dashboard (vanilla JS)
  scripts/serve.py      # run the platform
  scripts/demo_offline.py  # full offline demo, zero API cost
  tests/                # offline test suite + stub experiment loop
```

**Key integration facts** (verified against upstream):
- Upstream runs each generation inside a Docker container built from the
  experiment workspace, with `network_mode="host"` and the workspace
  bind-mounted at `/hyperagents` (`utils/docker_utils.build_container`). So:
  in-container LLM calls reach the platform proxy at `127.0.0.1:<port>/proxy/v1`
  and in-container usage logs stream to the host in real time.
- `setup_initial_gen` excludes `misc/` from the evolvable agent repo — platform
  files live in `misc/hacp/` and stay out of the meta-agent's world.
- The instrumentation patch on `agent/llm.py` becomes part of the experiment's
  base commit, so lineage diffs stay clean. Without `misc/hacp/platform_config.json`
  the patched file is **byte-for-byte behavior-identical to upstream**.
- All provider keys live only in the platform process; experiments hold
  per-experiment HACP tokens. The budget kill-switch is enforced server-side
  and cannot be bypassed by a self-modifying agent.

## Run (Replicas-first, portable)

```bash
python3 -m venv platform/.venv
platform/.venv/bin/pip install -r platform/requirements.txt

# offline mode (default): zero API cost
platform/.venv/bin/python platform/scripts/serve.py          # dashboard at http://localhost:8420

# real experiments (AFTER explicit approval + OPENROUTER_API_KEY):
HACP_UPSTREAM=openrouter platform/.venv/bin/python platform/scripts/serve.py
```

Environment: `HACP_PORT`, `HACP_HOME`, `HACP_UPSTREAM` (stub|openrouter),
`HACP_SHELL_MODEL`, `HACP_UPSTREAM_REPO`, `HACP_PRICE_OVERRIDES` (JSON).

## First controlled experiment (paper_review)

1. Create the experiment in the dashboard (pick meta/task models + budgets).
2. `POST /api/experiments/{id}/seed` — runs upstream `setup_initial.sh`
   equivalent steps (real LLM calls via the proxy — billable, hence gated).
3. `POST /api/experiments/{id}/start` — launches the **unmodified**
   `generate_loop.py` in the experiment's workspace (needs a Docker daemon).
4. Monitor live: generations, scores, tokens, cost; stop/resume anytime.

### Runtime image (build once per host)

Experiments run inside containers built from the image named `hyperagents`.
Build it from a prepared experiment workspace (never from the repo root),
then add the platform runtime layer:

```bash
WS=<experiment workspace>; cd "$WS"
docker build --network=host -t hyperagents-buildtest .   # Meta's unmodified Dockerfile
platform/docker/build_runtime_image.sh                  # adds git safe.directory -> hyperagents
```

The runtime layer exists because upstream bind-mounts the user-owned snapshot
into root containers; without `safe.directory` git refuses in-container
operations (exit 128). It adds only git runtime config — no Meta code changes.

### Host-side system dependencies (for the runner venv)

Upstream `generate_loop.py` and its seed/analysis commands run with the
platform venv, which therefore also needs: `matplotlib networkx scipy
hydra-core GitPython numpy pandas` (+ system `graphviz` and pip `pygraphviz`
for archive visualizations). See `platform/requirements.txt`.

### Operational notes

- Experiment containers write into the bind-mounted workspace as **root**;
  rebuilding a workspace afterwards may need `sudo rm -rf` (upstream's
  `git clean` removes most residue at each generation boundary).
- Budgets are **cumulative per experiment** across seeds, runs and resumes —
  enforced server-side by the proxy (see `tests/test_bypass_resistance.py`).
- Resuming a `finished` experiment (after raising `max_generation`) continues
  the same lineage via upstream `--resume_from`.

## Tests (offline, no API charges)

```bash
cd platform && .venv/bin/python -m pytest
```
