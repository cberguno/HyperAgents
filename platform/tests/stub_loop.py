"""Offline stub experiment loop.

Used ONLY when an experiment has config.loop='stub' (offline tests and the
demo). It faithfully mimics the observable filesystem behavior of upstream
generate_loop.py — archive.jsonl appends, gen_<n>/metadata.json, per-domain
report.json with improving scores, generate.log, and real LLM calls through
the platform proxy (stub upstream) so usage accounting and the budget
kill-switch are exercised end-to-end with ZERO API cost.

Signals: SIGINT finishes the current generation, writes the archive line and
exits cleanly (mirroring upstream's generation-boundary checkpoint semantics).
Supports --resume-from like upstream generate_loop.py.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import httpx

STOP = False


def _handle_sigint(signum, frame):
    global STOP
    STOP = True


def write_report(gen_dir: Path, accuracy: float, domain: str = "paper_review"):
    eval_dir = gen_dir / f"{domain}_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "report.json").write_text(
        json.dumps(
            {
                "overall_accuracy": round(accuracy, 4),
                "total_correct": int(accuracy * 100),
                "total": 100,
                "accuracy_by_ground_truth": {
                    "accept": {"precision": accuracy, "recall": accuracy, "correct": 50, "total": 100},
                    "reject": {"precision": accuracy, "recall": accuracy, "correct": 50, "total": 100},
                },
            },
            indent=2,
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--max-generation", type=int, default=3)
    ap.add_argument("--sleep", type=float, default=1.2)
    ap.add_argument("--resume-from", default=None)
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    ws = Path(args.workspace)
    cfg_path = ws / "misc" / "hacp" / "platform_config.json"
    cfg = json.loads(cfg_path.read_text())
    proxy = cfg["llm_api_base"]
    token = cfg.get("task_token") or f"hacp-{cfg['experiment_id']}-task"

    if args.resume_from:
        out_dir = Path(args.resume_from)
        archive = json.loads(
            [ln for ln in (out_dir / "archive.jsonl").read_text().splitlines() if ln.strip()][-1]
        )["archive"]
    else:
        out_dir = ws / "outputs" / f"generate_{args.run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        archive = []

    def log(msg: str):
        with open(out_dir / "generate.log", "a") as f:
            f.write(msg + "\n")
        print(msg, flush=True)

    def archive_append(gen):
        archive.append(gen)
        with open(out_dir / "archive.jsonl", "a") as f:
            f.write(json.dumps({"current_genid": gen, "archive": archive}) + "\n")

    # --- generation 'initial' -------------------------------------------------
    if not archive:
        gen_dir = out_dir / "gen_initial"
        gen_dir.mkdir(exist_ok=True)
        (gen_dir / "metadata.json").write_text(
            json.dumps({"gen_output_dir": str(gen_dir), "run_eval": True}, indent=2)
        )
        write_report(gen_dir, 0.52)
        archive_append("initial")
        log("generation initial completed, scores paper_review train=0.52")

    start_gen = len(archive)

    for genid in range(start_gen, args.max_generation + 1):
        if STOP:
            log("interrupted — stopping at generation boundary (resume supported)")
            sys.exit(0)

        gen_dir = out_dir / f"gen_{genid}"
        gen_dir.mkdir(exist_ok=True)
        agent_output = gen_dir / "agent_output"
        agent_output.mkdir(exist_ok=True)
        (agent_output / "meta_agent_chat_history.md").write_text(
            f"# Meta-agent chat history (stub)\n\n"
            f"Instruction: Modify any part of the codebase at `./`.\n\n"
            f"Generation {genid}: [stub offline reasoning] Improved the task agent prompt.\n"
        )
        (agent_output / "model_patch.diff").write_text(
            f"--- a/task_agent.py\n+++ b/task_agent.py\n@@ -10,3 +10,4 @@\n"
            f" # stub patch for generation {genid}\n+ # improved review heuristic v{genid}\n"
        )

        # Simulate the meta-agent + eval LLM calls through the real proxy path.
        try:
            r = httpx.post(
                f"{proxy}/chat/completions",
                json={
                    "model": cfg.get("meta_model", "stub/tiny"),
                    "messages": [{"role": "user", "content": f"stub meta-agent prompt for generation {genid}"}],
                },
                headers={"Authorization": f"Bearer {cfg.get('meta_token') or token}"},
                timeout=60,
            )
            log(f"meta agent call status={r.status_code}")
            if r.status_code == 429:
                log("budget exhausted — refusing to continue")
                sys.exit(2)
            r2 = httpx.post(
                f"{proxy}/chat/completions",
                json={
                    "model": cfg.get("task_model", "stub/task"),
                    "messages": [{"role": "user", "content": f"stub eval batch generation {genid}"}],
                },
                headers={"Authorization": f"Bearer {token}"},
                timeout=60,
            )
            log(f"task eval call status={r2.status_code}")
            if r2.status_code == 429:
                log("budget exhausted — refusing to continue")
                sys.exit(2)
        except httpx.HTTPError as e:
            log(f"proxy call failed: {e}")
            sys.exit(3)

        accuracy = 0.52 + 0.013 * genid
        parent = archive[-1] if archive else None
        (gen_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "gen_output_dir": str(gen_dir),
                    "current_genid": genid,
                    "parent_genid": parent,
                    "run_eval": True,
                    "run_full_eval": True,
                    "parent_agent_success": True,
                    "valid_parent": True,
                },
                indent=2,
            )
        )
        write_report(gen_dir, accuracy)
        archive_append(genid)
        log(f"generation {genid} completed, parent {parent}, scores paper_review train={accuracy:.4f}")
        time.sleep(args.sleep)

    log(f"all generations complete (initial + 1..{args.max_generation})")
    sys.exit(0)


if __name__ == "__main__":
    main()
