"""Conversational shell: an LLM assistant that can inspect experiments.

M1 scope: read-only inspection tools over the platform store (experiments,
scores, usage/cost, logs, failure context). Control actions (start/pause/stop)
are intentionally *not* auto-executed by the shell — it returns the exact API
call instead (full control integration is M4).

The shell's model is configured independently of experiment models
(ServiceConfig.shell_model), routes through the platform proxy, and its usage
is accounted in the same ledger (role='shell').
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from . import paths
from .store import Store

TOOL_PROMPT = """You are the assistant shell of the HyperAgents Control Platform.
You can inspect experiments and explain what is happening. You have these tools:

{tools}

To use a tool, respond ONLY with a <json> block:
<json>
{{"tool_name": "...", "tool_input": {{...}}}}
</json>

After you have the information you need (or none is required), respond with a
normal answer WITHOUT a <json> block. Be concise and precise. If asked to
perform a control action (start/pause/stop/resume), explain that you are
read-only in the current milestone and give the exact API call to use."""


def _tool(name: str, description: str, fn: Callable):
    return {"name": name, "description": description, "fn": fn}


def build_tools(store: Store, home: Path):
    def list_experiments():
        return [
            {
                "id": e["id"], "name": e["name"], "status": e["status"],
                "domains": e["config"].get("domains"),
                "meta_model": e["config"].get("meta_model"),
                "task_model": e["config"].get("task_model"),
                "current_gen": e["current_gen"],
                "total_tokens": e["usage"]["total_tokens"],
                "cost_usd": e["usage"]["cost_usd"],
            }
            for e in store.list_experiments()
        ]

    def get_experiment(experiment_id: str):
        e = store.get_experiment(experiment_id)
        if not e:
            return {"error": f"experiment {experiment_id} not found"}
        return {
            "id": e["id"], "name": e["name"], "status": e["status"],
            "config": e["config"], "current_gen": e["current_gen"],
            "budget_state": e["budget_state"], "usage": e["usage"],
            "started_at": e["started_at"], "finished_at": e["finished_at"],
            "error": e["error"],
        }

    def get_scores(experiment_id: str):
        events = store.list_events(experiment_id, limit=500)
        gens = None
        for ev in events:
            if ev["type"] == "generations":
                gens = ev["data"].get("generations")
        if gens is None:
            return {"note": "no generation scores recorded yet"}
        return [{"gen": g["gen"], "scores": g["scores"]} for g in gens]

    def get_usage(experiment_id: str = ""):
        if experiment_id:
            return {
                "totals": store.usage_totals(experiment_id),
                "breakdown": store.usage_breakdown(experiment_id),
            }
        out = {}
        for e in store.list_experiments():
            out[e["name"]] = store.usage_totals(e["id"])
        return out

    def tail_logs(experiment_id: str, lines: int = 20):
        logs = store.list_events(experiment_id, limit=1000)
        log_lines = [ev["data"].get("line", "") for ev in logs if ev["type"] == "log"]
        return {"lines": log_lines[-lines:]}

    def explain_failure(experiment_id: str, gen: str = ""):
        """Assemble failure context: error message, recent logs, meta-agent
        chat history and diff for the given generation (LLM summarizes)."""
        e = store.get_experiment(experiment_id)
        if not e:
            return {"error": "experiment not found"}
        ctx = {"status": e["status"], "error": e["error"]}
        if e.get("output_dir"):
            out_dir = Path(e["output_dir"])
            gen_name = f"gen_{gen}" if gen else None
            candidates = [out_dir / gen_name] if gen_name else sorted(
                out_dir.glob("gen_*"), key=lambda p: p.name
            )
            for gd in reversed(candidates):
                chat = gd / "agent_output" / "meta_agent_chat_history.md"
                patch = gd / "agent_output" / "model_patch.diff"
                if chat.exists():
                    ctx["meta_chat_history_tail"] = chat.read_text()[-4000:]
                if patch.exists():
                    ctx["model_patch_head"] = patch.read_text()[:4000]
                break
            gl = out_dir / "generate.log"
            if gl.exists():
                ctx["generate_log_tail"] = gl.read_text()[-4000:]
        return ctx

    def get_lineage(experiment_id: str):
        """Evolutionary lineage: each generation's parent, scores, duration."""
        gens = store.latest_generations(experiment_id)
        if not gens:
            return {"note": "no generations recorded yet"}
        return [
            {
                "gen": g["gen"], "parent": g.get("parent"),
                "duration_s": g.get("duration_s"),
                "train_scores": {
                    d: s.get("train") for d, s in (g.get("scores") or {}).items()
                },
                "valid_parent": g.get("valid_parent"),
            }
            for g in gens
        ]

    def get_generation(experiment_id: str, gen: str):
        """Memory + history of one generation: metadata, files, the
        meta-agent's diff and chat-history tail (truncated)."""
        e = store.get_experiment(experiment_id)
        if not e:
            return {"error": "experiment not found"}
        if not e.get("output_dir"):
            return {"note": "no output directory yet"}
        gd = Path(e["output_dir"]) / f"gen_{gen}"
        if not gd.exists():
            return {"error": f"generation {gen} not found"}
        out = {"gen": gen, "files": sorted(p.name for p in gd.rglob("*") if p.is_file())}
        patch = gd / "agent_output" / "model_patch.diff"
        chat = gd / "agent_output" / "meta_agent_chat_history.md"
        if patch.exists():
            out["model_patch"] = patch.read_text()[:3000]
        if chat.exists():
            out["meta_chat_history_tail"] = chat.read_text()[-3000:]
        return out

    return [
        _tool("list_experiments", "List all experiments with status, models, tokens and cost.", list_experiments),
        _tool("get_experiment", "Full details of one experiment by id.", get_experiment),
        _tool("get_scores", "Per-generation evaluation scores for one experiment.", get_scores),
        _tool("get_usage", "Token/cost usage for one experiment (or all if omitted).", get_usage),
        _tool("tail_logs", "Recent experiment log lines.", tail_logs),
        _tool("explain_failure", "Assemble failure context (logs, chat history, diff) for explanation.", explain_failure),
        _tool("get_lineage", "Evolutionary lineage of an experiment (parents, scores, generation durations).", get_lineage),
        _tool("get_generation", "Generation detail: metadata, files, meta-agent diff and chat history.", get_generation),
    ]


class ShellAgent:
    def __init__(self, client, store: Store, home: Path, max_tool_calls: int = 8):
        self.client = client
        self.store = store
        self.home = home
        self.max_tool_calls = max_tool_calls
        self.tools = {t["name"]: t for t in build_tools(store, home)}

    def run(self, message: str, history: list[dict] | None = None) -> dict:
        """One conversational turn. Returns reply + transcript + tool calls made."""
        messages = list(history or [])
        system = TOOL_PROMPT.format(
            tools="\n".join(
                f"- {t['name']}: {t['description']}" for t in self.tools.values()
            )
        )
        messages = [{"role": "system", "content": system}, *messages,
                   {"role": "user", "content": message}]
        tool_calls_made = []
        for _ in range(self.max_tool_calls + 1):
            reply = self.client.chat(messages)
            match = re.search(r"<json>\s*(\{.*?\})\s*</json>", reply, re.DOTALL)
            if not match:
                return {"reply": reply, "tool_calls": tool_calls_made,
                        "history": messages[1:] + [{"role": "assistant", "content": reply}]}
            try:
                call = json.loads(match.group(1))
            except json.JSONDecodeError:
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": "Your <json> block was malformed. Reply with a corrected tool call or a final answer."})
                continue
            name = call.get("tool_name")
            inp = call.get("tool_input") or {}
            if not isinstance(inp, dict):
                inp = {}
            tool = self.tools.get(name)
            if tool is None:
                result = {"error": f"unknown tool '{name}'"}
            else:
                try:
                    result = tool["fn"](**inp)
                except Exception as e:
                    result = {"error": str(e)}
            tool_calls_made.append({"tool_name": name, "tool_input": inp, "result": result})
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": "Tool result (JSON):\n```json\n"
                            + json.dumps(result, default=str)[:6000] + "\n```",
            })
        return {
            "reply": "Tool call limit reached without a final answer.",
            "tool_calls": tool_calls_made,
            "history": messages[1:],
        }
