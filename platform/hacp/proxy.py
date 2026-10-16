"""OpenAI-compatible LLM gateway with recording, cost accounting and budget enforcement.

All experiment LLM traffic (meta-agent, task-agent, shell) flows through this
proxy. This is the platform's server-side control point:

* **Key isolation** — provider keys (OpenRouter) never leave the platform
  process. Experiment containers/agents only hold per-experiment HACP tokens.
* **Accounting** — every response's usage is recorded in the ledger with role,
  model and estimated cost (authoritative source for budgets and dashboards).
* **Budget kill-switch** — once an experiment's budget is exhausted the proxy
  refuses requests (HTTP 429, code ``hacp_budget_exhausted``). A self-modifying
  agent cannot bypass this: it is enforced outside the agent's world.

Upstream containers run with ``network_mode="host"`` (upstream behavior), so
in-container calls reach this proxy at ``http://127.0.0.1:<port>/proxy/v1``.

Upstreams:
* ``openrouter`` — forwards to https://openrouter.ai/api/v1 (single key for all
  models; needs OPENROUTER_API_KEY and explicit user approval before use).
* ``stub`` — offline deterministic responses; zero API cost. Used by tests and
  the offline demo.

Tokens look like ``hacp-<experiment_id>-<role>`` (e.g. ``hacp-e3f9a1b2c3d4-meta``)
or the platform-level ``hacp-platform-shell``.
"""
from __future__ import annotations

import time
from typing import Optional, Protocol

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import (
    PROVIDER_KEY_VARS,
    ServiceConfig,
    normalize_model,
    openrouter_slug,
    resolve_model,
)
from .store import Store

PROVIDER_KEY_VARS  # re-exported for backwards imports (defined in config.py)

PLATFORM_TOKEN = "hacp-platform-shell"


def parse_token(token: str) -> tuple[Optional[str], str]:
    """Return (experiment_id, role) for an HACP token."""
    if token == PLATFORM_TOKEN:
        return None, "shell"
    parts = token.split("-")
    if len(parts) == 3 and parts[0] == "hacp" and parts[2] in ("meta", "task", "shell"):
        return parts[1], parts[2]
    raise ValueError(f"invalid HACP token format")


class Upstream(Protocol):
    name: str

    async def complete(self, payload: dict) -> tuple[dict, int]:
        """Return (response_body_dict, http_status)."""


class OpenRouterUpstream:
    name = "openrouter"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, referer: str = "https://hyperagents.local", title: str = "HyperAgents Control Platform"):
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": referer,
                "X-Title": title,
            },
            timeout=600.0,
        )

    async def complete(self, payload: dict) -> tuple[dict, int]:
        # Translate canonical model ids to OpenRouter slugs (the ledger keeps
        # the canonical id; only the forwarded payload is rewritten).
        payload = {**payload, "model": openrouter_slug(payload.get("model", ""))}
        resp = await self._client.post("/chat/completions", json=payload)
        try:
            return resp.json(), resp.status_code
        except Exception:
            return {"error": {"message": f"openrouter returned {resp.status_code}"}}, 502


class StubUpstream:
    """Deterministic offline 'LLM'. Zero API cost, no network.

    Replies with a short echo-style message and OpenAI-format usage derived
    from the request, so the entire accounting path is exercised for real.
    """

    name = "stub"

    async def complete(self, payload: dict) -> tuple[dict, int]:
        if payload.get("stream"):
            return {"error": {"message": "stub upstream does not support streaming"}}, 400
        model = payload.get("model", "stub/tiny")
        messages = payload.get("messages", [])
        prompt_tokens = sum(len(str(m.get("content", ""))) // 4 + 8 for m in messages)
        if model.endswith("stub/patcher") and len(messages) <= 1:
            # Rehearsal mode: behave like a meta-agent that uses its bash tool
            # to append a harmless marker line, producing a real (non-empty)
            # model_patch.diff so the full pipeline — diff capture, compilation
            # checks, in-container evaluation — is exercised end-to-end.
            content = (
                "<json>\n"
                '{"tool_name": "bash", "tool_input": '
                "{\"command\": \"echo '# hacp-rehearsal-marker' >> task_agent.py\"}}\n"
                "</json>"
            )
        else:
            content = (
                f"[stub:{model}] Offline deterministic response "
                f"({len(messages)} message(s) received). No real API call was made."
            )
        completion_tokens = len(content) // 4 + 6
        body = {
            "id": f"chatcmpl-stub-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        # tiny sleep so latency accounting has signal
        time.sleep(0.02)
        return body, 200


def create_upstream(service: ServiceConfig) -> Upstream:
    if service.upstream == "openrouter":
        import os

        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise RuntimeError(
                "HACP_UPSTREAM=openrouter requires OPENROUTER_API_KEY in the "
                "environment of the platform process. Experiments never see this key."
            )
        return OpenRouterUpstream(key)
    return StubUpstream()


def create_proxy_router(store: Store, bus, service: ServiceConfig) -> APIRouter:
    """Build the /proxy router (OpenAI-compatible, HACP-token authenticated)."""
    router = APIRouter()
    upstream = create_upstream(service)

    def _auth(authorization: Optional[str]) -> tuple[Optional[str], str]:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing HACP bearer token")
        token = authorization[len("Bearer "):].strip()
        try:
            exp_id, role = parse_token(token)
        except ValueError:
            raise HTTPException(401, "invalid HACP token")
        if exp_id is not None:
            exp = store.get_experiment(exp_id)
            if exp is None:
                raise HTTPException(401, f"unknown experiment in token: {exp_id}")
        return exp_id, role

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request, authorization: Optional[str] = Header(None)):
        exp_id, role = _auth(authorization)
        payload = await request.json()
        model = normalize_model(payload.get("model", ""))

        # Server-side model enforcement (optional; default off to preserve
        # research freedom of the evolving agent).
        if service.enforce_models and exp_id is not None:
            exp = store.get_experiment(exp_id)
            allowed = {
                "meta": exp["config"].get("meta_model"),
                "task": exp["config"].get("task_model"),
            }.get(role)
            if allowed and model != normalize_model(allowed):
                return JSONResponse(
                    status_code=403,
                    content={"error": {"code": "hacp_model_not_allowed", "message": f"model {model} not allowed for role {role}"}},
                )

        # Budget kill-switch (server-side, authoritative).
        if exp_id is not None:
            exp = store.get_experiment(exp_id)
            bstate = store.budget_state(exp)
            if bstate["exceeded"]:
                store.add_usage(
                    exp_id, role, model, 0, 0, 0, cost_usd=0.0,
                    status="blocked", source="proxy",
                )
                bus.publish(exp_id, "budget_blocked", {"reason": bstate["exceeded_reason"], "model": model})
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "code": "hacp_budget_exhausted",
                            "message": (
                                f"Experiment budget exceeded ({bstate['exceeded_reason']}). "
                                "The platform has stopped LLM access for this experiment."
                            ),
                        }
                    },
                )

        t0 = time.monotonic()
        try:
            body, status = await upstream.complete(payload)
        except Exception as e:
            store.add_usage(exp_id, role, model, 0, 0, 0, status="error", source="proxy")
            bus.publish(exp_id, "llm_error", {"model": model, "error": str(e)})
            raise HTTPException(502, f"upstream error: {e}")
        latency = time.monotonic() - t0

        if status == 200 and isinstance(body, dict):
            usage = body.get("usage") or {}
            pt = int(usage.get("prompt_tokens") or 0)
            ct = int(usage.get("completion_tokens") or 0)
            info = resolve_model(model)
            store.add_usage(
                exp_id, role, model, pt, ct,
                latency_s=round(latency, 4), source="proxy",
            )
            bus.publish(
                exp_id,
                "llm_call",
                {
                    "role": role,
                    "model": model,
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "latency_s": round(latency, 4),
                    "known_pricing": info.known_pricing,
                },
            )
            if not info.known_pricing:
                bus.publish(exp_id, "pricing_warning", {"model": model})
            return body
        # upstream error: pass through with error record
        store.add_usage(exp_id, role, model, 0, 0, 0, status="error", source="proxy")
        return JSONResponse(status_code=status, content=body)
    return router
