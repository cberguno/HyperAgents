"""Platform-side LLM client (used by the conversational shell).

Distinct from the experiment path: this client is platform code calling the
platform's own proxy with the platform-level token (role='shell'), so shell
usage is accounted in the same ledger with its own independently configurable
model (``HACP_SHELL_MODEL``). Never touches upstream agent code.
"""
from __future__ import annotations

from typing import Optional, Protocol

import httpx


class LLMClient(Protocol):
    def chat(self, messages: list[dict], model: Optional[str] = None,
             temperature: float = 0.2, max_tokens: int = 4096) -> str: ...


class ProxyLLMClient:
    """Synchronous client for the platform's OpenAI-compatible proxy."""

    def __init__(self, proxy_v1_base: str, token: str, default_model: str, timeout: float = 120.0):
        self.base = proxy_v1_base.rstrip("/")
        self.token = token
        self.default_model = default_model
        self.timeout = timeout

    def chat(self, messages: list[dict], model: Optional[str] = None,
             temperature: float = 0.2, max_tokens: int = 4096) -> str:
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = httpx.post(
            f"{self.base}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            try:
                detail = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                detail = resp.text
            raise RuntimeError(f"LLM proxy error {resp.status_code}: {detail}")
        body = resp.json()
        return body["choices"][0]["message"]["content"]


class FakeLLMClient:
    """Scripted client for offline tests/demos of the shell."""

    def __init__(self, responses: list[str], default_model: str = "stub/shell"):
        self.responses = list(responses)
        self.default_model = default_model
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], model: Optional[str] = None,
             temperature: float = 0.2, max_tokens: int = 4096) -> str:
        self.calls.append({"messages": [dict(m) for m in messages], "model": model})
        if not self.responses:
            raise RuntimeError("FakeLLMClient ran out of scripted responses")
        return self.responses.pop(0)
