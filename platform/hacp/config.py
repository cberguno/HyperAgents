"""Platform configuration: model registry with pricing, experiment and service config.

Pricing is expressed per 1M tokens (USD) and is an *estimate* used for budget
enforcement and cost dashboards. Prices change over time; overrides can be
supplied via ``HACP_PRICE_OVERRIDES`` (JSON: {"<model-id>": [in, out]}).
Models not present in the registry are tracked with zero cost and flagged as
unknown pricing (never silently guessed).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelInfo:
    id: str  # canonical id (also the OpenRouter model id)
    provider: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    label: str
    known_pricing: bool = True


# Approximate list prices per 1M tokens (input, output), USD.
_PRICING: dict[str, tuple[str, float, float, str]] = {
    # --- Anthropic ---
    "anthropic/claude-sonnet-4-5-20250929": ("anthropic", 3.00, 15.00, "Claude Sonnet 4.5"),
    "anthropic/claude-haiku-4-5": ("anthropic", 1.00, 5.00, "Claude Haiku 4.5"),
    "anthropic/claude-3-5-sonnet-20241022": ("anthropic", 3.00, 15.00, "Claude 3.5 Sonnet"),
    "anthropic/claude-3-haiku-20240307": ("anthropic", 0.25, 1.25, "Claude 3 Haiku"),
    # --- OpenAI ---
    "openai/gpt-4o": ("openai", 2.50, 10.00, "GPT-4o"),
    "openai/gpt-4o-mini": ("openai", 0.15, 0.60, "GPT-4o mini"),
    "openai/o3": ("openai", 2.00, 8.00, "o3"),
    "openai/o3-mini": ("openai", 1.10, 4.40, "o3-mini"),
    "openai/o4-mini": ("openai", 1.10, 4.40, "o4-mini"),
    "openai/gpt-5": ("openai", 1.25, 10.00, "GPT-5"),
    "openai/gpt-5-mini": ("openai", 0.25, 2.00, "GPT-5 mini"),
    "openai/gpt-5.2": ("openai", 1.25, 10.00, "GPT-5.2"),
    # --- Google ---
    "gemini/gemini-2.5-pro": ("google", 1.25, 10.00, "Gemini 2.5 Pro"),
    "gemini/gemini-2.5-flash": ("google", 0.30, 2.50, "Gemini 2.5 Flash"),
    "gemini/gemini-3-pro-preview": ("google", 2.00, 12.00, "Gemini 3 Pro (preview)"),
    # --- Offline stub (never billed) ---
    "stub/tiny": ("stub", 0.0, 0.0, "Offline stub model (no cost)"),
    "stub/task": ("stub", 0.0, 0.0, "Offline stub model (no cost)"),
}

_ALIASES: dict[str, str] = {
    # convenience names -> canonical ids
    "anthropic/claude-sonnet-4.5": "anthropic/claude-sonnet-4-5-20250929",
    "anthropic/claude-sonnet-4-5": "anthropic/claude-sonnet-4-5-20250929",
    "anthropic/claude-haiku-4.5": "anthropic/claude-haiku-4-5",
    "openai/claude-sonnet-4.5": "anthropic/claude-sonnet-4-5-20250929",
}


def _price_overrides() -> dict[str, tuple[float, float]]:
    raw = os.environ.get("HACP_PRICE_OVERRIDES", "")
    out: dict[str, tuple[float, float]] = {}
    if raw:
        try:
            data = json.loads(raw)
            out = {k: (float(v[0]), float(v[1])) for k, v in data.items()}
        except Exception:
            pass
    return out


def normalize_model(name: str) -> str:
    """Normalize a model id (handles OpenRouter prefixing and common aliases)."""
    n = (name or "").strip()
    if n.startswith("openrouter/"):
        n = n[len("openrouter/"):]
    return _ALIASES.get(n, n)


def resolve_model(name: str) -> ModelInfo:
    n = normalize_model(name)
    ov = _price_overrides().get(n)
    if n in _PRICING:
        provider, i, o, label = _PRICING[n]
        if ov:
            i, o = ov
        return ModelInfo(n, provider, i, o, label, True)
    if ov:
        return ModelInfo(n, "custom", ov[0], ov[1], n, True)
    return ModelInfo(n, "unknown", 0.0, 0.0, n, False)


def estimate_cost(model: str, prompt_tokens: int | float, completion_tokens: int | float) -> float:
    info = resolve_model(model)
    return (
        float(prompt_tokens) / 1e6 * info.input_usd_per_mtok
        + float(completion_tokens) / 1e6 * info.output_usd_per_mtok
    )


def model_catalog() -> list[dict]:
    ov = _price_overrides()
    out = []
    for mid, (provider, i, o, label) in _PRICING.items():
        if ov.get(mid):
            i, o = ov[mid]
        out.append(
            {
                "id": mid,
                "provider": provider,
                "label": label,
                "input_usd_per_mtok": i,
                "output_usd_per_mtok": o,
                "approx_cost_per_10k_prompt_tokens": round(i / 100, 6),
                "approx_cost_per_10k_completion_tokens": round(o / 100, 6),
            }
        )
    return sorted(out, key=lambda m: (m["provider"], m["id"]))


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------

class BudgetConfig(BaseModel):
    """Hard limits enforced by the platform for one experiment.

    The LLM proxy refuses requests once a limit is exceeded (server-side,
    impossible for a self-modifying agent to bypass) and the runner stops
    the evolution loop (status ``paused_budget``).
    """

    max_cost_usd: Optional[float] = Field(
        default=None, description="Maximum total estimated LLM cost in USD."
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum total tokens (prompt + completion)."
    )
    max_wallclock_hours: Optional[float] = Field(
        default=None, description="Maximum experiment wall-clock duration in hours."
    )

    def limits_str(self) -> str:
        parts = []
        if self.max_cost_usd is not None:
            parts.append(f"cost<= ${self.max_cost_usd:g}")
        if self.max_tokens is not None:
            parts.append(f"tokens<= {self.max_tokens}")
        if self.max_wallclock_hours is not None:
            parts.append(f"time<= {self.max_wallclock_hours:g}h")
        return ", ".join(parts) if parts else "unlimited"


class ExperimentConfig(BaseModel):
    """Configuration for one experiment (maps onto upstream generate_loop.py flags)."""

    name: str = Field(min_length=1, max_length=120)
    domains: list[str] = Field(default_factory=lambda: ["paper_review"])
    meta_model: str = "anthropic/claude-sonnet-4-5-20250929"
    task_model: str = "openai/gpt-4o"  # mirrors upstream paper_review default
    max_generation: int = Field(default=10, ge=1, le=500)
    parent_selection: str = Field(
        default="score_child_prop",
        pattern="^(random|latest|best|score_prop|score_child_prop)$",
    )
    optimize_option: str = Field(
        default="only_agent", pattern="^(both_agent_ensemble|only_agent|only_ensemble)$"
    )
    eval_samples: list[int] = Field(default_factory=list, description="-1 per domain if empty")
    eval_workers: int = Field(default=10, ge=1, le=64)
    eval_test: bool = False
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    seed: Optional[int] = None
    notes: str = ""
    # Platform-internal:
    loop: str = Field(
        default="generate",
        description="'generate' = real upstream generate_loop.py (needs Docker); "
        "'stub' = offline simulated loop (tests/demos, no API cost).",
    )

    @field_validator("meta_model", "task_model")
    @classmethod
    def _canonical_model(cls, v: str) -> str:
        """Store canonical model ids (e.g. 'anthropic/claude-sonnet-4.5' ->
        'anthropic/claude-sonnet-4-5-20250929') so catalog lookups and UI
        prefill always match."""
        return normalize_model(v)

    def generate_loop_args(self, run_id: str, resume_from: str | None) -> list[str]:
        """Map this config onto upstream generate_loop.py CLI flags (unmodified)."""
        domains = self.domains
        samples = self.eval_samples or [-1] * len(domains)
        args = [
            "generate_loop.py",
            "--domains", *domains,
            "--run_id", run_id,
            "--max_generation", str(self.max_generation),
            "--eval_samples", *[str(s) for s in samples],
            "--eval_workers", str(self.eval_workers),
            "--parent_selection", self.parent_selection,
            "--optimize_option", self.optimize_option,
        ]
        if self.eval_test:
            args.append("--eval_test")
        if resume_from:
            args += ["--resume_from", resume_from]
        return args

    def initial_eval_dirs(self) -> list[str]:
        """Upstream output dirs that must exist before generate_loop can start
        (produced by setup_initial.sh / the runner's seed phase)."""
        out = []
        for d in self.domains:
            if d == "polyglot":
                out.append(f"initial_{d}_0")
                continue
            out.append(f"initial_{d}_filtered_100_train_0")
        return out


class ServiceConfig(BaseModel):
    """Platform service configuration (env-driven, portable)."""

    host: str = "127.0.0.1"
    port: int = 8420
    home: str = ""  # default: <platform>/home
    upstream: str = Field(default="stub", pattern="^(stub|openrouter)$")
    upstream_repo: str = ""  # default: repo root containing platform/
    initial_slots: int = Field(default=5, ge=1, le=100)
    shell_model: str = "openai/gpt-4o-mini"
    shell_max_tool_calls: int = 8
    # Filled at runtime (host the in-container LLM calls reach via host networking):
    proxy_public_base: str = ""  # e.g. http://127.0.0.1:8420 — where the proxy listens
    enforce_models: bool = False

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        cfg = cls(
            host=os.environ.get("HACP_HOST", "127.0.0.1"),
            port=int(os.environ.get("HACP_PORT", "8420")),
            home=os.environ.get("HACP_HOME", ""),
            upstream=os.environ.get("HACP_UPSTREAM", "stub"),
            upstream_repo=os.environ.get("HACP_UPSTREAM_REPO", ""),
            shell_model=os.environ.get("HACP_SHELL_MODEL", "openai/gpt-4o-mini"),
        )
        if not cfg.upstream_repo:
            from .paths import upstream_repo_default

            cfg.upstream_repo = str(upstream_repo_default())
        if not cfg.home:
            from .paths import platform_home

            cfg.home = str(platform_home())
        if not cfg.proxy_public_base:
            cfg.proxy_public_base = f"http://{cfg.host}:{cfg.port}"
        return cfg

    @property
    def proxy_v1_base(self) -> str:
        return f"{self.proxy_public_base}/proxy/v1"
