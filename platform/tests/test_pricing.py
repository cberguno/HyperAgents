"""Pricing registry and cost estimation tests (offline)."""
import pytest

from hacp.config import estimate_cost, model_catalog, normalize_model, resolve_model


def test_normalize_openrouter_prefix():
    assert normalize_model("openrouter/openai/gpt-4o") == "openai/gpt-4o"
    assert normalize_model("openai/gpt-4o") == "openai/gpt-4o"


def test_aliases():
    assert normalize_model("anthropic/claude-sonnet-4.5") == "anthropic/claude-sonnet-4-5-20250929"
    assert normalize_model("openrouter/anthropic/claude-sonnet-4.5") == "anthropic/claude-sonnet-4-5-20250929"


def test_cost_math():
    # gpt-4o: $2.50/M in, $10/M out
    cost = estimate_cost("openrouter/openai/gpt-4o", 1_000_000, 500_000)
    assert cost == pytest.approx(2.5 + 5.0)
    # claude sonnet 4.5: 3/15
    cost = estimate_cost("anthropic/claude-sonnet-4-5-20250929", 1000, 1000)
    assert cost == pytest.approx((3 + 15) * 1000 / 1e6)


def test_unknown_model_is_flagged_not_guessed():
    info = resolve_model("totally/unknown-model")
    assert not info.known_pricing
    assert estimate_cost("totally/unknown-model", 1e9, 1e9) == 0.0


def test_price_overrides_env(monkeypatch):
    monkeypatch.setenv("HACP_PRICE_OVERRIDES", '{"custom/model": [1.0, 2.0]}')
    info = resolve_model("custom/model")
    assert info.known_pricing
    assert info.input_usd_per_mtok == 1.0
    assert estimate_cost("custom/model", 500_000, 500_000) == pytest.approx(0.5 + 1.0)


def test_stub_models_are_free():
    assert estimate_cost("stub/tiny", 10**9, 10**9) == 0.0


def test_gpt6_luna_pricing_matches_openrouter():
    # OpenRouter /models 2026-10-16: $0.10/M in, $0.50/M out
    cost = estimate_cost("openai/gpt-6-luna", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.60)
    assert resolve_model("openai/gpt-6-luna").known_pricing is True


def test_catalog_shape():
    cat = model_catalog()
    assert any(m["id"] == "openai/gpt-4o-mini" for m in cat)
    for m in cat:
        assert "input_usd_per_mtok" in m and "id" in m
