"""Auto-fetch model pricing from OpenRouter for token cost calculation."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request
from urllib.error import URLError

from .config import Config

log = logging.getLogger(__name__)

_CACHE_DIR = Path(".cache")
_CACHE_FILE = _CACHE_DIR / "openrouter_pricing.json"
_CACHE_TTL_SECONDS = 24 * 60 * 60
_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
_MILLION = 1_000_000
_PREFIX = ["anthropic/", "openai/", "google/", "meta/", "z-ai/", "deepseek/", "moonshotai/", "minimax/", "x-ai/"]


def resolve_model_name(cli_model: str | None) -> str | None:
    """Return the model name from CLI arg or env var fallback."""
    if cli_model:
        return cli_model
    env_model = os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
    if env_model:
        return env_model
    return None


def fetch_model_pricing(model_id: str) -> dict | None:
    """Fetch pricing for *model_id* from OpenRouter (with local cache).

    Returns either ``{"token_price_tiers": [...]}`` or scalar keys
    ``input_price``, ``cache_creation_price``, ``cache_read_price``,
    ``output_price`` (all per 1M tokens, float), or ``None`` on failure.
    """
    data = _load_models_data()
    if data is None:
        return None

    # OpenRouter model IDs are like "anthropic/claude-sonnet-4.6".
    # Try exact match first, then prefix match, then normalized match.
    model_info = None
    for m in data:
        if m.get("id") == model_id:
            model_info = m
            break

    if model_info is None:
        for prefix in _PREFIX:
            for m in data:
                if m.get("id") == f"{prefix}{model_id}":
                    model_info = m
                    break
            if model_info:
                break

    if model_info is None:
        # Match dash/dot version variants such as "claude-sonnet-4-6".
        normalized = _normalize_model_id(model_id)
        candidates = [normalized]
        for prefix in _PREFIX:
            candidates.append(f"{prefix}{normalized}")
        for m in data:
            m_norm = _normalize_model_id(m.get("id", ""))
            if m_norm in candidates:
                model_info = m
                break

    if model_info is None:
        log.warning("Model %r not found in OpenRouter pricing data", model_id)
        return None

    result = _extract_openrouter_pricing(model_info.get("pricing", {}))
    if result is None:
        log.warning("Model %r has incomplete OpenRouter pricing data", model_id)
        return None

    if result.get("token_price_tiers"):
        log.info(
            "Resolved tiered pricing for %s: %d tiers",
            model_info.get("id", model_id),
            len(result["token_price_tiers"]),
        )
    else:
        log.info(
            "Resolved pricing for %s: input=%.4f cache_create=%.4f cache_read=%.4f output=%.4f (per 1M tokens)",
            model_info.get("id", model_id),
            result["input_price"] or 0,
            result["cache_creation_price"] or 0,
            result["cache_read_price"] or 0,
            result["output_price"] or 0,
        )
    return result


def has_billable_pricing(config: Config) -> bool:
    """Return True when input+output prices are known."""
    if config.token_price_tiers:
        return bool(normalize_token_price_tiers(config.token_price_tiers))
    return config.input_token_price is not None and config.output_token_price is not None


def normalize_token_price_tiers(tiers: Any) -> list[dict[str, float | int | None]] | None:
    """Normalize tier fields to input/output/cache_read/cache_creation."""
    if not isinstance(tiers, list) or not tiers:
        return None

    normalized: list[dict[str, float | int | None]] = []
    previous_max = -1
    saw_open_ended = False
    for tier in tiers:
        if not isinstance(tier, dict):
            return None
        max_context = tier.get("max_context_tokens")
        if max_context is None:
            normalized_max: int | None = None
            saw_open_ended = True
        elif isinstance(max_context, int) and max_context > previous_max:
            normalized_max = max_context
            previous_max = max_context
        else:
            return None

        input_price = _coerce_price(tier.get("input"))
        output_price = _coerce_price(tier.get("output"))
        if input_price is None or output_price is None:
            return None
        cache_read = _coerce_price(tier.get("cache_read"))
        cache_creation = _coerce_price(tier.get("cache_creation"))
        normalized.append(
            {
                "max_context_tokens": normalized_max,
                "input": input_price,
                "output": output_price,
                "cache_read": input_price if cache_read is None else cache_read,
                "cache_creation": input_price if cache_creation is None else cache_creation,
            }
        )

    if not saw_open_ended or normalized[-1]["max_context_tokens"] is not None:
        return None
    return normalized


def normalize_scalar_pricing(
    *,
    input_price: float | None,
    output_price: float | None,
    cache_read_price: float | None = None,
    cache_creation_price: float | None = None,
) -> dict[str, float] | None:
    """Normalize scalar fields to input/output/cache_read/cache_creation."""
    input_value = _coerce_price(input_price)
    output_value = _coerce_price(output_price)
    if input_value is None or output_value is None:
        return None
    cache_read = _coerce_price(cache_read_price)
    cache_creation = _coerce_price(cache_creation_price)
    return {
        "input": input_value,
        "output": output_value,
        "cache_read": input_value if cache_read is None else cache_read,
        "cache_creation": input_value if cache_creation is None else cache_creation,
    }


def missing_pricing_message(model: str | None) -> str:
    model_label = model or "the selected model"
    return (
        "EurekAgent could not resolve both input and output token prices for "
        f"{model_label}.\n"
        "Live token/cost display will show N/A if you continue.\n\n"
        "Recommended: exit and add token pricing to your run script:\n"
        "  Scalar pricing:\n"
        "    --input-token-price <per-1M> --output-token-price <per-1M>\n"
        "    [--cache-read-token-price <per-1M>]\n"
        "    [--cache-creation-token-price <per-1M>]\n"
        "  Tiered pricing:\n"
        "    --token-price-tiers '[{\"max_context_tokens\":32768,\"input\":...,"
        "\"output\":...,\"cache_read\":...,\"cache_creation\":...},"
        "{\"max_context_tokens\":null,...}]'"
    )


def _extract_openrouter_pricing(pricing: Any) -> dict | None:
    if not isinstance(pricing, dict):
        return None

    tiers = _extract_openrouter_tiers(pricing)
    if tiers:
        return {"token_price_tiers": tiers}

    scalar = _openrouter_price_fields(pricing)
    normalized = normalize_scalar_pricing(
        input_price=scalar.get("input"),
        output_price=scalar.get("output"),
        cache_read_price=scalar.get("cache_read"),
        cache_creation_price=scalar.get("cache_creation"),
    )
    if normalized is None:
        return None
    return {
        "input_price": normalized["input"],
        "cache_creation_price": normalized["cache_creation"],
        "cache_read_price": normalized["cache_read"],
        "output_price": normalized["output"],
    }


def _extract_openrouter_tiers(
    pricing: dict,
) -> list[dict[str, float | int | None]] | None:
    overrides = pricing.get("overrides")
    if not isinstance(overrides, list) or not overrides:
        return None

    ordered = [
        override
        for override in overrides
        if (
            isinstance(override, dict)
            and isinstance(override.get("min_prompt_tokens"), int)
        )
    ]
    ordered.sort(key=lambda override: override["min_prompt_tokens"])
    if not ordered:
        return None

    tiers: list[dict[str, float | int | None]] = []
    current = _openrouter_price_fields(pricing)
    first_min = int(ordered[0]["min_prompt_tokens"])
    tiers.append({"max_context_tokens": first_min, **current})

    for index, override in enumerate(ordered):
        current = {
            **current,
            **_openrouter_price_fields(override, include_missing=False),
        }
        next_max = (
            int(ordered[index + 1]["min_prompt_tokens"])
            if index + 1 < len(ordered)
            else None
        )
        tiers.append({"max_context_tokens": next_max, **current})

    return normalize_token_price_tiers(tiers)


def _openrouter_price_fields(
    pricing: dict,
    *,
    include_missing: bool = True,
) -> dict[str, float | None]:
    fields = {
        "input": _parse_price(pricing.get("prompt")),
        "output": _parse_price(pricing.get("completion")),
        "cache_read": _parse_price(
            pricing.get("input_cache_read") or pricing.get("cache_read")
        ),
        "cache_creation": _parse_price(
            pricing.get("input_cache_write") or pricing.get("cache_creation")
        ),
    }
    if include_missing:
        return fields
    return {key: value for key, value in fields.items() if value is not None}


def _coerce_price(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or value < 0:
        return None
    return float(value)


def _parse_price(value: Any) -> float | None:
    """Parse a per-token price string to a per-1M-token float. None if empty."""
    if not value:
        return None
    try:
        per_token = float(value)
        return per_token * _MILLION
    except (ValueError, TypeError):
        return None


def _normalize_model_id(model_id: str) -> str:
    """Normalize model ID for fuzzy matching (e.g. claude-sonnet-4-6 → claude-sonnet-4.6)."""
    import re
    return re.sub(r'(\d)-(\d)', r'\1.\2', model_id)


def _load_models_data() -> list[dict] | None:
    """Load models data from cache or fetch from OpenRouter."""
    cached = _read_cache()
    if cached is not None:
        return cached

    fetched = _fetch_from_api()
    if fetched is not None:
        return fetched

    # Fetch failed — try stale cache as last resort
    stale = _read_cache(ignore_ttl=True)
    if stale is not None:
        log.warning("OpenRouter fetch failed; using stale cached pricing data")
        return stale

    log.warning("OpenRouter fetch failed and no cached pricing data available")
    return None


def _read_cache(*, ignore_ttl: bool = False) -> list[dict] | None:
    """Read cached models data. Returns None if missing or expired."""
    if not _CACHE_FILE.is_file():
        return None
    try:
        raw = json.loads(_CACHE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    fetched_at = raw.get("fetched_at", 0)
    if not ignore_ttl and (time.time() - fetched_at > _CACHE_TTL_SECONDS):
        return None

    return raw.get("data")


def _fetch_from_api() -> list[dict] | None:
    """Fetch models data from OpenRouter API and update cache."""
    log.info("Fetching model pricing from OpenRouter ...")
    try:
        req = Request(_OPENROUTER_MODELS_URL, headers={"User-Agent": "EurekaLoop/1.0"})
        with urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read())
    except (URLError, OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to fetch OpenRouter pricing: %s", exc)
        return None

    data = body.get("data")
    if not isinstance(data, list):
        log.warning("Unexpected OpenRouter response format")
        return None

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _CACHE_FILE.write_text(json.dumps({
            "fetched_at": time.time(),
            "data": data,
        }, indent=2, sort_keys=False))
    except OSError as exc:
        log.warning("Failed to write pricing cache: %s", exc)

    return data
