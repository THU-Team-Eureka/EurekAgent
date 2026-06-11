"""Auto-fetch model pricing from OpenRouter for token cost calculation."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

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

    Returns a dict with keys ``input_price``, ``cache_creation_price``,
    ``cache_read_price``, ``output_price`` (all per 1M tokens, float),
    or ``None`` on failure.
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

    pricing = model_info.get("pricing", {})
    result = _extract_prices(pricing)
    log.info(
        "Resolved pricing for %s: input=%.4f cache_create=%.4f cache_read=%.4f output=%.4f (per 1M tokens)",
        model_info.get("id", model_id),
        result["input_price"] or 0,
        result["cache_creation_price"] or 0,
        result["cache_read_price"] or 0,
        result["output_price"] or 0,
    )
    return result


def _extract_prices(pricing: dict) -> dict:
    """Convert OpenRouter per-token string prices to per-1M-token floats."""
    prompt = _parse_price(pricing.get("prompt", "0"))
    completion = _parse_price(pricing.get("completion", "0"))
    # OpenRouter uses "input_cache_write" / "input_cache_read" for cache pricing
    cache_creation = _parse_price(
        pricing.get("input_cache_write") or pricing.get("cache_creation", "")
    )
    cache_read = _parse_price(
        pricing.get("input_cache_read") or pricing.get("cache_read", "")
    )

    # Fallbacks when cache fields are missing
    if cache_creation is None:
        cache_creation = prompt
    if cache_read is None:
        cache_read = round(prompt * 0.1, 8) if prompt is not None else None

    return {
        "input_price": prompt,
        "cache_creation_price": cache_creation,
        "cache_read_price": cache_read,
        "output_price": completion,
    }


def _parse_price(value: str) -> float | None:
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
