from .base import Usage


def usage_dict_from_openai_usage(usage) -> Usage:
    """Map an OpenAI(-compatible) usage object onto our Usage TypedDict.

    Cache HITS are read from ``usage.prompt_tokens_details.cached_tokens`` and surface as
    ``cache_read_tokens`` so agent.py's "📦 r=…" line lights up. We never get a cache-*creation*
    count in the OpenAI usage shape, so ``cache_creation_tokens`` is always 0 here (the "w=…"
    counter stays quiet) — note this is just a reporting gap, not proof there was no write.

    Caveat for a Claude/Sonnet backend behind an OpenAI-compatible gateway: ``cached_tokens`` only
    shows real numbers if the gateway maps the backend's cache usage into OpenAI's usage shape
    (LiteLLM-style proxies map Anthropic's cache_read_input_tokens → cached_tokens; others may
    report nothing even when a hit occurred). So 0 here means "no hit OR not reported" — confirm
    against the gateway's docs before treating 0 as "caching is off".
    """
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details is not None else 0
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cache_read_tokens": cached or 0,
        "cache_creation_tokens": 0,
    }
