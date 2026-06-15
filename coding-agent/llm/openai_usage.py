from .base import Usage


def usage_dict_from_openai_usage(usage) -> Usage:
    """Map an OpenAI(-compatible) usage object onto our Usage TypedDict.

    OpenAI prompt caching is *automatic* — there is no cache_control breakpoint like Anthropic.
    Cache HITS are reported back in ``usage.prompt_tokens_details.cached_tokens`` and surface as
    ``cache_read_tokens`` so agent.py's "📦 r=…" line lights up. There is no cache-*creation*
    metric (the discount is applied automatically, with no separate write cost), so
    ``cache_creation_tokens`` is always 0 — that is expected, not a bug, and lets the agent's
    "w=…" counter stay quiet on this provider.

    ``cached_tokens`` is the single signal that tells us whether caching is live on the gateway:
    0 on the first turn, > 0 on later turns that reuse a stable prefix.
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
