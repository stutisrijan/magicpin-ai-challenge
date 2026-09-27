"""Runtime configuration, read once from environment variables.

Every knob has a safe default so the bot boots with zero configuration
(deterministic composer only). Add an LLM key to enable LLM composition.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    return default if val is None or val.strip() == "" else val.strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env(name, str(default))))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str) -> list[str]:
    return [p.strip() for p in _env(name, default).split(",") if p.strip()]


@dataclass
class Settings:
    # --- identity (GET /v1/metadata) ---
    team_name: str = field(default_factory=lambda: _env("TEAM_NAME", "Stuti Srijan"))
    team_members: list[str] = field(default_factory=lambda: _env_list("TEAM_MEMBERS", "Stuti Srijan"))
    contact_email: str = field(default_factory=lambda: _env("CONTACT_EMAIL", ""))
    bot_version: str = field(default_factory=lambda: _env("BOT_VERSION", "1.0.0"))
    submitted_at: str = field(default_factory=lambda: _env("SUBMITTED_AT", "2026-09-26T00:00:00Z"))

    # --- LLM providers, tried in this order; unknown / keyless providers are skipped ---
    llm_providers: list[str] = field(default_factory=lambda: _env_list("LLM_PROVIDERS", "gemini,groq,openai,anthropic,openrouter"))
    llm_disabled: bool = field(default_factory=lambda: _env_bool("LLM_DISABLED", False))

    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY", _env("GOOGLE_API_KEY", "")))
    gemini_models: list[str] = field(default_factory=lambda: _env_list("GEMINI_MODELS", "gemini-flash-latest,gemini-flash-lite-latest,gemini-3.5-flash-lite"))
    gemini_rpm: int = field(default_factory=lambda: _env_int("GEMINI_RPM", 10))

    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "openai/gpt-oss-120b,qwen/qwen3.8-27b"))
    groq_rpm: int = field(default_factory=lambda: _env_int("GROQ_RPM", 30))

    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    openai_rpm: int = field(default_factory=lambda: _env_int("OPENAI_RPM", 60))

    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-5"))
    anthropic_rpm: int = field(default_factory=lambda: _env_int("ANTHROPIC_RPM", 50))

    openrouter_api_key: str = field(default_factory=lambda: _env("OPENROUTER_API_KEY", ""))
    openrouter_model: str = field(default_factory=lambda: _env("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"))
    openrouter_rpm: int = field(default_factory=lambda: _env_int("OPENROUTER_RPM", 20))

    # --- latency budgets (seconds). Judge timeout is 30s; local simulator uses 15s. ---
    llm_timeout_s: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_S", 7.0))
    tick_deadline_s: float = field(default_factory=lambda: _env_float("TICK_DEADLINE_S", 9.0))
    reply_deadline_s: float = field(default_factory=lambda: _env_float("REPLY_DEADLINE_S", 8.0))
    precompose: bool = field(default_factory=lambda: _env_bool("PRECOMPOSE", True))
    precompose_concurrency: int = field(default_factory=lambda: _env_int("PRECOMPOSE_CONCURRENCY", 3))

    # --- engagement policy ---
    max_actions_per_tick: int = field(default_factory=lambda: _env_int("MAX_ACTIONS_PER_TICK", 20))
    auto_reply_cooldown_min: int = field(default_factory=lambda: _env_int("AUTO_REPLY_COOLDOWN_MIN", 60))
    urgent_bypass_urgency: int = field(default_factory=lambda: _env_int("URGENT_BYPASS_URGENCY", 4))

    # --- ops ---
    port: int = field(default_factory=lambda: _env_int("PORT", 8080))
    self_ping_url: str = field(default_factory=lambda: _env("SELF_PING_URL", ""))
    self_ping_interval_s: int = field(default_factory=lambda: _env_int("SELF_PING_INTERVAL_S", 600))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))


settings = Settings()


def reload_settings() -> Settings:
    """Re-read the environment (used by tests)."""
    global settings
    settings = Settings()
    return settings
