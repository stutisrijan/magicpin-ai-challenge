"""Vera bot entry point.

    uvicorn bot:app --host 0.0.0.0 --port 8080 --workers 1

`app` is the FastAPI service (app/main.py). `compose()` is the offline composition
contract from challenge-brief section 7.1, used to produce submission.jsonl.
"""

from __future__ import annotations

import importlib

from app.main import app

__all__ = ["app", "compose"]

_KEYS = ("body", "cta", "send_as", "suppression_key", "rationale")


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    """
    Inputs are the dicts loaded from the dataset JSON.
    Return a dict with keys: body, cta, send_as, suppression_key, rationale.
    Free to use any LLM, any prompting strategy, any retrieval.
    Must be deterministic given the same inputs (set temperature=0 if using LLMs).
    Must complete in < 30s per call.

    Implementation: facts are extracted deterministically from the four contexts, a
    per-kind playbook renders a grounded template draft, and (only when an LLM key is
    configured) an LLM at temperature 0 polishes the wording behind a validator that
    rejects any ungrounded number. Without a key the template draft is returned.
    """
    composer = importlib.import_module("app.composer")
    out = composer.compose(category or {}, merchant, trigger, customer)
    return {k: out.get(k) for k in _KEYS}
