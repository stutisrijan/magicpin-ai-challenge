"""Multi-turn handler (challenge brief section 7.4).

    respond(state, merchant_message) -> {"action": "send" | "wait" | "end", ...}

`state` describes the conversation so far. It is a dict (or any object with the same
attributes) with these keys, all optional:

    conversation_id   str
    merchant_id       str            (else merchant["merchant_id"])
    customer_id       str            (else customer["customer_id"])
    category          dict           CategoryContext
    merchant          dict           MerchantContext
    trigger           dict           TriggerContext that started the conversation
    customer          dict           CustomerContext for customer-facing conversations
    turns             [{"from" | "role": "vera" | "bot" | "merchant" | "customer", "body" | "message": str}]
    offer             str            what the opener offered to do (else derived from the trigger kind)
    now               str            ISO time of the reply (else the latest turn ts / wall clock)
    from_role         str            "merchant" (default) or "customer"
    use_llm           bool           default True: use the configured LLM if any, else templates

The same engine as POST /v1/reply runs on a fresh in-memory store seeded from `state`, so
the answer has exactly the /v1/reply shapes:

    {"action": "send", "body": str, "cta": str, "rationale": str}
    {"action": "wait", "wait_seconds": int, "rationale": str}
    {"action": "end", "rationale": str}

respond() is synchronous and never raises.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import importlib
import logging
from typing import Any

log = logging.getLogger(__name__)

_BOT_ROLES = {"vera", "bot", "assistant", "system"}
_CUSTOMER_ROLES = {"customer", "client", "patient", "member"}


def _get(state: Any, key: str, default: Any = None) -> Any:
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _s(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def respond(state: Any, merchant_message: str) -> dict:
    """Given the conversation so far + the merchant's latest message, produce the reply."""
    message = merchant_message if isinstance(merchant_message, str) else ("" if merchant_message is None
                                                                          else str(merchant_message))
    try:
        engine, req = _build(state or {}, message)
        return _run(engine.handle_reply(req))
    except Exception:
        log.exception("respond() failed; using the safe fallback")
        try:
            conversation = importlib.import_module("app.conversation")
            return conversation.fallback_reply({"message": message})
        except Exception:
            return {"action": "wait", "wait_seconds": 1800,
                    "rationale": "Could not process the reply; backing off 30 minutes."}


def _build(state: Any, message: str) -> tuple[Any, dict]:
    conversation = importlib.import_module("app.conversation")
    from app.schemas import SEND_AS_MERCHANT, SEND_AS_VERA
    from app.state import ContextStore, ConversationStore

    contexts = ContextStore()
    convs = ConversationStore()
    category = _get(state, "category") if isinstance(_get(state, "category"), dict) else None
    merchant = _get(state, "merchant") if isinstance(_get(state, "merchant"), dict) else None
    trigger = _get(state, "trigger") if isinstance(_get(state, "trigger"), dict) else None
    customer = _get(state, "customer") if isinstance(_get(state, "customer"), dict) else None

    merchant_id = _s(_get(state, "merchant_id")) or _s((merchant or {}).get("merchant_id")) or \
        _s((trigger or {}).get("merchant_id")) or None
    customer_id = _s(_get(state, "customer_id")) or _s((customer or {}).get("customer_id")) or \
        _s((trigger or {}).get("customer_id")) or None
    if category:
        contexts.put("category", _s(category.get("slug")) or "category", 1, category)
    if merchant and merchant_id:
        payload = {**merchant, "merchant_id": merchant.get("merchant_id") or merchant_id}
        contexts.put("merchant", merchant_id, 1, payload)
    trigger_id = None
    if trigger:
        trigger_id = _s(trigger.get("id")) or "trg_state"
        contexts.put("trigger", trigger_id, 1, {**trigger, "id": trigger_id})
    if customer and customer_id:
        contexts.put("customer", customer_id, 1, customer)

    from_role = (_s(_get(state, "from_role")) or ("customer" if customer_id else "merchant")).lower()
    kind = _s((trigger or {}).get("kind"))
    customer_facing = bool(customer_id) or from_role == "customer"
    cid = _s(_get(state, "conversation_id")) or f"conv_state_{merchant_id or 'unknown'}"
    now = _s(_get(state, "now"))
    convs.observe_time(now or None)

    conv = convs.create(
        cid, merchant_id=merchant_id, customer_id=customer_id, trigger_id=trigger_id, kind=kind or "unknown",
        send_as=SEND_AS_MERCHANT if customer_facing else SEND_AS_VERA,
        offer=_s(_get(state, "offer")) or _default_offer(kind),
        suppression_key=_s((trigger or {}).get("suppression_key")),
    )
    conv.meta["created_by"] = "state"

    turns = _get(state, "turns") or _get(state, "history") or []
    last_ts = None
    for turn in turns if isinstance(turns, (list, tuple)) else []:
        role, body, ts = _turn_fields(turn)
        if not body:
            continue
        last_ts = ts or last_ts
        if role == "bot":
            seq = next(conversation._SEQ)
            convs.add_turn(cid, "bot", body, ts=ts, seq=seq, move="history")
            if not conv.meta.get("ask"):
                conv.meta["ask"] = conversation.extract_ask(body)
            if "confirm" in body.lower() and ("draft" in body.lower() or "here" in body.lower()):
                conv.stage = "action"
            continue
        inbound_role = "customer" if (role == "customer" or customer_facing) else "merchant"
        intent = conversation.classify_inbound(body, conv, convs.merchant(merchant_id) if merchant_id else None)
        convs.add_turn(cid, inbound_role, body, ts=ts, seq=next(conversation._SEQ), intent=intent.label)
        if merchant_id and inbound_role == "merchant":
            norm = conversation.normalize_text(body)
            if len(norm.split()) >= 4:
                st = convs.merchant(merchant_id)
                st.auto_reply_texts[norm] = st.auto_reply_texts.get(norm, 0) + 1
        if conv.stage == "opened":
            conv.stage = "engaged"

    llm = None
    if _get(state, "use_llm", True):
        try:
            llm = importlib.import_module("app.llm").get_llm()
        except Exception:
            llm = None
    engine = conversation.ConversationEngine(contexts, convs, llm)
    req = {
        "conversation_id": cid, "merchant_id": merchant_id, "customer_id": customer_id,
        "from_role": "customer" if customer_facing else "merchant", "message": message,
        "received_at": now or last_ts, "turn_number": len(conv.turns) + 1,
    }
    return engine, req


def _turn_fields(turn: Any) -> tuple[str, str, str | None]:
    if isinstance(turn, dict):
        role = _s(turn.get("from") or turn.get("role") or turn.get("from_role")).lower()
        body = _s(turn.get("body") or turn.get("message") or turn.get("text"))
        ts = _s(turn.get("ts") or turn.get("received_at")) or None
    else:
        role = _s(getattr(turn, "role", "") or getattr(turn, "from_role", "")).lower()
        body = _s(getattr(turn, "body", "") or getattr(turn, "message", ""))
        ts = _s(getattr(turn, "ts", "")) or None
    if role in _BOT_ROLES:
        role = "bot"
    elif role in _CUSTOMER_ROLES:
        role = "customer"
    else:
        role = "merchant"
    return role, body, ts


def _default_offer(kind: str) -> str:
    if not kind:
        return ""
    try:
        return str(importlib.import_module("app.playbooks").get_playbook(kind).offer or "")
    except Exception:
        return ""


def _run(coro: Any) -> dict:
    """Run a coroutine to completion from sync code, even when called inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


__all__ = ["respond"]
