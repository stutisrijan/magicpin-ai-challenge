"""In-memory state: versioned contexts, conversations, per-merchant engagement memory.

Single-process by design: run exactly one worker so every request sees the same state.
All public methods are guarded by one re-entrant lock, so they are safe to call from
the event loop and from worker threads alike.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

SCOPES = ("category", "merchant", "customer", "trigger")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(value: str | None) -> datetime | None:
    """Parse ISO-8601 timestamps (with 'Z' or offsets). Returns aware UTC datetime or None."""
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        try:
            dt = datetime.strptime(value.strip()[:10], "%Y-%m-%d")
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# --------------------------------------------------------------------------- contexts


@dataclass
class StoredContext:
    scope: str
    context_id: str
    version: int
    payload: dict
    stored_at: str


class ContextStore:
    """Versioned context storage keyed by (scope, context_id).

    put() semantics (testing brief section 2.1):
      * new id, or higher version  -> replace atomically, status "accepted"
      * same or lower version      -> no-op, status "stale" (HTTP 409 at the API layer)
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: dict[tuple[str, str], StoredContext] = {}
        self._aliases: dict[tuple[str, str], str] = {}   # (scope, payload-internal id) -> context_id
        self._listeners: list = []

    # -- mutation -----------------------------------------------------------------
    def put(self, scope: str, context_id: str, version: int, payload: dict) -> tuple[str, int, str]:
        """Returns (status, current_version, stored_at). status in {"accepted", "stale"}."""
        with self._lock:
            key = (scope, context_id)
            cur = self._data.get(key)
            if cur is not None and version <= cur.version:
                return "stale", cur.version, cur.stored_at
            stored_at = utc_now_iso()
            self._data[key] = StoredContext(scope, context_id, version, payload, stored_at)
            internal_id = self._internal_id(scope, payload)
            if internal_id and internal_id != context_id:
                self._aliases[(scope, internal_id)] = context_id
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(scope, context_id, version, payload)
            except Exception:  # listeners must never break ingestion
                pass
        return "accepted", version, stored_at

    def add_listener(self, fn) -> None:
        """fn(scope, context_id, version, payload) called after every accepted put."""
        with self._lock:
            self._listeners.append(fn)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._aliases.clear()

    # -- queries --------------------------------------------------------------------
    def get(self, scope: str, context_id: str | None) -> dict | None:
        if not context_id:
            return None
        with self._lock:
            item = self._data.get((scope, context_id))
            if item is None:
                alias = self._aliases.get((scope, context_id))
                item = self._data.get((scope, alias)) if alias else None
            return item.payload if item else None

    def get_version(self, scope: str, context_id: str | None) -> int | None:
        if not context_id:
            return None
        with self._lock:
            item = self._data.get((scope, context_id))
            if item is None:
                alias = self._aliases.get((scope, context_id))
                item = self._data.get((scope, alias)) if alias else None
            return item.version if item else None

    def ids(self, scope: str) -> list[str]:
        with self._lock:
            return [cid for (s, cid) in self._data if s == scope]

    def counts(self) -> dict[str, int]:
        with self._lock:
            out = {s: 0 for s in SCOPES}
            for (s, _cid) in self._data:
                out[s] = out.get(s, 0) + 1
            return out

    def category_for(self, merchant: dict | None, trigger: dict | None = None) -> dict | None:
        """Category context for a merchant (by category_slug), falling back to trigger.payload.category."""
        slug = None
        if merchant:
            slug = merchant.get("category_slug") or (merchant.get("identity") or {}).get("category")
        if not slug and trigger:
            slug = (trigger.get("payload") or {}).get("category")
        return self.get("category", slug) if slug else None

    def customers_for(self, merchant_id: str) -> list[dict]:
        with self._lock:
            return [c.payload for (s, _cid), c in self._data.items()
                    if s == "customer" and c.payload.get("merchant_id") == merchant_id]

    @staticmethod
    def _internal_id(scope: str, payload: dict) -> str | None:
        if not isinstance(payload, dict):
            return None
        return {
            "category": payload.get("slug"),
            "merchant": payload.get("merchant_id"),
            "customer": payload.get("customer_id"),
            "trigger": payload.get("id"),
        }.get(scope)


# --------------------------------------------------------------------------- conversations


@dataclass
class Turn:
    role: str            # "bot" | "merchant" | "customer"
    body: str
    ts: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class Conversation:
    conversation_id: str
    merchant_id: str | None = None
    customer_id: str | None = None
    trigger_id: str | None = None
    kind: str = ""
    send_as: str = "vera"
    status: str = "active"          # "active" | "waiting" | "ended"
    stage: str = "opened"           # "opened" | "engaged" | "action" | "done"
    offer: str = ""                 # what the opening message offered to do
    suppression_key: str = ""
    language: str = ""              # last language code used/observed
    turns: list[Turn] = field(default_factory=list)
    auto_reply_streak: int = 0
    auto_reply_total: int = 0
    hostile_count: int = 0
    off_topic_count: int = 0
    wait_until: str | None = None
    ended_reason: str = ""
    created_at: str = ""
    updated_at: str = ""
    facts_digest: list[str] = field(default_factory=list)   # key fact texts from the opener, for grounding replies
    meta: dict[str, Any] = field(default_factory=dict)

    def bot_bodies(self) -> list[str]:
        return [t.body for t in self.turns if t.role == "bot" and t.body]

    def inbound(self) -> list[Turn]:
        return [t for t in self.turns if t.role != "bot"]


@dataclass
class MerchantState:
    merchant_id: str
    opted_out: bool = False
    opted_out_at: str | None = None
    cooldown_until: str | None = None          # simulated-time ISO; no low-urgency proactive sends before this
    auto_reply_texts: dict[str, int] = field(default_factory=dict)   # normalised inbound text -> count (all convs)
    last_sent_at: str | None = None
    sent_bodies: list[str] = field(default_factory=list)
    hostile_count: int = 0


class ConversationStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._convs: dict[str, Conversation] = {}
        self._merchants: dict[str, MerchantState] = {}
        self._sent_suppression: dict[str, str] = {}      # suppression_key -> conversation_id
        self._skipped: dict[str, str] = {}               # trigger_id -> reason (permanently skipped)
        self.sim_now: str | None = None                  # latest simulated time seen (tick.now / reply.received_at)

    # -- conversations ------------------------------------------------------------------
    def get(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            return self._convs.get(conversation_id)

    def create(self, conversation_id: str, **kwargs) -> Conversation:
        with self._lock:
            conv = Conversation(conversation_id=conversation_id, **kwargs)
            now = self.sim_now or utc_now_iso()
            conv.created_at = conv.created_at or now
            conv.updated_at = now
            self._convs[conversation_id] = conv
            return conv

    def get_or_create(self, conversation_id: str, **kwargs) -> Conversation:
        with self._lock:
            conv = self._convs.get(conversation_id)
            return conv if conv is not None else self.create(conversation_id, **kwargs)

    def exists(self, conversation_id: str) -> bool:
        with self._lock:
            return conversation_id in self._convs

    def add_turn(self, conversation_id: str, role: str, body: str, ts: str | None = None, **meta) -> None:
        with self._lock:
            conv = self._convs.get(conversation_id)
            if conv is None:
                return
            conv.turns.append(Turn(role=role, body=body, ts=ts or self.sim_now or utc_now_iso(), meta=meta))
            conv.updated_at = ts or self.sim_now or utc_now_iso()

    def all(self) -> list[Conversation]:
        with self._lock:
            return list(self._convs.values())

    def for_merchant(self, merchant_id: str) -> list[Conversation]:
        with self._lock:
            return [c for c in self._convs.values() if c.merchant_id == merchant_id]

    # -- merchant memory -----------------------------------------------------------------
    def merchant(self, merchant_id: str) -> MerchantState:
        with self._lock:
            st = self._merchants.get(merchant_id)
            if st is None:
                st = MerchantState(merchant_id=merchant_id)
                self._merchants[merchant_id] = st
            return st

    # -- suppression ---------------------------------------------------------------------
    def suppression_used(self, key: str | None) -> bool:
        if not key:
            return False
        with self._lock:
            return key in self._sent_suppression

    def mark_suppression(self, key: str | None, conversation_id: str) -> None:
        if not key:
            return
        with self._lock:
            self._sent_suppression[key] = conversation_id

    def mark_skipped(self, trigger_id: str, reason: str) -> None:
        with self._lock:
            self._skipped[trigger_id] = reason

    def skipped_reason(self, trigger_id: str) -> str | None:
        with self._lock:
            return self._skipped.get(trigger_id)

    # -- clock ---------------------------------------------------------------------------
    def observe_time(self, iso: str | None) -> None:
        """Track the latest simulated time we have seen."""
        dt = parse_iso(iso)
        if dt is None:
            return
        with self._lock:
            cur = parse_iso(self.sim_now)
            if cur is None or dt > cur:
                self.sim_now = dt.isoformat().replace("+00:00", "Z")

    def clear(self) -> None:
        with self._lock:
            self._convs.clear()
            self._merchants.clear()
            self._sent_suppression.clear()
            self._skipped.clear()
            self.sim_now = None
