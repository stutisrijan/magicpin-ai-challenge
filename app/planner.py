"""Tick planner: decide which available triggers become proactive sends this tick.

Pipeline per tick (DESIGN.md section 7):

    resolve ids -> skip permanently (suppressed / unusable consent / unaddressable)
                -> defer (missing contexts, merchant opted out, caps; merchant-facing only:
                          auto-reply cooldown and explicit waits, unless urgent)
                -> rank (urgency, real payload over placeholder, oldest deferral, input order)
                -> compose selected triggers concurrently under one shared deadline
                   (precompose cache -> in-flight precompose -> LLM compose -> template)
                -> record state (suppression, merchant memory, conversation opener)

Deferred triggers are remembered per ConversationStore and retried on later ticks even
when the judge's `available_triggers` no longer lists them (bounded by MAX_DEFER_TICKS).
State is only recorded in one synchronous block at the very end, so a cancelled tick
never leaves half-recorded sends behind.
"""

from __future__ import annotations

import asyncio
import importlib
import itertools
import logging
import re
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from app import config
from app.schemas import CTA_OPEN_ENDED, CTA_VALUES, SEND_AS_MERCHANT, SEND_AS_VERA
from app.state import ContextStore, ConversationStore, parse_iso

log = logging.getLogger("vera.planner")

MAX_DEFER_TICKS = 6              # how many later ticks a deferred trigger is retried
FALLBACK_RESERVE_S = 1.0         # time kept back for the template fallback
FINALIZE_RESERVE_S = 0.3         # time kept back for bookkeeping before the deadline
MIN_LLM_BUDGET_S = 0.3           # below this, go straight to the deterministic template

# Customer-facing kinds that are reminders (blocked by preferences.reminder_opt_in == False).
REMINDER_KINDS = {"recall_due", "appointment_tomorrow", "chronic_refill_due", "appointment_reminder",
                  "refill_due", "vaccination_due", "renewal_reminder"}
# Consent scopes that explicitly allow reminders (these win over a False reminder_opt_in).
REMINDER_SCOPES = {"recall_reminders", "appointment_reminders", "refill_reminders", "renewal_reminders",
                   "recall_alerts"}
OPTED_OUT_STATES = {"opted_out", "unsubscribed", "dnd", "blocked", "do_not_contact"}

# Cached / precomposed compositions carry the date part of the `now` they were composed for
# (facts use it for durations and seasonal beats); a tick only reuses one composed for its own date.
COMPOSED_FOR_KEY = "_composed_for"


def _composer():
    """Resolved lazily so tests (and a half-built tree) can substitute the module."""
    return importlib.import_module("app.composer")


# --------------------------------------------------------------------------- memory


@dataclass
class _Deferred:
    first_tick: int
    reason: str


@dataclass
class PlannerMemory:
    tick_count: int = 0
    deferred: dict[str, _Deferred] = field(default_factory=dict)   # insertion-ordered trigger ids


_MEMORY: "weakref.WeakKeyDictionary[ConversationStore, PlannerMemory]" = weakref.WeakKeyDictionary()
_conv_counter = itertools.count(1)


def memory_for(convs: ConversationStore) -> PlannerMemory:
    mem = _MEMORY.get(convs)
    if mem is None:
        mem = PlannerMemory()
        _MEMORY[convs] = mem
    return mem


def reset(convs: ConversationStore | None = None) -> None:
    """Forget deferred triggers (for one store, or all). Conversation ids are never reused."""
    if convs is None:
        _MEMORY.clear()
    else:
        _MEMORY.pop(convs, None)


def deferred_ids(convs: ConversationStore) -> list[str]:
    return list(memory_for(convs).deferred)


# --------------------------------------------------------------------------- shared helpers (also used by precompose)


def suppression_key_for(trigger: dict, merchant_id: str | None) -> str:
    key = trigger.get("suppression_key")
    if isinstance(key, str) and key.strip():
        return key.strip()
    return f"{trigger.get('kind') or 'generic'}:{merchant_id or ''}:{trigger.get('id') or ''}"


def trigger_customer_id(trigger: dict) -> str | None:
    cid = trigger.get("customer_id") or (trigger.get("payload") or {}).get("customer_id")
    return str(cid) if cid else None


def is_customer_scoped(trigger: dict) -> bool:
    return trigger.get("scope") == "customer" or bool(trigger_customer_id(trigger))


def resolve_inputs(trigger: dict, contexts: ContextStore) -> tuple[str | None, dict | None, dict, str | None, dict | None]:
    """(merchant_id, merchant, category, customer_id, customer) for a trigger. Missing parts are None/{}."""
    payload = trigger.get("payload") if isinstance(trigger.get("payload"), dict) else {}
    customer_id = trigger_customer_id(trigger)
    customer = contexts.get("customer", customer_id) if customer_id else None
    merchant_id = trigger.get("merchant_id") or payload.get("merchant_id") or (customer or {}).get("merchant_id")
    merchant_id = str(merchant_id) if merchant_id else None
    merchant = contexts.get("merchant", merchant_id) if merchant_id else None
    category = (contexts.category_for(merchant, trigger) if merchant else None) or {}
    return merchant_id, merchant, category, customer_id, customer


def consent_problem(customer: dict | None, kind: str) -> str | None:
    """Why a customer must not be messaged proactively (None means consent is usable)."""
    if not isinstance(customer, dict):
        return "no_customer"
    consent = customer.get("consent") if isinstance(customer.get("consent"), dict) else {}
    prefs = customer.get("preferences") if isinstance(customer.get("preferences"), dict) else {}
    if not consent.get("opted_in_at"):
        return "no_consent"
    if consent.get("opted_out_at") or consent.get("revoked_at"):
        return "consent_revoked"
    scope = consent.get("scope")
    if not scope:
        return "no_consent_scope"
    if str(customer.get("state") or "").lower() in OPTED_OUT_STATES:
        return "customer_opted_out"
    channel = str(prefs.get("channel") or "").strip().lower()
    if channel and (channel.startswith("none") or channel in {"no_channel", "unknown"}):
        return "no_channel"
    scopes = {str(s).lower() for s in scope} if isinstance(scope, (list, tuple, set)) else {str(scope).lower()}
    if kind in REMINDER_KINDS and prefs.get("reminder_opt_in") is False and not (scopes & REMINDER_SCOPES):
        return "reminders_opted_out"
    return None


def now_date(now: str | None) -> str:
    """'2026-04-26T10:30:00Z' -> '2026-04-26'; '' when unknown."""
    dt = parse_iso(now)
    return dt.date().isoformat() if dt else ""


def composed_for_ok(comp: Any, now: str | None) -> bool:
    """False when a cached composition was made for a different date than this tick's `now`
    (untagged compositions, and ticks without a usable `now`, always pass)."""
    tick_date = now_date(now)
    if not tick_date or not isinstance(comp, dict) or COMPOSED_FOR_KEY not in comp:
        return True
    return comp.get(COMPOSED_FOR_KEY) == tick_date


def trigger_urgency(trigger: dict) -> int:
    try:
        return int(float(trigger.get("urgency") or 1))
    except (TypeError, ValueError):
        return 1


def _is_placeholder(trigger: dict) -> bool:
    payload = trigger.get("payload")
    return not isinstance(payload, dict) or bool(payload.get("placeholder")) or not payload


def _short_id(raw: str | None, limit: int = 20) -> str:
    """'m_001_drmeera_dentist_delhi' -> 'm001_drmeera'; 'c_075_aditya_for_m_019...' -> 'c075_aditya'."""
    if not raw:
        return ""
    parts = [p for p in re.split(r"[^a-z0-9]+", str(raw).lower()) if p]
    if len(parts) >= 3 and len(parts[0]) == 1 and parts[1].isdigit():
        return f"{parts[0]}{parts[1]}_{parts[2]}"[:limit]
    return "_".join(parts)[:limit].strip("_")


def new_conversation_id(convs: ConversationStore, merchant_id: str | None, kind: str,
                        customer_id: str | None = None) -> str:
    """Readable, unique, never reused within the process: conv_<merchant>[_<customer>]_<kind>_<n>."""
    kind_slug = "_".join(p for p in re.split(r"[^a-z0-9]+", (kind or "msg").lower()) if p)[:24] or "msg"
    parts = ["conv", _short_id(merchant_id) or "merchant"]
    if customer_id:
        parts.append(_short_id(customer_id))
    parts.append(kind_slug.strip("_"))
    base = "_".join(p for p in parts if p)
    while True:
        cid = f"{base}_{next(_conv_counter)}"
        if not convs.exists(cid):
            return cid


# --------------------------------------------------------------------------- candidates


@dataclass
class Candidate:
    trigger_id: str
    trigger: dict
    kind: str
    urgency: int
    placeholder: bool
    order: int
    age_tick: int                    # tick the trigger was first deferred (older first at equal rank)
    merchant_id: str
    merchant: dict
    category: dict
    customer_id: str | None
    customer: dict | None
    suppression_key: str

    @property
    def customer_facing(self) -> bool:
        return self.customer is not None


def _merchant_waiting(convs: ConversationStore, merchant_id: str, now_dt: datetime) -> bool:
    """True when a merchant-facing conversation is in an explicit, unexpired wait."""
    for conv in convs.for_merchant(merchant_id):
        if conv.customer_id or conv.status != "waiting":
            continue
        until = parse_iso(conv.wait_until)
        if until is not None and until > now_dt:
            return True
    return False


def _evaluate(trigger_id: str, order: int, contexts: ContextStore, convs: ConversationStore,
              now_dt: datetime, age_tick: int = 0) -> tuple[str, Candidate | None, str]:
    """Classify one trigger id as ("ok" | "skip" | "defer" | "ignore", candidate, reason)."""
    trigger = contexts.get("trigger", trigger_id)
    if not isinstance(trigger, dict):
        return "ignore", None, "unknown_trigger"
    kind = str(trigger.get("kind") or "generic")
    merchant_id, merchant, category, customer_id, customer = resolve_inputs(trigger, contexts)
    if not merchant_id:
        return "skip", None, "no_merchant_id"
    if not isinstance(merchant, dict):
        return "defer", None, "merchant_context_missing"
    skey = suppression_key_for(trigger, merchant_id)
    if convs.suppression_used(skey):
        return "skip", None, "suppressed"
    if is_customer_scoped(trigger):
        if not customer_id:
            return "skip", None, "no_customer_id"
        if not isinstance(customer, dict):
            return "defer", None, "customer_context_missing"
        problem = consent_problem(customer, kind)
        if problem:
            return "skip", None, problem
    urgency = trigger_urgency(trigger)
    mstate = convs.merchant(merchant_id)
    if mstate.opted_out:
        return "defer", None, "merchant_opted_out"
    if not is_customer_scoped(trigger):
        customer_id, customer = None, None
        # Auto-reply cooldowns and explicit waits are about nudging the merchant; customer-facing
        # sends (recalls, reminders) go to the customer and are not held back by them.
        if urgency < config.settings.urgent_bypass_urgency:
            cooldown = parse_iso(mstate.cooldown_until)
            if cooldown is not None and cooldown > now_dt:
                return "defer", None, "merchant_cooldown"
            if _merchant_waiting(convs, merchant_id, now_dt):
                return "defer", None, "merchant_asked_to_wait"
    cand = Candidate(
        trigger_id=trigger_id, trigger=trigger, kind=kind, urgency=urgency,
        placeholder=_is_placeholder(trigger), order=order, age_tick=age_tick, merchant_id=merchant_id,
        merchant=merchant, category=category, customer_id=customer_id, customer=customer,
        suppression_key=skey,
    )
    return "ok", cand, ""


def _select(cands: list[Candidate], cap: int) -> tuple[list[Candidate], list[Candidate]]:
    """At most one merchant-facing action per merchant and one per customer; global cap."""
    ranked = sorted(cands, key=lambda c: (-c.urgency, c.placeholder, c.age_tick, c.order))
    chosen: list[Candidate] = []
    rest: list[Candidate] = []
    merchants: set[str] = set()
    customers: set[str] = set()
    keys: set[str] = set()
    for c in ranked:
        if len(chosen) >= cap or c.suppression_key in keys:
            rest.append(c)
            continue
        if c.customer_facing:
            if c.customer_id in customers:
                rest.append(c)
                continue
            customers.add(c.customer_id or "")
        else:
            if c.merchant_id in merchants:
                rest.append(c)
                continue
            merchants.add(c.merchant_id)
        keys.add(c.suppression_key)
        chosen.append(c)
    return chosen, rest


# --------------------------------------------------------------------------- composition


def _usable(comp: Any, prior_bodies: list[str]) -> bool:
    if not isinstance(comp, dict):
        return False
    body = comp.get("body")
    return isinstance(body, str) and bool(body.strip()) and body.strip() not in prior_bodies


async def _compose_one(c: Candidate, *, now: str | None, prior_bodies: list[str], cache: Any,
                       inflight: Mapping[str, "asyncio.Future"] | None, deadline_at: float,
                       fallback_reserve: float, finalize_reserve: float) -> dict | None:
    """Cache -> in-flight precompose -> compose (LLM allowed) -> template-only fallback."""
    loop = asyncio.get_running_loop()
    composer = _composer()
    args = (c.category, c.merchant, c.trigger, c.customer)

    fp = None
    try:
        fp = composer.input_fingerprint(*args)
    except Exception:
        log.debug("fingerprint failed for %s", c.trigger_id, exc_info=True)
    if fp and cache is not None:
        try:
            cached = cache.get(fp)
        except Exception:
            cached = None
        if _usable(cached, prior_bodies) and composed_for_ok(cached, now):
            return cached

    def llm_budget() -> float:
        return deadline_at - fallback_reserve - finalize_reserve - loop.time()

    if fp and inflight:
        task = inflight.get(fp)
        if task is not None and llm_budget() > 0.1:
            try:
                res = await asyncio.wait_for(asyncio.shield(task), timeout=llm_budget())
                if _usable(res, prior_bodies) and composed_for_ok(res, now):
                    return res
            except Exception:
                pass

    budget = llm_budget()
    if budget >= MIN_LLM_BUDGET_S:
        try:
            comp = await asyncio.wait_for(
                composer.compose_async(*args, now=now, prior_bodies=list(prior_bodies), timeout=budget),
                timeout=budget + 0.5 * fallback_reserve,
            )
            if _usable(comp, prior_bodies):
                if fp and cache is not None and (comp.get("meta") or {}).get("source") == "llm":
                    try:
                        cache.put(fp, {**comp, COMPOSED_FOR_KEY: now_date(now)})
                    except Exception:
                        pass
                return comp
        except Exception as exc:
            log.warning("compose failed for %s (%s); using template", c.trigger_id, type(exc).__name__)

    remaining = deadline_at - finalize_reserve - loop.time()
    if remaining <= 0.05:
        return None
    try:
        comp = await asyncio.wait_for(
            composer.compose_async(*args, now=now, prior_bodies=list(prior_bodies), use_llm=False,
                                   timeout=max(0.05, remaining - 0.05)),
            timeout=remaining,
        )
        return comp if _usable(comp, []) else None       # a duplicate body is caught (and skipped) by the caller
    except Exception as exc:
        log.warning("template compose failed for %s (%s)", c.trigger_id, type(exc).__name__)
        return None


def _finalize(c: Candidate, comp: dict) -> dict:
    """Normalise a composition into a complete tick action (conversation_id filled later)."""
    send_as = SEND_AS_MERCHANT if c.customer_facing else SEND_AS_VERA
    cta = comp.get("cta") if comp.get("cta") in CTA_VALUES else CTA_OPEN_ENDED
    body = comp["body"].strip()
    template_name = comp.get("template_name")
    if not isinstance(template_name, str) or not template_name.strip():
        template_name = f"{'merchant' if c.customer_facing else 'vera'}_{c.kind}_v1"
    params = comp.get("template_params")
    if not isinstance(params, (list, tuple)) or not params:
        params = [body]
    rationale = comp.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        rationale = f"{c.kind.replace('_', ' ')} for this merchant; message anchored on their own data with one clear ask."
    skey = comp.get("suppression_key")
    return {
        "conversation_id": "",
        "merchant_id": c.merchant_id,
        "customer_id": c.customer_id,
        "send_as": send_as,
        "trigger_id": c.trigger_id,
        "template_name": template_name.strip(),
        "template_params": [str(p) for p in params],
        "body": body,
        "cta": cta,
        "suppression_key": skey.strip() if isinstance(skey, str) and skey.strip() else c.suppression_key,
        "rationale": rationale.strip(),
    }


# --------------------------------------------------------------------------- tick


def _as_ids(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if isinstance(v, (str, int)) and not isinstance(v, bool):
            s = str(v).strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
    return out


async def plan_tick(now: str | None, available_triggers: list[str], *, contexts: ContextStore,
                    convs: ConversationStore, engine: Any, deadline_s: float | None = None,
                    cache: Any = None, inflight: Mapping[str, "asyncio.Future"] | None = None) -> list[dict]:
    """Return this tick's actions (possibly empty). Never raises for per-trigger problems.

    `cache` (ComposeCache-like get/put) and `inflight` (fingerprint -> precompose task) are
    optional hooks from the precompose layer.
    """
    loop = asyncio.get_running_loop()
    deadline = float(deadline_s if deadline_s is not None else config.settings.tick_deadline_s)
    deadline_at = loop.time() + deadline
    fallback_reserve = min(FALLBACK_RESERVE_S, deadline * 0.15)
    finalize_reserve = min(FINALIZE_RESERVE_S, deadline * 0.05)

    convs.observe_time(now)
    now_iso = now if parse_iso(now) else convs.sim_now
    now_dt = parse_iso(now_iso) or datetime.now(timezone.utc)

    mem = memory_for(convs)
    mem.tick_count += 1
    listed = _as_ids(available_triggers)
    listed_set = set(listed)
    carried = [tid for tid in mem.deferred if tid not in listed_set]
    candidates: list[Candidate] = []
    deferred: dict[str, str] = {}
    for order, tid in enumerate(listed + carried):
        entry = mem.deferred.get(tid)
        age = entry.first_tick if entry is not None else mem.tick_count
        status, cand, reason = _evaluate(tid, order, contexts, convs, now_dt, age)
        if status == "ok" and cand is not None:
            candidates.append(cand)
        elif status == "defer":
            deferred[tid] = reason
        elif status == "skip":
            convs.mark_skipped(tid, reason)
            mem.deferred.pop(tid, None)
            log.info("skip %s: %s", tid, reason)
        else:
            mem.deferred.pop(tid, None)

    chosen, capped = _select(candidates, max(0, int(config.settings.max_actions_per_tick)))
    for c in capped:
        deferred[c.trigger_id] = "tick_cap"

    tasks = []
    priors: list[list[str]] = []
    for c in chosen:
        prior = list(convs.merchant(c.merchant_id).sent_bodies)
        priors.append(prior)
        tasks.append(asyncio.ensure_future(_compose_one(
            c, now=now_iso, prior_bodies=prior, cache=cache, inflight=inflight, deadline_at=deadline_at,
            fallback_reserve=fallback_reserve, finalize_reserve=finalize_reserve)))
    results: list[dict | None] = [None] * len(tasks)
    if tasks:
        _done, pending = await asyncio.wait(tasks, timeout=max(0.05, deadline_at - finalize_reserve - loop.time()))
        for t in pending:
            t.cancel()
        for i, t in enumerate(tasks):
            if t.done() and not t.cancelled() and t.exception() is None:
                results[i] = t.result()

    # ---- record state: one synchronous block (no awaits) so cancellation can't split it
    actions: list[dict] = []
    for c, comp, prior in zip(chosen, results, priors):
        if not isinstance(comp, dict) or not _usable(comp, []):
            deferred[c.trigger_id] = "compose_unavailable"
            continue
        comp = {k: v for k, v in comp.items() if k != COMPOSED_FOR_KEY}
        mstate = convs.merchant(c.merchant_id)
        body = comp["body"].strip()
        if body in prior:                       # the composer knew this body and still repeated it
            convs.mark_skipped(c.trigger_id, "duplicate_body")
            mem.deferred.pop(c.trigger_id, None)
            continue
        if body in mstate.sent_bodies:          # collided with a send earlier in this tick: retry next tick
            deferred[c.trigger_id] = "duplicate_in_tick"
            continue
        action = _finalize(c, comp)
        action["conversation_id"] = new_conversation_id(convs, c.merchant_id, c.kind, c.customer_id)
        for key in {c.suppression_key, action["suppression_key"]}:
            convs.mark_suppression(key, action["conversation_id"])
        if not c.customer_facing:
            mstate.last_sent_at = now_iso
        mstate.sent_bodies.append(body)
        _open_conversation(engine, convs, action, {**comp, **{k: action[k] for k in ("body", "cta", "send_as")}},
                           now_iso)
        mem.deferred.pop(c.trigger_id, None)
        actions.append(action)

    for tid, reason in deferred.items():
        entry = mem.deferred.get(tid)
        if entry is None:
            mem.deferred[tid] = _Deferred(first_tick=mem.tick_count, reason=reason)
        elif mem.tick_count - entry.first_tick >= MAX_DEFER_TICKS:
            mem.deferred.pop(tid, None)
            log.info("drop deferred %s after %d ticks (%s)", tid, MAX_DEFER_TICKS, reason)
        else:
            entry.reason = reason
    if actions or deferred:
        log.info("tick: %d action(s), %d deferred, %d candidate(s)", len(actions), len(deferred), len(candidates))
    return actions


def _open_conversation(engine: Any, convs: ConversationStore, action: dict, composition: dict,
                       now: str | None) -> None:
    """Register the opener with the engine; guarantee the conversation id exists either way."""
    if engine is not None:
        try:
            engine.open_from_action(action, composition, now)
        except Exception:
            log.exception("open_from_action failed for %s", action["conversation_id"])
    if not convs.exists(action["conversation_id"]):
        meta = composition.get("meta") if isinstance(composition.get("meta"), dict) else {}
        convs.create(
            action["conversation_id"], merchant_id=action["merchant_id"], customer_id=action["customer_id"],
            trigger_id=action["trigger_id"], kind=str(meta.get("kind") or ""), send_as=action["send_as"],
            offer=str(meta.get("offer") or ""), suppression_key=action["suppression_key"],
            language=str(meta.get("language") or ""),
            facts_digest=[str(f) for f in (meta.get("facts") or [])][:8],
        )
        convs.add_turn(action["conversation_id"], "bot", action["body"], ts=now)


def plan_tick_sync(now: str | None, available_triggers: list[str], **kwargs: Any) -> list[dict]:
    """Blocking wrapper for scripts and tests (must not be called from a running event loop)."""
    return asyncio.run(plan_tick(now, available_triggers, **kwargs))
