"""Composition pipeline: one proactive message from the four contexts.

    facts = facts.build_facts(...)                      deterministic, verified in code
    pb    = playbooks.get_playbook(kind, scope, cat)    what the message must achieve
    draft = templates.render(facts, pb)                 deterministic, always computed
    if an LLM is available and time allows:
        candidate = LLM(prompts.compose_prompt(facts, pb, draft))
        validator.validate_body(candidate) == []  -> use it
        else one repair call listing the issues   -> use it if clean
    otherwise the draft

compose_async() never raises (except cancellation) and always returns the full output
contract (spec section 3). Results are cached in-process by an input fingerprint, so
identical inputs return identical output. compose() is a sync wrapper that works both
inside and outside a running event loop.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import hashlib
import importlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from app import config
from app.facts import build_facts
from app.numbers import extract_number_tokens, normalize_number
from app.prompts import COMPOSE_SCHEMA, compose_prompt, repair_prompt
from app.schemas import (
    CTA_BINARY_YES_NO,
    CTA_OPEN_ENDED,
    CTA_VALUES,
    SEND_AS_MERCHANT,
    SEND_AS_VERA,
    Draft,
    Fact,
    FactSheet,
    Playbook,
)
from app.validator import normalize_for_compare, validate_body

log = logging.getLogger(__name__)

__all__ = ["compose_async", "compose", "input_fingerprint", "ComposeCache", "clear_cache"]

# --------------------------------------------------------------------------- budgets (seconds)

MIN_LLM_BUDGET_S = 1.5          # do not start an LLM call with less time than this
REPAIR_MIN_S = 2.5              # a repair call needs at least this much time left
SAFETY_S = 0.3                  # kept back from every LLM call for validation + normalisation
REPAIR_RESERVE_S = 3.0          # added to llm_timeout_s when compose_async gets no explicit timeout
SYNC_TIMEOUT_S = 20.0           # compose() default: the offline contract allows < 30s per call
LLM_MAX_TOKENS = 900

_TEMPLATE_PARAM_LIMIT = 6
_META_FACTS_LIMIT = 8


# --------------------------------------------------------------------------- cache


class ComposeCache:
    """Thread-safe LRU of compositions keyed by fingerprint. get() returns a copy."""

    def __init__(self, max_entries: int = 1024) -> None:
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.RLock()
        self._data: OrderedDict[str, dict] = OrderedDict()

    def get(self, fp: str) -> dict | None:
        with self._lock:
            item = self._data.get(fp)
            if item is None:
                return None
            self._data.move_to_end(fp)
            return copy.deepcopy(item)

    def put(self, fp: str, comp: dict) -> None:
        if not fp or not isinstance(comp, dict):
            return
        with self._lock:
            self._data[fp] = copy.deepcopy(comp)
            self._data.move_to_end(fp)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, fp: object) -> bool:
        with self._lock:
            return fp in self._data


_CACHE = ComposeCache(max_entries=512)


def clear_cache() -> None:
    """Forget cached compositions (tests, /v1/teardown)."""
    _CACHE.clear()


def input_fingerprint(category: Any, merchant: Any, trigger: Any, customer: Any = None, *,
                      now: str | None = None, prior_bodies: Iterable[str] = ()) -> str:
    """sha256 over the canonical JSON of the four contexts (plus, optionally, now's date and prior bodies)."""
    parts: list[Any] = [category, merchant, trigger, customer]
    prior = [p for p in (prior_bodies or ()) if isinstance(p, str)]
    if now or prior:
        parts += [_date_part(now), prior]
    try:
        blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    except (TypeError, ValueError):          # mixed-type keys or cycles: still stable within the process
        blob = repr(parts)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _date_part(now: str | None) -> str:
    s = str(now or "").strip()
    return s[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", s) else s


# --------------------------------------------------------------------------- public API


async def compose_async(
    category: dict | None,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
    *,
    now: str | None = None,
    prior_bodies: Iterable[str] = (),
    use_llm: bool | None = True,
    timeout: float | None = None,
) -> dict:
    """One composed message (spec section 3). use_llm False = template only; True/None = LLM if available."""
    started = time.monotonic()
    budget = float(timeout) if timeout is not None else float(config.settings.llm_timeout_s) + REPAIR_RESERVE_S
    deadline = started + max(0.0, budget)
    prior = _clean_prior(prior_bodies)
    try:
        key = input_fingerprint(category, merchant, trigger, customer, now=now, prior_bodies=prior)
        llm = _llm_client() if use_llm is not False else None
        cached = _CACHE.get(key)
        if cached is not None and (_source(cached) == "llm" or llm is None):
            return cached

        prep = _prepare(category, merchant, trigger, customer, now, prior)
        result = None
        if llm is not None:
            result = await _refine_with_llm(llm, prep, deadline)
        if result is None:
            result = cached if cached is not None else _template_result(prep)
        _CACHE.put(key, result)
        return copy.deepcopy(result)
    except asyncio.CancelledError:
        raise
    except Exception:
        log.exception("compose_async failed; using safe composition")
        return _safe_composition(category, merchant, trigger, customer, now, prior)


def compose(
    category: dict | None,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
    *,
    now: str | None = None,
    use_llm: bool | None = None,
    prior_bodies: Iterable[str] = (),
    timeout: float | None = None,
) -> dict:
    """Sync wrapper around compose_async; safe inside or outside a running event loop. Never raises."""
    budget = SYNC_TIMEOUT_S if timeout is None else max(0.0, float(timeout))
    prior = _clean_prior(prior_bodies)

    def run() -> dict:
        return asyncio.run(compose_async(category, merchant, trigger, customer, now=now, prior_bodies=prior,
                                         use_llm=use_llm, timeout=budget))

    try:
        if not _loop_running():
            return run()
        # A loop is already running in this thread: run ours in a worker thread instead.
        future = _executor().submit(run)
        return future.result(timeout=budget + 5.0)
    except Exception:
        log.exception("compose failed; using safe composition")
        return _safe_composition(category, merchant, trigger, customer, now, prior)


def _loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()


def _executor() -> concurrent.futures.ThreadPoolExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="compose")
        return _EXECUTOR


# --------------------------------------------------------------------------- preparation


@dataclass
class _Prep:
    facts: FactSheet
    pb: Playbook
    draft: Draft
    trigger: dict
    merchant: dict
    prior: list[str]
    customer_facing: bool
    draft_issues: list[str] = field(default_factory=list)
    llm_attempts: int = 0                                   # LLM calls that returned a usable body
    llm_rejected: list[str] = field(default_factory=list)   # validator issues of rejected LLM bodies


@dataclass
class _Candidate:
    body: str
    cta: str = ""
    rationale: str = ""
    params: list[str] = field(default_factory=list)
    used: list[str] = field(default_factory=list)


def _prepare(category: Any, merchant: Any, trigger: Any, customer: Any, now: str | None,
             prior: list[str]) -> _Prep:
    m = merchant if isinstance(merchant, dict) else {}
    t = trigger if isinstance(trigger, dict) else {}
    facts = build_facts(category if isinstance(category, dict) else None, m, t,
                        customer if isinstance(customer, dict) else None, now=now)
    pb = _get_playbook(facts)
    draft = _render(facts, pb)
    draft.body = _clean_body(draft.body)
    customer_facing = facts.send_as == SEND_AS_MERCHANT
    prep = _Prep(facts=facts, pb=pb, draft=draft, trigger=t, merchant=m, prior=prior,
                 customer_facing=customer_facing)
    prep.draft_issues = validate_body(draft.body, facts, prior_bodies=prior, customer_facing=customer_facing)
    if prep.draft_issues:
        log.info("template draft for %s has issues: %s", facts.trigger_id, prep.draft_issues)
    return prep


_MISSING_WARNED: set[str] = set()


def _module(name: str) -> Any:
    """Import a sibling module at call time (it may be absent in partial builds or replaced in tests)."""
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        if name not in _MISSING_WARNED:
            _MISSING_WARNED.add(name)
            log.warning("%s is not available; using the built-in fallback", name)
        return None


def _get_playbook(facts: FactSheet) -> Playbook:
    try:
        mod = _module("app.playbooks")
        pb = mod.get_playbook(facts.kind, facts.scope, facts.category_slug) if mod else None
        if isinstance(pb, Playbook):
            return pb
    except Exception:
        log.exception("get_playbook failed for kind %r", facts.kind)
    return _fallback_playbook(facts)


def _render(facts: FactSheet, pb: Playbook) -> Draft:
    try:
        mod = _module("app.templates")
        draft = mod.render(facts, pb) if mod else None
        if isinstance(draft, Draft) and isinstance(draft.body, str) and draft.body.strip():
            return draft
        if mod:
            log.warning("templates.render returned no body for %s", facts.trigger_id)
    except Exception:
        log.exception("templates.render failed for kind %r", facts.kind)
    return _generic_draft(facts, pb)


def _llm_client() -> Any:
    """The LLM client when one is configured and available, else None."""
    try:
        mod = _module("app.llm")
        client = mod.get_llm() if mod else None
        return client if client is not None and client.available() else None
    except Exception:
        log.debug("LLM client unavailable", exc_info=True)
        return None


# --------------------------------------------------------------------------- LLM refinement


async def _refine_with_llm(llm: Any, prep: _Prep, deadline: float) -> dict | None:
    """A validated LLM composition, or None to fall back to the template draft."""
    def remaining() -> float:
        return deadline - time.monotonic()

    if remaining() < MIN_LLM_BUDGET_S:
        return None
    facts, pb, draft = prep.facts, prep.pb, prep.draft
    system, user = compose_prompt(facts, pb, draft, prior_bodies=prep.prior)
    first_budget = min(remaining() - SAFETY_S, max(1.0, float(config.settings.llm_timeout_s)))
    cand = await _call_llm(llm, system, user, first_budget)
    if cand is None:
        return None
    prep.llm_attempts = 1
    issues = _check(cand.body, prep)
    if not issues:
        return _llm_result(prep, cand, attempts=1)

    log.info("LLM draft for %s rejected: %s", facts.trigger_id, issues)
    prep.llm_rejected = list(issues)
    if remaining() <= REPAIR_MIN_S:
        return None
    system2, user2 = repair_prompt(facts, pb, draft, cand.body, issues, prior_bodies=prep.prior)
    fixed = await _call_llm(llm, system2, user2, remaining() - SAFETY_S)
    if fixed is None:
        return None
    prep.llm_attempts = 2
    issues2 = _check(fixed.body, prep)
    if issues2:
        log.info("LLM repair for %s rejected: %s", facts.trigger_id, issues2)
        prep.llm_rejected += [i for i in issues2 if i not in prep.llm_rejected]
        return None
    return _llm_result(prep, fixed, attempts=2, rejected=issues)


def _check(body: str, prep: _Prep) -> list[str]:
    """Validator issues plus composer-only guards that keep an LLM rewrite at least as good as the draft."""
    issues = validate_body(body, prep.facts, prior_bodies=prep.prior, mode="compose",
                           customer_facing=prep.customer_facing)
    issues += _regression_issues(body, prep)
    return issues


_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
_DATE_RE = re.compile(rf"\b\d{{1,2}}\s{_MONTHS}(?![a-z])|\b{_MONTHS}\s\d{{1,2}}\b")
_CLOCK_12_RE = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s?(am|pm)\b", re.IGNORECASE)
_CLOCK_24_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b(?!\s?(?:am|pm))", re.IGNORECASE)
# A closing sentence that asks for something (English + Hinglish): "...?", "Reply YES", "CONFIRM likhiye", "batayein".
_ASK_RE = re.compile(
    r"\?\s*\W*$|\b(?:reply|type|say|likh\w*|bhej\w* (?:do|dijiye)|batayein|bata dijiye|bataiye|confirm|"
    r"let me know|tell us)\b", re.IGNORECASE)
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]{3,}")


def _regression_issues(body: str, prep: _Prep) -> list[str]:
    """Ways an otherwise valid LLM body can be worse than the draft: vaguer, uncited, invented times, wrong language."""
    facts, draft = prep.facts, prep.draft.body
    issues: list[str] = []
    norm_body = normalize_for_compare(body)

    # research / compliance: the source citation must survive the rewrite
    source = next((f.text for f in facts.anchor_facts if f.key == "digest.source" and f.text.strip()), "")
    if source and normalize_for_compare(source) in normalize_for_compare(draft) \
            and normalize_for_compare(source) not in norm_body:
        issues.append(f"missing_source: cite the source exactly as given: '{source}'")

    # specificity: the draft had a concrete fact, so the rewrite must keep at least one
    if _has_specifics(draft) and not _has_specifics(body):
        issues.append("low_specificity: include at least one concrete fact from the facts "
                      "(a number, date, price or source), copied exactly")

    # clock times are not number-checked by the validator; they must still come from the data
    allowed = _clock_set(_grounding_text(prep))
    invented = sorted(_clock_set(body) - allowed)
    if invented:
        issues.append(f"ungrounded_time: {', '.join(invented[:4])} is not in the facts; use only the given "
                      f"times or slots, or ask what time works")

    # the single ask closes the message (the draft always does; a rewrite that trails off is weaker)
    if prep.pb.cta != "none" and _ends_with_ask(draft) and not _ends_with_ask(body):
        issues.append("cta_not_last: end the message with the single ask (a question or 'Reply YES ...')")

    # language plan: if the draft reads in the planned language, the rewrite must too
    expected = _expected_languages(facts)
    if expected and _detect_language(draft) in expected and _detect_language(body) not in expected:
        issues.append(f"language: write it in the planned language ({facts.language.code}): "
                      f"{facts.language.instruction}")
    if facts.language.script != "devanagari" and _DEVANAGARI_RE.search(body):
        issues.append("script: write in Roman script only (no Devanagari)")
    return issues


def _ends_with_ask(text: str) -> bool:
    sentences = _split_sentences(text or "")
    return bool(sentences) and bool(_ASK_RE.search(sentences[-1]))


def _has_specifics(text: str) -> bool:
    return bool(_significant_numbers(text) or _DATE_RE.search(text or "") or re.search(r"₹\s?\d|\d\s?%", text or ""))


def _clock_set(text: str) -> set[str]:
    """Clock times in canonical 12-hour form: "6 PM", "6:00pm" and "18:00" all become "6pm"."""
    out: set[str] = set()
    for m in _CLOCK_12_RE.finditer(text or ""):
        hour, minute, half = int(m.group(1)), m.group(2), m.group(3).lower()
        out.add(f"{hour}{':' + minute if minute and minute != '00' else ''}{half}")
    for m in _CLOCK_24_RE.finditer(text or ""):
        hour, minute = int(m.group(1)), m.group(2)
        half = "am" if hour < 12 else "pm"
        hour12 = hour % 12 or 12
        out.add(f"{hour12}{':' + minute if minute != '00' else ''}{half}")
    return out


def _grounding_text(prep: _Prep) -> str:
    """Every text the message may take times from: facts, slots, offers, notes, history and the draft."""
    facts = prep.facts
    parts = [f.text for f in facts.all_facts()] + list(facts.slots) + list(facts.offers_active)
    parts += list(facts.catalog_offers) + list(facts.notes) + [prep.draft.body, facts.seasonal_note, facts.trend_note]
    for turn in facts.history or []:
        if isinstance(turn, dict):
            parts.append(str(turn.get("body") or turn.get("message") or ""))
    for item in (facts.digest_item, facts.content_item):
        if isinstance(item, dict):
            parts.append(json.dumps(item, ensure_ascii=False, default=str))
    return "\n".join(str(p or "") for p in parts)


def _expected_languages(facts: FactSheet) -> set[str] | None:
    code = facts.language.code if facts.language else "en"
    if facts.language and facts.language.script == "devanagari":
        return None                     # templates stay romanised; do not second-guess a script switch
    if code in ("hinglish", "hi"):
        return {"hinglish", "hi"}
    if code == "en":
        return {"en"}
    return None                         # regional mixes (ta-en, ...) are English plus a greeting


def _detect_language(text: str) -> str:
    try:
        return importlib.import_module("app.language").detect_message_language(text)
    except Exception:
        return ""


async def _call_llm(llm: Any, system: str, user: str, budget: float) -> _Candidate | None:
    if budget < 1.0:
        return None
    try:
        raw = await asyncio.wait_for(
            llm.complete_json(system, user, schema=COMPOSE_SCHEMA, timeout=budget, max_tokens=LLM_MAX_TOKENS,
                              temperature=0.0),
            timeout=budget + 0.2,
        )
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        log.info("LLM call timed out after %.1fs", budget)
        return None
    except Exception:
        log.warning("LLM call failed", exc_info=True)
        return None
    return _parse_candidate(raw)


def _parse_candidate(raw: Any) -> _Candidate | None:
    data = raw
    if isinstance(raw, str):
        try:
            data = importlib.import_module("app.llm").extract_json(raw)
        except Exception:
            data = None
    if not isinstance(data, dict):
        return None
    body = data.get("body")
    if not isinstance(body, str):
        return None
    body = _clean_body(body)
    if not body:
        return None
    cta = data.get("cta") if isinstance(data.get("cta"), str) else ""
    rationale = data.get("rationale") if isinstance(data.get("rationale"), str) else ""
    params = [p.strip() for p in data.get("template_params") or [] if isinstance(p, str) and p.strip()] \
        if isinstance(data.get("template_params"), list) else []
    used = [u.strip() for u in data.get("used_facts") or [] if isinstance(u, str) and u.strip()] \
        if isinstance(data.get("used_facts"), list) else []
    return _Candidate(body=body, cta=cta.strip(), rationale=rationale.strip(), params=params, used=used)


# --------------------------------------------------------------------------- results


def _llm_result(prep: _Prep, cand: _Candidate, *, attempts: int, rejected: list[str] | None = None) -> dict:
    params = cand.params if _params_match(cand.params, cand.body) else []
    return _finalize(prep, body=cand.body, cta=cand.cta, rationale=cand.rationale, params=params,
                     used_keys=cand.used, source="llm",
                     extra_meta={"llm_attempts": attempts, "rejected_issues": list(rejected or [])})


def _template_result(prep: _Prep) -> dict:
    draft = prep.draft
    body = draft.body
    params = list(draft.template_params)
    issues = list(prep.draft_issues)
    if _is_repeat(body, prep.prior):
        body = _vary(body, prep)
        params = []                                   # the old params no longer match the varied body
        issues = validate_body(body, prep.facts, prior_bodies=prep.prior, customer_facing=prep.customer_facing)
    extra: dict[str, Any] = {"draft_issues": issues}
    if prep.llm_attempts:
        extra.update({"llm_attempts": prep.llm_attempts, "rejected_issues": list(prep.llm_rejected)})
    return _finalize(prep, body=body, cta=draft.cta, rationale=draft.rationale, params=params, used_keys=[],
                     source="template", extra_meta=extra)


def _finalize(prep: _Prep, *, body: str, cta: str, rationale: str, params: list[str], used_keys: list[str],
              source: str, extra_meta: dict | None = None) -> dict:
    facts, pb, draft = prep.facts, prep.pb, prep.draft
    body = body.strip() or draft.body.strip() or _generic_draft(facts, pb).body
    send_as = facts.send_as if facts.send_as in (SEND_AS_VERA, SEND_AS_MERCHANT) else SEND_AS_VERA
    if cta not in CTA_VALUES:
        cta = pb.cta if pb.cta in CTA_VALUES else CTA_OPEN_ENDED
    used_facts = _resolve_facts(facts, used_keys, body)
    meta = {
        "offer": pb.offer or draft.offer or "",
        "kind": facts.kind or str(prep.trigger.get("kind") or ""),
        "source": source,
        "language": facts.language.code if facts.language else "en",
        "facts": [f.text for f in used_facts][:_META_FACTS_LIMIT],
        "fact_keys": _uniq(f.key for f in used_facts)[:_META_FACTS_LIMIT],
        "trigger_id": facts.trigger_id or str(prep.trigger.get("id") or ""),
        "merchant_id": facts.merchant_id or str(prep.merchant.get("merchant_id") or ""),
        "customer_id": facts.customer_id,
        "consent_ok": bool(facts.consent_ok),
    }
    meta.update(extra_meta or {})
    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": _suppression_key(prep.trigger, prep.merchant, facts),
        "rationale": _rationale(rationale, draft, facts, pb, used_facts, cta),
        "template_name": _template_name(pb, facts, send_as),
        "template_params": _template_params(params, body, facts),
        "meta": meta,
    }


def _suppression_key(trigger: dict, merchant: dict, facts: FactSheet) -> str:
    key = trigger.get("suppression_key")
    if isinstance(key, str) and key.strip():
        return key.strip()
    payload = trigger.get("payload") if isinstance(trigger.get("payload"), dict) else {}
    merchant_id = (trigger.get("merchant_id") or payload.get("merchant_id") or merchant.get("merchant_id")
                   or facts.merchant_id or "")
    kind = trigger.get("kind") or facts.kind or "generic"
    trigger_id = trigger.get("id") or facts.trigger_id or ""
    return f"{kind}:{merchant_id}:{trigger_id}"


def _template_name(pb: Playbook, facts: FactSheet, send_as: str) -> str:
    name = (pb.template_name or "").strip() or f"vera_{facts.kind or 'generic'}_v1"
    if send_as == SEND_AS_MERCHANT and not name.startswith("merchant_"):
        name = "merchant_" + name.removeprefix("vera_")
    elif send_as == SEND_AS_VERA and name.startswith("merchant_"):
        name = "vera_" + name.removeprefix("merchant_")
    return name


def _template_params(params: list[str], body: str, facts: FactSheet) -> list[str]:
    clean = [str(p).strip() for p in params or [] if isinstance(p, str) and str(p).strip()]
    if clean:
        return clean[:_TEMPLATE_PARAM_LIMIT]
    return _params_from_body(body, facts)


def _params_match(params: list[str], body: str) -> bool:
    """LLM template params are kept only when every piece really appears in the body."""
    if not params:
        return False
    norm_body = normalize_for_compare(body)
    return all(normalize_for_compare(p) and normalize_for_compare(p) in norm_body for p in params)


_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "vs", "p", "no", "st", "e.g", "i.e", "approx", "rs", "sr", "jr", "ft"}
_SENTENCE_END_RE = re.compile(r"[.!?]+[\"')\]]*\s+")


def _split_sentences(text: str) -> list[str]:
    """Sentence split that keeps "Dr. Meera", "p. 14" and "Rs. 499" intact; newlines always split."""
    out: list[str] = []
    for line in re.split(r"\n+", text or ""):
        start = 0
        for m in _SENTENCE_END_RE.finditer(line):
            if m.group(0).startswith("."):
                before = re.search(r"([A-Za-z.]+)$", line[:m.start()])
                prev = before.group(1).lower().strip(".") if before else ""
                if prev in _ABBREVIATIONS or (len(prev) == 1 and prev.isalpha()):
                    continue
            piece = line[start:m.end()].strip()
            if piece:
                out.append(piece)
            start = m.end()
        tail = line[start:].strip()
        if tail:
            out.append(tail)
    return out


def _params_from_body(body: str, facts: FactSheet) -> list[str]:
    """[salutation, core line, ask line] derived from the body itself (always consistent with it)."""
    body = (body or "").strip()
    if not body:
        return [facts.salutation or facts.owner_name or "there"]
    sentences = _split_sentences(body)
    params: list[str] = []
    name = facts.salutation or facts.recipient_name
    if name and sentences:
        first = sentences[0]
        idx = first.lower().find(name.lower())
        if 0 <= idx <= 40:
            end = idx + len(name)
            params.append(first[:end].strip(" ,:-—"))
            rest = first[end:].lstrip(" ,:-—!").strip()
            sentences = ([rest] if rest else []) + sentences[1:]
    if len(sentences) >= 2:
        params += [" ".join(sentences[:-1]), sentences[-1]]
    elif sentences:
        params.append(sentences[0])
    return [p for p in params if p][:_TEMPLATE_PARAM_LIMIT] or [body]


def _resolve_facts(facts: FactSheet, used_keys: list[str], body: str) -> list[Fact]:
    """Facts the message relies on: the LLM's declared keys, else facts visibly present in the body."""
    all_facts = facts.all_facts()
    picked: list[Fact] = []
    for key in used_keys or []:
        k = key.strip().strip("[]")
        for f in all_facts:
            if f.key == k and f not in picked:
                picked.append(f)
                break
    if picked:
        return picked
    body_low = (body or "").lower()
    body_nums = _significant_numbers(body)
    for f in all_facts:
        text_hit = len(f.text) >= 8 and f.text.lower().rstrip(".") in body_low
        nums = _significant_numbers(f.text)
        if text_hit or (nums and nums <= body_nums):
            picked.append(f)
    return picked or list(facts.anchor_facts[:3])


_COMMON_NUMBERS = {"7", "30", "2024", "2025", "2026", "2027"}


def _significant_numbers(text: str) -> set[str]:
    """Numbers that identify a fact (not small counts, 7/30-day windows or years)."""
    nums = {normalize_number(t) for t in extract_number_tokens(text or "")} - {None}
    return {n for n in nums if n not in _COMMON_NUMBERS and not (n.isdigit() and int(n) <= 10)}


def _rationale(preferred: str, draft: Draft, facts: FactSheet, pb: Playbook, used: list[Fact], cta: str) -> str:
    for text in (preferred, draft.rationale):
        if isinstance(text, str) and text.strip():
            return _clip(re.sub(r"(?<=\w)_(?=\w)", " ", text.strip()), 480)
    kind = (facts.kind or "update").replace("_", " ")
    who = facts.salutation or facts.owner_name or facts.merchant_name or "the recipient"
    anchor = used[0].text if used else (facts.anchor_facts[0].text if facts.anchor_facts else "their own data")
    lever = (pb.levers[0] if pb.levers else "specificity").replace("_", " ")
    ask = {CTA_BINARY_YES_NO: "a single yes/no ask", CTA_OPEN_ENDED: "one open question"}.get(cta, "one clear ask")
    return _clip(f"Why now: {kind} for {who}. Anchored on '{anchor}', using {lever}, closing with {ask}.", 480)


# --------------------------------------------------------------------------- fallbacks


def _fallback_playbook(facts: FactSheet) -> Playbook:
    customer = facts.send_as == SEND_AS_MERCHANT
    kind = facts.kind or "generic"
    return Playbook(
        kind=kind, family="customer" if customer else "generic", customer_facing=customer,
        goal="Share the one specific reason for writing now and offer one concrete next step.",
        framing="Lead with the most specific fact, support it with one real number, end with one ask.",
        levers=["specificity", "effort_externalization"], cta=CTA_BINARY_YES_NO,
        cta_hint="Reply YES and we'll take care of it." if customer else "Want me to draft the next step?",
        offer="book a convenient time" if customer else "draft the next step for them",
        template_name=f"{'merchant' if customer else 'vera'}_{kind}_v1",
    )


def _generic_draft(facts: FactSheet, pb: Playbook) -> Draft:
    """A plain but grounded message built only from fact texts (used if templates are unavailable)."""
    anchor = facts.anchor_facts[0].text.rstrip(". ") if facts.anchor_facts else ""
    support = next((f.text.rstrip(". ") for f in facts.support_facts
                    if f.key.startswith(("perf.", "agg.", "customer.")) and f.text.rstrip(". ") != anchor), "")
    hinglish = facts.language.code == "hinglish" if facts.language else False
    if facts.send_as == SEND_AS_MERCHANT:
        greeting = (facts.language.greeting if facts.language else "") or "Hi"
        name = facts.salutation
        signer = facts.signer or facts.merchant_name or "your clinic"
        opener = f"{greeting} {name}, {signer} here." if name else f"{greeting}, {signer} here."
        core = f" {_sentence(anchor)}" if anchor else " A quick update from our side."
        ask = " Reply YES and we'll set it up for you, or tell us a time that works."
        body, cta = opener + core + ask, CTA_BINARY_YES_NO
    else:
        name = facts.salutation or facts.owner_name or "there"
        core = f"{name}, quick update: {_lower_first(anchor)}." if anchor else f"{name}, a quick update on your listing."
        extra = f" {_sentence(support)}" if support else ""
        ask = (" Kya main next step aapke liye draft kar doon?" if hinglish
               else " Want me to draft the next step for you?")
        body, cta = core + extra + ask, CTA_BINARY_YES_NO
    params = _params_from_body(body, facts)
    return Draft(body=body.strip(), cta=cta, rationale="", template_params=params, offer=pb.offer, source="template")


def _safe_composition(category: Any, merchant: Any, trigger: Any, customer: Any, now: str | None,
                      prior: list[str]) -> dict:
    """Template-only composition that cannot fail; last resort when the main path raised."""
    try:
        prep = _prepare(category, merchant, trigger, customer, now, prior)
        return _template_result(prep)
    except Exception:
        log.exception("template composition failed; using minimal message")
    try:
        facts = build_facts(category if isinstance(category, dict) else None,
                            merchant if isinstance(merchant, dict) else {},
                            trigger if isinstance(trigger, dict) else {},
                            customer if isinstance(customer, dict) else None, now=now)
        pb = _fallback_playbook(facts)
        draft = _generic_draft(facts, pb)
        prep = _Prep(facts=facts, pb=pb, draft=draft, trigger=trigger if isinstance(trigger, dict) else {},
                     merchant=merchant if isinstance(merchant, dict) else {}, prior=prior,
                     customer_facing=facts.send_as == SEND_AS_MERCHANT)
        return _template_result(prep)
    except Exception:
        log.exception("minimal composition failed")
    t = trigger if isinstance(trigger, dict) else {}
    m = merchant if isinstance(merchant, dict) else {}
    customer_facing = str(t.get("scope") or "") == "customer" and isinstance(customer, dict)
    kind = str(t.get("kind") or "generic")
    merchant_id = str(t.get("merchant_id") or m.get("merchant_id") or "")
    body = ("Hi, a quick update from our side about your next visit. Reply YES and we'll set it up for you."
            if customer_facing else
            "Hi, a quick update on your business listing this week. Want me to share the details?")
    return {
        "body": body,
        "cta": CTA_BINARY_YES_NO,
        "send_as": SEND_AS_MERCHANT if customer_facing else SEND_AS_VERA,
        "suppression_key": (t.get("suppression_key") if isinstance(t.get("suppression_key"), str)
                            and t.get("suppression_key").strip() else f"{kind}:{merchant_id}:{t.get('id') or ''}"),
        "rationale": f"Fallback message for {kind.replace('_', ' ')}: composition failed, so a neutral, "
                     f"fact-free note with one ask was used.",
        "template_name": f"{'merchant' if customer_facing else 'vera'}_generic_update_v1",
        "template_params": [body],
        "meta": {"offer": "", "kind": kind, "source": "template", "language": "en", "facts": [],
                 "fact_keys": [], "trigger_id": str(t.get("id") or ""), "merchant_id": merchant_id,
                 "customer_id": (customer or {}).get("customer_id") if isinstance(customer, dict) else None,
                 "consent_ok": not customer_facing},
    }


# --------------------------------------------------------------------------- anti-repetition

_VARIATIONS = {
    ("merchant", "en"): (
        "Following up since this is still open.",
        "Sharing this again in case it got buried.",
        "Bumping this up in case it slipped past.",
        "Circling back on this one.",
    ),
    ("merchant", "hinglish"): (
        "Yeh abhi bhi open hai, isliye dobara share kar rahi hoon.",
        "Shayad pichhla message miss ho gaya, isliye phir se bhej rahi hoon.",
        "Ek baar phir yaad dila rahi hoon.",
    ),
    ("customer", "en"): (
        "Just a gentle reminder from our side.",
        "Sharing this again in case you missed it.",
        "A quick follow-up from our side.",
    ),
    ("customer", "hinglish"): (
        "Bas ek chhota sa reminder hamari taraf se.",
        "Shayad pichhla message miss ho gaya ho, isliye dobara bhej rahe hain.",
        "Hamari taraf se ek chhota follow-up.",
    ),
}


def _is_repeat(body: str, prior: list[str]) -> bool:
    norm = normalize_for_compare(body)
    return any(normalize_for_compare(p) == norm for p in prior)


def _vary(body: str, prep: _Prep) -> str:
    """Make a repeated body new: insert a short follow-up line before the closing ask."""
    code = prep.facts.language.code if prep.facts.language else "en"
    options = _VARIATIONS[("customer" if prep.customer_facing else "merchant",
                           "hinglish" if code in ("hinglish", "hi") else "en")]
    sentences = _split_sentences(body)
    seed = int(hashlib.sha256(body.encode("utf-8")).hexdigest(), 16)
    for i in range(len(options)):
        line = options[(seed + i) % len(options)]
        if len(sentences) >= 2:
            candidate = " ".join(sentences[:-1] + [line, sentences[-1]])
        else:
            candidate = f"{body.rstrip()} {line}"
        if not _is_repeat(candidate, prep.prior):
            return candidate
    return f"{body.rstrip()} ({len(prep.prior) + 1})"


# --------------------------------------------------------------------------- helpers


def _clean_prior(prior_bodies: Iterable[str] | None) -> list[str]:
    try:
        return [p for p in (prior_bodies or ()) if isinstance(p, str) and p.strip()]
    except TypeError:
        return []


def _source(comp: dict) -> str:
    meta = comp.get("meta") if isinstance(comp, dict) else None
    return str((meta or {}).get("source") or "")


def _clean_body(text: str) -> str:
    """Tidy LLM / template output without changing its meaning."""
    s = str(text or "").replace("\r\n", "\n").replace("\\n", "\n").strip()
    s = re.sub(r"^(?:body|message)\s*:\s*", "", s, flags=re.IGNORECASE)
    if len(s) >= 2 and s[0] in "\"'“" and s[-1] in "\"'”" and s.count(s[0]) <= 2:
        s = s[1:-1].strip()
    s = re.sub(r"\bDr\b\.?\s*(?:Dr\b\.?\s*)+", "Dr. ", s, flags=re.IGNORECASE)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _sentence(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def _lower_first(text: str) -> str:
    """Lower-case the first letter unless it starts an acronym or a proper-looking word (CTR, Diwali)."""
    if not text:
        return text
    first = text.split()[0]
    if first.isupper() or (len(first) > 1 and not first[1:].islower()):
        return text
    common = {"calls", "profile", "views", "last", "new", "a", "the", "your", "salon", "dental", "current", "due"}
    return text[0].lower() + text[1:] if first.lower() in common else text


def _clip(text: str, limit: int) -> str:
    s = re.sub(r"\s+", " ", text or "").strip()
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _uniq(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    for it in items:
        if it not in out:
            out.append(it)
    return out
