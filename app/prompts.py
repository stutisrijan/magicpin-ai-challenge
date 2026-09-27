"""LLM prompts for message composition.

compose_prompt() turns (FactSheet, Playbook, Draft) into a compact (system, user) pair:
the system prompt carries who Vera is, the non-negotiable rules, the judging rubric and
the JSON output contract; the user prompt carries this message's facts in labelled
sections, with the deterministic template draft as a baseline to beat.

The LLM only writes wording around facts it is handed. Every number it may use appears
in a fact line; the validator rejects anything else. Prompts stay well under ~2,500
tokens so free-tier models answer fast.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from app.numbers import extract_number_tokens, normalize_number
from app.schemas import (
    CTA_BINARY_CONFIRM,
    CTA_BINARY_YES_NO,
    CTA_MULTI_CHOICE_SLOT,
    CTA_NONE,
    CTA_OPEN_ENDED,
    CTA_VALUES,
    SEND_AS_MERCHANT,
    Draft,
    Fact,
    FactSheet,
    Playbook,
)

__all__ = ["compose_prompt", "repair_prompt", "COMPOSE_SCHEMA", "length_target", "MAX_PROMPT_CHARS"]

# Keys ordered so the model picks facts, then plans, then writes.
COMPOSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "used_facts": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Keys (the [bracketed] ids) of the facts the message uses.",
        },
        "rationale": {
            "type": "string",
            "description": "1-2 sentences: why now (the trigger), the anchor fact, the lever, the ask.",
        },
        "body": {"type": "string", "description": "The WhatsApp message text."},
        "cta": {"type": "string", "enum": sorted(CTA_VALUES), "description": "Type of the single closing ask."},
        "template_params": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The body split in order: [salutation, core line(s), closing ask].",
        },
    },
    "required": ["body", "cta", "rationale"],
}

MAX_PROMPT_CHARS = 9000          # system + user, ~2,300 tokens
_MAX_SUPPORT = 14
_MAX_NOTES = 8
_MAX_HISTORY = 4

_CTA_HELP = {
    CTA_BINARY_YES_NO: "one yes/no ask as the last sentence (e.g. 'Want me to ...?' or 'Reply YES and I'll ...')",
    CTA_BINARY_CONFIRM: "one confirm-style ask as the last sentence (confirm / cancel)",
    CTA_OPEN_ENDED: "one easy open question as the last sentence",
    CTA_MULTI_CHOICE_SLOT: "offer the exact slot labels as numbered choices (Reply 1 / 2), or ask for a time that works",
    CTA_NONE: "no ask needed; close with a short, useful line",
}

_CATEGORY_VOICE = {
    "dentists": "clinical peer: address as 'Dr. <name>', technical vocabulary welcome, cite sources, no hype",
    "salons": "warm and practical, like a friendly expert; service + price beats discounts",
    "restaurants": "fellow operator: covers, AOV, footfall, delivery vs dine-in; brisk and practical",
    "gyms": "coach: energetic but disciplined; members, retention, footfall; no body-shaming or quick-fix claims",
    "pharmacies": "trustworthy and precise neighbourhood pharmacist; exact molecules, batches, dates; no overclaims",
}

_SYSTEM_CORE = """You write ONE WhatsApp message for Vera, magicpin's AI assistant for Indian local businesses (dentists, salons, restaurants, gyms, pharmacies). Merchant-facing messages go from Vera to the business owner. Customer-facing messages are sent from the business's own WhatsApp number, in the business's voice.

NON-NEGOTIABLE RULES
1. Grounding: every number, date, time, name, price, offer, slot, source and claim must come from the facts in the user message, copied exactly. Never invent names, numbers, percentages, prices, offers, slots, dates, times, research, sources, competitors or customer counts. If something is not in the facts, write around it. Numbers inside tone examples are not facts about this business.
2. Offers: quote only the merchant's ACTIVE OFFERS, exactly as written. Catalog ideas may only be suggested to the merchant ("you could run ..."), never presented as existing, and never mentioned to customers.
3. No URLs, links, web addresses or email addresses.
4. Never use a taboo word listed in the user message, and no hype ("!!", "amazing deal", "guaranteed").
5. First sentence = the recipient's name + the why-now hook. No preamble ("Hope you're doing well", "I am reaching out"), no greeting-only line, no self-introduction (at most "Vera here" when the notes say it is a first conversation).
6. Exactly one call to action, and it is the last sentence. Never two asks; at most 2 question marks in the whole message.
7. No internal jargon: no snake_case, no field names, no fact keys, never the words trigger, payload, placeholder or signal.
8. Customer-facing: speak as the business, never mention Vera or magicpin, no medical claims or guarantees, respectful to seniors (Namaste, ji), no guilt about a lapse.
9. Keep every fact from the BASELINE DRAFT correct; you may drop weak ones and choose better ones from the list.
10. Plain WhatsApp text: short sentences, no markdown headers, at most one emoji. A drafted list (tiers, steps) may use short lines.

HOW THE MESSAGE IS JUDGED (0-10 each, strict)
- Specificity: concrete, verifiable facts (numbers, dates, source citations, prices) from the facts.
- Category fit: voice and vocabulary fit the business type.
- Merchant fit: their own numbers, offers, locality, history; their language preference honoured.
- Trigger relevance: it is obvious why this message is being sent now.
- Engagement compulsion: one or two levers (loss aversion, curiosity, social proof from real data, effort externalisation such as "I've drafted it, just say go", reciprocity, asking their opinion) and a low-friction single ask.
Penalties: any fabricated data, jargon, URLs, repeating an earlier message.

OUTPUT
Return one JSON object only, no prose, with the keys:
"used_facts": [the bracketed keys of the facts you used],
"rationale": "1-2 sentences: why now + anchor fact + lever + the ask",
"body": "the message",
"cta": one of "binary_yes_no" | "binary_confirm_cancel" | "open_ended" | "multi_choice_slot" | "none",
"template_params": ["salutation", "core line(s)", "closing ask"] (pieces of the body, in order)."""


# --------------------------------------------------------------------------- public API


def length_target(facts: FactSheet, pb: Playbook) -> tuple[int, int]:
    """(min, max) characters to aim for (spec section 4 rule 11)."""
    if pb.customer_facing or facts.send_as == SEND_AS_MERCHANT:
        return 200, 420
    if pb.family == "planning" or facts.kind in ("active_planning_intent",):
        return 350, 700
    return 250, 500


def compose_prompt(facts: FactSheet, pb: Playbook, draft: Draft, *,
                   prior_bodies: Iterable[str] = ()) -> tuple[str, str]:
    """(system, user) prompts for one composition."""
    facts = facts if isinstance(facts, FactSheet) else FactSheet()
    pb = pb if isinstance(pb, Playbook) else Playbook()
    draft = draft if isinstance(draft, Draft) else Draft()
    system = _system(facts, pb)
    support_n, notes_n, framing_n = _MAX_SUPPORT, _MAX_NOTES, 900
    user = _user(facts, pb, draft, prior_bodies, support_n, notes_n, framing_n)
    # shrink the least important sections until the pair fits the budget
    while len(system) + len(user) > MAX_PROMPT_CHARS and (support_n > 4 or notes_n > 3 or framing_n > 400):
        support_n = max(4, support_n - 2)
        notes_n = max(3, notes_n - 1)
        framing_n = max(400, framing_n - 100)
        user = _user(facts, pb, draft, prior_bodies, support_n, notes_n, framing_n)
    return system, user


def repair_prompt(facts: FactSheet, pb: Playbook, draft: Draft, candidate: str, issues: list[str], *,
                  prior_bodies: Iterable[str] = ()) -> tuple[str, str]:
    """Same task, plus the rejected attempt and the exact problems to fix."""
    system, user = compose_prompt(facts, pb, draft, prior_bodies=prior_bodies)
    problems = "\n".join(f"- {_clip(i, 220)}" for i in list(issues)[:8]) or "- (unspecified)"
    user += (
        "\n\nYOUR PREVIOUS ATTEMPT WAS REJECTED\n"
        f"Attempt: {_clip(candidate, 900)!r}\n"
        f"Problems:\n{problems}\n"
        "Rewrite it so that every problem is fixed and every rule holds. Use only numbers that appear in the "
        "facts above; if unsure about a number, leave it out. Return the full JSON object again."
    )
    return system, user


# --------------------------------------------------------------------------- system


def _system(facts: FactSheet, pb: Playbook) -> str:
    lo, hi = length_target(facts, pb)
    customer = pb.customer_facing or facts.send_as == SEND_AS_MERCHANT
    who = (f"the business ({facts.signer or facts.merchant_name or 'the business'}), to its customer"
           if customer else "Vera, to the business owner")
    lang = facts.language.instruction if facts.language else "Write in clear, simple English."
    return (
        f"{_SYSTEM_CORE}\n\n"
        f"THIS MESSAGE: written as {who}. Language: {lang} Length: about {lo}-{hi} characters "
        f"(hard limit 900)."
    )


# --------------------------------------------------------------------------- user


def _user(facts: FactSheet, pb: Playbook, draft: Draft, prior_bodies: Iterable[str],
          support_n: int, notes_n: int, framing_n: int = 900) -> str:
    customer = pb.customer_facing or facts.send_as == SEND_AS_MERCHANT
    parts: list[str] = []

    # RECIPIENT & SENDER
    lines = []
    place = ", ".join(x for x in (facts.locality, facts.city) if x)
    business = f"{facts.merchant_name}{' (' + place + ')' if place else ''}" if facts.merchant_name else place
    if customer:
        if facts.recipient_name or facts.salutation:
            lines.append(f"Recipient: customer, address as '{facts.salutation or facts.recipient_name}'")
        else:
            lines.append("Recipient: customer with no name on file: do not address by name")
        if facts.about_name and facts.about_name != (facts.salutation or facts.recipient_name):
            lines.append(f"The message is about: {facts.about_name} (third person)")
        lines.append(f"Business: {business or 'the business'}; owner {facts.owner_name or 'unknown'}")
    else:
        lines.append(f"Recipient: the owner, address exactly as '{facts.salutation or facts.owner_name or 'there'}'"
                     f" (never 'Dr. Dr.')")
        if business:
            lines.append(f"Business: {business}")
    lines.append(f"Category: {facts.category_slug or 'unknown'}")
    parts.append(_section("RECIPIENT & SENDER", lines))

    # SEND AS
    if customer:
        parts.append(_section("SEND AS", [
            f"merchant_on_behalf: speak as \"{facts.signer or facts.merchant_name or 'the business'}\" "
            f"(e.g. \"{facts.signer or facts.merchant_name or 'the business'} here\"); never mention Vera or magicpin"]))
    else:
        parts.append(_section("SEND AS", ["vera: Vera writing to the owner (peer, helpful, no sales pitch)"]))

    # LANGUAGE
    lang = facts.language
    parts.append(_section("LANGUAGE", [
        f"{lang.code}: {lang.instruction}",
        f"Greeting word that fits: {lang.greeting}" + (" (Devanagari script)" if lang.script == "devanagari" else ""),
    ]))

    # CATEGORY VOICE
    voice = []
    cat_voice = _CATEGORY_VOICE.get(_cat_key(facts.category_slug))
    if cat_voice:
        voice.append(f"Voice: {cat_voice}")
    tone = ", ".join(x for x in (facts.voice_tone, facts.voice_register) if x)
    if tone:
        voice.append(f"Tone: {tone}")
    if facts.tone_examples:
        voice.append("Tone examples (style only; their numbers are NOT facts): "
                     + " | ".join(_clip(x, 110) for x in facts.tone_examples[:3]))
    if facts.vocab_allowed:
        voice.append("Vocabulary you may use: " + ", ".join(facts.vocab_allowed[:12]))
    if facts.taboos:
        voice.append("TABOO (never write): " + ", ".join(facts.taboos))
    parts.append(_section("CATEGORY VOICE", voice))

    # TRIGGER / WHY NOW
    why = [f"Event: {_humanize(facts.kind) or 'update'} (urgency {facts.urgency}/5)"]
    if facts.payload_is_placeholder:
        why.append("The event carries no specific data: anchor on the real facts below; invent nothing.")
    if pb.goal:
        why.append(f"Goal: {_clip(pb.goal, 300)}")
    if pb.framing:
        why.append(f"Framing: {_clip(pb.framing, framing_n)}")
    if pb.levers:
        why.append("Levers: " + ", ".join(_humanize(x) for x in pb.levers[:4]))
    why.append(f"Closing ask ({pb.cta}): {pb.cta_hint or _CTA_HELP.get(pb.cta, _CTA_HELP[CTA_OPEN_ENDED])}")
    if pb.offer:
        why.append(f"If they say yes, Vera will: {_clip(pb.offer, 160)}")
    parts.append(_section("TRIGGER / WHY NOW", why))

    # FACTS
    anchors = facts.anchor_facts[:8]
    parts.append(_section("ANCHOR FACTS (why now; use 1-2, numbers exactly as written)",
                          [_fact_line(f"A{i}", f) for i, f in enumerate(anchors, 1)] or ["(none)"]))
    support = _select_support(facts, anchors, customer, support_n, draft.body)
    parts.append(_section("SUPPORT FACTS (pick what strengthens the message)",
                          [_fact_line(f"S{i}", f) for i, f in enumerate(support, 1)] or ["(none)"]))
    known = " ".join(f.text for f in anchors + support).lower()
    extra = []
    if not customer:            # market trends are merchant intelligence, not customer copy
        if facts.seasonal_note and facts.seasonal_note.lower() not in known:
            extra.append(f"[seasonal.note] {_clip(facts.seasonal_note, 200)}")
        if facts.trend_note and facts.trend_note.lower() not in known:
            extra.append(f"[trend.top] {_clip(facts.trend_note, 200)}")
    if facts.content_item and facts.content_item.get("title"):
        extra.append(f"[content.item] Shareable explainer the business has: "
                     f"'{_clip(str(facts.content_item.get('title')), 120)}'")
    if extra:
        parts.append(_section("CONTEXT", extra))

    # OFFERS
    parts.append(_section("MERCHANT'S ACTIVE OFFERS (quotable exactly)",
                          [f"- {o}" for o in facts.offers_active[:6]] or
                          ["(none: quote no price, discount or freebie)" if customer else "(none)"]))
    active = {o.strip().lower() for o in facts.offers_active}
    ideas = [o for o in facts.catalog_offers if o.strip().lower() not in active]
    if not customer and ideas:
        parts.append(_section("CATALOG IDEAS (suggest to the merchant only; not live offers)",
                              [f"- {o}" for o in ideas[:6]]))
    if facts.slots:
        parts.append(_section("SLOTS (offer these exact labels)", [f"{i}. {s}" for i, s in enumerate(facts.slots[:4], 1)]))
    elif customer:
        parts.append(_section("SLOTS", ["(none: do not invent times; ask what time works)"]))

    # HISTORY
    hist = [] if customer else _history_lines(facts.history)   # Vera<->owner chat is not customer context
    prior = [p for p in (prior_bodies or ()) if isinstance(p, str) and p.strip()][-2:]
    hist += [f"- already sent: '{_clip(p, 200)}'" for p in prior]
    if hist:
        parts.append(_section("RECENT CONVERSATION (context; never repeat an earlier message)", hist))

    # NOTES
    if facts.notes:
        parts.append(_section("JUDGMENT NOTES", [f"- {_clip(n, 240)}" for n in facts.notes[:notes_n]]))

    # BASELINE
    if draft.body.strip():
        parts.append("BASELINE DRAFT (safe and valid; write a clearly better one, keep its facts correct):\n"
                     + draft.body.strip())

    opener = ("opens with a greeting and the business name (no customer name is on file)"
              if customer and not (facts.salutation or facts.recipient_name) else "opens with the recipient's name")
    parts.append("OUTPUT FORMAT: one JSON object with keys used_facts, rationale, body, cta, template_params. "
                 f"The body {opener}, uses only the facts above, and ends with the single ask.")
    return "\n\n".join(p for p in parts if p)


# --------------------------------------------------------------------------- helpers


def _section(title: str, lines: list[str]) -> str:
    lines = [ln for ln in lines if ln]
    if not lines:
        return ""
    return f"{title}\n" + "\n".join(lines)


def _fact_line(label: str, fact: Fact) -> str:
    return f"{label} [{fact.key}] {_clip(fact.text, 260)}"


# max facts per key family, so twelve performance lines never crowd out reviews or cohorts
_GROUP_CAPS = {"perf": 6, "agg": 3, "review": 2, "signal": 2, "merchant": 2, "customer": 6, "offer": 2,
               "seasonal": 1, "trend": 1, "peer": 2, "history": 2}


def _perf_rank(fact: Fact) -> int:
    key = fact.key
    return 0 if key.endswith("_vs_peer") else 1 if ".delta_" in key else 2


def _identifying_numbers(text: str) -> set[str]:
    """Numbers that identify a fact: not small counts, 7/30-day windows or years."""
    nums = {normalize_number(t) for t in extract_number_tokens(text or "")} - {None}
    return {n for n in nums if not (n.isdigit() and (int(n) <= 10 or int(n) == 30 or 2000 <= int(n) <= 2100))}


def _in_draft(fact: Fact, draft_low: str, draft_nums: set[str]) -> bool:
    if not draft_low:
        return False
    nums = _identifying_numbers(fact.text)
    return fact.text.lower().rstrip(".") in draft_low or bool(nums and nums <= draft_nums)


def _select_support(facts: FactSheet, anchors: list[Fact], customer: bool, limit: int,
                    draft_body: str = "") -> list[Fact]:
    """Strongest, most varied support facts: those the baseline draft uses, then weight, then a per-family cap."""
    anchor_texts = {f.text.lower() for f in anchors}
    draft_low, draft_nums = (draft_body or "").lower(), _identifying_numbers(draft_body)
    pool = []
    for i, f in enumerate(facts.support_facts):
        group = f.key.split(".", 1)[0]
        if f.text.lower() in anchor_texts:
            continue
        if f.key == "offer.active":                 # listed in the offers section
            continue
        if customer and group in ("offer", "seasonal", "trend", "peer", "perf", "agg", "signal", "history",
                                  "merchant", "review"):
            continue
        if group == "history" and facts.history:     # shown in the conversation section
            continue
        pool.append((i, group, f))
    pool.sort(key=lambda x: (not _in_draft(x[2], draft_low, draft_nums), -int(x[2].weight or 0),
                             _perf_rank(x[2]) if x[1] == "perf" else 0, x[0]))
    counts: dict[str, int] = {}
    out: list[Fact] = []
    for _i, group, f in pool:
        if counts.get(group, 0) >= _GROUP_CAPS.get(group, 3):
            continue
        counts[group] = counts.get(group, 0) + 1
        out.append(f)
        if len(out) >= limit:
            break
    return out


def _history_lines(history: list[dict] | None) -> list[str]:
    out = []
    for turn in list(history or [])[-_MAX_HISTORY:]:
        if not isinstance(turn, dict):
            continue
        who = str(turn.get("from") or turn.get("role") or "").lower()
        body = turn.get("body") or turn.get("message") or turn.get("text")
        if not isinstance(body, str) or not body.strip():
            continue
        label = "merchant" if who in ("merchant", "owner", "customer") else "vera"
        tag = " (do not repeat)" if label == "vera" else ""
        out.append(f"- {label}{tag}: '{_clip(body, 200)}'")
    return out


def _humanize(value: Any) -> str:
    return re.sub(r"[_\s]+", " ", str(value or "")).strip()


def _cat_key(slug: str) -> str:
    s = (slug or "").lower()
    for key in _CATEGORY_VOICE:
        if s.startswith(key[:4]):
            return key
    return s


def _clip(text: Any, limit: int) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    return s if len(s) <= limit else s[: max(0, limit - 1)].rstrip() + "…"
