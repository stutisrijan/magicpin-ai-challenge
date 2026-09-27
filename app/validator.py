"""Message validator: the last gate before any body leaves the bot.

validate_body() checks one WhatsApp body against the FactSheet it was written from and
returns a list of issue strings ("code: detail"); an empty list means the body is safe
to send. The composer uses it to accept or reject LLM output (and to phrase a repair
request), and the conversation engine uses it for reply bodies.

Checks are deliberately precise: good template copy must pass untouched, so every rule
targets a concrete failure the judge penalises (fabricated numbers, URLs, jargon, taboo
words, preambles, multiple asks, repeats) rather than stylistic taste.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

from app.numbers import clock_times, extract_number_tokens, normalize_number
from app.schemas import FactSheet

log = logging.getLogger(__name__)

__all__ = [
    "validate_body", "extract_number_tokens", "normalize_number", "issue_code", "normalize_for_compare",
    "ungrounded_numbers", "MODES",
]

MODES = ("compose", "reply", "reply_action")
MIN_CHARS = 40
MAX_CHARS = {"compose": 900, "reply": 700, "reply_action": 1100}
NAME_WINDOW = 80                 # the addressee must appear this early (merchant-facing)
NAME_WINDOW_CUSTOMER = 120       # customer-facing openers may lead with "Namaste 🙏 <business> here"
MAX_QUESTION_MARKS = 2

# Small integers and common durations read naturally without being "data".
SMALL_INT_MAX = 10
TIME_UNIT_NUMBERS = {"15", "20", "24", "30", "45", "48", "60", "90"}
_TIME_UNIT_RE = re.compile(
    r"^\s?-?\s?(?:mins?|minutes?|hrs?|hours?|h|ghante|ghanta|secs?|seconds?)\b", re.IGNORECASE)

# Time expressions that are not data claims: "24x7", "24/7", "7-9pm", "10:30", "18:00".
_ALWAYS_MASK = [
    re.compile(r"\b24\s?[x×/]\s?7\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:-|–|—|to)\s?\d{1,2}(?::\d{2})?\s?(?:am|pm)\b", re.IGNORECASE),
    re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b"),
]

# URLs and web addresses (the testing brief penalises every URL).
_URL_RE = re.compile(r"https?://\S*|\bwww\.\S*|\bwa\.me/\S*", re.IGNORECASE)
_DOMAIN_RE = re.compile(
    r"(?<![\w@.])[A-Za-z0-9][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)*"
    r"\.(?:com|in|org|net|io|co|ly|app|info|biz|me|xyz|link|site|online|store|shop|page|gl)"
    r"(?![A-Za-z0-9])(?:/\S*)?"
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")

# Internal jargon: snake_case, dotted fact keys, raw field names.
_SNAKE_RE = re.compile(r"(?<![\w@])[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+\b")
_FACT_KEY_RE = re.compile(
    r"\b(?:digest|trigger|perf|agg|offer|review|history|signal|customer|seasonal|trend|peer|merchant|derived)"
    r"\.[a-z][a-z_]*\b"
)
_JARGON_WORDS_RE = re.compile(r"\b(?:payload|placeholder|suppression|context[ _]id)\b", re.IGNORECASE)
# "trigger" as a noun is internal; as a verb ("heat can trigger dehydration") it is fine.
_TRIGGER_RE = re.compile(r"\btriggers?\b", re.IGNORECASE)
_TRIGGER_VERB_BEFORE = re.compile(
    r"\b(?:can|may|might|could|will|would|to|often|that|which|also|not|won't|don't|doesn't|usually)\s+$",
    re.IGNORECASE)
_UNFILLED_RE = re.compile(r"\{\{?\s*[\w.]*\s*\}?\}|\b(?:null|undefined|NaN)\b|\[object|<[a-z_]+>")

# Openers the judge reads as filler.
_PREAMBLE_RE = re.compile(
    r"\b(?:hope you|i hope|i am reaching out|i'm reaching out|i am writing to|i'm writing to|trust you are|"
    r"trust you're|hope this message finds|hope (?:all|everything) is (?:well|good|fine)|umeed hai)",
    re.IGNORECASE)
_GREETINGS_RE = re.compile(r"\bgreetings\b", re.IGNORECASE)   # "Greetings!", "warm greetings from ..."
_PREAMBLE_WINDOW = 140
_GREETINGS_WINDOW = 60           # "greetings" is filler only in the opening line
_SELF_INTRO_RE = re.compile(r"\b(?:i'm vera|i am vera|this is vera|vera here|myself vera|vera this side)\b",
                            re.IGNORECASE)
_BRAND_RE = re.compile(r"\b(?:vera|magic\s?pin)\b", re.IGNORECASE)
_DR_DR_RE = re.compile(r"\bdr\.?\s*dr\b", re.IGNORECASE)
_HYPE_RE = re.compile(r"!{2,}|\b(?:amazing deal|act now|click here|once in a lifetime|don't miss out)\b",
                      re.IGNORECASE)

# Calls to action: "Reply YES", "just say GO", "Reply 1 for Wed". STOP/NO are opt-outs, not asks.
_DIRECTIVE_RE = re.compile(
    r"\b(?:reply|type|text|send|say|message|bhejein|bhejo|likhein|likho)\s+(?:with\s+|us\s+)?[\"'“‘*]*"
    r"(yes|haan|ok|okay|go|confirm|y)(?![\w'])",
    re.IGNORECASE,
)
_NUMERIC_DIRECTIVE_RE = re.compile(      # slot choices: "Reply 1", "type 2"
    r"\b(?:reply|type|press)\s+(?:with\s+)?[\"'“‘*]*(\d)(?![\d.,%x])(?!\s*(?:min|hour|hr|day|week|month))",
    re.IGNORECASE,
)
_REVERSE_DIRECTIVE_RE = re.compile(      # Hinglish word order: "YES reply karein", "bas GO likh dijiye"
    r"(?<![\w'])[\"'“‘*]*(yes|haan|ok|go|confirm)[\"'”’*]*\s+(?:reply|likh|bhej|type)\w*", re.IGNORECASE)

# Qualifying phrases the local judge fails in action mode (plain substring match, like the judge).
REPLY_ACTION_FORBIDDEN = ("would you", "do you", "can you tell", "what if", "how about")
REPLY_ACTION_WORDS = ("done", "sending", "draft", "here", "confirm", "proceed", "next")

_NAME_STOPWORDS = {"dr", "ji", "team", "mr", "mrs", "ms", "the", "and", "from", "sri", "shri", "smt", "doc",
                   "there", "clinic", "salon", "gym", "pharmacy", "restaurant", "cafe"}


# --------------------------------------------------------------------------- public API


def validate_body(
    body: str,
    facts: FactSheet,
    *,
    prior_bodies: Iterable[str] = (),
    mode: str = "compose",
    customer_facing: bool = False,
) -> list[str]:
    """Issues found in `body` (empty list = OK). mode: "compose" | "reply" | "reply_action"."""
    try:
        return _validate(body, facts, prior_bodies, mode, customer_facing)
    except Exception as exc:  # a validator bug must never let an unchecked body through silently
        log.exception("validator failed")
        return [f"validator_error: {type(exc).__name__}"]


def issue_code(issue: str) -> str:
    """The stable code part of an issue string ("ungrounded_number: ..." -> "ungrounded_number")."""
    return str(issue).split(":", 1)[0].strip()


def normalize_for_compare(text: str) -> str:
    """Case-, punctuation- and whitespace-insensitive form used for repeat detection."""
    s = str(text or "").lower().replace("’", "'")
    s = re.sub(r"[^\w₹%]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def ungrounded_numbers(body: str, allowed: set[str] | frozenset[str]) -> list[str]:
    """Numeric tokens in `body` that the contexts do not support (spec section 10)."""
    text = str(body or "")
    for rx in _ALWAYS_MASK:
        text = rx.sub(lambda m: " " * len(m.group(0)), text)
    for clock in clock_times(text):                 # same-length masking keeps positions aligned
        text = text.replace(clock, " " * len(clock))
    bad: list[str] = []
    pos = 0
    for tok in extract_number_tokens(text):
        # locate the token (not glued to a word) to inspect what follows it (time units)
        m = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(tok)).search(text, pos)
        after = text[m.end(): m.end() + 12] if m else ""
        if m:
            pos = m.end()
        norm = normalize_number(tok)
        if norm is None or norm in allowed:
            continue
        low = tok.lower()
        marked = "%" in low or "₹" in low or low.startswith(("rs", "inr")) or low.rstrip().endswith("x")
        if not marked:
            if norm.isdigit() and int(norm) <= SMALL_INT_MAX:
                continue
            if norm in TIME_UNIT_NUMBERS and _TIME_UNIT_RE.match(after):
                continue
        if tok.strip() not in bad:
            bad.append(tok.strip())
    return bad


# --------------------------------------------------------------------------- checks


def _validate(body: Any, facts: Any, prior_bodies: Iterable[str], mode: str, customer_facing: bool) -> list[str]:
    mode = mode if mode in MODES else "compose"
    text = body if isinstance(body, str) else ""
    stripped = text.strip()
    if not stripped:
        return ["empty: the message body is empty"]

    sheet = facts if isinstance(facts, FactSheet) else FactSheet()
    low = stripped.lower().replace("’", "'")
    issues: list[str] = []

    # length
    if len(stripped) < MIN_CHARS:
        issues.append(f"too_short: {len(stripped)} chars; write at least {MIN_CHARS}")
    limit = MAX_CHARS[mode]
    if len(stripped) > limit:
        issues.append(f"too_long: {len(stripped)} chars; keep it under {limit}")

    # links
    urls = [m.group(0) for m in _URL_RE.finditer(stripped)]
    rest = _URL_RE.sub(" ", stripped)
    urls += [m.group(0) for m in _EMAIL_RE.finditer(rest)]
    urls += [m.group(0) for m in _DOMAIN_RE.finditer(_EMAIL_RE.sub(" ", rest))]
    if urls:
        issues.append(f"url: remove links/web addresses ({', '.join(_uniq(urls)[:3])})")

    # taboo vocabulary
    hits = _taboo_hits(stripped, sheet.taboos)
    if hits:
        issues.append(f"taboo: never use {', '.join(repr(h) for h in hits)}")

    # fabricated numbers
    bad = ungrounded_numbers(stripped, sheet.allowed_numbers or set())
    if bad:
        issues.append(f"ungrounded_number: {', '.join(bad[:6])} not found in the facts; use only numbers "
                      f"given in the facts")

    # customer-facing prices come only from the merchant's own offers / the customer's facts
    if customer_facing:
        prices = _unapproved_prices(stripped, sheet)
        if prices:
            issues.append(f"unapproved_price: {', '.join(prices[:4])} is not one of the merchant's active offers; "
                          f"quote only active-offer prices to customers")

    # internal jargon / template leftovers
    jargon = _jargon_hits(stripped)
    if jargon:
        issues.append(f"jargon: internal terms {', '.join(repr(j) for j in jargon[:4])} must not appear")
    unfilled = _uniq(m.group(0) for m in _UNFILLED_RE.finditer(stripped))
    if unfilled:
        issues.append(f"unfilled_placeholder: {', '.join(repr(u) for u in unfilled[:3])}")

    # openers
    head = low[:_PREAMBLE_WINDOW]
    pre = _uniq(m.group(0).lower() for m in _PREAMBLE_RE.finditer(head))
    if _GREETINGS_RE.search(low[:_GREETINGS_WINDOW]):
        pre.append("greetings")
    if pre:
        issues.append(f"preamble: drop filler openers ({', '.join(pre)}); open with the name and the reason")
    if mode in ("reply", "reply_action") and _SELF_INTRO_RE.search(low):
        issues.append("self_intro: do not re-introduce Vera in an ongoing conversation")
    if customer_facing:
        brands = _uniq(m.group(0) for m in _BRAND_RE.finditer(stripped))
        if brands:
            issues.append(f"brand_mention: customer messages speak as the business; remove {', '.join(brands)}")
    if _DR_DR_RE.search(stripped):
        issues.append("dr_dr: 'Dr. Dr.' duplicated honorific")
    if _HYPE_RE.search(stripped):
        issues.append("hype: no promotional hype ('!!', 'amazing deal', 'act now')")

    # addressee
    if mode == "compose":
        missing = _missing_name(stripped, sheet, customer_facing)
        if missing:
            issues.append(missing)

    # asks
    n_ctas = _count_ctas(stripped)
    if n_ctas > 1:
        issues.append(f"multiple_ctas: {n_ctas} separate reply directives; keep exactly one ask")
    q = stripped.count("?")
    if q > MAX_QUESTION_MARKS:
        issues.append(f"too_many_questions: {q} question marks; ask one thing")

    # repetition
    norm = normalize_for_compare(stripped)
    for prior in prior_bodies or ():
        if isinstance(prior, str) and prior.strip() and normalize_for_compare(prior) == norm:
            issues.append("repeat: identical to a message already sent; say something new")
            break

    # action mode (after the merchant said yes)
    if mode == "reply_action":
        found = [p for p in REPLY_ACTION_FORBIDDEN if p in low]
        if found:
            issues.append(f"qualifying: action replies must not ask {', '.join(repr(f) for f in found)}; "
                          f"deliver the thing now")
        if not any(w in low for w in REPLY_ACTION_WORDS):
            issues.append("no_action_word: say what is done / being sent / drafted and what happens next")
    return issues


_PRICE_RE = re.compile(r"(?:₹|\brs\.?\s?|\binr\s?)\s?\d[\d,]*(?:\.\d+)?", re.IGNORECASE)


def _unapproved_prices(text: str, sheet: FactSheet) -> list[str]:
    """Rupee amounts in a customer message that are not in active offers, slots or fact texts."""
    sources = list(sheet.offers_active) + list(sheet.slots) + [f.text for f in sheet.all_facts()]
    approved: set[str] = set()
    for src in sources:
        for tok in extract_number_tokens(str(src or "")):
            n = normalize_number(tok)
            if n is not None:
                approved.add(n)
    bad: list[str] = []
    for m in _PRICE_RE.finditer(text):
        n = normalize_number(m.group(0))
        if n is not None and n not in approved and m.group(0).strip() not in bad:
            bad.append(m.group(0).strip())
    return bad


def _taboo_hits(text: str, taboos: Iterable[str]) -> list[str]:
    hits: list[str] = []
    for raw in taboos or ():
        word = str(raw or "").split("(")[0].strip()
        if not word:
            continue
        pattern = re.escape(word).replace(r"\ ", r"\s+")
        if re.search(rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])", text, re.IGNORECASE):
            if word not in hits:
                hits.append(word)
    return hits


def _jargon_hits(text: str) -> list[str]:
    found = [m.group(0) for m in _SNAKE_RE.finditer(text)]
    found += [m.group(0) for m in _FACT_KEY_RE.finditer(text)]
    found += [m.group(0) for m in _JARGON_WORDS_RE.finditer(text)]
    for m in _TRIGGER_RE.finditer(text):
        before = text[max(0, m.start() - 20): m.start()]
        if not _TRIGGER_VERB_BEFORE.search(before):
            found.append(m.group(0))
    return _uniq(found)


def _name_tokens(*names: str) -> list[str]:
    out: list[str] = []
    for name in names:
        for tok in re.findall(r"[^\W\d_][\w'-]*", str(name or "")):
            t = tok.strip("'-")
            if len(t) >= 3 and t.lower() not in _NAME_STOPWORDS and t.lower() not in (o.lower() for o in out):
                out.append(t)
    return out


def _missing_name(body: str, sheet: FactSheet, customer_facing: bool) -> str | None:
    names = [n for n in (sheet.salutation, sheet.recipient_name, sheet.about_name) if n]
    if not customer_facing and sheet.owner_name:
        names.append(sheet.owner_name)
    names = [n for n in names if not _is_placeholder_name(n)]
    if not names:
        return None                       # nobody to address by name ("there", "Doc"): nothing to check
    tokens = _name_tokens(*names)
    window = body[: (NAME_WINDOW_CUSTOMER if customer_facing else NAME_WINDOW)]
    if not tokens:
        # names made only of short words ("Dr. Li"): fall back to a plain substring test
        if any(n.lower() in window.lower() for n in names):
            return None
        return f"missing_name: open with the recipient's name ({names[0]})"
    for tok in tokens:
        if re.search(rf"(?<![\w]){re.escape(tok)}(?![\w])", window, re.IGNORECASE):
            return None
    return f"missing_name: open with the recipient's name ({sheet.salutation or names[0]}) in the first sentence"


def _is_placeholder_name(name: str) -> bool:
    """Stand-in addressees from facts ("there", "Doc", "team") that carry no real name."""
    words = re.findall(r"[^\W\d_]+", str(name or "").lower())
    return not words or all(w in _NAME_STOPWORDS for w in words)


def _count_ctas(text: str) -> int:
    verbal = len(_DIRECTIVE_RE.findall(text)) + len(_REVERSE_DIRECTIVE_RE.findall(text))
    # a numbered slot list ("Reply 1 for Wed, 2 for Thu") is one choice, not several asks
    numeric = 1 if _NUMERIC_DIRECTIVE_RE.search(text) else 0
    return verbal + numeric


def _uniq(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    for it in items:
        if it not in out:
            out.append(it)
    return out
