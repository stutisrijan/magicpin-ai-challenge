"""Deterministic message templates: render(facts, playbook) -> Draft. No LLM.

This is the always-available writer. The composer tries an LLM on top of it, but when
the free tier is rate-limited this draft is what the merchant receives, so every kind
gets hand-written copy:

    sentence 1   salutation + why-now hook (the trigger, with its anchor fact)
    body         1-2 anchor facts with exact numbers / sources, one merchant-specific
                 support fact, one compulsion lever
    last         exactly one CTA (binary ask, slot choice, CONFIRM, or one open question)

Rules the copy follows (and the tests enforce):
  * every number comes from a Fact text / value on the sheet (so it is in allowed_numbers);
    small counts (0-10) are the only numbers written freely, and a final grounding pass
    drops any optional sentence whose numbers are not in facts.allowed_numbers
  * no URLs, no snake_case, no taboo words, no "Dr. Dr.", no question before the CTA
  * customer-facing copy speaks as the business, never mentions Vera or magicpin, quotes
    prices only from the merchant's active offers and offers the exact slot labels
  * language follows facts.language: Hinglish (Roman code-mix, facts kept in English),
    Hindi-dominant Roman for "hi" customers, English (+ regional greeting) otherwise
  * phrasing varies across merchants by a stable hash, so identical inputs always give
    identical output while different merchants don't get identical skeletons
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.numbers import extract_number_tokens, normalize_number
from app.playbooks import canonical_kind, category_key
from app.schemas import (
    CTA_BINARY_CONFIRM,
    CTA_BINARY_YES_NO,
    CTA_MULTI_CHOICE_SLOT,
    CTA_OPEN_ENDED,
    Draft,
    Fact,
    FactSheet,
    Playbook,
)

log = logging.getLogger(__name__)

MERCHANT_MAX = 500          # spec: 250-500 chars merchant-facing
PLANNING_MAX = 760          # a drafted artifact may run longer
CUSTOMER_MAX = 420          # spec: 200-420 chars customer-facing

_PEOPLE = {"dentists": "patients", "salons": "clients", "restaurants": "customers", "gyms": "members",
           "pharmacies": "customers"}
_BIZ = {"dentists": "clinic", "salons": "salon", "restaurants": "restaurant", "gyms": "gym", "pharmacies": "pharmacy"}
_BIZ_NOUN = {"dentists": "dental clinic", "salons": "salon", "restaurants": "restaurant", "gyms": "gym",
             "pharmacies": "pharmacy"}
_BIZ_PLURAL = {"dentists": "dental clinics", "salons": "salons", "restaurants": "restaurants", "gyms": "gyms",
               "pharmacies": "pharmacies"}
_EMOJI = {"dentists": "🦷", "salons": "✨", "restaurants": "🍽️", "gyms": "💪", "pharmacies": ""}
# what a customer "comes in for" per category: (noun, Hindi gender "f"/"m")
_VISIT = {"dentists": ("visit", "f"), "salons": ("visit", "f"), "restaurants": ("visit", "f"),
          "gyms": ("session", "m"), "pharmacies": ("visit", "f")}
# curious-ask topic per category: (noun, Hindi gender)
_ASK_TOPIC = {"dentists": ("treatment", "m"), "salons": ("service", "f"), "restaurants": ("dish", "f"),
              "gyms": ("class", "f"), "pharmacies": ("product", "m")}

# first words of fact texts that read naturally in lower case mid-sentence
_LC_WORDS = {
    "Calls", "Profile", "Direction", "Leads", "Reviews", "Bookings", "Orders", "Footfall", "Crossed", "Plan",
    "Trial", "Due", "Last", "Next", "Routine", "Regular", "Monthly", "Oral-care", "Hair", "Follow-up", "Salon",
    "Dental", "Table", "Training", "Pharmacy", "Appointment", "Open", "Estimated", "Verification", "Window",
    "Current", "Delivery", "Peer", "Peers", "Previous", "Was", "Wedding", "Bridal", "Season", "Expected", "Likely",
    "Performance", "Sharp", "Seasonal", "Past", "Prefers", "Multi-center", "Maximum", "Standard", "Gym", "Most",
    "Worth", "Audit", "Reassess", "Move", "Pull", "Set", "Update", "Consider", "Add", "Apply", "Enable",
    "Position", "Run", "Push", "Pause", "Package", "Document", "On", "Yoga", "Workout", "Refill", "Visit",
    "A", "An", "The", "Your", "Specific", "Age", "New", "Recently", "Active", "Lapsed", "Has", "Check", "Keep", "Offer",
}
_KEEP_CASE = {"Pro", "Basic", "Free", "Premium", "Google", "CTR", "IPL", "ORS", "GST", "DCI", "JIDA", "IDA"}

_URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
_DATE_RE = re.compile(r"\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) \d{4}\b")
_PCT_RE = re.compile(r"[+\-]?\d+(?:\.\d+)?%")
_AGO_RE = re.compile(r"\((about \d+ months|\d+ days) ago\)")
_CLOCK_RE = re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b", re.IGNORECASE)
_DAY_RANGE_RE = re.compile(r"\b(mon|tue|wed|thu|fri|sat|sun)[a-z]*\s*-\s*(mon|tue|wed|thu|fri|sat|sun)[a-z]*\b", re.I)
_FESTIVE_RE = re.compile(r"\b(?:festival|festive|diwali|holi|christmas|new[- ]year|valentine'?s?|eid|navratri|"
                         r"wedding|pongal|onam|durga|ganesh)\b", re.IGNORECASE)
_TREND_STOP = {"near", "me", "price", "cost", "in", "for", "the", "a", "of", "best", "delhi", "mumbai", "bangalore",
               "bengaluru", "hyderabad", "chennai", "pune", "jaipur", "lucknow", "chandigarh", "ahmedabad",
               "searches", "offer", "offers"}

# numeric grounding (mirrors spec section 10 so the draft always passes the validator)
_NUM_MASKS = [
    re.compile(r"\b24\s?[x×/]\s?7\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:-|–|—|to)\s?\d{1,2}(?::\d{2})?\s?(?:am|pm)\b", re.IGNORECASE),
    re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b"),
    _CLOCK_RE,
]
_TIME_UNIT_AFTER = re.compile(r"^\s?-?\s?(?:mins?|minutes?|hrs?|hours?|h|ghante|ghanta|secs?|seconds?)\b", re.I)
_FREE_DURATIONS = {"15", "20", "24", "30", "45", "48", "60", "90"}


# --------------------------------------------------------------------------- small text helpers


def _lc(text: str) -> str:
    """Lower-case the first letter when the fact starts with a generic word ("Calls down ..." -> "calls down ...")."""
    if not text:
        return text
    first = text.split(" ", 1)[0].rstrip(":,")
    if first in _KEEP_CASE:
        return text
    if first in _LC_WORDS:
        return text[0].lower() + text[1:]
    return text


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _end(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text if text[-1] in ".!?:…" or _ends_with_emoji(text) else text + "."


def _ends_with_emoji(text: str) -> bool:
    return bool(text) and ord(text[-1]) > 0x2600


def _strip_end(text: str) -> str:
    return (text or "").strip().rstrip(".").strip()


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z'\"(])", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _after_colon(text: str) -> str:
    return text.split(":", 1)[1].strip() if ":" in (text or "") else (text or "").strip()


def _date_in(text: str) -> str:
    m = _DATE_RE.search(text or "")
    return m.group(0) if m else ""


def _short_date(date: str) -> str:
    """'1 Sep 2025' -> 'Sep 2025' (for 'with us since ...')."""
    parts = date.split()
    return " ".join(parts[1:]) if len(parts) == 3 else date


def _pct_in(text: str) -> str:
    m = _PCT_RE.search(text or "")
    return m.group(0) if m else ""


def _ago_in(text: str) -> str:
    m = _AGO_RE.search(text or "")
    return f"{m.group(1)} ago" if m else ""


def _join(items: list[str], conj: str = "and") -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} {conj} {items[1]}"
    return ", ".join(items[:-1]) + f" {conj} {items[-1]}"


def _num(value: Any) -> str:
    """Format a raw numeric value the way facts.py does (thousands separators, no trailing .0)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if v.is_integer():
        return f"{int(v):,}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _lead_int(text: str) -> int:
    m = re.match(r"\s*(\d[\d,]*)", text or "")
    return int(m.group(1).replace(",", "")) if m else 0


def _verbed(text: str) -> str:
    """"Calls down 50% over ..." -> "calls are down 50% over ..." (EN clauses need a verb)."""
    m = re.match(r"^(?P<subj>[A-Za-z][A-Za-z -]*?) (?P<dir>up|down) (?P<rest>\d.*)$", text or "")
    if not m:
        return _lc(text)
    subj = m.group("subj")
    verb = "is" if subj.lower() in ("ctr", "performance", "profile performance", "footfall") else "are"
    return _lc(f"{subj} {verb} {m.group('dir')} {m.group('rest')}")


def _trend_are(text: str) -> str:
    """"'x' searches up 45% YoY" -> "'x' searches are up 45% YoY"."""
    return re.sub(r"\bsearches (up|down)\b", r"searches are \1", text or "", count=1)


def _trend_keyword(text: str) -> str:
    """The quoted query of a trend fact minus filler words: "'clear aligners delhi' ..." -> "clear aligners"."""
    m = re.search(r"'([^']+)'", text or "")
    if not m:
        return ""
    words = [w for w in m.group(1).split() if w.lower() not in _TREND_STOP]
    return " ".join(words)


def _words(text: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z]{4,}", (text or "").lower()):
        out.add(w)
        out.add(w.rstrip("s"))
    return out


def _is_festive(text: str) -> bool:
    return bool(_FESTIVE_RE.search(text or ""))


def _beat_split(text: str) -> tuple[str, str]:
    """'Oct-Dec: wedding season' -> ('Oct-Dec', 'wedding season')."""
    if ": " in text:
        rng, note = text.split(": ", 1)
        if re.search(r"[A-Z][a-z]{2}", rng) and len(rng) <= 14:
            return rng.strip(), note.strip()
    return "", text.strip()


def _slot_pref_match(slots: list[str], pref: str) -> bool:
    """True when every slot label fits the stated preference (weekday / Saturday / evening / morning ...)."""
    if not slots or not pref:
        return False
    p = pref.lower()
    days = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    for s in slots:
        low = s.lower()
        day = next((d for d in days if re.search(rf"\b{d}", low)), None)
        clock = _CLOCK_RE.search(low)
        hour = None
        if clock:
            hm = re.match(r"(\d{1,2})", clock.group(0))
            hour = int(hm.group(1)) % 12 + (12 if "pm" in clock.group(0).lower() else 0) if hm else None
        if "weekday" in p and (day is None or days[day] > 4):
            return False
        if ("saturday" in p or "sat" in p.split()) and day != "sat":
            return False
        if "sunday" in p and day != "sun":
            return False
        if "evening" in p and (hour is None or hour < 16):
            return False
        if "morning" in p and (hour is None or hour >= 12):
            return False
        if "afternoon" in p and (hour is None or not 12 <= hour < 17):
            return False
    return any(k in p for k in ("weekday", "saturday", "sunday", "evening", "morning", "afternoon"))


def _ungrounded(text: str, allowed: set[str]) -> list[str]:
    """Numeric tokens in text that the fact sheet does not support (same rule as the validator)."""
    s = text or ""
    for rx in _NUM_MASKS:
        s = rx.sub(lambda m: " " * len(m.group(0)), s)
    bad: list[str] = []
    pos = 0
    for tok in extract_number_tokens(s):
        m = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(tok)).search(s, pos)
        after = s[m.end(): m.end() + 12] if m else ""
        if m:
            pos = m.end()
        norm = normalize_number(tok)
        if norm is None or norm in allowed:
            continue
        low = tok.lower()
        marked = "%" in low or "₹" in low or low.startswith(("rs", "inr")) or low.rstrip().endswith("x")
        if not marked:
            if norm.isdigit() and int(norm) <= 10:
                continue
            if norm in _FREE_DURATIONS and _TIME_UNIT_AFTER.match(after):
                continue
        bad.append(tok.strip())
    return bad


# --------------------------------------------------------------------------- message model


@dataclass
class _Msg:
    first: str                                    # salutation + why-now hook (always kept)
    cta: str                                      # the one ask, always the last sentence
    parts: list[tuple[str, int]] = field(default_factory=list)   # (sentence, priority 0 keep .. 3 drop first)
    cta_type: str = CTA_BINARY_YES_NO
    offer: str = ""
    why: str = ""                                 # rationale: what triggered the message
    anchor: str = ""                              # rationale: the anchor fact used
    lever: str = ""                               # rationale: the compulsion lever
    max_len: int = 0
    newline_parts: bool = False                   # planning drafts keep their line breaks

    def add(self, text: str, prio: int = 2) -> None:
        if text and text.strip():
            self.parts.append((text, prio))


class _W:
    """Read-only view over a FactSheet with the helpers every renderer needs."""

    def __init__(self, facts: FactSheet, pb: Playbook) -> None:
        self.f = facts
        self.pb = pb
        self.kind = canonical_kind(facts.kind or pb.kind)
        self.cat = category_key(facts.category_slug)
        self.customer = facts.scope == "customer"
        self.yoga = "yoga" in (facts.merchant_name or "").lower()
        code = (facts.language.code or "en").lower()
        self.code = code
        self.hi = code == "hi" and self.customer
        self.hg = code in ("hinglish", "hi") and not self.hi
        greeting = (facts.language.greeting or "").strip()
        self.regional = code.endswith("-en") and bool(greeting) and greeting.lower() not in ("hi", "hello")
        self.greeting = greeting if self.regional else ""
        self.people = _PEOPLE.get(self.cat, "customers")
        self.biz = "studio" if self.yoga else _BIZ.get(self.cat, "business")
        self.biz_noun = "yoga studio" if self.yoga else _BIZ_NOUN.get(self.cat, "business")
        self.biz_plural = "yoga studios" if self.yoga else _BIZ_PLURAL.get(self.cat, "businesses")
        self.taboos = [t.lower() for t in (facts.taboos or []) if t and t.strip()]
        self.allowed = set(facts.allowed_numbers or set())
        self._seed = f"{facts.merchant_id}|{facts.customer_id or ''}|{self.kind}|{facts.trigger_id}"

    # -- language ---------------------------------------------------------------------
    def L(self, en: str, hg: str | None = None, hi: str | None = None) -> str:
        if self.hi:
            return hi if hi is not None else (hg if hg is not None else en)
        if self.hg:
            return hg if hg is not None else en
        return en

    def join(self, items: list[str]) -> str:
        return _join(items, self.L("and", "aur", "aur"))

    def join_or(self, items: list[str]) -> str:
        return _join(items, self.L("or", "ya", "ya"))

    def pick(self, options: list[str], salt: str = "") -> str:
        options = [o for o in options if o]
        if not options:
            return ""
        h = int(hashlib.md5(f"{self._seed}|{salt}".encode()).hexdigest(), 16)
        return options[h % len(options)]

    # -- facts ------------------------------------------------------------------------
    def fact(self, key: str) -> Fact | None:
        for f in self.f.all_facts():
            if f.key == key and self.safe(f.text):
                return f
        return None

    def t(self, key: str) -> str:
        f = self.fact(key)
        return f.text if f else ""

    def v(self, key: str) -> Any:
        f = self.fact(key)
        return f.value if f else None

    def facts(self, prefix: str) -> list[Fact]:
        return [f for f in self.f.all_facts() if f.key.startswith(prefix) and self.safe(f.text)]

    def safe(self, text: str) -> bool:
        low = (text or "").lower()
        return not any(re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", low) for t in self.taboos)

    def grounded(self, text: str) -> bool:
        return not _ungrounded(text, self.allowed)

    def num_ok(self, value: Any) -> bool:
        n = normalize_number(str(value))
        return n is not None and (n in self.allowed or (n.isdigit() and int(n) <= 10))

    # -- addressing -------------------------------------------------------------------
    @property
    def sal(self) -> str:
        return (self.f.salutation or self.f.recipient_name or "").strip()

    def m_open(self) -> str:
        """Merchant-facing opener: "Dr. Meera,", "Hi Lakshmi,", "Suresh ji,", "Vanakkam Padma,"."""
        sal = self.sal or self.f.owner_name or ""
        if not sal or sal == "there":
            return f"{self.greeting or 'Hi'} there,"
        if self.regional:
            return f"{self.greeting} {sal},"
        if self.cat == "dentists" or sal.startswith("Dr.") or sal.endswith(" team"):
            return f"{sal},"
        if self.hg:
            return self.pick([f"{sal},", f"Hi {sal},", f"{sal} ji,"], "open")
        return self.pick([f"{sal},", f"Hi {sal},"], "open")

    def c_greet(self) -> str:
        if self.regional:
            return self.greeting
        if self.hi or (self.f.language.greeting or "").lower() == "namaste":
            return "Namaste"
        return "Hi"

    def c_open(self) -> str:
        """Customer-facing opener. Family relays ("Sharma ji" via son) get a bare Namaste."""
        greet = self.c_greet()
        if self.f.about_name and not self.f.recipient_name:
            return f"{greet} 🙏" if greet == "Namaste" else f"{greet},"
        sal = self.sal
        if not sal:
            return f"{greet},"
        if self.hi and not sal.endswith(" ji") and not sal.startswith(("Mr.", "Mrs.", "Ms.", "Dr.")):
            return f"{greet} {sal} ji,"
        return f"{greet} {sal},"

    @property
    def signer(self) -> str:
        s = (self.f.signer or "").strip()
        if not s or s in ("there", "Doc"):
            s = self.f.merchant_name or ""
        return s

    @property
    def biz_name(self) -> str:
        return self.f.merchant_name or self.signer

    @property
    def emoji(self) -> str:
        if self.hi or self.f.language.greeting == "Namaste":
            return ""
        return "🧘" if self.yoga else _EMOJI.get(self.cat, "")

    def intro(self, hg_noun: str = "ek message", hi_noun: str | None = None) -> str:
        """A complete intro sentence: "Dr. Meera's Dental Clinic here 🦷" / "X ki taraf se ek reminder."."""
        e = self.emoji
        if not self.biz_name:
            return ""
        if self.hi or self.hg:
            noun = hi_noun if (self.hi and hi_noun) else hg_noun
            return f"{self.biz_name} ki taraf se {noun}" + (f" {e}" if e else ".")
        lead = self.pick([f"{self.signer} here", f"this is {self.signer}"], "intro")
        return lead + (f" {e}" if e else ".")

    # -- merchant data helpers --------------------------------------------------------
    def pos_review(self) -> Fact | None:
        cands = [f for f in self.facts("review.") if " praise" in f.text]
        return max(cands, key=lambda f: _lead_int(f.text)) if cands else None

    def peer(self, direction: str, metrics: tuple[str, ...] = ("ctr", "calls", "views", "directions")) -> Fact | None:
        for metric in metrics:
            f = self.fact(f"perf.{metric}_vs_peer")
            if f and f"{direction} peers" in f.text:
                return f
        return None

    def offer_for(self, hint: str = "", prefer_free: bool = False, strict: bool = False) -> str:
        offers = [o for o in self.f.offers_active if self.safe(o)]
        if not offers:
            return ""
        hw = _words(hint)
        for o in offers:
            if hw & _words(o):
                return o
        if strict:
            return ""
        if prefer_free:
            for o in offers:
                if "free" in o.lower():
                    return o
        return offers[0]

    def priced_offer(self) -> str:
        return next((o for o in self.f.offers_active if "₹" in o and self.safe(o)), "")

    def catalog_idea(self, hint: str = "", strict: bool = False) -> str:
        ideas = [o for o in self.f.catalog_offers if "@" in o and self.safe(o) and self.grounded(o)]
        if not ideas:
            return ""
        hw = _words(hint)
        for o in ideas:
            if hw & _words(o):
                return o
        return "" if strict else ideas[0]

    def agg_count(self, key: str) -> str:
        v = self.v(key)
        return _num(v) if v is not None else ""

    def pref(self) -> str:
        return _lc(self.t("customer.preferred_slots").removeprefix("Prefers ").strip())

    def visits(self) -> tuple[int, str]:
        """(visit count, first-visit date) from 'N visits since D'."""
        t = self.t("customer.visits")
        m = re.match(r"(\d+) visits? since (.+)$", t)
        if m:
            return int(m.group(1)), m.group(2).strip()
        return _lead_int(t), ""

    def retention(self) -> tuple[str, str, str] | None:
        """('32%', '3-month', '55%') from '32% 3-month retention vs 55% peer average (...)'."""
        for f in self.facts("agg.retention"):
            m = re.match(r"(\d+(?:\.\d+)?%) (\S+) retention(?: vs (\d+(?:\.\d+)?%) peer average)?", f.text)
            if m:
                return m.group(1), m.group(2), m.group(3) or ""
        return None


def _cta_draft(w: _W, en: str, hg: str) -> str:
    """Binary "want me to X?" ask in the right language, with a little phrasing variety."""
    return w.L(
        en=w.pick([f"Want me to {en}?", f"Should I {en}?", f"Shall I {en}?"], "cta"),
        hg=w.pick([f"Kya main {hg}?", f"Bolein toh main {hg}?", f"Main {hg}?"], "cta"),
    )


# --------------------------------------------------------------------------- merchant-facing renderers


def _m_research(w: _W) -> _Msg:
    title = w.t("digest.title")
    if not title:
        return _m_generic(w)
    src, stat, trial = w.t("digest.source"), w.t("digest.stat"), w.t("digest.trial_n")
    summary, act, seg = w.t("digest.summary"), w.t("digest.actionable"), w.t("digest.segment")
    o = w.m_open()
    cohort = w.agg_count("agg.high_risk_adult_count") if "high-risk" in f"{seg} {title}".lower() else ""
    src_en = src or "this week's digest"
    src_hg = src or "is hafte ke digest"
    if cohort:
        first = w.L(
            en=w.pick([f"{o} {src_en} has a finding that speaks to your {cohort} high-risk adult patients: {title}.",
                       f"{o} one from {src_en} for your {cohort} high-risk adult patients: {title}."], "hook"),
            hg=w.pick([f"{o} {src_hg} mein aapke {cohort} high-risk adult patients ke kaam ki finding aayi hai: {title}.",
                       f"{o} {src_hg} ka naya item seedha aapke {cohort} high-risk adult patients se juda hai: {title}."],
                      "hook"))
    else:
        yours = {"dentists": "your practice", "pharmacies": "your counter"}.get(w.cat, f"your {w.biz}")
        first = w.L(
            en=w.pick([f"{o} {src_en} has one worth two minutes of your time: {title}.",
                       f"{o} new in {src_en}, and relevant to {yours}: {title}."], "hook"),
            hg=w.pick([f"{o} {src_hg} mein ek kaam ki cheez aayi hai: {title}.",
                       f"{o} {src_hg} ka ek naya update aapke {w.biz} ke kaam ka hai: {title}."], "hook"))
    m = _Msg(first=first, cta="", why=f"new research digest item ({src or 'category digest'})",
             lever="curiosity plus reciprocity (Vera pulls it and drafts the customer note)")
    sents = _sentences(summary)
    anchor = stat or (sents[0].rstrip(".") if sents else "")
    if anchor and trial and trial.split("-")[0] not in anchor:
        anchor = f"{anchor} ({trial})"
    m.add(anchor, 0)
    m.anchor = _strip_end(anchor) or title
    if cohort:
        m.anchor += f"; merchant's {cohort} high-risk adult patients"
    details = [s for s in sents if _strip_end(s) not in (stat or "") and _strip_end(s) not in anchor]
    for d in details[:1]:
        m.add(d, 1 if cohort else 2)
    if act:
        m.add(w.L(en=f"Practical takeaway: {_lc(_strip_end(act))}.",
                  hg=f"Aapke liye practical step: {_lc(_strip_end(act))}."), 2)
    audience = {"dentists": "patient", "pharmacies": "customer", "salons": "client", "gyms": "member"}.get(
        w.cat, "customer")
    m.cta = _cta_draft(w, en=f"pull the key points and draft a {audience}-education WhatsApp you can forward",
                       hg=f"iske key points nikaal ke aapke {w.people} ke liye ek WhatsApp note draft kar doon")
    m.offer = w.pb.offer
    return m


def _m_regulation(w: _W) -> _Msg:
    title = w.t("digest.title")
    deadline = w.t("trigger.deadline")
    if not title and not deadline:
        return _m_generic(w)
    src, stat, summary, act = w.t("digest.source"), w.t("digest.stat"), w.t("digest.summary"), w.t("digest.actionable")
    o = w.m_open()
    what = title or _strip_end(deadline)
    frm = (f" from the {src}" if not src.lower().startswith("the ") else f" from {src}") if src else ""
    first = w.L(
        en=w.pick([f"{o} compliance heads-up{frm}: {what}.", f"{o} a rule change to get ahead of{frm}: {what}."],
                  "hook"),
        hg=f"{o} {src + ' se ' if src else ''}ek zaroori compliance update: {what}.")
    m = _Msg(first=first, cta="", why=f"regulation change ({src or 'category digest'})", anchor=what,
             lever="loss aversion (deadline) plus effort externalisation")
    if stat:
        m.add(stat, 0)
        m.anchor = _strip_end(stat)
    rest = [s for s in _sentences(summary) if _strip_end(s) != _strip_end(stat)]
    if rest:
        m.add(" ".join(rest[:2]), 1)
    days = re.search(r"\((\d[\d,]*) days away\)", deadline)
    if days:
        m.add(w.L(en=f"That leaves {days.group(1)} days to get the setup checked and documented.",
                  hg=f"Setup check aur document karne ke liye {days.group(1)} din bache hain."), 1)
    elif deadline and _date_in(deadline) and _date_in(deadline) not in f"{title} {stat}":
        m.add(w.L(en=f"{_strip_end(deadline)}.", hg=f"{_strip_end(deadline)}."), 1)
    if act:
        m.add(w.L(en=f"The practical step: {_lc(_strip_end(act))}.", hg=f"Practical step: {_lc(_strip_end(act))}."), 2)
    audit = {"dentists": "X-ray and SOP audit checklist", "pharmacies": "register audit checklist",
             "restaurants": "packaging-cost and GST checklist"}.get(w.cat, "compliance checklist")
    m.cta = _cta_draft(w, en=f"send a 5-point {audit} you can run with your team this week",
                       hg=f"aapke setup ke liye ek 5-point {audit} bhej doon")
    m.offer = w.pb.offer
    return m


def _m_cde(w: _W) -> _Msg:
    title = w.t("digest.title")
    if not title:
        return _m_generic(w)
    src, date, summary = w.t("digest.source"), w.t("digest.date"), w.t("digest.summary")
    act, credits, fee = w.t("digest.actionable"), w.t("digest.credits"), w.t("digest.fee")
    o = w.m_open()
    when = f", {_lc(date)}" if date else ""
    cr = f" ({credits})" if credits else ""
    first = w.L(
        en=w.pick([f"{o} {src or 'your chapter'} has a session worth blocking: {title}{when}{cr}.",
                   f"{o} one for your calendar: {title}{when}{cr}" + (f", listed on the {src}." if src else ".")],
                  "hook"),
        hg=f"{o} {src or 'chapter'} ka ek session aapke calendar ke layak hai: {title}{when}{cr}.")
    m = _Msg(first=first, cta="", why=f"CDE / learning opportunity ({src or 'digest'})",
             anchor=f"{title}{when}{cr}", lever="curiosity plus effort externalisation")
    m.add(summary, 1)
    if act and ("₹" in act or "free" in act.lower()):
        m.add(act, 1)
    elif fee:
        m.add(fee, 1)
    city = (w.f.city or "").lower()
    if city and city in f"{src} {title}".lower():
        m.add(w.L(en=f"It's your own {w.f.city} chapter, so an easy evening to fit in.",
                  hg=f"Yeh aapka apna {w.f.city} chapter hai, toh shaam ko aana easy rahega."), 3)
    m.cta = _cta_draft(w, en="block the slot in your calendar and send the registration details on the day",
                       hg="calendar mein slot block karke registration details bhej doon")
    m.offer = w.pb.offer
    return m


def _m_supply(w: _W) -> _Msg:
    batches = w.v("trigger.batches") or []
    if not batches:
        bt = _after_colon(w.t("trigger.batches"))
        batches = [b.strip() for b in bt.split(",") if b.strip()] if bt else []
    mol = _after_colon(w.t("trigger.molecule"))
    mfr = _after_colon(w.t("trigger.manufacturer"))
    title, src, summary = w.t("digest.title"), w.t("digest.source"), w.t("digest.summary")
    if not batches and not title:
        return _m_generic(w)
    o = w.m_open()
    by = f" by {mfr}" if mfr else ""
    at = f" ({src})" if src else ""
    if batches:
        b = _join([str(x) for x in batches])
        what = f"{mol} batches {b}{by}" if mol else f"batches {b}{by}"
        first = w.L(en=w.pick([f"{o} urgent: voluntary recall on {what}{at}.",
                               f"{o} urgent recall to act on today: {what}{at}."], "hook"),
                    hg=f"{o} urgent recall alert: {what}{at}.")
        anchor = f"recall of {what}"
    else:
        what = re.sub(r"\s+by manufacturer [A-Z]\b", "", title)
        what = _lc(what.split(":", 1)[1].strip()) if what.lower().startswith("voluntary recall:") else what
        first = w.L(en=f"{o} urgent: voluntary recall on {what}{at}." if "recall" in title.lower() else
                    f"{o} urgent: {what}{at}.",
                    hg=f"{o} urgent recall alert: {what}{at}.")
        anchor = what
    m = _Msg(first=first, cta="", why=f"supply / recall alert ({src or 'regulator'})", anchor=anchor,
             lever="urgency (loss aversion) plus effort externalisation of the whole workflow")
    clean = re.sub(r"\s*\([^)]*\)", "", summary)
    sents = _sentences(clean)
    risk = [s for s in sents if "risk" in s.lower()]
    other = [s for s in sents if s not in risk]
    if other:
        m.add(other[0], 1)
    if risk:
        m.add(risk[0], 1)
    if len(other) > 1:
        m.add(other[1], 3)
    rx = w.t("agg.chronic_rx_count")
    if rx:
        pool = rx.replace(" on repeat refills", "")
        m.add(w.L(en=f"I can check your {pool} against these batches right away.",
                  hg=f"Aapke {pool} ki list main abhi in batches se match kar sakti hoon."), 0)
        m.anchor += f"; pool of {pool}"
    else:
        m.add(w.L(en="I can filter your repeat-prescription list for these batches right away.",
                  hg="Aapki repeat-prescription list main abhi in batches ke liye filter kar sakti hoon."), 0)
    m.cta = _cta_draft(w, en="draft the customer WhatsApp note and the replacement-pickup steps now",
                       hg="customers ke liye WhatsApp note aur replacement-pickup steps abhi draft kar doon")
    m.offer = w.pb.offer
    return m


def _m_seasonal(w: _W) -> _Msg:
    trends = w.t("trigger.trends")
    title, src, act = w.t("digest.title"), w.t("digest.source"), w.t("digest.actionable")
    if not trends and not title:
        return _m_generic(w)
    season = w.t("trigger.season_note").removeprefix("Season:").strip()
    o = w.m_open()
    at = f" ({src})" if src else ""
    if trends:
        first = w.L(en=f"{o} the {season + ' ' if season else ''}demand shift is here{at}: {_lc(trends)}.",
                    hg=f"{o} {season + ' ka ' if season else ''}demand shift shuru ho gaya hai{at}: {_lc(trends)}.")
        anchor = trends
    else:
        first = w.L(en=f"{o} the seasonal demand shift is here{at}: {title}.",
                    hg=f"{o} seasonal demand shift aa gaya hai{at}: {title}.")
        anchor = title
    m = _Msg(first=first, cta="", why=f"seasonal demand shift ({src or 'category data'})", anchor=anchor,
             lever="loss aversion (missed seasonal demand) plus effort externalisation")
    if act:
        m.add(w.L(en=f"The one move this week: {_lc(_strip_end(act))}.",
                  hg=f"Is hafte ka ek move: {_lc(_strip_end(act))}."), 0)
    offer = w.offer_for("delivery home")
    if offer:
        m.add(w.L(en=f"Your {offer} offer is the natural hook for these seasonal baskets.",
                  hg=f"Aapka {offer} offer in seasonal orders ke liye perfect hook hai."), 1)
    rep = w.t("agg.repeat_customer_pct")
    if rep:
        head = rep.split(" vs ")[0]
        m.add(w.L(en=f"With {_lc(head)}, a heads-up to your regulars will land well.",
                  hg=f"{head} ke saath, regulars ko ek heads-up bhejna kaam karega."), 2)
    if w.cat == "pharmacies":
        en, hg = ("draft the counter-display list plus a WhatsApp broadcast for your regulars",
                  "counter-display list aur regulars ke liye WhatsApp broadcast draft kar doon")
    else:
        en, hg = ("draft a seasonal post plus a WhatsApp broadcast for your regulars",
                  "ek seasonal post aur regulars ke liye WhatsApp broadcast draft kar doon")
    m.cta = _cta_draft(w, en=en, hg=hg)
    m.offer = w.pb.offer
    return m


def _m_trend(w: _W) -> _Msg:
    trend = w.t("trigger.trends") or w.t("trend.top") or w.t("digest.title")
    if not trend:
        return _m_generic(w)
    o = w.m_open()
    first = w.L(en=f"{o} a search trend worth acting on this week: {_trend_are(trend)}.",
                hg=f"{o} ek search trend jo aapke kaam ka hai: {_trend_are(trend)}.")
    m = _Msg(first=first, cta="", why="category trend movement", anchor=trend,
             lever="curiosity plus loss aversion (competitors capture the demand)")
    src = w.t("digest.source")
    act = w.t("digest.actionable")
    if act:
        m.add(w.L(en=f"{src + ' suggests' if src else 'The practical move'}: {_lc(_strip_end(act))}.",
                  hg=f"Practical move: {_lc(_strip_end(act))}."), 2)
    kw = _trend_keyword(trend) or trend
    offer = w.offer_for(kw, strict=True)
    if offer:
        m.add(w.L(en=f"Your {offer} fits it well, and your profile copy should say so.",
                  hg=f"Aapka {offer} is trend pe fit baithta hai; profile mein yeh dikhna chahiye."), 1)
    else:
        idea = w.catalog_idea(kw, strict=True)
        if idea:
            m.add(w.L(en=f"A clear service-and-price offer such as {idea} would put you in front of that demand.",
                      hg=f"{idea} jaisa ek clear offer aapko is demand ke saamne le aayega."), 1)
    m.cta = _cta_draft(w, en="update your Google profile line and draft a post around it",
                       hg="aapki Google profile line update karke iska ek post draft kar doon")
    m.offer = w.pb.offer
    return m


_FESTIVAL_MOVE = {
    "gyms": ("Line up a shape-up offer for past members now, so it's live the day the window opens.",
             "Past members ke liye abhi se ek shape-up offer ready rakhein, taaki window khulte hi live ho."),
    "pharmacies": ("Stock up and plan a health-check push before the season starts.",
                   "Season shuru hone se pehle stock aur ek health-check push plan kar lein."),
    "salons": ("Build the package now, so it's live before bookings peak.",
               "Package abhi se bana lein, taaki bookings peak se pehle live ho."),
    "restaurants": ("Set up the festive and bulk-order menu now, so orders come to you first.",
                    "Festive aur bulk-order menu abhi set kar lein, taaki orders pehle aapke paas aayein."),
    "dentists": ("Blocking a few dedicated slots now catches that demand before other clinics do.",
                 "Abhi se kuch dedicated slots block karne se yeh demand pehle aapko milegi."),
}


def _m_festival(w: _W) -> _Msg:
    fest = w.fact("trigger.festival")
    if fest and not w.f.payload_is_placeholder:
        return _m_festival_named(w, fest)
    beats = [f.text for f in w.facts("trigger.season_note")]
    if not beats:
        return _m_generic(w)
    o = w.m_open()
    rng0, note0 = _beat_split(beats[0])
    if _is_festive(note0) and rng0:
        first = w.L(en=w.pick([f"{o} the next festive window for {w.biz_plural} is {rng0}: {note0}.",
                               f"{o} worth planning ahead: {rng0} is the festive window for {w.biz_plural} ({note0})."],
                              "hook"),
                    hg=f"{o} {w.biz_plural} ke liye agla festive window {rng0} hai: {note0}.")
    elif rng0:
        first = w.L(en=w.pick([f"{o} {rng0} is the season to plan for right now: {note0}.",
                               f"{o} the {rng0} season is here for {w.biz_plural}: {note0}."], "hook"),
                    hg=f"{o} {w.biz_plural} ke liye abhi {rng0} ka season chal raha hai: {note0}.")
    else:
        first = w.L(en=f"{o} the season is turning: {note0}.", hg=f"{o} season badal raha hai: {note0}.")
    m = _Msg(first=first, cta="", why=f"upcoming festive / seasonal window ({rng0 or 'category calendar'})",
             anchor=beats[0], lever="loss aversion (plan before the rush) plus effort externalisation")
    if len(beats) > 1:
        rng1, note1 = _beat_split(beats[1])
        m.add(w.L(en=f"Right now it's {rng1}: {note1}." if rng1 else f"Right now: {note1}.",
                  hg=f"Abhi {rng1} chal raha hai: {note1}." if rng1 else f"Abhi: {note1}."), 2)
    move = _FESTIVAL_MOVE.get(w.cat, ("Planning it now puts you ahead of the rush.",
                                      "Abhi plan karne se aap rush se aage rahenge."))
    m.add(w.L(en=move[0], hg=move[1]), 1)
    offer = w.offer_for(note0)
    if offer:
        m.add(w.L(en=f"Your {offer} can anchor it.", hg=f"Aapka {offer} iska centrepiece ban sakta hai."), 1)
    else:
        idea = w.catalog_idea(note0)
        if idea:
            m.add(w.L(en=f"Something like {idea} could anchor it.",
                      hg=f"{idea} jaisa offer iska centrepiece ban sakta hai."), 1)
    _add_strength(w, m, prio=3, style="plain")
    en_cta = {"gyms": "draft the shape-up offer post and a comeback note to past members",
              "pharmacies": "draft the seasonal health-check post and a WhatsApp note for regulars",
              "restaurants": "draft the festive menu post and a WhatsApp broadcast"}.get(
        w.cat, "draft the seasonal plan and a post for your review")
    hg_cta = {"gyms": "shape-up offer post aur past members ke liye comeback note draft kar doon",
              "pharmacies": "seasonal health-check post aur regulars ke liye WhatsApp note draft kar doon",
              "restaurants": "festive menu post aur WhatsApp broadcast draft kar doon"}.get(
        w.cat, "seasonal plan aur ek post aapke review ke liye draft kar doon")
    m.cta = _cta_draft(w, en=en_cta, hg=hg_cta)
    m.offer = w.pb.offer
    return m


def _m_festival_named(w: _W, fest: Fact) -> _Msg:
    val = fest.value if isinstance(fest.value, dict) else {}
    name = str(val.get("festival") or fest.text.split(" on ")[0]).strip()
    date = _date_in(fest.text)
    days_f = w.fact("trigger.days_until")
    days = days_f.value if days_f and isinstance(days_f.value, int) else None
    o = w.m_open()
    on = f" on {date}" if date else ""
    if days is not None and days > 45:
        first = w.L(
            en=w.pick([f"{o} {name} is{on}, {days} days out, so this is the planning window, not the rush.",
                       f"{o} {name} falls{on} ({days} days to go): early enough to plan it properly."], "hook"),
            hg=f"{o} {name}{' ' + date + ' ko' if date else ''} hai, yaani {days} din baaki; rush se pehle plan karne "
               f"ka yahi sahi time hai.")
    elif days is not None:
        first = w.L(en=f"{o} {name} is {days} days away{f' ({date})' if date else ''}, so the window to get a package "
                       f"live is now.",
                    hg=f"{o} {name} sirf {days} din door hai{f' ({date})' if date else ''}, package live karne ka time "
                       f"abhi hai.")
    else:
        first = w.L(en=f"{o} {name} is coming up{on}, and it's worth planning for now.",
                    hg=f"{o} {name}{' ' + date + ' ko' if date else ''} aa raha hai; abhi se plan karna worth hai.")
    m = _Msg(first=first, cta="", why=f"upcoming festival ({name}{on})", anchor=fest.text,
             lever="early-planning advantage (loss aversion) plus effort externalisation")
    season = w.t("seasonal.note")
    if season:
        rng, note = _beat_split(season)
        m.add(w.L(en=f"It lands in a key window for {w.biz_plural}: {rng}, {note}." if rng else
                  f"Seasonal pattern: {note}.",
                  hg=f"Yeh {w.biz_plural} ke liye ek bade window mein aata hai ({rng}: {note})." if rng else
                  f"Season: {note}."), 1)
        m.anchor += f"; {season}"
    offers = [x for x in w.f.offers_active if w.safe(x)]
    if offers:
        lead = max(offers, key=lambda x: _lead_int(re.sub(r"^.*₹", "", x)) if "₹" in x else 0)
        extra = next((x for x in offers if x != lead), "")
        m.add(w.L(en=f"A {name} combo built on your {lead}" + (f", with {extra} as the add-on," if extra else "")
                     + " gives festive shoppers a ready choice.",
                  hg=f"Aapke {lead}" + (f" aur {extra}" if extra else "") + f" se ek {name} combo ban sakta hai, "
                     f"taaki customers ke paas ready option ho."), 0)
    else:
        idea = w.catalog_idea(season)
        if idea:
            m.add(w.L(en=f"Something like {idea} could anchor a {name} package.",
                      hg=f"{idea} jaisa offer {name} package ka centrepiece ban sakta hai."), 1)
    if days is not None and days > 45:
        m.add(w.L(en="Set it up now and it's live well before the rush, with no scramble later.",
                  hg="Abhi set ho jaye toh rush se kaafi pehle live rahega, baad mein bhaag-daud nahi."), 2)
    m.cta = _cta_draft(w, en=f"draft the {name} package post for your review",
                       hg=f"aapke review ke liye {name} package ka post draft kar doon")
    m.offer = w.pb.offer
    return m


def _m_ipl(w: _W) -> _Msg:
    match = _after_colon(w.t("trigger.match"))
    if not match:
        return _m_ipl_season(w)
    venue = w.t("trigger.venue").split(",")[0].strip()
    clock = _CLOCK_RE.search(w.t("trigger.match_time"))
    clock_s = clock.group(0) if clock else ""
    weeknight = w.v("trigger.weeknight")
    summary, src = w.t("digest.summary"), w.t("digest.source")
    sents = _sentences(summary)
    weekend_s = next((s for s in sents if "saturday" in s.lower() and "%" in s), "")
    wk_s = next((s for s in sents if "weeknight" in s.lower() and "%" in s), "")
    o = w.m_open()
    at = f" at {venue}" if venue else ""
    tm = f", {clock_s}" if clock_s else ""
    offer = w.offer_for("pizza thali combo biryani")
    m = _Msg(first="", cta="", why=f"IPL match today ({match}{tm})", lever="loss aversion plus effort externalisation")
    if weeknight is False:
        where = f" ({venue}{tm})" if venue else tm
        m.first = w.L(en=w.pick([f"{o} {match}{at} tonight{tm}, and it's a weekend game, so play it differently.",
                                 f"{o} heads-up for tonight: {match}{where} is a weekend match."], "hook"),
                      hg=f"{o} aaj raat {match}{where} hai, aur yeh weekend match hai.")
        pct = _pct_in(weekend_s)
        if weekend_s and pct:
            by = f"{src}: " if src else ""
            m.add(w.L(en=f"{by}on Saturday match nights restaurant covers drop {pct.lstrip('+-')}, as fans watch at home.",
                      hg=f"{by}Saturday match nights pe restaurant covers {pct.lstrip('+-')} gir jaate hain, log ghar pe "
                         f"dekhte hain."), 0)
            m.anchor = _strip_end(weekend_s)
        elif weekend_s:
            m.add(f"{src}: {_strip_end(weekend_s)}." if src else _end(weekend_s), 0)
            m.anchor = _strip_end(weekend_s)
        m.add(w.L(en="So skip the dine-in promo tonight and go delivery-first.",
                  hg="Isliye aaj dine-in promo skip karke delivery pe focus karein."), 0)
        wk_pct = _pct_in(wk_s)
        if offer:
            weekday_only = bool(_DAY_RANGE_RE.search(offer)) or "weekday" in offer.lower()
            if weekday_only:
                m.add(w.L(en=f"Your {offer} doesn't cover tonight anyway, so save it for the next weeknight match"
                             + (f", where the same data shows {wk_pct} covers." if wk_pct else "."),
                          hg=f"Aapka {offer} aaj valid nahi, use agle weeknight match ke liye rakhein"
                             + (f", wahan covers {wk_pct} badhte hain." if wk_pct else ".")), 0)
            else:
                m.add(w.L(en=f"Push your {offer} as tonight's delivery special instead.",
                          hg=f"Aaj apna {offer} delivery special ki tarah push karein."), 1)
        dv, dn = w.v("agg.delivery_orders_30d"), w.v("agg.dine_in_orders_30d")
        if dv and dn:
            m.add(w.L(en=f"Your own mix backs it: {_num(dv)} delivery vs {_num(dn)} dine-in orders in the last 30 days.",
                      hg=f"Aapka apna data bhi yahi kehta hai: last 30 days mein {_num(dv)} delivery vs {_num(dn)} "
                         f"dine-in orders."), 2)
        m.cta = _cta_draft(w, en="draft a delivery-only banner and an Insta story for tonight",
                           hg="aaj raat ke liye delivery-only banner aur Insta story draft kar doon")
        m.offer = f"draft a delivery-only banner and an Insta story for tonight's {match} match"
    else:
        m.first = w.L(en=f"{o} {match}{at} tonight{tm}, a weeknight game, which is when match-night offers pay off.",
                      hg=f"{o} aaj {match}{at}{tm} hai, weeknight match, aur aise din match-night offers chalte hain.")
        if wk_s:
            m.add(f"{src}: {_strip_end(wk_s)}." if src else _end(wk_s), 0)
            m.anchor = _strip_end(wk_s)
        combo = next((c for c in w.f.catalog_offers if "match" in c.lower() and w.safe(c) and w.grounded(c)), "")
        if combo:
            m.add(w.L(en=f"A {combo} is the format that works on nights like this" +
                         (f", running alongside your {offer}." if offer else "."),
                      hg=f"{combo} jaisa format aise din best chalta hai" + (f", aapke {offer} ke saath." if offer
                                                                             else ".")), 1)
        elif offer:
            m.add(w.L(en=f"Push your {offer} as the match-night special.",
                      hg=f"Apna {offer} match-night special ki tarah push karein."), 1)
        m.cta = _cta_draft(w, en="set up the match-night post and an Insta story for today",
                           hg="aaj ke liye match-night post aur Insta story set kar doon")
        m.offer = f"draft a match-night combo post and an Insta story for today's {match} match"
    if not m.anchor:
        m.anchor = f"{match}{at}{tm}"
    return m


def _m_ipl_season(w: _W) -> _Msg:
    """IPL trigger without match details: season-level advice from the digest (no invented fixture)."""
    sents = _sentences(w.t("digest.summary"))
    weekend_s = next((x for x in sents if "saturday" in x.lower() and "%" in x), "")
    wk_s = next((x for x in sents if "weeknight" in x.lower() and "%" in x), "")
    if not weekend_s and not wk_s:
        return _m_generic(w)
    src = w.t("digest.source")
    o = w.m_open()
    wk_pct, we_pct = _pct_in(wk_s), _pct_in(weekend_s).lstrip("+-")
    facts_en = _join([f"weeknight matches drive {wk_pct} covers" if wk_pct else "",
                      f"Saturday matches cut covers {we_pct}" if we_pct else ""])
    facts_hg = _join([f"weeknight matches pe covers {wk_pct} badhte hain" if wk_pct else "",
                      f"Saturday matches pe covers {we_pct} girte hain" if we_pct else ""], "aur")
    first = w.L(en=f"{o} IPL season playbook from {src or 'the latest order data'}: {facts_en}.",
                hg=f"{o} IPL season ka playbook ({src or 'latest order data'}): {facts_hg}.")
    m = _Msg(first=first, cta="", why="IPL match-day window", anchor=_strip_end(wk_s or weekend_s),
             lever="loss aversion plus effort externalisation")
    m.add(w.L(en="So run match-night combos on Tue/Wed/Thu match dates and go delivery-first on weekends.",
              hg="Isliye match-night combos Tue/Wed/Thu match dates pe chalayein, aur weekends pe delivery-first rakhein."), 0)
    offer = w.offer_for("pizza thali combo biryani")
    if offer:
        m.add(w.L(en=f"Your {offer} is the natural match-night hook.",
                  hg=f"Aapka {offer} match-night ke liye natural hook hai."), 1)
    else:
        combo = next((c for c in w.f.catalog_offers if "match" in c.lower() and w.safe(c) and w.grounded(c)), "")
        if combo:
            m.add(w.L(en=f"A {combo} is the format that works on those nights.",
                      hg=f"{combo} jaisa format un raaton mein best chalta hai."), 1)
    m.cta = _cta_draft(w, en="set up a weeknight match-night post you can reuse all season",
                       hg="poore season ke liye ek weeknight match-night post set kar doon")
    m.offer = "draft a reusable weeknight match-night post and an Insta story"
    return m


def _add_strength(w: _W, m: _Msg, prio: int = 2, style: str = "edge") -> str:
    """One real strength of the merchant (praised review theme, above-peer metric, or active offer)."""
    rev = w.pos_review()
    if rev:
        if style == "edge":
            m.add(w.L(en=f"Your edge is what a newcomer can't copy: {_lc(rev.text)}.",
                      hg=f"Aapki strength wahi hai jo koi naya copy nahi kar sakta: {_lc(rev.text)}."), prio)
        else:
            m.add(w.L(en=f"You start from strength: {_lc(rev.text)}.",
                      hg=f"Aapki strength already hai: {_lc(rev.text)}."), prio)
        return rev.text
    above = w.peer("above")
    if above:
        m.add(w.L(en=f"Your listing is already ahead: {above.text}.",
                  hg=f"Aapki listing already aage hai: {above.text}."), prio)
        return above.text
    offer = w.offer_for()
    if offer and style == "edge":
        m.add(w.L(en=f"Lead with your {offer}.", hg=f"Apne {offer} ko aage rakhein."), prio)
        return offer
    return ""


def _m_competitor(w: _W) -> _Msg:
    comp = w.fact("trigger.competitor")
    if not comp:
        return _m_generic(w)
    o = w.m_open()
    their = _after_colon(w.t("trigger.their_offer"))
    dist = w.t("trigger.distance")
    opened = _date_in(w.t("trigger.opened_date"))
    real_name = comp.text.startswith("New competitor nearby:")
    m = _Msg(first="", cta="", why="new competitor listing nearby", anchor=comp.text,
             lever="loss aversion plus social proof from their own reviews")
    if real_name:
        name = _after_colon(comp.text)
        where = f" {dist}" if dist else " nearby"
        when = f" on {opened}" if opened else ""
        lists = f" and is listing {their}" if their else ""
        where_hg = where.replace(" away", " door").replace(" nearby", " paas mein")
        m.first = w.L(en=f"{o} {name} opened{where}{when}{lists}.",
                      hg=f"{o} {name}{' ' + opened + ' ko' if opened else ''}{where_hg} khula hai" +
                         (f" aur {their} list kar raha hai." if their else "."))
        m.anchor = f"{name} opened{where}{when}{lists}"
        mine = w.offer_for(their) if their else ""
        if mine and their:
            gap = ""
            tp, mp = re.search(r"₹\s?([\d,]+)", their), re.search(r"₹\s?([\d,]+)", mine)
            if tp and mp:
                diff = int(mp.group(1).replace(",", "")) - int(tp.group(1).replace(",", ""))
                if diff > 0 and w.num_ok(diff):
                    gap = f"₹{diff:,}"
            if gap:
                m.add(w.L(en=f"That's {gap} under your {mine}, but a price war isn't the answer.",
                          hg=f"Yeh aapke {mine} se {gap} kam hai, lekin price war sahi jawab nahi hai."), 0)
            else:
                m.add(w.L(en=f"That's up against your {mine}, but a price war isn't the answer.",
                          hg=f"Yeh aapke {mine} ke against hai, lekin price war sahi jawab nahi hai."), 0)
    else:
        loc = w.f.locality or w.f.city
        m.first = w.L(en=f"{o} {_lc(comp.text)}, so the same local searches now show one more option.",
                      hg=f"{o} {loc + ' mein ' if loc else 'aapke paas '}ek nayi {w.biz_noun} listing khuli hai, yaani "
                         f"wahi local searches mein ab ek aur option dikhega.")
    _add_strength(w, m, prio=1)
    stale = w.t("signal.stale_posts")
    if stale:
        m.add(w.L(en=f"One gap to close first: {_lc(stale)}.", hg=f"Ek gap pehle band karna hoga: {_lc(stale)}."), 1)
    else:
        freq = w.t("peer.avg_post_freq_days")
        mm = re.search(r"every (\d+) days", freq)
        if mm:
            m.add(w.L(en=f"{freq}, so a fresh post now keeps you ahead of the new listing.",
                      hg=f"Peers har {mm.group(1)} din mein Google post karte hain; abhi ek fresh post aapko nayi "
                         f"listing se aage rakhega."), 2)
    m.cta = _cta_draft(w, en="draft a Google post that leads with your strengths this week",
                       hg="is hafte ke liye aapki strengths wala Google post draft kar doon")
    m.offer = w.pb.offer
    return m


def _perf_anchor(w: _W) -> tuple[Fact | None, str]:
    f = w.fact("trigger.metric_delta") or w.fact("derived.metric_delta")
    if not f:
        return None, ""
    kind = (f.value or {}).get("kind") if isinstance(f.value, dict) else None
    if f.key == "trigger.metric_delta" or kind == "delta_7d":
        kind = "delta"
    return f, kind or "delta"


def _visible_gaps(w: _W) -> list[tuple[str, str]]:
    gaps: list[tuple[str, str]] = []
    stale = w.t("signal.stale_posts")
    if stale:
        gaps.append((_lc(stale), _lc(stale)))
    if w.v("merchant.verified") is False:
        gaps.append(("your Google profile isn't verified yet", "Google profile abhi verified nahi hai"))
    if not w.f.offers_active:
        gaps.append(("no offer is live on your profile", "profile pe koi offer live nahi hai"))
    sub = w.t("merchant.subscription")
    if "expired" in sub.lower():
        gaps.append((f"your {_lc(sub)}", f"aapka {_lc(sub)}"))
    if not gaps and w.t("signal.no_recent_post"):
        gaps.append(("there's no recent Google post", "koi recent Google post nahi hai"))
    return gaps


def _m_perf_dip(w: _W) -> _Msg:
    f, kind = _perf_anchor(w)
    if not f:
        return _m_generic(w)
    o = w.m_open()
    text = f.text
    if kind == "peer_gap":
        first = w.L(en=f"{o} one number on your profile needs attention: {text}.",
                    hg=f"{o} aapke profile ka ek number dhyaan maangta hai: {text}.")
    elif kind == "expired_plan":
        first = w.L(en=f"{o} a risk building on your profile: {_lc(text)}.",
                    hg=f"{o} aapke profile pe ek risk ban raha hai: {_lc(text)}.")
    elif kind == "flat":
        first = w.L(en=f"{o} a quick look at your numbers: {text}.", hg=f"{o} aapke numbers pe ek nazar: {text}.")
    else:
        first = w.L(en=w.pick([f"{o} heads-up: {_verbed(text)}.", f"{o} flagging a drop early: {_verbed(text)}."], "hook"),
                    hg=w.pick([f"{o} aapke profile pe ek signal dikh raha hai: {_lc(text)}.",
                               f"{o} ek drop jaldi flag kar rahi hoon: {_lc(text)}."], "hook"))
    m = _Msg(first=first, cta="", why="performance dip on the profile", anchor=text,
             lever="loss aversion plus effort externalisation (Vera drafts the fix)")
    gaps = _visible_gaps(w)
    if gaps:
        m.add(w.L(en=f"What I can see from here: {_join([g[0] for g in gaps[:2]])}.",
                  hg=f"Profile pe yeh dikh raha hai: {_join([g[1] for g in gaps[:2]], 'aur')}."), 0)
    offer = w.offer_for()
    if offer:
        m.add(w.L(en=f"A fresh post featuring your {offer} is the quickest lever to pull.",
                  hg=f"Aapke {offer} ke saath ek fresh post sabse quick lever hai."), 1)
    else:
        idea = w.catalog_idea()
        if idea:
            m.add(w.L(en=f"Putting a clear offer like {idea} live, plus one fresh post, is the fastest fix.",
                      hg=f"{idea} jaisa ek clear offer live karna aur ek fresh post daalna sabse fast fix hai."), 1)
    ret = w.retention()
    if ret and ret[2] and "below peers" in " ".join(x.text for x in w.facts("agg.retention")):
        m.add(w.L(en=f"It matters more because {ret[1]} retention is {ret[0]} vs a {ret[2]} peer average.",
                  hg=f"Yeh isliye bhi zaroori hai kyunki {ret[1]} retention {ret[0]} hai, peer average {ret[2]}."), 3)
    elif kind != "peer_gap":
        below = w.peer("below")
        if below:
            m.add(w.L(en=f"For context, {_lc(below.text)}.", hg=f"Context ke liye: {below.text}."), 3)
    m.cta = _cta_draft(w, en="draft the post and the offer line for your OK today",
                       hg="post aur offer line draft karke aapke OK ke liye bhej doon")
    m.offer = w.pb.offer
    return m


def _m_perf_spike(w: _W) -> _Msg:
    f, kind = _perf_anchor(w)
    if not f:
        return _m_generic(w)
    o = w.m_open()
    text = f.text
    if kind == "peer_gap":
        first = w.L(en=f"{o} your profile is outperforming the category: {text}.",
                    hg=f"{o} aapka profile category se aage chal raha hai: {text}.")
    elif kind == "flat":
        first = w.L(en=f"{o} a quick look at your numbers: {text}.", hg=f"{o} aapke numbers pe ek nazar: {text}.")
    else:
        first = w.L(en=w.pick([f"{o} good news: {_verbed(text)}.", f"{o} something's working: {_verbed(text)}."], "hook"),
                    hg=w.pick([f"{o} achhi khabar: {_lc(text)}.", f"{o} kuch sahi chal raha hai: {_lc(text)}."],
                              "hook"))
    m = _Msg(first=first, cta="", why="performance spike on the profile", anchor=text,
             lever="curiosity plus loss aversion (momentum fades) and effort externalisation")
    driver = _strip_end(_after_colon(w.t("trigger.likely_driver")))
    if driver:
        d = re.sub(r"^the\s+", "", driver)
        m.add(w.L(en=f"The likely driver is your {d}.", hg=f"Iski wajah shayad aapka {d} hai."), 0)
        m.anchor += f"; driver: {d}"
    m.add(w.L(en="Momentum like this fades if the profile goes quiet, so this is the week for a follow-up.",
              hg="Profile shaant ho gaya toh yeh momentum utar jata hai, isliye follow-up isi hafte hona chahiye."), 1)
    conv = next((x for x in w.facts("agg.") if "above peers" in x.text), None)
    if conv:
        head = conv.text.split(" (")[0]
        m.add(w.L(en=f"With {_lc(head)}, every extra enquiry is worth more to you than to most.",
                  hg=f"{head} ke saath, har extra enquiry aapke liye zyada valuable hai."), 2)
    elif kind != "peer_gap":
        above = w.peer("above")
        if above:
            m.add(w.L(en=f"And the base is strong: {above.text}.", hg=f"Base bhi strong hai: {above.text}."), 2)
    if w.v("merchant.verified") is False:
        m.add(w.L(en="Your Google profile isn't verified yet, so there's more headroom here.",
                  hg="Google profile abhi verified nahi hai, yaani aur growth ki gunjaaish hai."), 3)
    noun = "bookings" if w.cat in ("dentists", "salons", "gyms") else "orders"
    m.cta = _cta_draft(w, en=f"draft a follow-up post that turns this interest into {noun}",
                       hg=f"ek follow-up post draft kar doon taaki yeh interest {noun} mein badle")
    m.offer = w.pb.offer
    return m


def _m_seasonal_dip(w: _W) -> _Msg:
    f, kind = _perf_anchor(w)
    season = w.t("trigger.season_note")
    beat = w.t("seasonal.note")
    if not f and not season and not beat:
        return _m_generic(w)
    o = w.m_open()
    phrase = _after_colon(season)
    if phrase.endswith(" window"):
        phrase = phrase[: -len(" window")] + " lull"
    rng, note = _beat_split(beat) if beat else ("", "")
    head, _, tail = note.partition(" — ")
    m = _Msg(first="", cta="", why="expected seasonal dip", anchor="",
             lever="reassurance plus a redirect to retention (effort externalisation)")
    if f and kind == "delta":
        delta = _verbed(f.text)
        what = f"the expected {phrase}" if phrase else (f"the usual {rng} {head}" if rng and head else
                                                        "the expected seasonal lull")
        m.first = w.L(en=w.pick([f"{o} {delta}, but that's {what}, not something you did wrong.",
                                 f"{o} before this worries you: {delta}, and it's {what}."], "hook"),
                      hg=f"{o} {_lc(f.text)}, lekin yeh {what.replace('the expected ', 'expected ').replace('the usual ', '')} "
                         f"hai, aapki koi galti nahi.")
        m.anchor = f"{f.text}; {season}".strip("; ")
    elif rng and head:
        m.first = w.L(en=f"{o} a heads-up on the season: across {w.biz_plural}, {rng} is the {head}.",
                      hg=f"{o} season ka ek heads-up: sabhi {w.biz_plural} ke liye {rng} {head} hota hai.")
        m.anchor = beat
        if f:
            m.add(w.L(en=f"Your numbers right now: {_lc(f.text)}.", hg=f"Aapke numbers abhi: {f.text}."), 1)
    else:
        m.first = w.L(en=f"{o} a quick look at your numbers before the slower months: {f.text}.",
                      hg=f"{o} slow season se pehle aapke numbers pe ek nazar: {f.text}.")
        m.anchor = f.text
    act, src = w.t("digest.actionable"), w.t("digest.source")
    if act:
        m.add(w.L(en=f"{src or 'The category data'} is clear on the play: {_lc(_strip_end(act))}.",
                  hg=f"{src or 'Category data'} ke hisaab se: {_lc(_strip_end(act))}."), 0)
    elif tail and f and kind == "delta":
        m.add(w.L(en=f"Across {w.biz_plural}, {rng} is the {head}, so the play is to {_lc(tail)}.",
                  hg=f"Sabhi {w.biz_plural} ke liye {rng} {head} hota hai, isliye {tail}."), 0)
    elif tail:
        m.add(w.L(en=f"The play for now: {_lc(tail)}.", hg=f"Abhi ka play: {tail}."), 0)
    members = w.agg_count("agg.total_active_members")
    churn = w.t("agg.monthly_churn_pct")
    if members:
        tail_en = f" (churn is {churn.split(' (')[0].replace(' monthly churn', ' a month')})" if churn else ""
        tail_hg = f" ({churn.split(' (')[0]})" if churn else ""
        m.add(w.L(en=f"The lever that matters now is keeping your {members} active members{tail_en}.",
                  hg=f"Abhi asli lever aapke {members} active members ko retain karna hai{tail_hg}."), 0)
        m.anchor += f"; {members} active members"
    elif w.v("agg.total_unique_ytd"):
        n = _num(w.v("agg.total_unique_ytd"))
        m.add(w.L(en=f"The lever that matters now is the {n} {w.people} you already have this year.",
                  hg=f"Abhi asli lever is saal ke aapke {n} {w.people} hain."), 1)
    who = "members" if w.cat == "gyms" else w.people
    m.cta = _cta_draft(w, en=f"draft a 4-week attendance challenge to keep your {who} coming through the dip",
                       hg=f"{who} ke liye ek 4-week attendance challenge draft kar doon, taaki dip mein bhi woh aate rahein")
    m.offer = w.pb.offer
    return m


def _milestone_phrase(w: _W, f: Fact) -> tuple[str, str]:
    """(en, hg) clause for a milestone fact, e.g. 'crossed 700 profile views in the last 30 days (787 now)'."""
    text = f.text
    val = f.value if isinstance(f.value, dict) else {}
    if f.key.startswith("derived.milestone") and val.get("threshold") and val.get("actual"):
        th, act = _num(val["threshold"]), _num(val["actual"])
        label = re.sub(r"^Crossed [\d,]+ ", "", text).split(" (")[0]
        return (f"you crossed {th} {label} ({act} now)",
                f"aapne {th} {label.replace('in the last', 'pichhle').replace(' days', ' din mein')} cross kar liye "
                f"(abhi {act})")
    if text.startswith("Crossed"):
        return _lc(text), _lc(text)
    return text, text


def _m_milestone(w: _W) -> _Msg:
    f = w.fact("trigger.milestone") or w.fact("derived.milestone")
    if not f:
        return _m_generic(w)
    o = w.m_open()
    text = f.text
    reviews = "review" in text.lower()
    near = re.match(r"([\d,]+) (.+?) now, ([\d,]+) away from ([\d,]+)", text)
    if near:
        cur, label, gap, goal = near.groups()
        first = w.L(en=w.pick([f"{o} {w.biz_name} is at {cur} {label}, just {gap} short of {goal}.",
                               f"{o} {w.biz_name} is {gap} {label} away from {goal} ({cur} now)."], "hook"),
                    hg=f"{o} {w.biz_name} ab {cur} {label} pe hai, {goal} se sirf {gap} door.")
    else:
        en, hg = _milestone_phrase(w, f)
        first = w.L(en=w.pick([f"{o} a milestone worth a moment: {en}.", f"{o} quick win to celebrate: {en}."], "hook"),
                    hg=f"{o} ek milestone celebrate karne layak: {hg}.")
    m = _Msg(first=first, cta="", why="milestone reached / imminent", anchor=text,
             lever="social proof plus effort externalisation")
    rev = w.pos_review()
    if rev:
        m.add(w.L(en=f"Your regulars are clearly happy: {_lc(rev.text)}.",
                  hg=f"Aapke regulars khush hain: {_lc(rev.text)}."), 1)
    if reviews:
        m.add(w.L(en="A thank-you note to this week's regulars with a gentle review request can close that gap quickly.",
                  hg="Is hafte ke regulars ko ek thank-you note aur halki si review request se yeh gap jaldi band ho "
                     "sakta hai."), 1)
    else:
        above = w.peer("above", ("ctr",))
        if above:
            m.add(w.L(en=f"And people who find you do click: {above.text}.",
                      hg=f"Aur jo aapko dekhte hain, woh click bhi karte hain: {above.text}."), 1)
        cust = w.fact("derived.milestone_customers")
        if cust:
            en2, hg2 = _milestone_phrase(w, cust)
            en2 = en2.replace("you crossed", "you've also crossed")
            m.add(w.L(en=f"{_cap(en2)}." if en2.startswith("you") else f"Another one: {en2}.",
                      hg=f"Saath hi {hg2.replace('aapne', 'aapne is saal').replace(' this year', '')}."), 2)
        m.add(w.L(en="A thank-you post plus a review request to happy customers keeps that momentum compounding.",
                  hg="Ek thank-you post aur khush customers se review request is momentum ko aage badhayega."), 2)
    noun = {"restaurants": "regular diners", "salons": "regular clients", "gyms": "members",
            "dentists": "happy patients", "pharmacies": "regular customers"}.get(w.cat, "regulars")
    m.cta = _cta_draft(w, en=f"draft the thank-you post and a review-request message for your {noun}",
                       hg=f"aapke {noun} ke liye thank-you post aur review-request message draft kar doon")
    m.offer = w.pb.offer
    return m


def _m_review(w: _W) -> _Msg:
    theme = w.fact("trigger.theme")
    o = w.m_open()
    if theme:
        first = w.L(en=w.pick([f"{o} a pattern in your reviews: {_lc(theme.text)}.",
                               f"{o} flagging a review trend early: {_lc(theme.text)}."], "hook"),
                    hg=f"{o} aapke reviews mein ek pattern dikh raha hai: {_lc(theme.text)}.")
        m = _Msg(first=first, cta="", why="review theme emerged", anchor=theme.text,
                 lever="loss aversion (ratings) plus effort externalisation")
        quote = _after_colon(w.t("trigger.quote")).strip("'")
        if quote:
            m.add(w.L(en=f"One customer put it as '{quote}'.", hg=f"Ek customer ne likha: '{quote}'."), 0)
        rev = w.pos_review()
        if rev and theme.value and str(theme.value).lower() not in rev.key:
            m.add(w.L(en=f"On the plus side, {_lc(rev.text)}, so this is very fixable.",
                      hg=f"Achhi baat yeh hai ki {_lc(rev.text)}, toh yeh fix ho sakta hai."), 1)
        m.add(w.L(en="A calm public reply to each, plus one operational fix, stops it from compounding.",
                  hg="Har review ka shaant public reply aur ek operational fix, isse badhne se rokega."), 2)
        m.cta = _cta_draft(w, en="draft the public replies and the fix note for your OK",
                           hg="in reviews ke polite public replies aur fix note draft kar doon")
        m.offer = w.pb.offer
        return m
    bench = w.t("derived.review_benchmark")
    f, kind = _perf_anchor(w)
    if not bench and not f:
        return _m_generic(w)
    bm = re.search(r"average (?:a )?([\d.]+★) rating(?: and ([\d,]+) reviews)?", bench)
    if bench and bm:
        hg_bench = (f"is category mein peer {w.biz_plural} ka average {bm.group(1)} rating"
                    + (f" aur {bm.group(2)} reviews" if bm.group(2) else "") + " hai")
    else:
        hg_bench = _lc(bench or (f.text if f else ""))
    lead = bench or f.text
    first = w.L(en=w.pick([f"{o} time for a quick reviews check at {w.biz_name}: {_lc(lead)}.",
                           f"{o} a quick reviews check for {w.biz_name}: {_lc(lead)}."], "hook"),
                hg=f"{o} {w.biz_name} ke reviews pe ek nazar daalne ka sahi time hai: {hg_bench}.")
    m = _Msg(first=first, cta="", why="review check-in (no specific review theme in the data)", anchor=lead,
             lever="loss aversion plus effort externalisation")
    m.add(w.L(en="Reviews are often the tie-breaker when people compare options nearby.",
              hg="Log paas ke options compare karte waqt aksar reviews dekh ke decide karte hain."), 2)
    if f and bench:
        if kind == "expired_plan":
            m.add(w.L(en=f"And with your {_lc(f.text.split(',')[0])}, profile upkeep has been paused.",
                      hg=f"Aur {_lc(f.text.split(',')[0])}, toh profile upkeep ruka hua hai."), 1)
        else:
            m.add(w.L(en=f"With {_lc(f.text)}, it's worth knowing what recent reviewers are saying.",
                      hg=f"Saath hi {_lc(f.text)}, isliye recent reviewers kya keh rahe hain, yeh jaanna zaroori hai."),
                  1)
    m.cta = _cta_draft(w, en="pull your latest reviews into a short summary and draft replies for your OK",
                       hg="aapke latest reviews ka short summary bana ke har review ka reply draft kar doon")
    m.offer = "pull the latest reviews into a short summary and draft replies for their OK"
    return m


def _perf_summary(w: _W) -> str:
    bits = []
    for key, label in (("perf.views", "profile views"), ("perf.calls", "calls"), ("perf.leads", "leads")):
        v = w.v(key)
        if v:
            bits.append(f"{_num(v)} {label}")
    return w.join(bits)


def _m_renewal(w: _W) -> _Msg:
    days_f = w.fact("trigger.days_remaining")
    exp_f = w.fact("trigger.days_since_expiry")
    if not days_f and not exp_f:
        return _m_generic(w)
    o = w.m_open()
    amount = w.v("trigger.renewal_amount")
    amt = f"₹{_num(amount)}" if amount is not None and w.num_ok(amount) else ""
    summary = _perf_summary(w)
    m = _Msg(first="", cta="", why="plan renewal window", lever="loss aversion plus effort externalisation")
    if exp_f:
        m.first = w.L(en=f"{o} your {_lc(exp_f.text)}, and profile upkeep has been paused since.",
                      hg=f"{o} aapka {_lc(exp_f.text)}, aur tab se profile upkeep ruka hua hai.")
        m.anchor = exp_f.text
        if summary:
            m.add(w.L(en=f"Even so, the profile pulled {summary} in the last 30 days; restarting keeps that from slipping.",
                      hg=f"Phir bhi last 30 days mein profile pe {summary} aaye; restart se yeh momentum bacha rahega."),
                  0)
        _renewal_extras(w, m)
        m.cta = _cta_draft(w, en="restart the plan for your confirmation", hg="plan restart karke confirmation ke liye bhej doon")
        m.offer = "restart the plan for their confirmation and resume profile upkeep"
        return m
    days = days_f.value if isinstance(days_f.value, int) else _lead_int(days_f.text)
    trial = days_f.text.lower().startswith("trial")
    plan = _after_colon(w.t("trigger.plan")).removesuffix(" plan") if w.t("trigger.plan") else ""
    if trial:
        what_en, what_hg = f"your trial ends in {days} days", f"aapka trial {days} din mein khatam ho raha hai"
    else:
        what_en = f"your {plan + ' ' if plan else ''}plan has {days} days left"
        what_hg = f"aapke {plan + ' ' if plan else ''}plan mein {days} din bache hain"
    m.anchor = days_f.text + (f", renewal {amt}" if amt else "")
    if days <= 30:
        m.first = w.L(en=f"{o} {what_en}" + (f" (renewal {amt})" if amt else "") + ".",
                      hg=f"{o} {what_hg}" + (f" (renewal {amt})" if amt else "") + ".")
        if summary:
            m.add(w.L(en=f"In the last 30 days the profile brought in {summary}.",
                      hg=f"Last 30 days mein profile se {summary} aaye."), 0)
        dip = w.v("perf.delta_calls")
        dip_txt = w.t("perf.delta_calls")
        if isinstance(dip, (int, float)) and dip <= -0.1:
            m.add(w.L(en=f"If it lapses, posts and profile upkeep pause right when {_verbed(dip_txt).replace(' are ', ' are already ', 1).replace(' is ', ' is already ', 1)}.",
                      hg=f"Plan lapse hua toh posts aur profile upkeep ruk jayenge, woh bhi aise waqt jab "
                         f"{_lc(dip_txt)}."), 1)
        else:
            m.add(w.L(en="If it lapses, posts and profile upkeep pause and that momentum slips.",
                      hg="Plan lapse hua toh posts aur profile upkeep ruk jayenge aur momentum chala jayega."), 1)
        _renewal_extras(w, m)
        m.cta = w.L(en=w.pick(["Reply YES and I'll set up the renewal for your confirmation.",
                               "Want me to set up the renewal for your confirmation?"], "cta"),
                    hg=w.pick(["Bas YES reply karein, main renewal set up karke confirmation ke liye bhej dungi.",
                               "Kya main renewal set up karke aapke confirmation ke liye bhej doon?"], "cta"))
        m.offer = w.pb.offer
    else:
        m.first = w.L(en=f"{o} a quick value check on your plan ({days} days left, nothing due yet).",
                      hg=f"{o} aapke plan ka ek quick value check ({days} din bache hain, abhi kuch due nahi).")
        if summary:
            m.add(w.L(en=f"Last 30 days: {summary}.", hg=f"Last 30 days mein: {summary}."), 0)
        _renewal_extras(w, m)
        m.cta = _cta_draft(w, en="send you a short results summary like this every month",
                           hg="aapko har mahine aisa ek short results summary bhej doon")
        m.offer = "send a short monthly results summary"
    return m


def _renewal_extras(w: _W, m: _Msg) -> None:
    above = w.peer("above")
    if above:
        m.add(w.L(en=f"That's ahead of the category: {above.text}.", hg=f"Yeh category se aage hai: {above.text}."), 2)
    elif w.v("agg.total_unique_ytd"):
        n = _num(w.v("agg.total_unique_ytd"))
        m.add(w.L(en=f"That's on top of {n} unique {w.people} so far this year.",
                  hg=f"Saath hi is saal ab tak {n} unique {w.people} aaye hain."), 3)


def _m_winback(w: _W) -> _Msg:
    exp_f = w.fact("trigger.days_since_expiry")
    if not exp_f:
        return _m_renewal(w)
    o = w.m_open()
    n = exp_f.value if isinstance(exp_f.value, int) else _lead_int(re.sub(r"^\D+", "", exp_f.text))
    dip = _pct_in(w.t("trigger.perf_dip"))
    lapsed = w.v("trigger.lapsed_added")
    neg = next((x for x in w.facts("perf.delta_") if " down " in f" {x.text} "), None)
    showing = bool(dip or lapsed or neg)
    first = w.L(en=f"{o} it's been {n} days since your plan expired" +
                   (", and the gap is showing in your numbers." if showing else ", and profile upkeep has been paused since."),
                hg=f"{o} plan expire hue {n} din ho gaye hain" +
                   (", aur iska asar numbers pe dikh raha hai." if showing else ", aur tab se profile upkeep ruka hua hai."))
    m = _Msg(first=first, cta="", why=f"win-back window ({exp_f.text})", anchor=exp_f.text,
             lever="loss aversion plus effort externalisation (one-step restart)")
    bits_en, bits_hg = [], []
    if dip:
        bits_en.append(f"profile performance is down {dip.lstrip('-')} since then")
        bits_hg.append(f"profile performance {dip.lstrip('-')} gir gayi hai")
    if lapsed:
        bits_en.append(f"{lapsed} more {w.people} have lapsed")
        bits_hg.append(f"{lapsed} aur {w.people} lapse ho gaye hain")
    if bits_en:
        m.add(w.L(en=_cap(_join(bits_en)) + ".", hg=_cap(_join(bits_hg, "aur")) + "."), 0)
        m.anchor += "; " + _join(bits_en)
    elif neg:
        m.add(w.L(en=f"{_cap(neg.text)}.", hg=f"{_cap(neg.text)}."), 0)
        m.anchor += f"; {neg.text}"
    else:
        summary = _perf_summary(w)
        if summary:
            m.add(w.L(en=f"Even so, the profile still pulled {summary} in the last 30 days, and that's what slips "
                         f"without upkeep.",
                      hg=f"Phir bhi last 30 days mein profile pe {summary} aaye; upkeep ke bina yahi girega."), 0)
    ret = w.retention()
    if ret:
        m.add(w.L(en=f"{_cap(ret[1])} retention is at {ret[0]}" + (f" vs a {ret[2]} peer average." if ret[2] else "."),
                  hg=f"{_cap(ret[1])} retention abhi {ret[0]} hai" + (f", peer average {ret[2]}." if ret[2] else ".")), 2)
    m.add(w.L(en=f"Restarting is one step, and I can have a win-back message for your lapsed {w.people} ready the same day.",
              hg=f"Restart ek step ka kaam hai, aur main lapsed {w.people} ke liye win-back message usi din ready kar "
                 f"dungi."), 1)
    m.cta = _cta_draft(w, en="reactivate the plan and draft that message", hg="plan reactivate karke woh message draft kar doon")
    m.offer = w.pb.offer
    return m


def _m_dormant(w: _W) -> _Msg:
    o = w.m_open()
    lapsed = next(iter(w.facts("agg.lapsed_")), None)
    perf, kind = _perf_anchor(w)
    m = _Msg(first="", cta="", why="merchant has gone quiet", lever="curiosity (one fresh number) plus reciprocity")
    if lapsed:
        mm = re.match(r"(\d[\d,]*) (\w+) lapsed (\d+)\+ days", lapsed.text)
        n = mm.group(1) if mm else ""
        fresh = f"{n} {mm.group(2)} haven't been back in {mm.group(3)}+ days" if mm else _lc(lapsed.text)
        fresh_hg = f"{n} {mm.group(2)} {mm.group(3)}+ din se wapas nahi aaye" if mm else lapsed.text
        m.first = w.L(en=w.pick([f"{o} one number from your profile worth 30 seconds: {fresh}.",
                                 f"{o} quick one, no long pitch: {fresh}."], "hook"),
                      hg=w.pick([f"{o} aapke profile ka ek number dekhne layak hai: {fresh_hg}.",
                                 f"{o} ek quick baat, lamba pitch nahi: {fresh_hg}."], "hook"))
        m.anchor = lapsed.text
        ret = w.retention()
        if ret:
            m.add(w.L(en=f"{_cap(ret[1])} retention is at {ret[0]}" + (f" vs a {ret[2]} peer average." if ret[2] else "."),
                      hg=f"{_cap(ret[1])} retention abhi {ret[0]} hai" + (f", jabki peer average {ret[2]} hai."
                                                                          if ret[2] else ".")), 1)
        who = f"those {n} {w.people}" if n else f"your lapsed {w.people}"
        who_hg = f"un {n} {w.people}" if n else f"lapsed {w.people}"
        m.add(w.L(en="A short, warm comeback message usually brings a good share of them back.",
                  hg="Ek chhota, warm comeback message aksar kaafi logon ko wapas le aata hai."), 2)
        m.cta = _cta_draft(w, en=f"draft a comeback message for {who} so you can take a look",
                           hg=f"{who_hg} ke liye ek comeback message draft karke aapko dikha doon")
        m.offer = "draft a comeback WhatsApp message for their lapsed customers"
        return m
    if perf:
        up = " up " in f" {perf.text} " or "above peers" in perf.text
        if up:
            m.first = w.L(en=w.pick([f"{o} good news from your profile: {_verbed(perf.text)}.",
                                     f"{o} quick one from your profile: {_verbed(perf.text)}."], "hook"),
                          hg=f"{o} aapke profile se ek achhi khabar: {_lc(perf.text)}.")
            move = w.L(en="A fresh post now builds on that while it's moving.",
                       hg="Abhi ek fresh post isi momentum ko aage le jayega.")
        else:
            m.first = w.L(en=f"{o} quick one from your profile: {_verbed(perf.text)}.",
                          hg=f"{o} aapke profile se ek quick update: {_lc(perf.text)}.")
            move = w.L(en="A fresh post is the quickest way to get it moving again.",
                       hg="Ek fresh post isse dobara chalane ka sabse quick tarika hai.")
        m.anchor = perf.text
        ctx = w.peer("above") if up else (w.peer("below") or w.peer("above"))
        if ctx:
            m.add(w.L(en=f"For context, {_lc(ctx.text)}.",
                      hg=f"Aur peers ke comparison mein: {ctx.text}."), 2)
        freq = w.t("peer.avg_post_freq_days")
        mm = re.search(r"every (\d+) days", freq)
        if mm:
            m.add(w.L(en=f"{freq}, so a fresh post this week is the quickest way to "
                         + ("build on it." if up else "get it moving again."),
                      hg=f"Peers har {mm.group(1)} din mein Google post karte hain, toh is hafte ek fresh post "
                         + ("isi momentum ko aage le jayegi." if up else "sabse quick fix hai.")), 1)
        else:
            m.add(move, 1)
        m.cta = _cta_draft(w, en="draft that post for you to check", hg="woh post draft karke aapko dikha doon")
        m.offer = w.pb.offer
        return m
    return _m_generic(w)


def _m_gbp(w: _W) -> _Msg:
    o = w.m_open()
    up = _pct_in(w.t("trigger.uplift"))
    path = w.t("trigger.verification_path").removeprefix("Verification by ").strip()
    name = w.biz_name
    poss = f"{name}'" if name.endswith("s") else f"{name}'s"
    first = w.L(en=f"{o} {poss} Google profile still isn't verified" +
                   (f", and verifying brings an estimated {up} uplift." if up else "."),
                hg=f"{o} {name} ka Google profile abhi verified nahi hai" +
                   (f", aur verify karne se estimated {up} uplift milta hai." if up else "."))
    m = _Msg(first=first, cta="", why="unverified Google Business Profile",
             anchor="profile not verified" + (f"; estimated {up} uplift" if up else ""),
             lever="loss aversion plus effort externalisation (Vera walks them through)")
    if path:
        m.add(w.L(en=f"It's a single step: verification by {path}.", hg=f"Bas ek step hai: verification by {path}."), 0)
    trend = w.t("trend.top")
    if trend:
        m.add(w.L(en=f"With {trend}, being verified is what gets you shown for those searches.",
                  hg=f"{_cap(_trend_are(trend))}, aur verified profile hi in searches mein aage dikhta hai."), 1)
    views = w.t("perf.views")
    if views:
        m.add(w.L(en=f"You're already getting {_lc(views)} without it.",
                  hg=f"Bina verification ke bhi {views} aa rahe hain."), 2)
    m.cta = _cta_draft(w, en="walk you through it right now", hg="abhi aapko step-by-step verification karwa doon")
    m.offer = w.pb.offer
    return m


def _m_curious(w: _W) -> _Msg:
    o = w.m_open()
    trend = w.t("trend.top")
    topic, gender = _ASK_TOPIC.get(w.cat, ("service", "f"))
    guess = _trend_keyword(trend)
    rev = w.pos_review()
    hint = _trend_are(trend)
    first = w.L(
        en=w.pick([f"{o} this week's quick question, with a hint from the data: {hint}." if trend else
                   f"{o} this week's quick question for {w.biz_name}, no report attached.",
                   f"{o} a quick one for this week: {hint}." if trend else f"{o} a quick one for this week."], "hook"),
        hg=f"{o} is hafte ka quick sawaal, data se ek hint ke saath: {hint}." if trend else
        f"{o} is hafte ka ek quick sawaal.")
    m = _Msg(first=first, cta="", cta_type=CTA_OPEN_ENDED, why="weekly curious-ask cadence",
             anchor=trend or "weekly check-in", lever="asking the merchant plus reciprocity (a free artifact back)")
    offer = w.offer_for(guess)
    views = w.t("perf.views")
    if offer and guess and (_words(guess) & _words(offer)):
        m.add(w.L(en=f"Your {offer} sits right on that trend.", hg=f"Aapka {offer} bilkul isi trend pe hai."), 1)
    elif rev:
        m.add(w.L(en=f"And {_lc(rev.text)}.", hg=f"Aur {_lc(rev.text)}."), 2)
    elif views:
        m.add(w.L(en=f"With {_lc(views)}, the right post will get seen.",
                  hg=f"Aapke profile pe {_lc(views)} aaye hain, toh sahi post zaroor dikhegi."), 2)
    m.add(w.L(en="Whatever the answer, I'll turn it into a Google post plus a ready WhatsApp reply for when customers "
                 "ask about price.",
              hg="Aap jo bhi batayein, main usse ek Google post aur customers ke liye ek ready WhatsApp reply bana "
                 "dungi."), 0)
    which = "kaunsi" if gender == "f" else "kaunsa"
    g = f": {guess}, or something else?" if guess else "?"
    g_hg = f": {guess} ya kuch aur?" if guess else "?"
    if w.cat == "pharmacies":
        m.cta = w.L(en=f"What are customers asking for most this week{g}",
                    hg=f"Is hafte customers sabse zyada kya maang rahe hain{g_hg}")
    else:
        m.cta = w.L(en=f"Which {topic} are {w.people} asking for most this week{g}",
                    hg=f"Is hafte {w.people} sabse zyada {which} {topic} maang rahe hain{g_hg}")
    m.offer = w.pb.offer
    return m


def _history_snippet(w: _W) -> str:
    """A concrete, non-question sentence from Vera's last message (e.g. an outline with numbers)."""
    f = w.fact("history.last_vera")
    if not f:
        return ""
    m = re.search(r":\s*'(.*)'\s*$", f.text)
    body = m.group(1) if m else ""
    for s in re.split(r"(?<=[.!?])\s+", body):
        s = s.strip()
        if s and not s.endswith("?") and re.search(r"\d", s) and "…" not in s:
            return re.sub(r"^(suggest|suggested|maybe|how about)\s+", "", s, flags=re.I).rstrip(".")
    return ""


def _m_planning(w: _W) -> _Msg:
    topic_f = w.fact("trigger.intent_topic")
    topic = topic_f.text.removeprefix("Planning:").strip() if topic_f else ""
    if not topic:
        return _m_planning_starter(w)
    o = w.m_open()
    first = w.L(en=w.pick([f"{o} here's a first cut of the {topic}, ready for you to edit:",
                           f"{o} you asked what the {topic} would look like, so here's a draft to edit:"], "hook"),
                hg=f"{o} yeh raha {topic} ka pehla draft, aap edit kar sakte hain:")
    m = _Msg(first=first, cta="", why=f"merchant asked what the {topic} would look like",
             anchor=f"a ready draft of the {topic}", lever="effort externalisation (the draft is already done)",
             max_len=PLANNING_MAX, newline_parts=True)
    offer = w.priced_offer() or w.offer_for()
    tw = _words(topic)
    snippet = _history_snippet(w)
    members = w.agg_count("agg.total_active_members")
    loc = w.f.locality
    if w.cat == "restaurants" and tw & {"corporate", "bulk", "office", "catering", "party", "thali", "package"}:
        item = next((x for x in ("thali", "biryani", "pizza", "meal", "combo", "lunch") if x in topic.lower()), "meal")
        price = re.search(r"₹[\d,]+", offer or "")
        lines = [
            f"• Office pack: 10+ {item}s per order" + (f" at your {price.group(0)} {item} price" if price else "")
            + (f", free delivery within {loc}" if loc else ", free delivery"),
            "• Weekly plan: a fixed weekday menu for teams, one consolidated bill every Friday",
            "• Ordering: WhatsApp the order by 11am, delivered by 1pm",
            f"• Bulk rate: your call on a per-{item} price" + (f" below {price.group(0)}" if price else "")
            + " for bigger orders; tell me the number and I'll plug it in",
        ]
        for line in lines:
            m.add(line, 0)
        m.anchor = f"draft corporate pack built on {offer}" if offer else m.anchor
        if snippet:
            m.add(w.L(en=f"The base is proven: {_lc(snippet)}.", hg=f"Base already strong hai: {snippet}."), 1)
        trend = w.t("trend.top")
        if trend:
            m.add(w.L(en=f"Demand is there too: {_trend_are(trend)}.", hg=f"Demand bhi hai: {_trend_are(trend)}."), 2)
        m.cta = _cta_draft(w, en="turn this into a Google post and a WhatsApp note for office admins nearby",
                           hg="ise Google post aur office admins ke liye WhatsApp note mein badal doon")
        m.offer = "turn the corporate pack draft into a Google post and a WhatsApp note for office admins"
        return m
    if w.cat == "gyms":
        lines = [f"• Format: {snippet} (the outline we discussed)" if snippet else
                 "• Format: 4 weeks, 3 classes a week, small batches"]
        small = w.fact("review.small_classes")
        lines.append("• Batches: weekend mornings plus one weekday evening, kept small"
                     + (" (your reviews already praise the small class sizes)" if small else ""))
        free = next((x for x in w.f.offers_active if "free" in x.lower() and w.safe(x)), "")
        if free and "kid" in topic.lower():
            lines.append(f"• Parent hook: a {free} for any parent who enrols a child")
        elif free:
            lines.append(f"• Hook: a {free} for everyone who signs up in the first week")
        elif offer:
            lines.append(f"• Price anchor: your {offer}; set the programme fee and I'll plug it in")
        lines.append("• Launch: Google post, Insta carousel" + (f" and a WhatsApp note to your {members} active members"
                                                               if members else " and a WhatsApp note to members"))
        for line in lines:
            m.add(line, 0)
        m.anchor = f"draft {topic} ({snippet})" if snippet else m.anchor
        trend = w.t("trend.top")
        if trend:
            m.add(w.L(en=f"Timing works: {_trend_are(trend)}.", hg=f"Timing sahi hai: {_trend_are(trend)}."), 2)
        m.cta = _cta_draft(w, en="publish the post and send the note to your members",
                           hg="post publish karke members ko note bhej doon")
        m.offer = f"publish the {topic} post and send the member note"
        return m
    total = w.t("agg.total_unique_ytd")
    mine = w.offer_for(topic, strict=True)
    idea = "" if mine else w.catalog_idea(topic, strict=True)
    base = mine or idea or offer
    what = (f", built around your {mine}" if mine else
            f", with {idea} as the entry option (a standard format in your category, your call)" if idea else
            f", priced off your {offer}" if offer else ", as a fixed-price package")
    lines = [
        f"• What: the {topic}{what}",
        f"• Who: your existing {w.people} first" + (f" ({_lc(total)})" if total else "") + f", then new {w.people} "
        f"from Google" + (f" in {loc}" if loc else ""),
        "• Launch: a Google post plus a WhatsApp note to regulars",
        "• Price: " + (f"anchored on {base}; " if base else "") + "tell me the final rate and I'll plug it in",
    ]
    for line in lines:
        m.add(line, 0)
    if base:
        m.anchor = f"draft {topic} built on {base}"
    m.cta = _cta_draft(w, en="turn this into the post and the WhatsApp note", hg="ise post aur WhatsApp note mein badal doon")
    m.offer = w.pb.offer
    return m


def _m_planning_starter(w: _W) -> _Msg:
    """active_planning_intent without a topic: a starter draft from real merchant data, one open ask."""
    o = w.m_open()
    offer = w.priced_offer() or w.offer_for()
    idea = "" if offer else w.catalog_idea()
    total = w.v("agg.total_unique_ytd")
    loc = w.f.locality
    trend = w.t("trend.top")
    first = w.L(en=f"{o} to get your new plan moving, here's a starter draft to react to:",
                hg=f"{o} aapke naye plan ke liye yeh raha ek starter draft:")
    m = _Msg(first=first, cta="", cta_type=CTA_OPEN_ENDED, why="merchant is actively planning something new",
             anchor=offer or idea or "starter draft", lever="effort externalisation plus asking the merchant",
             max_len=PLANNING_MAX, newline_parts=True)
    if offer:
        m.add(f"• Lead offer: your {offer}", 0)
    elif idea:
        m.add(f"• Lead offer: {idea} (a standard format in your category; your call on the price)", 0)
    m.add(f"• Who: your {_num(total) + ' ' if total else ''}{w.people} this year first, then Google searchers"
          + (f" in {loc}" if loc else ""), 0)
    m.add("• Launch: a Google post plus a WhatsApp note to regulars", 0)
    if trend:
        m.add(f"• Demand signal: {_trend_are(trend)}", 1)
    m.cta = w.L(en="What's the one idea you're planning, so I can turn this into the full draft today?",
                hg="Aap kaunsa idea plan kar rahe hain, ek line mein batayenge taaki main aaj hi ise full draft bana doon?")
    m.offer = "turn the starter into a full draft plan once the merchant names the idea"
    return m


_EVENT_MOVES = {
    "restaurants": ("Days like this shift orders to delivery, so lead with that.",
                    "Aise din orders delivery pe shift hote hain, toh wahi aage rakhein."),
    "pharmacies": ("Keep ORS and summer essentials at the counter today.",
                   "Aaj ORS aur summer essentials counter pe rakhein."),
    "gyms": ("Nudge members towards the cooler morning slots.", "Members ko thande morning slots ki taraf nudge karein."),
    "salons": ("Push indoor, cooling services and off-peak slots today.",
               "Aaj indoor, cooling services aur off-peak slots push karein."),
    "dentists": ("Let patients know appointments are running as normal.",
                 "Patients ko batayein ki appointments normal chal rahe hain."),
}


def _m_event(w: _W) -> _Msg:
    """weather_heatwave / local_news_event: the event fact plus one practical move."""
    city = (w.f.city or "").lower()
    evs = [f for f in w.facts("trigger.") if not (f.key == "trigger.generic.city" and _after_colon(f.text).lower() == city)]
    ev = w.fact("trigger.weather") or next((f for f in evs if f.key.startswith("trigger.generic.")), None) or \
        next(iter(evs), None)
    if not ev:
        return _m_generic(w)
    o = w.m_open()
    heat = w.kind == "weather_heatwave"
    what = _after_colon(ev.text) if ev.key.startswith("trigger.generic.") else ev.text
    first = w.L(en=f"{o} {'heat alert' if heat else 'local update'} for today: {_lc(what)}.",
                hg=f"{o} aaj ka {'heat alert' if heat else 'local update'}: {_lc(what)}.")
    m = _Msg(first=first, cta="", why=w.kind.replace("_", " "), anchor=what,
             lever="timeliness plus effort externalisation")
    for extra in [f for f in evs if f is not ev][:2]:
        m.add(extra.text, 2)
    move = _EVENT_MOVES.get(w.cat, ("One timely post keeps customers informed.",
                                    "Ek timely post customers ko informed rakhega."))
    if heat or w.cat == "restaurants":
        m.add(w.L(en=move[0], hg=move[1]), 1)
    else:
        m.add(w.L(en="A quick post today tells nearby customers you're open and how to reach you.",
                  hg="Aaj ek quick post se paas ke customers ko pata chalega ki aap open hain aur kaise pahunchein."), 1)
    offer = w.offer_for()
    if offer:
        m.add(w.L(en=f"Your {offer} fits today well.", hg=f"Aaj aapka {offer} fit baithta hai."), 2)
    m.cta = _cta_draft(w, en="draft a timely post and a WhatsApp broadcast for today",
                       hg="aaj ke liye ek timely post aur WhatsApp broadcast draft kar doon")
    m.offer = w.pb.offer
    return m


_FOLLOWUP_LABEL = {
    "recall_due": ("a {person} recall", "{person} recall"),
    "customer_lapsed_soft": ("a win-back check-in", "win-back check-in"),
    "customer_lapsed_hard": ("a win-back message", "win-back message"),
    "appointment_tomorrow": ("an appointment reminder for tomorrow", "kal ke appointment ka reminder"),
    "chronic_refill_due": ("a refill reminder", "refill reminder"),
    "trial_followup": ("a trial follow-up", "trial follow-up"),
    "wedding_package_followup": ("a bridal follow-up", "bridal follow-up"),
    "unplanned_slot_open": ("an open slot to fill", "khaali slot bharne ka mauka"),
}


def _m_customer_kind_for_merchant(w: _W) -> _Msg:
    """A customer-scoped trigger rendered to the merchant (no customer context): offer to send it for them."""
    o = w.m_open()
    person = {"dentists": "patient", "gyms": "member", "salons": "client"}.get(w.cat, "customer")
    en_l, hg_l = _FOLLOWUP_LABEL.get(w.kind, ("a customer follow-up", "customer follow-up"))
    en_l, hg_l = en_l.format(person=person), hg_l.format(person=person)
    anchors = [f for f in w.f.anchor_facts if w.safe(f.text) and f.key != "trigger.kind"]
    lead = re.sub(r"\s+due$", "", _strip_end(anchors[0].text)) if anchors else ""
    first = w.L(en=f"{o} {en_l} is due" + (f": {_lc(lead)}." if lead else "."),
                hg=f"{o} ek {hg_l} due hai" + (f": {_lc(lead)}." if lead else "."))
    m = _Msg(first=first, cta="", why=f"customer {w.kind.replace('_', ' ')} due", anchor=lead or en_l,
             lever="effort externalisation (Vera sends it from the business)")
    for f in anchors[1:3]:
        m.add(_end(f.text), 2)
    offer = w.offer_for()
    slots = list(w.f.slots)
    bits = []
    if offer:
        bits.append(w.L(en=f"your {offer} offer", hg=f"aapka {offer} offer"))
    if slots:
        bits.append(w.L(en=f"the open slots ({w.join_or(slots)})", hg=f"open slots ({w.join_or(slots)})"))
    m.add(w.L(en=f"I can send it from your number, speaking as {w.biz_name or 'the business'}" +
                 (f", with {_join(bits)}." if bits else "."),
              hg=f"Main ise aapke number se, {w.biz_name or 'aapke business'} ki taraf se bhej sakti hoon" +
                 (f", saath mein {_join(bits, 'aur')}." if bits else ".")), 0)
    m.cta = _cta_draft(w, en="send it on your behalf today", hg="aapki taraf se aaj hi bhej doon")
    m.offer = w.pb.offer
    return m


def _m_generic(w: _W) -> _Msg:
    o = w.m_open()
    anchors = [f for f in w.f.anchor_facts if w.safe(f.text) and "?" not in f.text and f.key != "trigger.kind"]
    generic = [f for f in anchors if f.key.startswith("trigger.generic.")]
    topic = w.kind.replace("_", " ") if w.kind and w.kind != "generic" else ""
    if generic:
        lead = "; ".join(_cap(f.text)[:1].lower() + f.text[1:] for f in generic[:3])
        first = w.L(en=f"{o} a quick heads-up" + (f" on {topic}" if topic else "") + f": {lead}.",
                    hg=f"{o} ek quick update" + (f" ({topic})" if topic else "") + f": {lead}.")
        rest = [f for f in anchors if f not in generic[:3]]
    elif anchors:
        lead = anchors[0].text
        first = w.L(en=f"{o} a quick update worth your attention: {_lc(lead)}.",
                    hg=f"{o} ek zaroori update: {_lc(lead)}.")
        rest = anchors[1:]
    else:
        lead = w.t("perf.views")
        first = w.L(en=f"{o} a quick look at your profile this week" + (f": {_lc(lead)}." if lead else "."),
                    hg=f"{o} is hafte aapke profile pe ek nazar" + (f": {lead}." if lead else "."))
        rest = []
    m = _Msg(first=first, cta="", why=topic or "update", anchor=lead or "profile snapshot",
             lever="specificity plus effort externalisation")
    for f in rest[:2]:
        m.add(_end(f.text), 1)
    support = w.peer("above") or w.peer("below")
    if support:
        m.add(w.L(en=f"For context, {_lc(support.text)}.", hg=f"Context ke liye: {support.text}."), 2)
    offer = w.offer_for()
    if offer:
        m.add(w.L(en=f"Your {offer} is the natural hook here.", hg=f"Aapka {offer} yahan natural hook hai."), 3)
    m.cta = _cta_draft(w, en="put together a quick plan for this", hg="iske liye ek quick plan bana doon")
    m.offer = w.pb.offer or "put together a short action plan based on this update"
    return m


# --------------------------------------------------------------------------- customer-facing renderers


def _svc(w: _W, key: str = "trigger.service_due") -> str:
    t = w.t(key)
    t = re.sub(r"\s+due$", "", t).strip()
    t = re.sub(r"^Due for an? ", "", t).strip()
    return _lc(t)


def _visit_noun(w: _W) -> tuple[str, str]:
    """(noun, Hindi gender) for 'your last ...'."""
    if w.yoga:
        return "class", "f"
    return _VISIT.get(w.cat, ("visit", "f"))


def _next_noun(w: _W) -> str:
    """What the customer books next: a class, a session, an appointment, a visit or an order."""
    if w.yoga:
        return "class"
    return {"gyms": "session", "salons": "appointment", "dentists": "visit", "restaurants": "visit",
            "pharmacies": "order"}.get(w.cat, "visit")


def _about(w: _W) -> str:
    """The person the message is about when it goes to someone else (a parent, a son)."""
    a = w.f.about_name
    return a if a and a != w.f.recipient_name else ""


def _poss(w: _W, gender: str = "m", cap: bool = False) -> str:
    """Relay-aware possessive: 'your' / "Aanya's" (en), 'aapka' / 'Sharma ji ki' (hg/hi)."""
    a = _about(w)
    if w.hi or w.hg:
        s = f"{a} {'ki' if gender == 'f' else 'ka'}" if a else ("aapki" if gender == "f" else "aapka")
    else:
        s = f"{a}'s" if a else "your"
    return _cap(s) if cap else s


def _last(w: _W, noun: str, gender: str, cap: bool = False) -> str:
    """'your last session' / 'Aanya ki last visit' / 'aapka pichhla session' (Hindi)."""
    if w.hi:
        word = "pichhli" if gender == "f" else "pichhla"
    elif w.hg:
        word = "last"
    else:
        return (_poss(w, gender, cap) + f" last {noun}")
    return f"{_poss(w, gender, cap)} {word} {noun}"


def _hg_was(gender: str) -> str:
    return "thi" if gender == "f" else "tha"


def _slot_short(slots: list[str]) -> list[str]:
    """Short handles for a numbered choice: the weekday + date when unique ("Wed 5 Nov"), else the full label."""
    heads = [s.split(",")[0].strip() for s in slots]
    return heads if len(set(heads)) == len(heads) and all(heads) else list(slots)


def _slot_cta(w: _W, slots: list[str], who: str = "") -> tuple[str, str]:
    """Slot-choice CTA with exact labels: 2+ slots -> reply 1/2/3; one slot -> reply YES."""
    if len(slots) >= 2:
        opts = slots[:3]
        short = _slot_short(opts)
        en = "Reply " + ", ".join(f"{i + 1} for {s}" for i, s in enumerate(short)) + \
             ", or tell us a time that works better."
        hg = ", ".join(f"{s} ke liye {i + 1}" for i, s in enumerate(short)) + \
            " reply karein, ya apna convenient time batayein."
        hi = ", ".join(f"{s} ke liye {i + 1}" for i, s in enumerate(short)) + \
            " likhiye, ya apna suvidhajanak samay batayein."
        return w.L(en=en, hg=_cap(hg), hi=_cap(hi)), CTA_MULTI_CHOICE_SLOT
    s = slots[0]
    en = f"Reply YES to book {s}" + (f" for {who}" if who else "") + ", or tell us a time that suits you."
    hg = f"{s} book karne ke liye YES reply karein" + (f" ({who} ke liye)" if who else "") + ", ya apna time batayein."
    hi = f"{s} pakka karne ke liye YES likhiye" + (f" ({who} ke liye)" if who else "") + ", ya apna samay batayein."
    return w.L(en=en, hg=hg, hi=hi), CTA_BINARY_YES_NO


def _c_offer_line(w: _W, offer: str) -> str:
    if not offer:
        return ""
    if re.search(r"\bfree\b", offer, re.IGNORECASE) and "@" not in offer:
        return w.L(en=f"Our {offer} offer is open to you when you come in.",
                   hg=f"Aapke liye hamara {offer} offer bhi available hai.",
                   hi=f"Aapke liye hamara {offer} offer bhi uplabdh hai.")
    return w.L(en=f"Our {offer} offer applies.", hg=f"Hamara {offer} offer apply hoga.",
               hi=f"Hamara {offer} offer laagu hoga.")


def _c_loyalty(w: _W) -> str:
    """'Thanks for your 22 visits with us since Sep 2025.' when the customer has a real history."""
    n, since = w.visits()
    if n < 2:
        return ""
    s = _short_date(since) if since else ""
    a = _about(w)
    if a:
        return w.L(en=f"Thank you for trusting us with {a}'s {n} visits" + (f" since {s}." if s else "."),
                   hg=f"{s + ' se ' if s else ''}{a} ki {n} visits ke liye shukriya.",
                   hi=f"{s + ' se ' if s else ''}{a} ki {n} visits ke liye dhanyavaad.")
    return w.L(en=f"Thanks for your {n} visits with us" + (f" since {s}." if s else "."),
               hg=f"{s + ' se ' if s else ''}{n} visits ke liye shukriya.",
               hi=f"{s + ' se ' if s else ''}aapki {n} visits ke liye dhanyavaad.")


def _c_pref_tail(w: _W) -> tuple[str, str]:
    """(' (weekday evenings)', ' (weekday evenings)') when a preferred time is known."""
    pref = w.pref()
    return (f" ({pref})", f" ({pref})") if pref else ("", "")


def _c_recall(w: _W) -> _Msg:
    o, slots = w.c_open(), list(w.f.slots)
    svc = _svc(w) or "next visit"
    due = _date_in(w.t("trigger.due_date"))
    last_t = w.t("trigger.last_service_date") or w.t("customer.last_visit")
    last = _date_in(last_t)
    ago = _ago_in(last_t)
    noun, gender = _visit_noun(w)
    about = _about(w)
    m = _Msg(first="", cta="", why=f"{svc} recall due", anchor=f"{svc} due" + (f" on {due}" if due else ""),
             lever="specificity plus effort externalisation (slots already held)")
    intro = w.intro("ek reminder", "ek chhota sa reminder")
    if w.cat == "gyms" and not due:
        spot = "the spot" if about else "your spot"
        if ago and last:
            en = f"It's been {ago.replace(' ago', '')} since {_last(w, noun, gender)} on {last}, and {spot} is waiting."
        elif last:
            en = f"{_last(w, noun, gender, cap=True)} was on {last}, and {spot} is waiting."
        else:
            en = f"It's time for {_poss(w)} next {noun}, and {spot} is waiting."
        m.first = w.L(en=f"{o} {intro} {en}",
                      hg=f"{o} {intro} {_last(w, noun, gender, cap=True)} {last} ko {_hg_was(gender)}, aur jagah ready "
                         f"hai." if last else f"{o} {intro} {_poss(w, 'f', cap=True)} next {noun} ka time ho gaya hai.",
                      hi=f"{o} {intro} {_last(w, noun, gender, cap=True)} {last} ko {_hg_was(gender)}; jagah taiyaar "
                         f"hai." if last else f"{o} {intro} {_poss(w, 'f', cap=True)} agli {noun} ka samay ho gaya hai.")
        m.anchor = f"last {noun} on {last}" if last else m.anchor
    else:
        when = f" by {due}" if due else ""
        prev = (f", and it's been {ago.replace(' ago', '')} since the last visit" if (ago and not due) else
                f" (last visit: {last})" if last and not due else "")
        m.first = w.L(en=f"{o} {intro} {_poss(w, cap=True)} {svc} is due{when}{prev}.",
                      hg=f"{o} {intro} {_poss(w, cap=True)} {svc} {due + ' tak ' if due else ''}due hai"
                         + (f"; last visit {last} ko thi" if last and not due else "") + ".",
                      hi=f"{o} {intro} {_poss(w, cap=True)} {svc} ka samay ho gaya hai{f' ({due} tak)' if due else ''}"
                         + (f"; pichhli visit {last} ko thi" if last and not due else "") + ".")
    offer = w.offer_for(svc, prefer_free=w.cat == "gyms")
    table = w.cat == "restaurants"
    if slots:
        pref = w.pref()
        match = _slot_pref_match(slots, pref)
        n = len(slots)
        label = w.join_or(slots)
        word = "table" if table else "slot"
        for_en = f"for {about}" if about else "for you"
        for_hg = f"{about} ke liye" if about else "aapke liye"
        m.add(w.L(en=f"We've kept {n} {word}{'s' if n > 1 else ''} {for_en}" +
                     ((", both " if n > 1 else ", ") + f"on {pref} as you prefer" if match else "") + f": {label}.",
                  hg=f"{n} slot{'s' if n > 1 else ''} {for_hg} hold kiye hain" +
                     (f" ({pref}, jaisa aapko pasand hai)" if match else "") + f": {label}.",
                  hi=f"{_cap(for_hg)} {n} samay rakhe hain: {label}."), 0)
        m.anchor += f"; slots {_join(slots, 'or')}"
        m.add(_c_offer_line(w, offer), 1)
        m.cta, m.cta_type = _slot_cta(w, slots, about)
    else:
        m.add(_c_offer_line(w, offer), 1)
        m.add(_c_loyalty(w), 2)
        tail_en, tail_hg = _c_pref_tail(w)
        who_en = f" for {about}" if about else ""
        who_hg = f" {about} ke liye" if about else " aapke liye"
        if w.cat == "pharmacies":
            m.add(w.L(en="We can keep the regular items packed so it's a quick in-and-out.",
                      hg="Regular saamaan hum pehle se pack karke rakh sakte hain.",
                      hi="Niyamit saamaan hum pehle se pack karke rakh sakte hain."), 1)
            m.cta = w.L(en="Reply YES and we'll have it ready at a time that suits you.",
                        hg="Bas YES reply karein, hum aapke convenient time pe sab ready rakhenge.",
                        hi="YES likhiye, hum aapke suvidhajanak samay par sab taiyaar rakhenge.")
        else:
            hold = {"gyms": "a class spot" if w.yoga else "a session", "restaurants": "a table"}.get(w.cat, "a slot")
            m.cta = w.L(en=w.pick([f"Reply YES and we'll hold {hold}{who_en} at a time that suits you{tail_en}.",
                                   f"Just reply YES and we'll hold {hold}{who_en}{tail_en}."], "cta"),
                        hg=f"Bas YES reply karein, hum{who_hg} ek slot hold kar denge{tail_hg}.",
                        hi=f"YES likhiye, hum{who_hg} suvidhajanak samay par jagah rakh denge.")
    m.offer = w.pb.offer
    return m


def _c_lapsed(w: _W) -> _Msg:
    o = w.c_open()
    hard = w.kind == "customer_lapsed_hard"
    days_f = w.fact("trigger.days_since_last_visit")
    days = days_f.value if days_f and isinstance(days_f.value, int) else None
    last = _date_in(w.t("trigger.last_service_date") or w.t("customer.last_visit"))
    noun, gender = _visit_noun(w)
    about = _about(w)
    intro = w.intro("ek message", "ek sandesh")
    m = _Msg(first="", cta="", why=f"customer {'lapsed a while' if hard else 'recently lapsed'}",
             lever="warmth, no-shame framing and a no-pressure restart")
    normal_en = " Breaks like that are completely normal." if hard else ""
    normal_hg = " Koi baat nahi, break sabke saath hota hai." if hard else ""
    normal_hi = " Koi baat nahi, aisa sabke saath hota hai." if hard else ""
    if days:
        m.first = w.L(en=f"{o} {intro} It's been {days} days since {_last(w, noun, gender)}" +
                         (f" on {last}." if last else ".") + normal_en,
                      hg=f"{o} {intro} {_last(w, noun, gender, cap=True)} ko {days} din ho gaye" +
                         (f" ({last})." if last else ".") + normal_hg,
                      hi=f"{o} {intro} {_last(w, noun, gender, cap=True)} ko {days} din ho gaye" +
                         (f" ({last})." if last else ".") + normal_hi)
        m.anchor = f"{days} days since the last {noun}"
    elif last:
        en_opts = [f"{o} {intro} It's been a while since {_last(w, noun, gender)} on {last}, so we thought we'd "
                   f"check in.", f"{o} {intro} Just checking in, as {_last(w, noun, gender)} with us was on {last}."]
        m.first = w.L(en=(en_opts[0].replace(", so we thought we'd check in.", ".") + normal_en) if hard else
                      w.pick(en_opts, "hook"),
                      hg=f"{o} {intro} {_last(w, noun, gender, cap=True)} {last} ko {_hg_was(gender)}" +
                         (f".{normal_hg}" if hard else ", toh socha ek baar haal-chaal pooch lein."),
                      hi=f"{o} {intro} {_last(w, noun, gender, cap=True)} {last} ko {_hg_was(gender)}" +
                         (f".{normal_hi}" if hard else ", isliye haal-chaal poochne ke liye sandesh bhej rahe hain."))
        m.anchor = f"last {noun} on {last}"
    else:
        m.first = w.L(en=f"{o} {intro} It's been a while since {_last(w, noun, gender)}, so we thought we'd check in.",
                      hg=f"{o} {intro} Kaafi time ho gaya, toh socha haal-chaal pooch lein.",
                      hi=f"{o} {intro} Kaafi samay ho gaya, isliye haal-chaal pooch rahe hain.")
        m.anchor = "lapsed customer"
    focus = _after_colon(w.t("trigger.previous_focus") or w.t("customer.focus"))
    months = w.v("trigger.previous_membership_months")
    offer = w.offer_for(focus or _svc(w), prefer_free=True)
    subj = about or "You"
    if focus:
        over = f" through {'their' if about else 'your'} {months} months with us" if months else ""
        plural_offer = offer.lower().endswith(("classes", "sessions"))
        if offer:
            m.add(w.L(en=f"{subj} trained for {focus}{over}; whenever you want to pick it back up, our {offer} "
                         + ("are" if plural_offer else "offer is") + " an easy, no-pressure restart.",
                      hg=f"Focus {focus} tha{f' ({months} months hamare saath)' if months else ''}; dobara shuru "
                         f"karna ho toh hamara {offer} offer ek aasaan, no-pressure restart hai.",
                      hi=f"Focus {focus} tha; dobara shuru karne ke liye hamara {offer} offer aasaan rahega."), 0)
            m.anchor += f"; previous focus {focus}; {offer}"
        else:
            m.add(w.L(en=f"{subj} trained for {focus}{over}, and the easiest restart is a couple of easy-paced "
                         f"sessions to find the rhythm again.",
                      hg=f"Focus {focus} tha; dobara shuru karne ka sabse aasaan tarika hai kuch halke sessions.",
                      hi=f"Focus {focus} tha; dobara shuru karne ke liye kuch halke sessions sabse aasaan rahenge."), 0)
            m.anchor += f"; previous focus {focus}"
    else:
        svc = _svc(w)
        if w.cat == "pharmacies":
            m.add(w.L(en="If any regular medicines or refills are due, we can keep them ready.",
                      hg="Agar koi regular medicine ya refill due hai, hum pehle se ready rakh sakte hain.",
                      hi="Agar koi niyamit dawa ya refill chahiye, hum pehle se taiyaar rakh sakte hain."), 1)
        elif svc:
            m.add(w.L(en=f"Whenever {'it suits' if about else 'you are ready'}, we'd be glad to see "
                         f"{about or 'you'} for a {svc}.",
                      hg=f"Jab bhi time mile, {svc} ke liye aaiye.",
                      hi=f"Jab bhi samay mile, {svc} ke liye padhariye."), 1)
        if offer:
            m.add(w.L(en=f"Our {offer} offer is open if you'd like an easy way back.",
                      hg=f"Aasaan restart ke liye hamara {offer} offer available hai.",
                      hi=f"Aasaan shuruaat ke liye hamara {offer} offer uplabdh hai."), 1)
        m.add(_c_loyalty(w), 2)
    tail_en, tail_hg = _c_pref_tail(w)
    who_en = f" for {about}" if about else ""
    who_hg = f" {about} ke liye" if about else " aapke liye"
    if w.cat == "pharmacies":
        m.cta = w.L(en="Reply YES and we'll have everything ready at a time that suits you.",
                    hg="Bas YES reply karein, hum aapke convenient time pe sab ready rakhenge.",
                    hi="YES likhiye, hum aapke suvidhajanak samay par sab taiyaar rakhenge.")
    elif hard or tail_en:
        said = "no-pressure" in " ".join(p for p, _ in m.parts)
        m.cta = w.L(en=f"Reply YES and we'll hold a spot{who_en} this week{tail_en}" + ("." if said else ", no pressure."),
                    hg=f"Bas YES reply karein, hum is hafte{who_hg} ek slot hold kar denge{tail_hg}, koi pressure nahi.",
                    hi=f"YES likhiye, hum is hafte{who_hg} ek samay rakh denge, koi dabav nahi.")
    else:
        m.cta = w.L(en=w.pick(["Reply YES and we'll book a time that suits you.",
                               "Just reply YES and we'll find a time that works for you."], "cta"),
                    hg="Bas YES reply karein, hum aapke convenient time pe book kar denge.",
                    hi="YES likhiye, hum aapke suvidhajanak samay par booking kar denge.")
    m.offer = w.pb.offer
    return m


def _c_appointment(w: _W) -> _Msg:
    o = w.c_open()
    due = w.t("trigger.due_date")
    svc = _lc(due.split(" scheduled")[0]) if " scheduled" in due else "appointment"
    date = _date_in(due)
    clock = _CLOCK_RE.search(due)
    at = f" at {clock.group(0)}" if clock else ""
    intro = w.intro("ek reminder", "ek chhota sa reminder")
    loc = w.f.locality
    m = _Msg(first="", cta="", cta_type=CTA_BINARY_CONFIRM, why=f"{svc} tomorrow",
             anchor=f"{svc} tomorrow" + (f", {date}" if date else ""), lever="commitment (one-word confirm)")
    m.first = w.L(en=w.pick([f"{o} {intro} A quick reminder that {_poss(w)} {svc} is tomorrow",
                             f"{o} {intro} Just a reminder: {_poss(w)} {svc} is tomorrow"], "hook")
                  + (f", {date}" if date else "") + f"{at}.",
                  hg=f"{o} {intro} Kal" + (f", {date} ko," if date else "") + f" {_poss(w)} {svc} hai.",
                  hi=f"{o} {intro} Kal" + (f", {date} ko," if date else "") + f" {_poss(w)} {svc} hai.")
    stylist = _after_colon(w.t("customer.stylist"))
    if stylist:
        m.add(w.L(en=f"{stylist} will be ready for you.", hg=f"{stylist} aapke liye ready rahengi.",
                  hi=f"{stylist} aapke liye taiyaar rahengi."), 1)
    if loc:
        m.add(w.L(en=f"See you at our {loc} {w.biz}; everything will be set up so it's quick and on time.",
                  hg=f"Hamare {loc} {w.biz} mein sab ready rahega, taaki time bache.",
                  hi=f"Hamare {loc} {w.biz} mein sab taiyaar rahega, taaki samay bache."), 2)
    else:
        m.add(w.L(en="Everything will be set up so it's quick and on time.",
                  hg="Hum sab ready rakhenge taaki time bache.",
                  hi="Hum sab taiyaar rakhenge taaki samay bache."), 2)
    m.add(_c_loyalty(w), 3)
    m.cta = w.L(en="Reply CONFIRM to lock it in, or tell us a better time and we'll move it.",
                hg="Confirm karne ke liye CONFIRM reply karein, ya naya time bata dijiye.",
                hi="Pakka karne ke liye CONFIRM likhiye, ya samay badalna ho to bata dijiye.")
    m.offer = w.pb.offer
    return m


def _c_refill(w: _W) -> _Msg:
    mols = w.v("trigger.molecules") or []
    if w.cat == "pharmacies" or mols:
        return _c_refill_pharmacy(w, mols)
    o = w.c_open()
    svc = _svc(w) or "regular refill"
    last = _date_in(w.t("trigger.last_service_date") or w.t("customer.last_visit"))
    intro = w.intro("ek chhota sa check-in", "ek chhota sa sandesh")
    m = _Msg(first="", cta="", why=f"{svc} check-in", anchor=f"{svc}; last visit {last}" if last else svc,
             lever="helpful check-in plus effort externalisation")
    lv_hg = f" (last visit {last} ko thi)" if last else ""
    if w.cat == "gyms":
        m.first = w.L(en=f"{o} {intro} {_poss(w, cap=True)} monthly membership is up for renewal" +
                         (f" (last visit: {last})." if last else "."),
                      hg=f"{o} {intro} {_poss(w, 'f', cap=True)} monthly membership renewal due hai{lv_hg}.",
                      hi=f"{o} {intro} {_poss(w, 'f', cap=True)} monthly membership renewal ka samay aa gaya hai"
                         + (f"; pichhli visit {last} ko thi." if last else "."))
        m.add(w.L(en="Renewing now keeps the routine going without a break.",
                  hg="Abhi renew karne se routine bina break ke chalta rahega.",
                  hi="Abhi renew karne se routine bina ruke chalta rahega."), 1)
        cta = ("Reply YES and we'll renew it and confirm the usual slot.",
               "Bas YES reply karein, hum renew karke usual slot confirm kar denge.",
               "Renew karne ke liye YES likhiye, baaki hum sambhal lenge.")
    elif w.cat == "restaurants":
        m.first = w.L(en=f"{o} {intro} It's been a while" + (f" since your last visit on {last}" if last else "") +
                         ", so we thought we'd check in about your regular order.",
                      hg=f"{o} {intro} Kaafi din ho gaye{lv_hg}, aapke regular order ke liye check-in kar rahe the.",
                      hi=f"{o} {intro} Kaafi din ho gaye{lv_hg}; aapke niyamit order ke liye pooch rahe the.")
        fav = _after_colon(w.t("customer.favourite"))
        if fav:
            m.add(w.L(en=f"Your {fav} is just a message away.", hg=f"Aapka {fav} bas ek message door hai.",
                      hi=f"Aapka {fav} bas ek sandesh door hai."), 1)
        cta = ("Reply YES and we'll have your usual order ready.",
               "Bas YES reply karein, hum aapka usual order ready rakhenge.",
               "YES likhiye, hum aapka order taiyaar rakhenge.")
    else:
        items = {"dentists": ("oral-care", "the usual oral-care items"),
                 "salons": ("hair and skin care", "the usual hair and skin care products")}.get(
            w.cat, ("regular", "the usual items"))
        since = f" since the last visit on {last}" if last else ""
        m.first = w.L(en=w.pick([f"{o} {intro} {_poss(w, cap=True)} {items[0]} refill is about due, so here's a "
                                 f"quick check-in{since}.",
                                 f"{o} {intro} It's about time for {_poss(w)} {items[0]} refill" +
                                 (f"; the last visit was on {last}." if last else ".")], "hook"),
                      hg=f"{o} {intro} {_poss(w, cap=True)} {items[0]} refill due hone wala hai{lv_hg}.",
                      hi=f"{o} {intro} {_poss(w, cap=True)} {items[0]} refill ka samay aa gaya hai{lv_hg}.")
        m.add(w.L(en=f"We can keep {items[1]} ready for pickup, or book a short check if anything needs a look.",
                  hg="Usual saamaan hum ready rakh sakte hain, ya kuch check karwana ho toh ek chhoti visit book kar "
                     "dete hain.",
                  hi="Saamaan hum taiyaar rakh sakte hain, ya kuch dikhana ho to ek chhoti visit rakh dete hain."), 0)
        cta = ("Reply YES and we'll set it up at a time that suits you.",
               "Bas YES reply karein, hum aapke convenient time pe sab set kar denge.",
               "YES likhiye, hum aapke suvidhajanak samay par sab taiyaar rakhenge.")
    offer = w.offer_for(svc, prefer_free=True)
    if offer:
        m.add(_c_offer_line(w, offer), 2)
    m.add(_c_loyalty(w), 3)
    m.cta = w.L(en=cta[0], hg=cta[1], hi=cta[2])
    m.cta_type = CTA_BINARY_YES_NO
    m.offer = w.pb.offer
    return m


def _c_refill_pharmacy(w: _W, mols: list) -> _Msg:
    o = w.c_open()
    runs = _date_in(w.t("trigger.stock_runs_out"))
    about = _about(w)
    n = len(mols)
    mol_s = ", ".join(str(x) for x in mols)
    loc = w.f.locality
    who_biz = f"{w.biz_name}, {loc}" if loc and loc.lower() not in w.biz_name.lower() else w.biz_name
    m = _Msg(first="", cta="", cta_type=CTA_BINARY_CONFIRM, why="chronic refill due",
             anchor=(f"{n} regular medicines ({mol_s})" if mols else "regular refill") + (f" run out on {runs}" if runs else ""),
             lever="specificity plus effort externalisation (packed and delivered)")
    if mols:
        intro = w.L(en=f"{w.signer} here.", hg=f"{who_biz} se.", hi=f"{who_biz} se.")
        m.first = w.L(
            en=f"{o} {intro} {_poss(w, cap=True)} {n} regular medicine{'s' if n != 1 else ''} ({mol_s}) " +
               (f"run out on {runs}." if runs else "are due for a refill."),
            hg=f"{o} {intro} {_poss(w, 'f', cap=True)} {n} regular medicines ({mol_s}) " +
               (f"{runs} ko khatam ho rahi hain." if runs else "ka refill due hai."),
            hi=(f"{o} {intro} {runs} ko {_poss(w, 'f')} {n} niyamit dawaiyan ({mol_s}) khatam ho rahi hain." if runs
                else f"{o} {intro} {_poss(w, 'f', cap=True)} {n} niyamit dawaiyan ({mol_s}) ka refill due hai."))
        m.add(w.L(en="We can have the same medicines packed and ready.",
                  hg="Wahi medicines hum pack karke ready rakh sakte hain.",
                  hi="Wahi dawaiyan hum pack karke taiyaar rakh sakte hain."), 0)
    else:
        svc = _svc(w) or "monthly medicine refill"
        last = _date_in(w.t("trigger.last_service_date") or w.t("customer.last_visit"))
        intro = w.intro("ek reminder", "ek chhota sa reminder")
        m.first = w.L(en=f"{o} {intro} {_poss(w, cap=True)} {svc} is due" + (f" (last visit: {last})." if last else "."),
                      hg=f"{o} {intro} {_poss(w, cap=True)} {svc} due hai" + (f" (last visit {last})." if last else "."),
                      hi=f"{o} {intro} {_poss(w, cap=True)} {svc} ka samay ho gaya hai" +
                         (f" (pichhli visit {last})." if last else "."))
        m.add(w.L(en="We can keep the regular medicines packed and ready.",
                  hg="Regular medicines hum pack karke ready rakh sakte hain.",
                  hi="Niyamit dawaiyan hum pack karke taiyaar rakh sakte hain."), 1)
    senior = next((x for x in w.f.offers_active if "senior" in x.lower() and w.safe(x)), "")
    delivery = next((x for x in w.f.offers_active if "deliver" in x.lower() and w.safe(x)), "")
    is_senior = bool(w.t("customer.senior")) or "ji" in (w.f.about_name or "").split()
    saved = bool(w.t("trigger.delivery_address") or w.t("customer.delivery"))
    morning = "morning" in w.pref()
    if senior and is_senior:
        m.add(w.L(en=f"{senior} applies.", hg=f"{senior} lagega.", hi=f"{senior} lagega."), 1)
    if delivery or saved:
        where_en = " to the saved address" if saved else ""
        where_hi = " saved address par" if saved else ""
        when_en, when_hi = (" in the morning", " subah") if morning else ("", "")
        m.add(w.L(en=(f"With {delivery}, " if delivery else "") + f"we can deliver{where_en}{when_en}.",
                  hg=(f"{delivery} ke saath " if delivery else "") + f"hum{where_hi}{when_hi} delivery kar denge.",
                  hi=(f"{delivery} ke saath " if delivery else "") + f"hum{where_hi}{when_hi} delivery kar denge."), 1)
    m.cta = w.L(en="Reply CONFIRM to dispatch, or tell us if the prescription has changed.",
                hg="Dispatch ke liye CONFIRM reply karein, ya prescription mein koi change ho to batayein.",
                hi="Dispatch ke liye CONFIRM likhiye, ya dose mein koi badlav ho to bata dijiye.")
    m.offer = w.pb.offer
    return m


def _c_trial(w: _W) -> _Msg:
    o, slots = w.c_open(), list(w.f.slots)
    about = _about(w)
    trial_t = w.t("trigger.trial_date")
    date = _date_in(trial_t) or _date_in(w.t("trigger.last_service_date") or w.t("customer.last_visit"))
    noun = _lc(trial_t.split(" on ")[0]) if " on " in trial_t else (
        _lc(w.t("trigger.next_step").removeprefix("Follow-up after the ").strip()) or "trial")
    nxt = _next_noun(w)
    intro = w.intro("ek message", "ek sandesh")
    m = _Msg(first="", cta="", why=f"follow-up after the {noun}", anchor=f"{noun} on {date}" if date else noun,
             lever="commitment plus effort externalisation")
    on = f" on {date}" if date else ""
    if about:
        m.first = w.L(en=f"{o} {intro} Thank you for bringing {about} to the {noun}{on}.",
                      hg=f"{o} {intro} {about} ko {noun} ke liye laane ka shukriya{f' ({date})' if date else ''}.",
                      hi=f"{o} {intro} {about} ko {noun} ke liye laane ka dhanyavaad{f' ({date})' if date else ''}.")
    elif w.cat == "pharmacies":
        m.first = w.L(en=f"{o} {intro} Thanks for your {noun} with us{on}.",
                      hg=f"{o} {intro} Hamare saath {noun} ke liye thank you{f' ({date})' if date else ''}.",
                      hi=f"{o} {intro} Hamare saath {noun} ke liye dhanyavaad{f' ({date})' if date else ''}.")
    else:
        m.first = w.L(en=f"{o} {intro} Thanks for coming in for your {noun}{on}.",
                      hg=f"{o} {intro} {date + ' ko ' if date else ''}{noun} ke liye aane ka thank you.",
                      hi=f"{o} {intro} {date + ' ko ' if date else ''}{noun} ke liye aane ka dhanyavaad.")
    if slots:
        pref = w.pref()
        match = _slot_pref_match(slots, pref)
        label = w.join_or(slots)
        kid = "kids " if about and w.cat == "gyms" else ""
        m.add(w.L(en=f"The next {kid}{nxt} is {label}" + (f", on {pref} as you prefer." if match else "."),
                  hg=f"Next {nxt} {label} hai" + (f" ({pref}, jaisa aapko pasand hai)." if match else "."),
                  hi=f"Agla {nxt} {label} ko hai."), 0)
        m.anchor += f"; next {nxt} {_join(slots, 'or')}"
    else:
        loc = w.f.locality
        at_en = f" at our {loc} {w.biz}" if loc else ""
        at_hg = f" hamare {loc} {w.biz} mein" if loc else ""
        if w.cat == "gyms":
            m.add(w.L(en=f"The first couple of weeks are when a new routine sticks best, so let's lock in the next "
                         f"{nxt}{at_en}.",
                      hg=f"Shuru ke do hafton mein routine sabse achhe se banta hai, toh agla {nxt}{at_hg} fix kar "
                         f"lete hain.",
                      hi=f"Shuru ke do hafton mein aadat sabse achhe se banti hai, isliye agla {nxt}{at_hg} tay kar "
                         f"lete hain."), 1)
        elif w.cat == "pharmacies":
            m.add(w.L(en="For anything needed regularly, we can keep it ready so there's never a wait.",
                      hg="Jo cheezein regularly chahiye, hum pehle se ready rakh sakte hain.",
                      hi="Jo cheezein niyamit chahiye, hum pehle se taiyaar rakh sakte hain."), 1)
        elif w.cat == "dentists":
            m.add(w.L(en=f"If a follow-up was advised, booking it early keeps things on track{at_en}.",
                      hg=f"Agar follow-up ki salah di gayi thi, toh use jaldi book karna behtar hai.",
                      hi=f"Agar follow-up ki salah di gayi thi, to use jaldi rakh lena achha rahega."), 1)
        else:
            stylist = _after_colon(w.t("customer.stylist"))
            m.add(w.L(en=f"We'd love to have you back{at_en}" + (f", with {stylist} again." if stylist else "."),
                      hg=f"Aapko{at_hg} dobara dekh ke khushi hogi.",
                      hi=f"Aapko{at_hg} phir se dekhkar khushi hogi."), 1)
    offer = w.offer_for("month membership first", prefer_free=False)
    if offer:
        m.add(w.L(en=f"If {about} enjoyed it, {offer} is the easy way to continue." if about else
                     f"If you'd like to continue, our {offer} offer makes it easy to start.",
                  hg=f"Continue karna ho toh {offer} se shuru karna easy hai.",
                  hi=f"Aage jaari rakhna ho to {offer} se shuruaat aasaan hai."), 1)
    if slots:
        m.cta, m.cta_type = _slot_cta(w, slots, about)
    elif w.cat == "pharmacies":
        m.cta = w.L(en="Reply YES and we'll keep the next order ready.",
                    hg="Bas YES reply karein, hum next order ready rakhenge.",
                    hi="YES likhiye, hum agla order taiyaar rakhenge.")
    else:
        what = {"restaurants": "a table for the next visit"}.get(w.cat, f"the next {nxt}")
        m.cta = w.L(en=f"Reply YES and we'll set up {what} at a time that suits you.",
                    hg=f"Bas YES reply karein, hum next {nxt} aapke convenient time pe book kar denge.",
                    hi=f"YES likhiye, hum agla {nxt} aapke suvidhajanak samay par rakh denge.")
    m.offer = w.pb.offer
    return m


def _c_wedding(w: _W) -> _Msg:
    o = w.c_open()
    wedding = _date_in(w.t("trigger.wedding_date") or w.t("customer.wedding_date"))
    days_f = w.fact("trigger.days_to_wedding")
    days = days_f.value if days_f and isinstance(days_f.value, int) else None
    trial = _date_in(w.t("trigger.trial_date"))
    nxt = w.t("trigger.next_step").removeprefix("Window now open for the ").strip()
    intro = f"{w.signer} here 💍" if not (w.hi or w.hg) else f"{w.biz_name} ki taraf se 💍"
    m = _Msg(first="", cta="", why="bridal follow-up window",
             anchor=(f"{days} days to the wedding on {wedding}" if days else f"wedding on {wedding}"),
             lever="anticipation plus effort externalisation")
    her_en, her_hg = _poss(w), _poss(w, "f", cap=True)
    if days and wedding:
        m.first = w.L(en=f"{o} {intro} {days} days to go until {her_en} wedding on {wedding}!",
                      hg=f"{o} {intro} {her_hg} wedding {wedding} ko hai, yaani {days} din baaki!",
                      hi=f"{o} {intro} {her_hg} shaadi {wedding} ko hai, yaani {days} din baaki!")
    elif wedding:
        m.first = w.L(en=f"{o} {intro} {_cap(her_en)} wedding on {wedding} is getting closer!",
                      hg=f"{o} {intro} {her_hg} wedding {wedding} ko hai!",
                      hi=f"{o} {intro} {her_hg} shaadi {wedding} ko hai!")
    else:
        m.first = w.L(en=f"{o} {intro} Checking in about {her_en} bridal prep.",
                      hg=f"{o} {intro} {her_hg} bridal prep ke liye check-in.",
                      hi=f"{o} {intro} {her_hg} bridal taiyaari ke liye check-in.")
    if nxt:
        since = f"Since your bridal trial on {trial}, " if trial else ""
        m.add(w.L(en=f"{since}this is the right time to start the {nxt}, so nothing is rushed closer to the day.",
                  hg=f"{'Bridal trial ' + trial + ' ko hua tha; ' if trial else ''}ab {nxt} shuru karne ka sahi time "
                     f"hai, taaki end mein koi rush na ho.",
                  hi=f"Ab {nxt} shuru karne ka sahi samay hai, taaki baad mein jaldbaazi na ho."), 0)
        m.anchor += f"; {nxt} window open"
    else:
        m.add(w.L(en="Starting your prep sessions early means nothing is rushed closer to the day.",
                  hg="Prep sessions jaldi shuru karne se end mein koi rush nahi hota.",
                  hi="Taiyaari jaldi shuru karne se baad mein jaldbaazi nahi hoti."), 1)
    pref = w.pref()
    if pref:
        m.add(w.L(en=f"We can keep every session on {pref}, as you prefer.",
                  hg=f"Aapke sessions {pref} pe rakh sakte hain, jaisa aapko pasand hai.",
                  hi=f"Aapke sessions {pref} ko rakh sakte hain."), 1)
    day = pref.rstrip("s") if pref and re.fullmatch(r"(sat|sun|mon|tue|wed|thu|fri)[a-z]*", pref.lower()) else ""
    m.cta = w.L(en=f"Reply YES and we'll block a {day} for your first session." if day else
                "Reply YES and we'll share the first available dates for your first session.",
                hg=f"Bas YES reply karein, hum pehle session ke liye ek {day} block kar denge." if day else
                "Bas YES reply karein, hum pehle session ki available dates bhej denge.",
                hi="YES likhiye, hum pehle session ki taarikh rakh denge.")
    m.offer = w.pb.offer
    return m


def _c_slot_open(w: _W) -> _Msg:
    o, slots = w.c_open(), list(w.f.slots)
    if not slots:
        return _c_generic(w)
    about = _about(w)
    intro = w.intro("ek update", "ek soochna")
    word = "table" if w.cat == "restaurants" else "slot"
    for_en = f" for {about}" if about else ""
    for_hg = f"{about} ke liye " if about else ""
    m = _Msg(first=w.L(en=f"{o} {intro} A {word} just opened up{for_en}: {slots[0]}.",
                       hg=f"{o} {intro} " + _cap(f"{for_hg}ek slot abhi khula hai: {slots[0]}."),
                       hi=f"{o} {intro} " + _cap(f"{for_hg}ek samay abhi khaali hua hai: {slots[0]}.")),
             cta="", why="a slot just opened", anchor=f"open slot {slots[0]}", lever="scarcity plus specificity")
    pref = w.pref()
    if pref and _slot_pref_match(slots[:1], pref):
        m.add(w.L(en=f"It's on {pref}, which we know suits you.", hg=f"Yeh {pref} pe hai, jo aapko suit karta hai.",
                  hi=f"Yeh {pref} ko hai, jo aapko theek rehta hai."), 1)
    offer = w.offer_for()
    if offer:
        m.add(_c_offer_line(w, offer), 2)
    m.add(w.L(en="Open slots like this usually go quickly.", hg="Aise slots jaldi bhar jaate hain.",
              hi="Aise samay jaldi bhar jaate hain."), 3)
    m.cta, m.cta_type = _slot_cta(w, slots, about)
    m.offer = w.pb.offer
    return m


def _c_generic(w: _W) -> _Msg:
    o = w.c_open()
    anchors = [f for f in w.f.anchor_facts if w.safe(f.text) and "?" not in f.text and f.key != "trigger.kind"
               and w.grounded(f.text)]
    lead = anchors[0].text if anchors else ""
    intro = w.intro("ek update", "ek soochna")
    m = _Msg(first=w.L(en=f"{o} {intro} " + (_end(lead) if lead else "A quick note from us."),
                       hg=f"{o} {intro} " + (_end(lead) if lead else ""),
                       hi=f"{o} {intro} " + (_end(lead) if lead else "")),
             cta="", why=w.kind.replace("_", " ") or "customer update", anchor=lead or "customer update",
             lever="specificity plus effort externalisation")
    for f in anchors[1:2]:
        m.add(_end(f.text), 2)
    m.add(_c_loyalty(w), 3)
    m.cta = w.L(en="Reply YES and we'll take care of it for you.",
                hg="Bas YES reply karein, baaki hum sambhal lenge.",
                hi="YES likhiye, baaki hum sambhal lenge.")
    m.offer = w.pb.offer
    return m


# --------------------------------------------------------------------------- dispatch + assembly

_MERCHANT: dict[str, Callable[[_W], _Msg]] = {
    "research_digest": _m_research, "regulation_change": _m_regulation, "cde_opportunity": _m_cde,
    "supply_alert": _m_supply, "category_seasonal": _m_seasonal, "category_trend_movement": _m_trend,
    "festival_upcoming": _m_festival, "ipl_match_today": _m_ipl, "weather_heatwave": _m_event,
    "local_news_event": _m_event, "competitor_opened": _m_competitor, "perf_dip": _m_perf_dip,
    "perf_spike": _m_perf_spike, "seasonal_perf_dip": _m_seasonal_dip, "milestone_reached": _m_milestone,
    "review_theme_emerged": _m_review, "renewal_due": _m_renewal, "winback_eligible": _m_winback,
    "dormant_with_vera": _m_dormant, "gbp_unverified": _m_gbp, "curious_ask_due": _m_curious,
    "scheduled_recurring": _m_curious, "active_planning_intent": _m_planning,
}
_CUSTOMER: dict[str, Callable[[_W], _Msg]] = {
    "recall_due": _c_recall, "customer_lapsed_soft": _c_lapsed, "customer_lapsed_hard": _c_lapsed,
    "appointment_tomorrow": _c_appointment, "chronic_refill_due": _c_refill, "trial_followup": _c_trial,
    "wedding_package_followup": _c_wedding, "unplanned_slot_open": _c_slot_open,
}
_CUSTOMER_KINDS = set(_CUSTOMER)

_CTA_DESC = {
    CTA_BINARY_YES_NO: "a single yes/no ask",
    CTA_BINARY_CONFIRM: "a one-word CONFIRM",
    CTA_MULTI_CHOICE_SLOT: "a slot choice with the exact labels",
    CTA_OPEN_ENDED: "one open question",
}


def render(facts: FactSheet, pb: Playbook) -> Draft:
    """Deterministic draft for (facts, playbook). Never raises; always returns a non-empty body."""
    facts = facts if isinstance(facts, FactSheet) else FactSheet()
    pb = pb if isinstance(pb, Playbook) else Playbook()
    try:
        w = _W(facts, pb)
    except Exception:
        log.exception("template view failed for %s", getattr(facts, "trigger_id", "?"))
        return Draft(body="Hi, a quick update from us. Reply YES and we'll take care of it for you.",
                     cta=CTA_BINARY_YES_NO, rationale="Fallback message.", template_params=["", "", ""],
                     offer=getattr(pb, "offer", ""), source="template")
    table = _CUSTOMER if w.customer else _MERCHANT
    fn = table.get(w.kind)
    if fn is None:
        if w.customer:
            fn = _c_generic
        elif w.kind in _CUSTOMER_KINDS:
            fn = _m_customer_kind_for_merchant
        else:
            fn = _m_generic
    for attempt in (fn, _c_generic if w.customer else _m_generic):
        try:
            msg = attempt(w)
            if not msg.cta:
                msg.cta = w.L(en="Want me to take care of it?", hg="Kya main ise sambhal loon?")
            return _finish(w, msg)
        except Exception:  # a bug in one renderer must not lose the message
            log.exception("template renderer %s failed for %s", getattr(attempt, "__name__", attempt), facts.trigger_id)
    return _minimal_draft(w)


def _minimal_draft(w: _W) -> Draft:
    who = w.sal or ("there" if not w.customer else "")
    body = (f"Hi {who}, " if who else "Hi, ") + (
        "a quick update from us. Reply YES and we'll take care of it for you." if w.customer
        else "a quick update on your profile this week. Want me to put together a short plan for it?")
    return Draft(body=body, cta=CTA_BINARY_YES_NO, rationale=f"Fallback message for {w.kind.replace('_', ' ')}.",
                 template_params=[who, body, ""], offer=w.pb.offer, source="template")


def _polish(text: str, w: _W) -> str:
    s = _URL_RE.sub("", text)
    s = s.replace("_", " ")
    s = re.sub(r"\bDr\.?\s+Dr\.?\s*", "Dr. ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" +([,.;:!?])", r"\1", s)
    s = re.sub(r"\.\.+", ".", s)
    s = re.sub(r"([.!?])\.", r"\1", s)
    s = re.sub(r",\.", ".", s)
    s = re.sub(r" *\n *", "\n", s)
    # a bare greeting ("Hi,") followed by a capitalised sentence start reads as a typo
    s = re.sub(r"^((?:Hi|Hello|Namaste|Vanakkam|Namaskaram|Namaskara|Namaskar)(?: there)?,) "
               r"(Your|It's|Just|A|The|Kal|Aapka|Aapki)\b", lambda m: f"{m.group(1)} {m.group(2).lower()}", s)
    return s.strip()


def _anchor_in_body(anchor: str, body: str) -> bool:
    """True when the anchor's numbers (or most of its words) appear in the body (keeps the rationale faithful)."""
    nums = [normalize_number(t) for t in extract_number_tokens(anchor)]
    body_nums = {normalize_number(t) for t in extract_number_tokens(body)}
    if nums:
        return all(n in body_nums for n in nums if n)
    words = [x for x in re.findall(r"[a-z]{4,}", anchor.lower())]
    if not words:
        return True
    low = body.lower()
    return sum(1 for x in words if x in low) >= 0.6 * len(words)


def _finish(w: _W, msg: _Msg) -> Draft:
    limit = msg.max_len or (CUSTOMER_MAX if w.customer else MERCHANT_MAX)
    first = _polish(_end(msg.first), w)
    cta = _polish(_end(msg.cta), w)
    for label, text in (("hook", first), ("cta", cta)):
        bad = _ungrounded(text, w.allowed)
        if bad:  # never expected: every renderer builds these from fact texts
            log.warning("template %s for %s has ungrounded numbers %s", label, w.f.trigger_id, bad)
    parts: list[tuple[int, str, int]] = []
    for i, (text, prio) in enumerate(msg.parts):
        t = _polish(text, w)
        if not t or not w.safe(t):
            continue
        if _ungrounded(t, w.allowed):
            log.info("dropping ungrounded sentence from %s draft: %r", w.f.trigger_id, t)
            continue
        if "?" in t:                        # the CTA is the only question in the message
            if prio > 0:
                continue
            t = t.replace("?", ".")
        parts.append((i, t if (msg.newline_parts and t.startswith("•")) else _end(t), prio))

    def assemble(ps: list[tuple[int, str, int]]) -> str:
        out = first
        prev_bullet = False
        for _, p, _ in sorted(ps):
            bullet = msg.newline_parts and p.startswith("•")
            out += ("\n" + p) if (bullet or prev_bullet) else (" " + p)
            prev_bullet = bullet
        return out + ("\n" if prev_bullet else " ") + cta

    body = assemble(parts)
    while len(body) > limit:
        droppable = [p for p in parts if p[2] > 0]
        if not droppable:
            break
        parts.remove(max(droppable, key=lambda p: (p[2], p[0])))
        body = assemble(parts)
    body = _polish(body, w)
    if body and body[0].islower():
        body = body[0].upper() + body[1:]

    opener = _polish(w.c_open() if w.customer else w.m_open(), w)
    if first.startswith(opener):
        hook = first[len(opener):].strip()
    else:
        head = first.split(",", 1)
        hook = head[1].strip() if len(head) > 1 and len(head[0]) <= 40 else first
    p1 = w.sal or opener.rstrip(",").strip()
    core = body[len(first):len(body) - len(cta)].strip() if body.startswith(first) and body.endswith(cta) else body
    joiner = "\n" if core.startswith("•") and msg.newline_parts else " "
    params = [p1, _polish(f"{hook}{joiner}{core}".strip(), w), cta]

    cta_desc = _CTA_DESC.get(msg.cta_type, "a single ask")
    offer = msg.offer or w.pb.offer
    why = _strip_end(msg.why) or w.kind.replace("_", " ")
    anchors = [a.strip() for a in _strip_end(msg.anchor).split("; ") if a.strip()]
    kept = [a for a in anchors if _anchor_in_body(a, body)] or [_strip_end(hook)[:140]]
    anchor = "; ".join(kept)[:200]
    rationale = f"Why now: {why}. Anchored on {anchor}; lever: {msg.lever or 'specificity'}; closes with {cta_desc}" + \
                (f" to {offer}." if offer and msg.cta_type != CTA_OPEN_ENDED else ".")
    rationale = rationale.replace("_", " ")
    return Draft(body=body, cta=msg.cta_type, rationale=rationale, template_params=params, offer=offer,
                 source="template")
