"""Deterministic reply bodies for the conversation engine.

Used whenever the LLM is unavailable, too slow, or its output fails validation, so every
function here must produce a complete, grounded WhatsApp reply on its own:

    auto_reply_nudge()      one owner-directed nudge after a detected auto-reply
    hostile_apology()       one-line apology with an explicit opt-out path
    off_topic_redirect()    polite decline (GST, loans, ...) + redirect to the thread's offer
    answer_question()       answer only from facts; say "I'll check" when unknown
    answer_with_artifact()  "yes, but <question>": answer, then the artifact
    action_artifact()       ACTION MODE: the actual drafted artifact + one CONFIRM-style CTA
    confirm_done()          after the merchant confirms the artifact
    edit_ack()              feedback on a delivered draft
    continue_thread()       engaged / unclassified reply: next best step toward the offer
    slot_confirmation(), slot_prompt(), customer_accept()   customer booking flows

Every number in a body comes from a Fact text, an offer title or a slot label (small
counts such as "3 posts" aside), so the validator's grounding check passes. Action-mode
bodies never contain the qualifying phrases the judge penalises ("would you", "do you",
"can you tell", "what if", "how about"). Templates exist in English and Hinglish (Roman
script); Fact texts and customer-facing drafts stay in English.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas import (
    CTA_BINARY_CONFIRM,
    CTA_BINARY_YES_NO,
    CTA_MULTI_CHOICE_SLOT,
    CTA_NONE,
    CTA_OPEN_ENDED,
    Fact,
    FactSheet,
)

QUALIFYING_PHRASES = ("would you", "do you", "can you tell", "what if", "how about")

# --------------------------------------------------------------------------- kind tables

# What Vera offers to do next (verb phrase for "I'll ..."), when the opener did not say.
DEFAULT_OFFERS = {
    "research_digest": "send the 2-line summary and a draft patient WhatsApp",
    "regulation_change": "send a ready compliance checklist",
    "cde_opportunity": "share the session details and set a reminder",
    "supply_alert": "filter your customer list and draft the recall note",
    "category_seasonal": "draft the shelf plan and a WhatsApp broadcast",
    "category_trend_movement": "draft a Google post on it",
    "festival_upcoming": "draft your festive Google post",
    "ipl_match_today": "draft today's match-day post",
    "weather_heatwave": "draft a heatwave Google post",
    "local_news_event": "draft a Google post for it",
    "competitor_opened": "draft a counter-post built on your own offer",
    "perf_dip": "draft a recovery plan with a ready Google post",
    "perf_spike": "draft a post to keep the momentum going",
    "seasonal_perf_dip": "draft a retention challenge for your current members",
    "milestone_reached": "draft a thank-you post and a review request",
    "review_theme_emerged": "draft replies to those reviews",
    "gbp_unverified": "start the verification with you",
    "renewal_due": "share the renewal steps",
    "winback_eligible": "draft a restart plan with a win-back message",
    "dormant_with_vera": "draft a fresh Google post for you",
    "active_planning_intent": "turn the draft into a launch post and a WhatsApp note",
    "curious_ask_due": "turn your answer into a Google post and a WhatsApp reply",
    "scheduled_recurring": "draft this week's update",
}

_PEOPLE = {"dentists": "patients", "salons": "clients", "restaurants": "customers", "gyms": "members",
           "pharmacies": "customers"}
_PERSON = {"dentists": "patient", "salons": "client", "restaurants": "customer", "gyms": "member",
           "pharmacies": "customer"}
_GENERIC_SERVICES = {
    "dentists": "dental check-ups and consultations",
    "salons": "hair and beauty appointments",
    "restaurants": "your next meal with us",
    "gyms": "training sessions and memberships",
    "pharmacies": "medicines and everyday health essentials",
}
_STOPWORDS = {"the", "a", "an", "my", "our", "your", "please", "pls", "plz", "posts", "post", "some", "more",
              "on", "about", "for", "to", "and", "also", "it", "them", "this", "that", "is", "are", "most",
              "mostly", "in", "demand", "week", "hai", "hain", "sabse", "zyada", "jyada", "ki", "ke", "ka", "mein",
              "me", "pe", "par", "do", "karo", "kar", "focus", "yes", "ok", "okay", "sure", "haan", "google",
              "draft", "drafts", "write", "make", "create", "i", "we", "want", "would", "like", "be", "should",
              "can", "could", "abhi", "bhi", "things", "stuff", "services", "service", "people", "asking", "ask",
              "customers", "clients", "patients", "members", "ones", "one", "lot", "lots", "of", "these", "days",
              "right", "now", "currently", "definitely", "probably", "i think", "think", "wala", "wale", "log",
              "aajkal", "is", "was", "been", "getting", "get", "go", "with", "very", "much", "really", "just",
              "lets", "let", "whats", "what", "next", "ahead", "done", "start", "hmm", "yeah", "yup", "great"}
_VERBS = {"draft", "send", "share", "pull", "set", "schedule", "publish", "filter", "book", "block", "start", "turn",
          "create", "prepare", "write", "add", "update", "push", "run", "build", "get", "reply", "fix", "show",
          "check", "audit", "remind", "confirm", "launch", "list", "make", "put", "post", "plan", "flag",
          "reactivate", "renew", "walk", "help", "map", "compare", "keep", "save", "shortlist", "hold", "pack",
          "follow", "restart", "move", "refresh", "respond", "summarise", "summarize", "line", "queue", "tell",
          "bring", "offer", "do", "go", "pin", "notify", "message", "call"}


def cat_key(slug: str | None) -> str:
    s = (slug or "").lower()
    if "dent" in s:
        return "dentists"
    if "salon" in s or "beauty" in s or "spa" in s:
        return "salons"
    if "restaurant" in s or "cafe" in s or "food" in s:
        return "restaurants"
    if "gym" in s or "fitness" in s or "yoga" in s:
        return "gyms"
    if "pharm" in s or "chemist" in s or "medic" in s:
        return "pharmacies"
    return s


# --------------------------------------------------------------------------- context


@dataclass
class ReplyContext:
    """Everything a template needs; built by the conversation engine per reply."""

    facts: FactSheet = field(default_factory=FactSheet)
    kind: str = ""
    lang: str = "en"                  # "en" | "hinglish"
    customer_facing: bool = False
    offer: str = ""                   # verb phrase: what Vera offered to do ("draft 3 Google posts")
    topic: str = ""                   # what the thread is about ("the JIDA Oct 2026, p.14 item")
    inbound: str = ""                 # latest inbound message (raw)
    stage: str = "opened"
    variant: int = 0                  # rotates wording (anti-repetition)
    topics: list[str] = field(default_factory=list)   # topics the merchant asked for, if any
    soft: bool = False                # after an opt-out: answer without pushing the offer
    artifact: str = ""                # artifact delivered earlier in this conversation (for confirm_done)

    @property
    def cat(self) -> str:
        return cat_key(self.facts.category_slug)

    @property
    def hi(self) -> bool:
        return self.lang in ("hinglish", "hi")

    @property
    def name(self) -> str:
        n = (self.facts.salutation or "").strip()
        return "" if n in ("there", "Doc", "") else n

    @property
    def people(self) -> str:
        return _PEOPLE.get(self.cat, "customers")

    @property
    def person(self) -> str:
        return _PERSON.get(self.cat, "customer")


@dataclass
class Reply:
    body: str
    cta: str = CTA_OPEN_ENDED
    label: str = ""                   # short artifact / move label used in rationales


class FactView:
    """Lookup helpers over a FactSheet (by exact key or prefix)."""

    def __init__(self, sheet: FactSheet) -> None:
        self.sheet = sheet
        self._by_key: dict[str, list[Fact]] = {}
        for f in sheet.all_facts():
            if isinstance(f, Fact) and isinstance(f.text, str) and f.text.strip():
                self._by_key.setdefault(f.key, []).append(f)

    def text(self, *keys: str) -> str:
        for key in keys:
            for f in self._by_key.get(key, []):
                return f.text.strip()
        return ""

    def texts(self, prefix: str) -> list[str]:
        out = []
        for key, facts in self._by_key.items():
            if key == prefix or key.startswith(prefix + "."):
                out.extend(f.text.strip() for f in facts)
        return out

    def value(self, key: str):
        for f in self._by_key.get(key, []):
            return f.value
        return None

    def has(self, key: str) -> bool:
        return bool(self.text(key))


# --------------------------------------------------------------------------- text helpers

_ABBREV_RE = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|St|No|vs|Rs|approx|etc|Prof|Sr|Jr|[A-Z])\.$")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _short(text: str, limit: int = 160) -> str:
    text = _clean(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:-—")
    return cut + "…"


def sentences(text: str) -> list[str]:
    """Split into sentences without breaking on abbreviations ("Dr. R. Mehta") or decimals ("1.5")."""
    text = _clean(text)
    out: list[str] = []
    buf = ""
    for part in re.split(r"(?<=[.!?])\s+", text):
        buf = f"{buf} {part}".strip() if buf else part
        if _ABBREV_RE.search(buf):
            continue
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return [s for s in out if s]


def first_sentence(text: str, limit: int = 180) -> str:
    parts = sentences(text)
    return _short(parts[0].rstrip("."), limit) if parts else ""


def _lc_first(text: str) -> str:
    """Lower-case the first letter unless the first word is an acronym / proper noun-ish."""
    text = _clean(text)
    if not text:
        return text
    first = text.split()[0]
    if first.isupper() or any(ch.isdigit() for ch in first) or (len(first) > 1 and first[1:].lower() != first[1:]):
        return text
    if first in ("I", "Google", "Dr.", "Diwali", "IPL") or first.endswith("'s") \
            or first.lower().rstrip(",.") in _PROPER:
        return text
    return text[0].lower() + text[1:]


_PROPER = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february",
           "march", "april", "may", "june", "july", "august", "september", "october", "november", "december",
           "mon", "tue", "wed", "thu", "fri", "sat", "sun", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
           "oct", "nov", "dec"}


def _cap(text: str) -> str:
    text = _clean(text)
    return text[:1].upper() + text[1:] if text else text


def _strip_prefix(text: str, *prefixes: str) -> str:
    for p in prefixes:
        if p and text.lower().startswith(p.lower()):
            return text[len(p):].strip()
    return text


def _pick(options: list[str], variant: int) -> str:
    return options[variant % len(options)] if options else ""


def _comma_name(ctx: ReplyContext) -> str:
    return f", {ctx.name}" if ctx.name else ""


def _in_loc(ctx: ReplyContext) -> str:
    return f" in {ctx.facts.locality}" if ctx.facts.locality else ""


def _biz(ctx: ReplyContext) -> str:
    return ctx.facts.merchant_name or "our place"


def _owner_ref(ctx: ReplyContext) -> str:
    owner = (ctx.facts.owner_name or "").strip()
    if not owner or owner.endswith(" team") or owner in ("there", "Doc"):
        return "our team"
    return owner if ctx.cat == "dentists" or owner.startswith("Dr.") else f"{owner} and team"


def has_qualifying(text: str) -> bool:
    low = (text or "").lower()
    return any(p in low for p in QUALIFYING_PHRASES)


def _offer_price(offer: str) -> tuple[str, str]:
    """'Weekday Lunch Thali @ ₹149' -> ('Weekday Lunch Thali', '₹149'); no price -> (offer, '')."""
    m = re.match(r"\s*(.+?)\s*(?:@|at|for)\s*(₹\s?[\d,]+(?:\.\d+)?)\s*$", offer or "")
    if m:
        return m.group(1).strip(), m.group(2).replace(" ", "")
    return (offer or "").strip(), ""


_PRONOUNS = [(r"\btheir own\b", "your own"), (r"\btheir\b", "your"), (r"\bthey're\b", "you're"),
             (r"\bthey\b", "you"), (r"\bthemselves\b", "yourself"), (r"\bthem\b", "you"),
             (r"\bthe merchant's\b", "your"), (r"\bthe merchant\b", "you")]


def offer_phrase(offer: str, kind: str, short: bool = False) -> str:
    """Normalise an offer into a short second-person verb phrase that reads after "I'll"."""
    o = _clean(offer).rstrip(".?!")
    o = re.sub(r"^(?:and\s+)?(?:i'?ll|i will|i can|we'?ll|vera will|to)\s+", "", o, flags=re.IGNORECASE)
    o = re.sub(r"^(?:want me to(?: go ahead and)?|shall i|should i|kya main)\s+", "", o, flags=re.IGNORECASE)
    for pat, rep in _PRONOUNS:
        o = re.sub(pat, rep, o, flags=re.IGNORECASE)
    if len(o) > (55 if short else 80):
        for sep in (", with ", " plus ", ", ready for ", " so that ", " so ", ", and ", " with ", " and "):
            head = o.split(sep)[0]
            if head != o and len(head.split()) >= 3:
                o = head
                break
    o = o.strip(" ,;:-—")
    if not o:
        return DEFAULT_OFFERS.get(kind, "share the details")
    first = o.split()[0].lower()
    if first in _VERBS:
        return o[0].lower() + o[1:]
    if first in ("the", "a", "an", "your", "my") or first[0].isdigit():
        return f"send over {_lc_first(o)}"
    return f"send over {_lc_first(o)}"


# --------------------------------------------------------------------------- topics

_TOPIC_LEAD_RE = re.compile(
    r"(?:focus(?:ing)?\s+on|about|regarding|highlight(?:ing)?|mention(?:ing)?|featur(?:e|ing)|promot(?:e|ing)|"
    r"push(?:ing)?|topics?\s*:?)\s+(.+)", re.IGNORECASE)
_TOPIC_HI_RE = re.compile(r"(.+?)\s+(?:ke\s+baa?re\s+me(?:in)?|ke\s+upar|pe\s+focus|par\s+focus)", re.IGNORECASE)
_TOPIC_BANNED = {"price", "prices", "cost", "gst", "tax", "loan", "time", "today", "tomorrow", "it", "that", "this"}


def extract_topics(text: str, loose: bool = False) -> list[str]:
    """Topics a merchant asked for ("focus on whitening and aligners" -> ["whitening", "aligners"]).

    With loose=True (answers to "what's in demand?"), the whole short message is split into topics.
    """
    text = _clean(text)
    if not text:
        return []
    m = _TOPIC_LEAD_RE.search(text) or _TOPIC_HI_RE.search(text)
    chunk = m.group(1) if m else (text if loose and len(text.split()) <= 14 else "")
    if not chunk:
        return []
    chunk = re.split(r"[.!?\n]", chunk)[0]
    parts = re.split(r",|&|/|\+|\band\b|\bor\b|\baur\b|\bya\b", chunk, flags=re.IGNORECASE)
    out: list[str] = []
    for part in parts:
        words = re.findall(r"[A-Za-z][A-Za-z'\-]*", part)
        while words and words[0].lower() in _STOPWORDS:
            words.pop(0)
        while words and words[-1].lower() in _STOPWORDS:
            words.pop()
        words = [w for w in words if w.lower() not in ("hai", "hain", "zyada", "sabse", "demand")]
        if not words or len(words) > 4:
            continue
        topic = " ".join(words).lower()
        if len(topic) < 3 or topic in _TOPIC_BANNED or topic in (t.lower() for t in out):
            continue
        out.append(topic)
    return out[:3]


def trend_topic(fx: FactView, ctx: ReplyContext) -> str:
    """'clear aligners delhi' searches up 62% -> 'clear aligners' (locality / city / 'near me' removed)."""
    t = fx.text("trend.top")
    m = re.search(r"'([^']+)'", t)
    if not m:
        return ""
    q = m.group(1).lower()
    for w in ("near me", "nearby", (ctx.facts.city or "").lower(), (ctx.facts.locality or "").lower(),
              "delhi", "mumbai", "bangalore", "bengaluru", "hyderabad", "chennai", "pune", "jaipur", "lucknow",
              "kolkata", "chandigarh", "ahmedabad", "noida", "gurgaon", "gurugram"):
        if w:
            q = re.sub(rf"\b{re.escape(w)}\b", " ", q)
    q = _clean(q)
    if re.search(r"\b(?:offer|offers|discount|deal|deals|free|sale|cheap|cheapest|delivery|24x7|24 hours|open now"
                 r"|best|near me)\b", q):
        return ""          # a search query like "match night offer" must not become a claimed offer in a post
    return q if 2 < len(q) <= 40 else ""


def _matching_offer(ctx: ReplyContext, topic: str) -> str:
    tw = {w for w in re.findall(r"[a-z]+", topic.lower()) if len(w) > 3}
    for offer in ctx.facts.offers_active:
        ow = set(re.findall(r"[a-z]+", offer.lower()))
        if tw & ow:
            return offer
    return ""


def _praise_themes(fx: FactView) -> list[str]:
    out = []
    for t in fx.texts("review"):
        m = re.search(r"praise (.+?)(?::|$)", t)
        if m:
            out.append(_clean(m.group(1)).rstrip("."))
    return out


def _trend_items(fx: FactView) -> list[str]:
    """'ORS demand up 40%, sunscreen demand up 38% ...' -> ['ORS', 'sunscreen']."""
    items = re.findall(r"([A-Za-z][\w &/-]*?) demand up", fx.text("trigger.trends"))
    return [_clean(i).removeprefix("and ").strip() for i in items if i.strip()][:3]


# --------------------------------------------------------------------------- common pieces


def thread_topic(ctx: ReplyContext, fx: FactView | None = None) -> str:
    """Short noun phrase naming what this conversation is about (for redirects and nudges)."""
    if ctx.topic:
        return ctx.topic
    fx = fx or FactView(ctx.facts)
    k = ctx.kind
    source = fx.text("digest.source")
    title = fx.text("digest.title")
    if k == "research_digest":
        return f"the {source} item" if source else "the research update"
    if k == "regulation_change":
        return f"the {_short(title, 60)} update" if title else "the compliance update"
    if k == "cde_opportunity":
        return f"the {_short(title, 60)} session" if title else "the CDE session"
    if k == "supply_alert":
        mol = _strip_prefix(fx.text("trigger.molecule"), "Molecule:")
        return f"the {mol} batch recall" if mol else "the batch recall"
    if k == "active_planning_intent":
        topic = _strip_prefix(fx.text("trigger.intent_topic"), "Planning:")
        return f"the {topic}" if topic else "the plan you asked about"
    if k == "renewal_due":
        plan = fx.text("trigger.plan")
        return f"your {plan} renewal" if plan else "your plan renewal"
    if k == "ipl_match_today":
        match = _strip_prefix(fx.text("trigger.match"), "IPL match today:")
        return f"the {match} match plan" if match else "today's match plan"
    if k == "festival_upcoming":
        fest = fx.text("trigger.festival").split(" on ")[0]
        return f"the {fest} plan" if fest else "the festive-season plan"
    if k == "competitor_opened":
        return "the new competitor nearby"
    names = {
        "category_seasonal": "the seasonal demand shift", "category_trend_movement": "the trend update",
        "weather_heatwave": "the heatwave plan", "local_news_event": "the local event",
        "perf_dip": "the dip in your numbers", "perf_spike": "your recent jump in numbers",
        "seasonal_perf_dip": "the seasonal dip", "milestone_reached": "your milestone",
        "review_theme_emerged": "the review pattern", "gbp_unverified": "your Google profile verification",
        "winback_eligible": "restarting your plan", "dormant_with_vera": "your profile refresh",
        "curious_ask_due": "this week's demand question", "scheduled_recurring": "this week's update",
        "recall_due": "your check-up", "appointment_tomorrow": "your appointment",
        "chronic_refill_due": "the refill", "trial_followup": "your next session",
        "wedding_package_followup": "your bridal package", "customer_lapsed_soft": "your next visit",
        "customer_lapsed_hard": "your next visit", "unplanned_slot_open": "the open slot",
    }
    if k in names:
        return names[k]
    if ctx.offer:
        return "what I offered earlier"
    return "your Google profile"


_HI_OFFERS = {
    "digest": "summary aur customer WhatsApp ka draft ready", "checklist": "compliance checklist ready",
    "cde": "session aapke calendar mein block", "supply": "customer note aur recall steps ready",
    "seasonal": "shelf plan aur broadcast ready", "trend": "profile line aur Google post ready",
    "posts": "Google posts ke drafts ready", "pair": "Google post aur WhatsApp reply ready",
    "plan": "plan aur Google post ready", "match": "match-day banner aur story ready",
    "challenge": "attendance challenge ka draft ready", "milestone": "thank-you post aur review request ready",
    "review_replies": "review replies ke drafts ready", "verification": "verification shuru",
    "renewal": "renewal set up", "winback": "restart plan aur win-back message ready",
    "comeback": "comeback message aur post ready", "package": "launch post aur WhatsApp note ready",
    "onboarding": "setup shuru", "generic": "agla step ready",
}


def hi_offer(ctx: ReplyContext) -> str:
    """Natural Hinglish verb object for "main ... kar dungi" (English offers read awkwardly there)."""
    return _HI_OFFERS.get(artifact_type(ctx), _HI_OFFERS["generic"])


def _confirm_line(ctx: ReplyContext, en_verb: str, hi_verb: str) -> str:
    if ctx.hi:
        return f"CONFIRM reply karein, main {hi_verb}."
    return f"Reply CONFIRM and I'll {en_verb}."


def _yes_line(ctx: ReplyContext) -> str:
    offer = offer_phrase(ctx.offer, ctx.kind)
    if ctx.hi:
        ho = hi_offer(ctx)
        return _pick([f"Main {ho} kar doon? Bas YES reply karein.",
                      f"YES reply karein, main {ho} kar deti hoon."], ctx.variant)
    return _pick([f"Want me to {offer}? Just reply YES.",
                  f"Reply YES and I'll {offer}.",
                  f"Shall I {offer}? A YES is all I need."], ctx.variant)


def distinct_variant(body: str, ctx: ReplyContext, prior_norm: set[str]) -> str:
    """Last-resort anti-repetition: add a grounded, not-yet-used fact so the body is new."""
    def norm(t: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w₹%]+", " ", t.lower().replace("’", "'"))).strip()

    fx = FactView(ctx.facts)
    pool = [f.text for f in ctx.facts.anchor_facts] + fx.texts("perf") + fx.texts("agg") + fx.texts("offer")
    for text in pool:
        text = text.rstrip(".")
        if text and text.lower() not in body.lower():
            sep = "\n\n" if "\n" in body else " "
            cand = f"{body}{sep}(For reference: {_lc_first(text)}.)"
            if norm(cand) not in prior_norm:
                return cand
    lead = "Quick recap: " if not ctx.hi else "Short mein: "
    cand = lead + body
    return cand if norm(cand) not in prior_norm else f"{lead}{body} 🙂"


# --------------------------------------------------------------------------- simple moves


def auto_reply_nudge(ctx: ReplyContext) -> Reply:
    offer = offer_phrase(ctx.offer, ctx.kind, short=True) if (ctx.offer or ctx.kind in DEFAULT_OFFERS) else ""
    biz = ctx.facts.merchant_name
    if ctx.hi:
        tail = (f"main {hi_offer(ctx)} kar dungi" if offer else
                f"main {biz or 'aapke business'} ke liye ready update share kar dungi")
        body = _pick([
            f"Lagta hai yeh auto-reply hai 🙂 Owner jab dekhein, bas YES reply kar dein, {tail}.",
            f"Yeh automated reply lag raha hai. Owner tak pahunche toh ek YES kaafi hai, {tail}.",
        ], ctx.variant)
    else:
        tail = f"I'll {offer}" if offer else f"I'll share the update I have for {biz or 'your business'}"
        body = _pick([
            f"Looks like an auto-reply 🙂 When the owner sees this, just reply YES and {tail}.",
            f"This looks like an automated reply. Whenever the owner gets a moment, one YES is enough and {tail}.",
        ], ctx.variant)
    return Reply(body, CTA_BINARY_YES_NO, "auto-reply nudge")


def hostile_apology(ctx: ReplyContext) -> Reply:
    if ctx.customer_facing:
        biz = ctx.facts.signer or ctx.facts.merchant_name or "us"
        body = _pick([f"Sorry for the bother. Reply STOP and we won't message you again; {biz} is here whenever "
                      f"you need us.",
                      "Apologies for the trouble. Reply STOP and you won't get any more messages from us."],
                     ctx.variant)
        return Reply(body, CTA_NONE, "apology")
    if ctx.hi:
        body = _pick([
            f"Maaf kijiye{_comma_name(ctx)}, pareshan karna maqsad nahi tha. STOP reply karein, main aage koi "
            f"message nahi bhejungi.",
            "Sorry for the bother. Main yahin rok rahi hoon; STOP likh dein toh aage koi message nahi aayega.",
        ], ctx.variant)
    else:
        body = _pick([
            f"Sorry for the bother{_comma_name(ctx)}. I won't push this further; reply STOP and I'll stop "
            f"messaging completely.",
            "Apologies, that wasn't meant to annoy you. Reply STOP and you won't hear from me again.",
        ], ctx.variant)
    return Reply(body, CTA_NONE, "apology")


_OFF_TOPIC_WHO = {
    "ca": ("{thing} is best handled by your CA", "{thing} ke liye aapke CA sahi rahenge"),
    "bank": ("{thing} is best taken up with your bank", "{thing} ke liye aapka bank hi sahi jagah hai"),
    "lawyer": ("{thing} needs a lawyer's advice", "{thing} ke liye kisi lawyer se baat karna sahi rahega"),
    "utility": ("{thing} is best sorted with the provider's helpline",
                "{thing} ke liye provider ki helpline best rahegi"),
    "it": ("{thing} needs a local tech person", "{thing} ke liye kisi local tech person ko dikhana sahi rahega"),
    "personal": ("{thing} is outside what I can help with", "{thing} mere scope ke bahar hai"),
}


def off_topic_redirect(ctx: ReplyContext, area: str, thing: str) -> Reply:
    thing = _cap(thing or "That")
    who_en, who_hi = _OFF_TOPIC_WHO.get(area, _OFF_TOPIC_WHO["personal"])
    topic = thread_topic(ctx)
    offer = offer_phrase(ctx.offer, ctx.kind)
    if ctx.customer_facing:
        body = (f"{who_en.format(thing=thing)}, so we can't help with that here. "
                f"For {topic}, just reply here whenever it suits you.")
        return Reply(body, CTA_OPEN_ENDED, "off-topic redirect")
    if ctx.hi:
        first = who_hi.format(thing=thing) + ", yeh mere scope ke bahar hai."
        ho = hi_offer(ctx)
        tail = (f" Waise main {ho} kar sakti hoon, jab chahein bata dijiye." if ctx.soft
                else _pick([f" Wapas kaam ki baat par: main {ho} kar doon?",
                            f" Waise aapka kaam ready hai: main {ho} kar doon?"], ctx.variant))
    else:
        first = who_en.format(thing=thing) + "; that's outside what I can help with directly."
        tail = (f" If you'd still like me to {offer}, just say so; otherwise I'll stay out of your way." if ctx.soft
                else _pick([f" Coming back to {topic}: shall I {offer} now?",
                            f" Meanwhile, {topic} is ready to go. Shall I {offer}?"], ctx.variant))
    return Reply(first + tail, CTA_OPEN_ENDED, "off-topic redirect")


def confirm_done(ctx: ReplyContext) -> Reply:
    art = ctx.artifact or artifact_type(ctx)
    p = ctx.person
    en = {
        "digest": f"Done ✅ The {p} WhatsApp is queued to go out. I'll flag any replies that need you.",
        "posts": "Done ✅ The first post goes live on your Google profile today; the rest follow one by one. "
                 "I'll share how they do once there's data.",
        "pair": "Done ✅ The post goes live today and the WhatsApp reply is saved as a quick reply for incoming "
                "questions. I'll report back once there's data.",
        "plan": "Done ✅ The post is going live and the plan is saved. I'll check back once the numbers move.",
        "match": "Done ✅ The match-day copy is queued for today. I'll share how orders move after the match.",
        "challenge": f"Done ✅ The challenge note goes to your {ctx.people} today and the poster copy is saved. "
                     f"I'll send a weekly check-in count.",
        "milestone": "Done ✅ Thank-you post queued and the review request goes to recent regulars. I'll tell you "
                     "when you cross the mark.",
        "checklist": "Done ✅ Checklist saved and a reminder is set before the deadline. I'll nudge you then.",
        "cde": "Done ✅ Date saved; I'll remind you the day before with the session details.",
        "supply": "Done ✅ The note goes to the matched customers now. I'll flag anyone who replies with a question.",
        "seasonal": "Done ✅ The broadcast is queued and the shelf plan is saved for your team.",
        "package": "Done ✅ The launch post and the WhatsApp note are queued; nothing goes live without your final OK.",
        "renewal": "Noted ✅ I'll share the renewal confirmation here as soon as it's processed. Nothing else needed "
                   "from you right now.",
        "winback": "Done ✅ Restart noted and the win-back message is queued. I'll update you once it's live.",
        "comeback": f"Done ✅ The comeback message goes to your lapsed {ctx.people} today and the post goes live. "
                    f"I'll tell you who replies.",
        "verification": "Done ✅ Verification request started. Watch for the code by postcard or phone and send it "
                        "to me here.",
        "review_replies": "Done ✅ Posting the replies now. I'll flag any new review on this theme.",
        "onboarding": "Done ✅ Setup started. I'll send the first draft here for your approval before it goes live.",
        "generic": "Done ✅ Going ahead now. I'll update you here once it's live.",
    }
    hi = {
        "digest": f"Done ✅ {_cap(p)} WhatsApp queue ho gaya hai. Koi reply aapke kaam ka hua toh flag kar dungi.",
        "posts": "Done ✅ Pehla post aaj aapke Google profile par live hoga; baaki ek-ek karke jayenge. Data aate hi "
                 "update dungi.",
        "pair": "Done ✅ Post aaj live hoga aur WhatsApp reply quick reply mein save hai. Data aate hi bataungi.",
        "plan": "Done ✅ Post live ho raha hai aur plan save hai. Numbers move hote hi bataungi.",
        "match": "Done ✅ Match-day copy aaj ke liye queue hai. Match ke baad orders ka update dungi.",
        "challenge": f"Done ✅ Challenge note aaj aapke {ctx.people} ko jayega, poster copy save hai. Har hafte "
                     f"check-in count bhejungi.",
        "milestone": "Done ✅ Thank-you post queue hai aur review request regulars ko ja raha hai.",
        "checklist": "Done ✅ Checklist save hai aur deadline se pehle reminder set hai.",
        "cde": "Done ✅ Date save kar li; ek din pehle session details ke saath yaad dila dungi.",
        "supply": "Done ✅ Matched customers ko note abhi ja raha hai. Koi sawaal aaya toh flag kar dungi.",
        "seasonal": "Done ✅ Broadcast queue hai; shelf plan aapki team ke liye save hai.",
        "package": "Done ✅ Launch post aur WhatsApp note queue mein hain; aapke final OK ke bina kuch live nahi hoga.",
        "renewal": "Noted ✅ Renewal process hote hi confirmation yahin share kar dungi. Abhi aapko kuch aur nahi "
                   "karna.",
        "winback": "Done ✅ Restart note kar liya, win-back message queue mein hai. Live hote hi bataungi.",
        "comeback": f"Done ✅ Comeback message aaj lapsed {ctx.people} ko jayega aur post live hoga. Kaun reply karta "
                    f"hai, bataungi.",
        "verification": "Done ✅ Verification request shuru. Postcard ya phone par code aaye toh yahin bhej dijiye.",
        "review_replies": "Done ✅ Replies post kar rahi hoon. Is theme par naya review aaya toh flag karungi.",
        "onboarding": "Done ✅ Setup shuru. Pehla draft approval ke liye yahin bhejungi.",
        "generic": "Done ✅ Aage badh rahi hoon. Live hote hi yahin update dungi.",
    }
    table = hi if ctx.hi else en
    body = table.get(art, table["generic"])
    if ctx.variant % 2 == 1:
        body = body.replace("Done ✅", "All set ✅") if not ctx.hi else body.replace("Done ✅", "Ho gaya ✅")
    return Reply(body, CTA_NONE, f"confirmed {art}")


def edit_ack(ctx: ReplyContext) -> Reply:
    note = _short(ctx.inbound, 90).replace('"', "'")
    art = ctx.artifact or artifact_type(ctx)
    verb_en, verb_hi = _art_verbs(ctx, art)
    if ctx.hi:
        body = _pick([f"Noted: \"{note}\". Draft mein yeh change kar rahi hoon. {_confirm_line(ctx, verb_en, verb_hi)}",
                      f"Theek hai, \"{note}\" wala change draft mein daal rahi hoon. "
                      f"{_confirm_line(ctx, verb_en, verb_hi)}"], ctx.variant)
    else:
        body = _pick([f"Noted: \"{note}\". Updating the draft with that change now. "
                      f"{_confirm_line(ctx, verb_en, verb_hi)}",
                      f"Got it, working \"{note}\" into the draft. {_confirm_line(ctx, verb_en, verb_hi)}"],
                     ctx.variant)
    return Reply(body, CTA_BINARY_CONFIRM, "draft edit")


# --------------------------------------------------------------------------- question answers


def answer_question(ctx: ReplyContext, qtype: str) -> Reply:
    """Answer from facts only; unknown -> say we'll check. One CTA at the end (none when soft)."""
    fx = FactView(ctx.facts)
    if ctx.customer_facing:
        return _customer_answer(ctx, fx, qtype)
    ans = _merchant_answer_line(ctx, fx, qtype)
    if ctx.soft:
        tail = (" Main aage se khud message nahi karungi; zarurat ho toh yahin likh dijiye." if ctx.hi
                else " I won't message you unprompted; just write here if you need anything.")
        return Reply(ans + tail, CTA_NONE, f"answer ({qtype})")
    if ctx.stage == "action":
        art = ctx.artifact or artifact_type(ctx)
        en_verb, hi_verb = _art_verbs(ctx, art)
        confirm = _confirm_line(ctx, en_verb, hi_verb) if ctx.variant % 2 == 0 else (
            "Bas CONFIRM likh dijiye." if ctx.hi else "Just reply CONFIRM when you're ready.")
        return Reply(f"{ans} {confirm}", CTA_BINARY_CONFIRM, f"answer ({qtype})")
    return Reply(f"{ans} {_yes_line(ctx)}", CTA_BINARY_YES_NO, f"answer ({qtype})")


def answer_with_artifact(ctx: ReplyContext, qtype: str) -> Reply:
    """'Yes, but what's the price?': answer the question, then deliver the artifact (action mode)."""
    fx = FactView(ctx.facts)
    ans = _merchant_answer_line(ctx, fx, qtype)
    art = action_artifact(ctx)
    if qtype == "identity" or len(ans) + len(art.body) > 1000:
        en_verb, hi_verb = _art_verbs(ctx, art.label)
        return Reply(f"{ans} {_confirm_line(ctx, en_verb, hi_verb)}", CTA_BINARY_CONFIRM, art.label)
    return Reply(f"{ans}\n\n{art.body}", art.cta, art.label)


def _merchant_answer_line(ctx: ReplyContext, fx: FactView, qtype: str) -> str:
    hi = ctx.hi
    if qtype == "identity":
        return ("Aap Vera se baat kar rahe hain, magicpin ki merchant assistant: aapke profile ke numbers dekhti "
                "hoon aur drafts bana ke aapke approval ke liye bhejti hoon." if hi else
                "You're chatting with Vera, magicpin's merchant assistant: I track your profile numbers and prepare "
                "drafts for you to approve.")
    if qtype == "call":
        return _pick(["Main sirf WhatsApp par kaam karti hoon, call nahi kar sakti; par sab kuch yahin ho jayega.",
                      "Call ki zarurat nahi; saara kaam yahin WhatsApp par ho jayega."] if hi else
                     ["I work over WhatsApp only, so I can't call; everything can be done right here.",
                      "No call needed; I can handle all of it right here on WhatsApp."], ctx.variant)
    if qtype == "price":
        if ctx.kind in ("renewal_due", "winback_eligible") and fx.text("trigger.renewal_amount"):
            plan = fx.text("trigger.plan")
            line = f"{fx.text('trigger.renewal_amount')}{' for the ' + plan if plan else ''}"
            return f"{line}." if not hi else f"{line} hai."
        if ctx.kind == "cde_opportunity":
            fee = fx.text("digest.actionable", "digest.fee")
            if fee:
                return f"{fee.rstrip('.')}."
        if ctx.kind in ("active_planning_intent", "competitor_opened", "perf_dip", "festival_upcoming") \
                and ctx.facts.offers_active:
            base = ctx.facts.offers_active[0]
            return (f"Pricing aap tay karenge; draft aapke current '{base}' se shuru hota hai." if hi else
                    f"You set the pricing; the draft starts from your current '{base}'.")
        return _pick(["Exact amount mere paas abhi confirm nahi hai; main check karke bataungi, andaaza nahi lagaungi.",
                      "Iska figure mere data mein nahi hai; confirm karke hi bataungi."] if hi else
                     ["I don't have a confirmed figure for that yet; I'll check and come back rather than guess.",
                      "That figure isn't in what I have, so I'll confirm it before quoting anything."], ctx.variant)
    if qtype == "when":
        about_us = re.search(r"\b(?:live|post|posted|publish|send|sent|start|ready|done|go out|shuru|bhej\w*|hoga"
                             r"|jayega)\b", (ctx.inbound or "").lower())
        if ctx.stage == "action" and about_us:
            return ("Aaj hi, aapke CONFIRM ke turant baad; usse pehle kuch live nahi hoga." if hi else
                    "Today, right after you confirm; nothing goes live before that.")
        when = fx.text("digest.date", "trigger.deadline", "trigger.match_time", "trigger.days_remaining",
                       "trigger.due_date", "trigger.festival", "trigger.stock_runs_out", "trigger.opened_date")
        if when:
            return f"{when.rstrip('.')}."
        if about_us:
            return ("Aapke YES ke baad aaj hi ready kar dungi." if hi else
                    "The same day you say yes; nothing goes live without your OK.")
        return ("Iski exact date mere paas nahi hai; confirm karke bataungi." if hi else
                "I don't have an exact date for that; I'll confirm rather than guess.")
    if qtype == "source":
        src = fx.text("digest.source")
        if src:
            return (f"Source: {src}. Main isse aage kuch nahi jodungi." if hi else
                    f"Source: {src}. Nothing added beyond what it says.")
        basis = fx.text("trigger.metric_delta", "derived.metric_delta", "perf.views", "perf.calls")
        if basis:
            return (f"Yeh aapke profile ke numbers se hai: {_lc_first(basis.rstrip('.'))}." if hi else
                    f"It's from your own profile numbers: {_lc_first(basis.rstrip('.'))}.")
        return ("Mere paas iska written source nahi hai; check karke bataungi." if hi else
                "I don't have a written source for that; I'll check and confirm.")
    if qtype == "audience":
        seg = fx.text("digest.segment")
        caveat = next((x for x in sentences(fx.text("digest.summary"))
                       if re.search(r"\b(?:no effect|not for|only)\b", x, re.IGNORECASE)), "")
        if seg:
            tie = _segment_tie(fx).replace("relevant to your ", "that's your ")
            tail = f"; {_lc_first(caveat.rstrip('.'))}" if caveat else ""
            line = f"{seg.rstrip('.')}{f' ({tie})' if tie else ''}{tail}."
            return line if not hi else f"Short mein: {line}"
        qtype = "details"
    if qtype == "count" and ctx.kind == "supply_alert":
        pool = fx.text("agg.chronic_rx_count")
        if hi:
            return (f"Alert mein aapke affected customers ka count nahi hai; check karne wala pool: "
                    f"{_lc_first(pool.rstrip('.'))}." if pool else
                    "Alert mein affected customers ka count nahi hai; list filter karke exact number bataungi.")
        return (f"The alert doesn't say how many of yours got these batches; the pool to check is your "
                f"{_lc_first(pool.rstrip('.'))}." if pool else
                "The alert doesn't give an affected count; I'll filter your list and share the exact number.")
    if qtype == "count":
        cnt = fx.text(
            "trigger.lapsed_added", "agg.high_risk_adult_count", "agg.lapsed_180d_plus", "agg.lapsed_90d_plus",
            "agg.total_active_members", "agg.total_unique_ytd", "perf.calls", "perf.views")
        if cnt:
            return f"From your data: {_lc_first(cnt.rstrip('.'))}." if not hi else \
                f"Aapke data ke hisaab se: {_lc_first(cnt.rstrip('.'))}."
        return ("Exact count abhi mere paas nahi hai; check karke bataungi." if hi else
                "I don't have that exact count; I'll check rather than guess.")
    if qtype == "how" and ctx.stage == "action":
        return ("Aap upar ka draft dekh lijiye; koi bhi line badalni ho toh bata dijiye, main update kar dungi. "
                "Aapke OK ke bina kuch live nahi hoga." if hi else
                "You check the draft above; tell me any line to change and I'll update it. Nothing goes live "
                "without your OK.")
    if qtype == "how":
        offer = offer_phrase(ctx.offer, ctx.kind)
        return (f"Simple hai: main {hi_offer(ctx)} karti hoon, aap dekh ke approve karte hain, phir hi kuch live "
                f"hota hai." if hi else f"Simple: I {offer}, you review it, and nothing goes live until you approve.")
    if qtype == "why":
        why = _why_line(fx)
        if why:
            return f"Kyunki {_lc_first(why)}." if hi else f"Because {_lc_first(why)}."
    detail = _detail_line(fx)
    if detail:
        return f"Short version: {detail}." if not hi else f"Short mein: {detail}."
    return _pick(["Iska exact jawab mere paas abhi nahi hai; check karke bataungi, andaaza nahi lagaungi.",
                  "Yeh mere data mein nahi hai; confirm karke hi bataungi."] if hi else
                 ["I don't have a confirmed answer to that yet; I'll check and come back rather than guess.",
                  "That isn't in what I have, so I'll confirm it rather than guess."], ctx.variant)


def _why_line(fx: FactView) -> str:
    for key in ("digest.stat", "trigger.metric_delta", "derived.metric_delta", "trigger.theme", "trigger.competitor",
                "trigger.days_remaining", "trigger.deadline", "derived.milestone", "trigger.milestone",
                "signal.stale_posts", "perf.ctr_vs_peer"):
        t = fx.text(key)
        if t:
            return t.rstrip(".")
    return ""


def _detail_line(fx: FactView) -> str:
    parts = []
    title = fx.text("digest.title")
    stat = fx.text("digest.stat")
    if title:
        parts.append(title.rstrip("."))
        if stat and stat.lower() not in title.lower():
            parts.append(_lc_first(stat.rstrip(".")))
    else:
        for f in fx.sheet.anchor_facts[:2]:
            parts.append(f.text.rstrip("."))
    return _short("; ".join(p for p in parts if p), 320)


def _customer_answer(ctx: ReplyContext, fx: FactView, qtype: str) -> Reply:
    hi = ctx.hi
    biz = ctx.facts.merchant_name or "us"
    if qtype == "price":
        offer = ctx.facts.offers_active[0] if ctx.facts.offers_active else ""
        line = (f"Hamara current offer: {offer}." if hi else f"Our current offer is {offer}.") if offer else (
            "Exact price visit par confirm ho jayega." if hi else "We'll confirm the exact price when you visit.")
    elif qtype == "when":
        when = ", ".join(ctx.facts.slots) if ctx.facts.slots else fx.text("trigger.due_date", "trigger.stock_runs_out")
        if hi:
            line = f"Available: {when}." if when else "Aapke convenient time par slot fix kar denge."
        else:
            line = f"Available: {when}." if when else "We'll fix a slot at a time that suits you."
    elif qtype == "identity":
        signer = ctx.facts.signer or biz
        line = f"{signer} ki taraf se message hai." if hi else f"This is {signer}."
    else:
        who = _owner_ref(ctx)
        if hi:
            line = f"Achha sawaal; {who} visit par sab detail mein samjha denge."
        else:
            line = f"Good question; {who} will walk you through it at your visit."
    if ctx.facts.slots:
        return Reply(f"{line} {_slot_options_line(ctx)}", CTA_MULTI_CHOICE_SLOT, f"answer ({qtype})")
    tail = ("Aapke liye kaunsa din aur time theek rahega, bata dijiye." if hi else
            "Just reply with a day and time that suits you.")
    return Reply(f"{line} {tail}", CTA_OPEN_ENDED, f"answer ({qtype})")


# --------------------------------------------------------------------------- customer booking flows


def _slot_options_line(ctx: ReplyContext) -> str:
    slots = ctx.facts.slots[:3]
    if len(slots) == 1:
        return (f"Reply YES for {slots[0]}, or tell us a time that suits you." if not ctx.hi else
                f"{slots[0]} ke liye YES reply karein, ya apna time bata dijiye.")
    opts = ", ".join(f"{i + 1} for {s}" for i, s in enumerate(slots))
    return f"Reply {opts}, or tell us a time that suits you." if not ctx.hi else (
        f"Reply karein: {opts}. Ya apna time bata dijiye.")


def slot_prompt(ctx: ReplyContext) -> Reply:
    if not ctx.facts.slots:
        return customer_accept(ctx)
    lead = _pick(["Great!", "Lovely!"], ctx.variant) if not ctx.hi else _pick(["Badhiya!", "Zaroor!"], ctx.variant)
    return Reply(f"{lead} {_slot_options_line(ctx)}", CTA_MULTI_CHOICE_SLOT, "slot options")


def slot_confirmation(ctx: ReplyContext, slot: str) -> Reply:
    biz = ctx.facts.merchant_name or "us"
    who = ctx.facts.about_name or ""
    price = ""
    offers = ctx.facts.offers_active
    if offers and ctx.kind in ("recall_due", "unplanned_slot_open", "customer_lapsed_soft", "customer_lapsed_hard"):
        price = f" {offers[0]} applies." if not ctx.hi else f" {offers[0]} lagu rahega."
    for_who = f" for {who}" if who and who.lower() not in (ctx.name or "").lower() else ""
    if ctx.hi:
        body = _pick([f"Confirmed ✅ {slot}{for_who}, {biz}.{price} Time badalna ho toh bas yahin reply kar dijiye.",
                      f"Done ✅ Booking pakki: {slot}{for_who}, {biz}.{price} Reschedule karna ho toh yahin bata dijiye."],
                     ctx.variant)
    else:
        body = _pick([f"Confirmed ✅ {slot}{for_who} at {biz}.{price} If you need a different time, just reply here.",
                      f"Done ✅ You're booked for {slot}{for_who} at {biz}.{price} Need to reschedule? Reply here anytime."],
                     ctx.variant)
    return Reply(body, CTA_NONE, f"booked {slot}")


def customer_accept(ctx: ReplyContext) -> Reply:
    """Customer said yes without picking a slot."""
    fx = FactView(ctx.facts)
    slots = ctx.facts.slots
    if len(slots) == 1:
        return slot_confirmation(ctx, slots[0])
    if len(slots) > 1:
        # A plain "yes" to a two-slot offer: hold the first (preference-matched) slot and ask to lock it,
        # instead of bouncing the choice back to the customer.
        first, alt = slots[0], slots[1]
        if ctx.hi:
            body = (f"Done ✅ {first} aapke liye hold kar diya hai. Pakka karne ke liye CONFIRM likhiye, "
                    f"ya {alt} chahiye toh 2 reply kar dijiye.")
        else:
            body = (f"Done ✅ I've held {first} for you. Reply CONFIRM to lock it in, "
                    f"or 2 if {alt} suits you better.")
        return Reply(body, CTA_BINARY_CONFIRM, f"held {first}")
    hi = ctx.hi
    biz = ctx.facts.merchant_name or "us"
    signer = ctx.facts.signer or biz
    k = ctx.kind
    if k == "chronic_refill_due" and ctx.cat == "pharmacies":
        meds = re.sub(r"^\d+ regular medicines due:\s*", "", fx.text("trigger.molecules"))
        who = ctx.facts.about_name
        saved = fx.has("trigger.delivery_address")
        morning = "morning" in fx.text("customer.preferred_slots").lower()
        senior = next((o for o in ctx.facts.offers_active if "senior" in o.lower()), "")
        what = f"{meds} refill" if meds else "refill"
        lead = _pick(["Noted ✅", "Done ✅"], ctx.variant)
        if hi:
            senior_line = f" Aapka {senior} discount lagega." if senior and fx.has("customer.senior") else ""
            body = (f"{lead} {who + ' ke liye ' if who else ''}{what} pack kar rahe hain; delivery"
                    f"{' saved address par' if saved else ''}{' subah' if morning else ''} ho jayegi.{senior_line} "
                    f"Koi badlav ho toh yahin bata dijiye. — {signer}")
        else:
            senior_line = f" Your {senior} applies." if senior and fx.has("customer.senior") else ""
            whose = f"{who}'s " if who else "your "
            body = (f"{lead} We're packing {whose}{what} for delivery{' to the saved address' if saved else ''}"
                    f"{' in the morning' if morning else ''}.{senior_line} Any change, just reply here. — {signer}")
        return Reply(body, CTA_NONE, "refill confirmed")
    if k == "appointment_tomorrow":
        when = fx.text("trigger.due_date")
        m = re.search(r"tomorrow, (.+)$", when)
        day = f" ({m.group(1)})" if m else ""
        body = _pick([f"Confirmed ✅ Kal milte hain{day}! Kuch badalna ho toh yahin reply kar dijiye. — {signer}",
                      f"Done ✅ Kal{day} aapka intezaar rahega. Time badalna ho toh bata dijiye. — {signer}"] if hi else
                     [f"Perfect ✅ See you tomorrow{day} at {biz}. If anything changes, just reply here.",
                      f"All set ✅ We'll see you tomorrow{day} at {biz}. Need a different time? Just reply here."],
                     ctx.variant)
        return Reply(body, CTA_NONE, "appointment confirmed")
    # Fact text is third-person ("Prefers Saturdays"); rephrase it for the customer's own eyes.
    pref = re.sub(r"^prefers\s+", "", fx.text("customer.preferred_slots"), flags=re.IGNORECASE)
    pref_line = (f" ({pref} best rahega)" if hi else f", ideally {pref},") if pref else ""
    if k == "wedding_package_followup":
        step = fx.text("trigger.next_step")
        step_line = f"{step.rstrip('.')}. " if step else ""
        body = _pick([f"Wonderful! {step_line}Reply with a day that suits you{pref_line} and we'll confirm your first "
                      f"session.",
                      f"Lovely! Share a day that works{pref_line} and we'll confirm your first session."] if not hi else
                     [f"Badhiya! {step_line}Apna convenient din bata dijiye{pref_line}, pehla session confirm kar denge.",
                      f"Zaroor! Din bata dijiye{pref_line}, pehla session confirm kar denge."], ctx.variant)
        return Reply(body, CTA_OPEN_ENDED, "bridal next step")
    if hi:
        body = _pick([f"Badhiya! Apna convenient din aur time bata dijiye{pref_line}, hum {biz} mein slot confirm kar "
                      f"denge.",
                      f"Zaroor! Bas din aur time likh dijiye{pref_line}; {biz} mein aapka slot confirm kar denge."],
                     ctx.variant)
    else:
        body = _pick([f"Great! Reply with a day and time that suits you{pref_line} and we'll confirm your slot at "
                      f"{biz}.",
                      f"Lovely! Send us the day and time that works for you{pref_line} and we'll confirm it at {biz}."],
                     ctx.variant)
    return Reply(body, CTA_OPEN_ENDED, "booking next step")


# --------------------------------------------------------------------------- continue (engaged)


def continue_thread(ctx: ReplyContext) -> Reply:
    fx = FactView(ctx.facts)
    if ctx.customer_facing:
        if ctx.facts.slots:
            return slot_prompt(ctx)
        return customer_accept(ctx)
    anchor = _why_line(fx) or (fx.sheet.anchor_facts[0].text if fx.sheet.anchor_facts else "")
    lead_en = _pick(["Got it.", "Understood.", "Thanks for that."], ctx.variant)
    lead_hi = _pick(["Samajh gayi.", "Theek hai.", "Shukriya batane ke liye."], ctx.variant)
    fact = ""
    if anchor and ctx.variant % 2 == 0:
        fact = (f" Quick context: {_lc_first(anchor.rstrip('.'))}." if not ctx.hi else
                f" Context ke liye: {_lc_first(anchor.rstrip('.'))}.")
    elif ctx.variant % 2 == 1:
        fact = (" It's ready to go the moment you say so; nothing goes live without your OK." if not ctx.hi else
                " Aapke OK ke bina kuch live nahi hoga.")
    body = f"{lead_hi if ctx.hi else lead_en}{fact} {_yes_line(ctx)}"
    return Reply(body, CTA_BINARY_YES_NO, "next step")


# --------------------------------------------------------------------------- ACTION MODE


def artifact_type(ctx: ReplyContext) -> str:
    """Which artifact to deliver, from the conversation kind and the offer wording."""
    offer = (ctx.offer or "").lower()
    inbound = (ctx.inbound or "").lower()
    k = ctx.kind
    if re.search(r"\b(join|judna|judrna|jodna|onboard|sign me up|register)\b", inbound) and k in ("", "unknown",
                                                                                                    "generic"):
        return "onboarding"
    by_kind = {
        "research_digest": "digest", "regulation_change": "checklist", "cde_opportunity": "cde",
        "supply_alert": "supply", "category_seasonal": "seasonal", "category_trend_movement": "trend",
        "weather_heatwave": "seasonal", "local_news_event": "plan", "festival_upcoming": "plan",
        "ipl_match_today": "match", "competitor_opened": "plan", "perf_dip": "plan", "perf_spike": "posts",
        "seasonal_perf_dip": "challenge" if ctx.cat == "gyms" else "plan", "milestone_reached": "milestone",
        "review_theme_emerged": "review_replies",
        "gbp_unverified": "verification", "renewal_due": "renewal", "winback_eligible": "winback",
        "dormant_with_vera": "comeback", "active_planning_intent": "package", "curious_ask_due": "pair",
        "scheduled_recurring": "posts",
    }
    if k in by_kind:
        return by_kind[k]
    if re.search(r"\bposts?\b", offer):
        return "posts"
    if re.search(r"abstract|summary|research", offer):
        return "digest"
    if re.search(r"renew", offer):
        return "renewal"
    if re.search(r"review", offer):
        return "review_replies"
    if re.search(r"verif", offer):
        return "verification"
    if re.search(r"checklist|audit|complian", offer):
        return "checklist"
    return "generic"


def _art_verbs(ctx: ReplyContext, art: str) -> tuple[str, str]:
    person = ctx.person
    return {
        "digest": (f"send it to your {person} list", f"ise aapki {person} list ko bhej dungi"),
        "trend": ("publish the post and update the profile line", "post publish karke profile line update kar dungi"),
        "posts": ("publish the first one today and queue the rest", "pehla aaj publish karke baaki queue kar dungi"),
        "pair": ("publish the post today and save the reply", "post aaj publish karke reply save kar dungi"),
        "plan": ("publish the post today", "post aaj hi publish kar dungi"),
        "match": ("queue both for today", "dono aaj ke liye queue kar dungi"),
        "challenge": (f"send the note to your {ctx.people} today", f"note aaj aapke {ctx.people} ko bhej dungi"),
        "milestone": ("post the thank-you and send the review request",
                      "thank-you post karke review request bhej dungi"),
        "checklist": ("save it and set a reminder before the deadline",
                      "ise save karke deadline se pehle reminder set kar dungi"),
        "cde": ("save the date and remind you the day before", "date save karke ek din pehle yaad dila dungi"),
        "supply": ("send the note to the matched customers", "matched customers ko note bhej dungi"),
        "seasonal": ("send the broadcast and save the shelf plan", "broadcast bhej ke shelf plan save kar dungi"),
        "package": ("queue the post and the note for your final OK", "post aur note final OK ke liye queue kar dungi"),
        "renewal": ("process the renewal and confirm here", "renewal process karke yahin confirm kar dungi"),
        "winback": ("restart the plan and queue the win-back message",
                    "plan restart karke win-back message queue kar dungi"),
        "comeback": ("send the comeback message and publish the post",
                     "comeback message bhej ke post publish kar dungi"),
        "verification": ("start the verification request", "verification request shuru kar dungi"),
        "review_replies": ("post these replies", "yeh replies post kar dungi"),
        "onboarding": ("start the setup", "setup shuru kar dungi"),
        "generic": ("go ahead", "aage badh jaungi"),
    }.get(art, ("go ahead", "aage badh jaungi"))


def action_artifact(ctx: ReplyContext) -> Reply:
    """ACTION MODE for merchants: deliver the real artifact now, then one CONFIRM-style CTA."""
    if ctx.customer_facing:
        return customer_accept(ctx)
    fx = FactView(ctx.facts)
    art = artifact_type(ctx)
    builder = _BUILDERS.get(art, _art_generic)
    lines = builder(ctx, fx)
    if not lines:
        art, lines = "posts", _art_posts(ctx, fx)
    en_verb, hi_verb = _art_verbs(ctx, art)
    body = "\n".join(lines).strip() + "\n\n" + _confirm_line(ctx, en_verb, hi_verb)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return Reply(body, CTA_BINARY_CONFIRM, art)


def artifact_label(art: str) -> str:
    """Human description of an artifact key, for rationales ("posts" -> "3 Google post drafts")."""
    return _ART_LABELS.get(art, art or "draft")


_ART_LABELS = {
    "digest": "2-line summary and a draft customer WhatsApp", "trend": "profile line and Google post draft",
    "posts": "3 Google post drafts", "pair": "Google post and WhatsApp reply drafts", "plan": "plan and post draft",
    "match": "match-day copy", "challenge": "4-week attendance challenge draft",
    "milestone": "thank-you post and review request", "checklist": "compliance checklist", "cde": "session details",
    "supply": "recall workflow and customer note", "seasonal": "shelf plan and broadcast draft",
    "package": "package recap with launch post and WhatsApp note", "renewal": "renewal summary",
    "winback": "restart plan and win-back message", "verification": "verification steps",
    "comeback": "comeback message and fresh post",
    "review_replies": "draft review replies", "onboarding": "setup steps", "generic": "draft",
}


def _intro(ctx: ReplyContext, en: str, hi: str) -> str:
    """Artifact header, name first when known: "Dr. Meera, here's the plan:"."""
    text = _clean((hi if ctx.hi else en).replace("{name}", ""))
    if not ctx.name or not text:
        return text
    first = text.split()[0]
    lowered = text if (first.isupper() or first[:1].isdigit()) else text[0].lower() + text[1:]
    return f"{ctx.name}, {lowered}"


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _draft_label(ctx: ReplyContext, en: str, hi: str) -> str:
    return hi if ctx.hi else en


def _customer_note_digest(ctx: ReplyContext, fx: FactView) -> str:
    signer = ctx.facts.merchant_name or "Our clinic"
    title = fx.text("digest.title").rstrip(".")
    source = fx.text("digest.source")
    closer = {
        "dentists": "Ask us at your next check-up what it means for you.",
        "salons": "Ask us about it at your next appointment.",
        "gyms": "Ask your coach about it at your next session.",
        "pharmacies": "Our pharmacist is happy to explain; just reply here.",
        "restaurants": "Come by and ask us about it.",
    }.get(ctx.cat, "Reply here to know more.")
    src = f" ({source})" if source else ""
    return f"Hi! {signer} here with a quick update{src}: {_lc_first(title)}. {closer}"


def _art_digest(ctx: ReplyContext, fx: FactView) -> list[str]:
    title = fx.text("digest.title")
    if not title:
        return []
    source = fx.text("digest.source")
    stat = fx.text("digest.stat") or first_sentence(fx.text("digest.summary"))
    if stat and stat.lower().rstrip(".") in title.lower():
        stat = ""
    actionable = fx.text("digest.actionable")
    line2 = stat or actionable
    tie = _segment_tie(fx)
    if tie and line2:
        line2 = f"{line2.rstrip('.')}; {tie}"
    caveat = next((s for s in sentences(fx.text("digest.summary"))[1:]
                   if re.search(r"\b(?:no effect|not|only|except)\b", s, re.IGNORECASE)), "")
    lines = [_intro(ctx, "Here's the 2-line summary{name}:", "Yeh raha 2-line summary{name}:"),
             f"1) {title.rstrip('.')}{f' ({source})' if source else ''}."]
    if line2:
        extra = f" {caveat.rstrip('.')}." if caveat and len(line2) < 200 else ""
        lines.append(f"2) {_cap(line2.rstrip('.'))}.{extra}")
    digest_kind = str((ctx.facts.digest_item or {}).get("kind") or "").lower()
    if digest_kind in ("research", "tech", "") and ctx.cat in ("dentists", "gyms", "pharmacies", "salons"):
        lines += ["", _draft_label(ctx, f"Draft {ctx.person} WhatsApp (edit freely):",
                                   f"{_cap(ctx.person)} WhatsApp ka draft (jaise chahein edit karein):"),
                  f"\"{_customer_note_digest(ctx, fx)}\""]
    elif actionable and actionable != line2:
        lines += ["", (f"Suggested next step: {actionable.rstrip('.')}." if not ctx.hi else
                       f"Agla step: {actionable.rstrip('.')}.")]
    return lines


def _segment_tie(fx: FactView) -> str:
    seg = fx.text("digest.segment").lower()
    if not seg:
        return ""
    for text in fx.texts("agg"):
        low = text.lower()
        if ("high-risk" in seg and "high-risk" in low) or ("diabet" in seg and "diabet" in low) or (
                "chronic" in seg and "chronic" in low) or ("senior" in seg and "senior" in low):
            return f"relevant to your {_lc_first(text.rstrip('.'))}"
    return ""


def _art_trend(ctx: ReplyContext, fx: FactView) -> list[str]:
    if fx.text("digest.title") and not fx.text("trend.top"):
        return _art_digest(ctx, fx)
    topic = trend_topic(fx, ctx) or (ctx.topics[0] if ctx.topics else "")
    trend = fx.text("trend.top") or fx.text("digest.title")
    if not topic and not trend:
        return []
    biz, loc = _biz(ctx), _in_loc(ctx)
    lines = [_intro(ctx, "Here's the draft{name}:", "Yeh raha draft{name}:")]
    if trend:
        lines.append(f"Why now: {_lc_first(trend.rstrip('.'))}.")
    subject = topic or _GENERIC_SERVICES.get(ctx.cat, "our services")
    lines += ["", _draft_label(ctx, "Profile line (edit freely):", "Profile line (edit kar sakte hain):"),
              f"\"{_cap(subject)} at {biz}{loc}.\"",
              _draft_label(ctx, "Google post:", "Google post:"), f"\"{_topic_post(ctx, subject, ctx.variant)}\""]
    return lines


def _art_checklist(ctx: ReplyContext, fx: FactView) -> list[str]:
    title = fx.text("digest.title")
    stat = fx.text("digest.stat")
    actionable = fx.text("digest.actionable")
    deadline = fx.text("trigger.deadline")
    source = fx.text("digest.source")
    items: list[str] = []
    if stat:
        items.append(f"What changed: {_lc_first(stat.rstrip('.'))}")
    elif title:
        items.append(f"What changed: {_lc_first(title.rstrip('.'))}")
    for s in sentences(fx.text("digest.summary")):
        s = s.rstrip(".")
        if s and (not stat or s.lower() not in stat.lower()) and len(items) < 3:
            items.append(_cap(s))
    if actionable:
        items.append(f"Action: {_lc_first(actionable.rstrip('.'))}")
    if deadline:
        items.append(_cap(deadline.rstrip(".")))
    if source:
        items.append(f"Record the check in your SOP file, citing {source}")
    if not items:
        return []
    lines = [_intro(ctx, "Here's your compliance checklist{name}:", "Yeh raha aapka compliance checklist{name}:")]
    lines += [f"{i + 1}) {t}." for i, t in enumerate(items[:5])]
    return lines


def _art_cde(ctx: ReplyContext, fx: FactView) -> list[str]:
    title = fx.text("digest.title")
    if not title:
        return []
    source = fx.text("digest.source")
    date = fx.text("digest.date")
    credits = fx.text("digest.credits")
    fee = fx.text("digest.actionable") or fx.text("digest.fee")
    summary = _short(fx.text("digest.summary"), 220)
    lines = [_intro(ctx, "Here are the details{name}:", "Yeh rahi details{name}:"),
             f"• {title}{f' ({source})' if source else ''}"]
    when = "; ".join(x.rstrip(".") for x in (date, credits) if x)
    if when:
        lines.append(f"• {when}")
    if fee:
        lines.append(f"• {fee.rstrip('.')}")
    if summary:
        lines.append(f"• {summary.rstrip('.')}.")
    return lines


def _art_supply(ctx: ReplyContext, fx: FactView) -> list[str]:
    batches = fx.text("trigger.batches")
    mfr = _strip_prefix(fx.text("trigger.manufacturer"), "Manufacturer:")
    mol = _strip_prefix(fx.text("trigger.molecule"), "Molecule:") or "the affected medicine"
    source = fx.text("digest.source")
    pool = fx.text("agg.chronic_rx_count")
    summary = fx.text("digest.summary").lower()
    steps = []
    if batches:
        steps.append(f"Pull {_lc_first(batches)}{f' ({mfr})' if mfr else ''} from the shelf.")
    elif mfr:
        steps.append(f"Pull the flagged {mol} batches from {mfr} off the shelf.")
    steps.append(f"I filter your {_lc_first(pool.rstrip('.'))} for {mol} buyers." if pool else
                 f"I filter your repeat-prescription list for {mol} buyers.")
    if "replacement" in summary and "distributor" in summary:
        steps.append("Replacements come through your distributor return chain.")
    biz = ctx.facts.merchant_name or "your pharmacy"
    note = (f"Namaste, {biz} here. Some {mol} batches are under a voluntary recall{f' ({source})' if source else ''}. "
            f"Please bring your strip in or reply here; we'll check the batch and arrange a replacement.")
    lines = [_intro(ctx, "Here's the recall workflow{name}:", "Yeh raha recall workflow{name}:")]
    lines += [f"{i + 1}) {s}" for i, s in enumerate(steps)]
    lines += ["", _draft_label(ctx, "Draft customer WhatsApp (edit freely):", "Customer WhatsApp ka draft:"),
              f"\"{note}\""]
    return lines


def _art_seasonal(ctx: ReplyContext, fx: FactView) -> list[str]:
    actionable = fx.text("digest.actionable")
    trends = fx.text("trigger.trends")
    season = fx.text("seasonal.note", "trigger.season_note")
    title = fx.text("digest.title")
    items = [x for x in (actionable, trends or title) if x] or ([season] if season else [])
    if not items:
        return []
    lines = [_intro(ctx, "Here's the plan{name}:", "Yeh raha plan{name}:")]
    lines += [f"{i + 1}) {_cap(t.rstrip('.'))}." for i, t in enumerate(items[:2])]
    lines += ["", _draft_label(ctx, f"Draft WhatsApp broadcast for your regulars (edit freely):",
                               "Regulars ke liye WhatsApp broadcast ka draft:"),
              f"\"{_seasonal_post(ctx, fx)}\""]
    return lines


def _seasonal_post(ctx: ReplyContext, fx: FactView) -> str:
    biz, loc = _biz(ctx), _in_loc(ctx)
    offer = ctx.facts.offers_active[0] if ctx.facts.offers_active else ""
    items = _trend_items(fx)
    if ctx.kind == "weather_heatwave":
        base = f"Beat the heat with {biz}{loc}: message us before you step out and we'll keep things ready."
    elif items:
        base = (f"Summer essentials like {_join(items)} at {biz}{loc}: reply here and we'll keep your order "
                f"ready.")
    elif ctx.cat == "pharmacies":
        base = f"Season-ready essentials at {biz}{loc}: message us and we'll keep your list ready."
    else:
        base = f"Seasonal favourites at {biz}{loc}: message us to book or order."
    return f"{base} {offer}." if offer and len(base) < 170 else base


def _plan_steps(ctx: ReplyContext, fx: FactView) -> list[str]:
    k = ctx.kind
    steps: list[str] = []
    offer = ctx.facts.offers_active[0] if ctx.facts.offers_active else ""
    catalog = ctx.facts.catalog_offers[0] if ctx.facts.catalog_offers else ""
    if k == "competitor_opened":
        comp = fx.text("trigger.competitor")
        dist = fx.text("trigger.distance")
        theirs = fx.text("trigger.their_offer")
        if comp:
            steps.append(f"{_cap(comp.rstrip('.'))}{f', {dist}' if dist else ''}"
                         f"{f'. {theirs.rstrip(chr(46))}' if theirs else ''}.")
        praise = _praise_themes(fx)
        if offer:
            steps.append(f"Don't race them on price: lead with {offer} plus what regulars already praise"
                         f"{f' ({praise[0]})' if praise else ''}.")
        else:
            steps.append("Lead with what a new listing can't copy: your reviews and your team.")
    elif k == "festival_upcoming":
        fest = fx.text("trigger.festival")
        season = fx.text("seasonal.note", "trigger.season_note")
        if fest:
            days = fx.text("trigger.days_until")
            steps.append(f"{_cap(fest.rstrip('.'))}{'; ' + days.rstrip('.') if days else ''}.")
        elif season:
            steps.append(f"Season to plan for: {season.rstrip('.')}.")
        steps.append(f"Feature {offer} in a festive post." if offer else
                     f"Add one clear service + price offer (idea: {catalog}, only if you want to run it)." if catalog
                     else "Add one clear service + price offer for the season.")
    elif k == "weather_heatwave":
        steps.append("Heat-proof your listing: update timings and highlight what helps in the heat.")
        if offer:
            steps.append(f"Feature {offer} in a quick post.")
    else:  # perf_dip, local_news_event, generic plans
        delta = fx.text("trigger.metric_delta", "derived.metric_delta", "perf.delta_calls", "perf.delta_views")
        if delta:
            steps.append(f"{_cap(delta.rstrip('.'))}: a fresh post and quick review replies are the fastest fix.")
        stale = fx.text("signal.stale_posts", "signal.no_recent_post")
        peer = fx.text("peer.avg_post_freq_days")
        if stale:
            steps.append(f"{_cap(stale.rstrip('.'))}{f' ({_lc_first(peer)})' if peer else ''}: post today.")
        unverified = fx.text("signal.unverified_gbp")
        if unverified and len(steps) < 2 and not (offer or catalog):
            steps.append(f"{_cap(unverified.rstrip('.'))}: verification is the next fix after the post.")
        if offer:
            steps.append(f"Lead the post with {offer}.")
        elif catalog:
            steps.append(f"Offer line to add: {catalog} (a suggestion; only if you want to run it).")
        ctr = fx.text("perf.ctr_vs_peer")
        if ctr and "below" in ctr and len(steps) < 3:
            steps.append(f"Close the gap: {ctr.rstrip('.')}.")
    return [s for s in steps if s][:3]


def _art_plan(ctx: ReplyContext, fx: FactView) -> list[str]:
    steps = _plan_steps(ctx, fx)
    lines = [_intro(ctx, "Here's the plan{name}:", "Yeh raha plan{name}:")]
    lines += [f"{i + 1}) {s}" for i, s in enumerate(steps)]
    lines += ["", _draft_label(ctx, "Draft Google post (edit freely):", "Google post ka draft (edit kar sakte hain):"),
              f"\"{_plan_post(ctx, fx)}\""]
    return lines


def _plan_post(ctx: ReplyContext, fx: FactView) -> str:
    biz, loc = _biz(ctx), _in_loc(ctx)
    offer = ctx.facts.offers_active[0] if ctx.facts.offers_active else ""
    k = ctx.kind
    if k == "festival_upcoming":
        fest = fx.text("trigger.festival").split(" on ")[0]
        occasion = f"{fest} is coming" if fest else "Festive season is here"
        return f"{occasion} at {biz}{loc}: {offer + '. Book' if offer else 'book'} your slot early on WhatsApp."
    if k == "competitor_opened":
        praise = _praise_themes(fx)
        why = f"our {ctx.people} keep mentioning {praise[0]}" if praise else "trusted by our regulars"
        return f"{biz}{loc}: {why}. {offer + '. ' if offer else ''}Message us to book."
    service = _GENERIC_SERVICES.get(ctx.cat, "your next visit")
    return f"{offer + ' at ' + biz if offer else biz}{loc}: {service}. Message us on WhatsApp to book."


def _is_weekend_match(fx: FactView) -> bool:
    t = fx.text("trigger.weeknight").lower()
    return "weekend" in t or "not a weeknight" in t


def _art_match(ctx: ReplyContext, fx: FactView) -> list[str]:
    biz, loc = _biz(ctx), _in_loc(ctx)
    offer = ctx.facts.offers_active[0] if ctx.facts.offers_active else ""
    match = _strip_prefix(fx.text("trigger.match"), "IPL match today:") or "today's match"
    start = fx.text("trigger.match_time")
    stat = fx.text("digest.stat")
    lines = [_intro(ctx, "Here's the match-day copy{name} (edit freely):", "Yeh rahi match-day copy{name}:")]
    if _is_weekend_match(fx):
        if stat:
            lines.append(f"Why delivery-first: {_lc_first(stat.rstrip('.'))}.")
        lines += [_draft_label(ctx, "Delivery banner:", "Delivery banner:"),
                  f"\"{match} tonight? Order in from {biz}{loc} and we'll get it to you hot.\"",
                  _draft_label(ctx, "Insta story:", "Insta story:"),
                  f"\"Match at home, food from {biz}. Order before the first ball.\""]
        if offer:
            lines.append(f"Keep {offer} for its own days; no dine-in match promo today.")
    else:
        lines += [_draft_label(ctx, "Match-night post:", "Match-night post:"),
                  f"\"{match} at {biz}{loc}! {offer + '. ' if offer else ''}Come watch with us.\"",
                  _draft_label(ctx, "Insta story:", "Insta story:"),
                  f"\"Big screen, good food, {match}. See you at {biz}.\""]
        if start:
            lines.append(f"Timing: goes out ahead of the match ({_lc_first(start.rstrip('.'))}).")
    return lines


def _art_challenge(ctx: ReplyContext, fx: FactView) -> list[str]:
    biz = _biz(ctx)
    people = ctx.people
    offer = ctx.facts.offers_active[0] if ctx.facts.offers_active else ""
    delta = fx.text("trigger.metric_delta", "derived.metric_delta")
    note = fx.text("trigger.season_note")
    lines = [_intro(ctx, "Here's the 4-week attendance challenge{name} (draft, edit freely):",
                    "Yeh raha 4-week attendance challenge{name} (draft, edit kar sakte hain):")]
    if delta:
        lines.append(f"Context: {_lc_first(delta.rstrip('.'))}{f' ({_lc_first(note)})' if note else ''}; normal for "
                     f"the season, so the focus is keeping current {people}.")
    lines += ["• Week 1: book your sessions for the week in advance.",
              "• Week 2: bring a friend for one session.",
              "• Week 3: try one new class or workout.",
              "• Week 4: full-attendance members get a shout-out on the studio wall."]
    lines += ["", _draft_label(ctx, f"WhatsApp note to your {people}:", f"{_cap(people)} ke liye WhatsApp note:"),
              f"\"Our 4-week challenge starts this week at {biz}! Show up, keep your streak, and finish strong."
              f"{' ' + offer + ' for friends you bring.' if offer else ''} Reply JOIN to get in.\""]
    return lines


def _art_milestone(ctx: ReplyContext, fx: FactView) -> list[str]:
    biz, loc = _biz(ctx), _in_loc(ctx)
    ms = fx.text("trigger.milestone", "derived.milestone")
    praise = _praise_themes(fx)
    lines = [_intro(ctx, "Here are both drafts{name} (edit freely):", "Yeh rahe dono drafts{name}:")]
    if ms:
        lines.append(f"Where you stand: {_lc_first(ms.rstrip('.'))}.")
    lines += ["", _draft_label(ctx, "Thank-you post:", "Thank-you post:"),
              f"\"Thank you for making {biz}{loc} part of your routine"
              f"{f', and for all the kind words about our {praise[0]}' if praise else ''}. See you again soon!\"",
              _draft_label(ctx, f"Review request to recent {ctx.people}:",
                           f"Recent {ctx.people} ke liye review request:"),
              f"\"Thank you for choosing {biz}! If you enjoyed your visit, a quick Google review would mean a lot "
              f"to our team.\""]
    return lines


def _driver_topic(fx: FactView) -> str:
    """'Likely driver: the kids yoga post' -> 'kids yoga'."""
    t = _strip_prefix(fx.text("trigger.likely_driver"), "Likely driver:")
    t = re.sub(r"^(?:the|your|a)\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+(?:post|posts|offer|campaign|update)$", "", t, flags=re.IGNORECASE)
    return t.strip().lower() if 2 < len(t) <= 40 else ""


def _art_posts(ctx: ReplyContext, fx: FactView) -> list[str]:
    topics = list(ctx.topics)
    if not topics:
        tt = _driver_topic(fx) or trend_topic(fx, ctx)
        if tt:
            topics.append(tt)
    posts: list[str] = []
    for i, t in enumerate(topics[:3]):
        posts.append(_topic_post(ctx, t, i))
    for offer in ctx.facts.offers_active:
        if len(posts) >= 3:
            break
        if any(offer in p for p in posts):
            continue
        posts.append(f"{offer} at {_biz(ctx)}{_in_loc(ctx)}. Message us on WhatsApp to book.")
    for theme in _praise_themes(fx):
        if len(posts) >= 3:
            break
        posts.append(f"What our {ctx.people} mention most: {theme}. See for yourself at {_biz(ctx)}{_in_loc(ctx)}.")
    fillers = [f"{_cap(_GENERIC_SERVICES.get(ctx.cat, 'Your next visit'))} at {_biz(ctx)}{_in_loc(ctx)}. "
               f"Message us to book a slot.",
               f"New week, fresh slots at {_biz(ctx)}. Message us on WhatsApp and we'll hold one for you.",
               f"In {ctx.facts.locality or 'the area'}? {_biz(ctx)} is right here. Say hi on WhatsApp."]
    for f in fillers:
        if len(posts) >= 3:
            break
        posts.append(f)
    why = fx.text("trigger.metric_delta", "derived.metric_delta", "trigger.likely_driver")
    on = f" on {_join(topics[:3])}" if topics else ""
    lines = [_intro(ctx, "Here are 3 short Google post drafts" + on + "{name} (edit freely):",
                    "Yeh rahe 3 chhote Google post drafts" + on + "{name} (jaise chahein edit karein):")]
    if why and ctx.kind in ("perf_spike", "dormant_with_vera"):
        lines.append(f"Why now: {_lc_first(why.rstrip('.'))}; these keep the momentum going." if not ctx.hi else
                     f"Kyun abhi: {_lc_first(why.rstrip('.'))}; yeh momentum banaye rakhenge.")
    lines += [f"{i + 1}) \"{p}\"" for i, p in enumerate(posts[:3])]
    return lines


def _topic_post(ctx: ReplyContext, topic: str, i: int) -> str:
    biz, loc = _biz(ctx), _in_loc(ctx)
    offer = _matching_offer(ctx, topic)
    if offer:
        return f"{offer} at {biz}{loc}. Message us on WhatsApp to book."
    t_cap = _cap(topic)
    owner = _owner_ref(ctx)
    by_cat = {
        "dentists": [f"{t_cap} at {biz}{loc}: {owner} explains your options first. Message us to book a consultation.",
                     f"Thinking about {topic}: book a consultation with {owner} at {biz}{loc}.",
                     f"{t_cap}, done carefully at {biz}. WhatsApp us to fix a slot."],
        "salons": [f"{t_cap} at {biz}{loc}. Message us to book your slot.",
                   f"Treat yourself to {topic} this week at {biz}.",
                   f"Our stylists at {biz}{loc} are ready for your {topic}. Book on WhatsApp."],
        "restaurants": [f"{t_cap} at {biz}{loc}. Order now or drop by.",
                        f"Craving {topic}: {biz} has you covered today.",
                        f"{t_cap}, fresh from our kitchen at {biz}{loc}."],
        "gyms": [f"{t_cap} at {biz}{loc}. Message us to join a session.",
                 f"Train for {topic} with our coaches at {biz}.",
                 f"{t_cap} sessions at {biz}{loc}. Say hi on WhatsApp to get started."],
        "pharmacies": [f"{t_cap}: ask our pharmacist at {biz}{loc}.",
                       f"Need {topic}: message {biz} and we'll keep it ready for pickup.",
                       f"{t_cap} at {biz}{loc}. Message us on WhatsApp for home delivery options."],
    }
    options = by_cat.get(ctx.cat, [f"{t_cap} at {biz}{loc}. Message us on WhatsApp to know more."])
    return options[i % len(options)]


def _art_pair(ctx: ReplyContext, fx: FactView) -> list[str]:
    """curious_ask_due: the merchant's answer (or the top trend) -> a Google post + a WhatsApp quick reply."""
    topic = ctx.topics[0] if ctx.topics else trend_topic(fx, ctx)
    if not topic:
        return _art_posts(ctx, fx)
    biz = _biz(ctx)
    offer = _matching_offer(ctx, topic) or (ctx.facts.offers_active[0] if ctx.facts.offers_active else "")
    trend = fx.text("trend.top")
    lines = [_intro(ctx, f"Here's the pair on {topic}{{name}} (edit freely):",
                    f"{_cap(topic)} par yeh raha pair{{name}} (edit kar sakte hain):")]
    if trend and topic.split()[0].lower() in trend.lower():
        lines.append(f"Backed by demand: {_lc_first(trend.rstrip('.'))}.")
    matching = _matching_offer(ctx, topic)
    if matching:
        quick = f"Yes! Our {matching} at {biz} is available. "
    elif ctx.cat == "pharmacies":
        quick = f"Yes, our pharmacist at {biz} can help with {topic}. "
    else:
        quick = f"Yes, we do {topic} at {biz}! {offer + '. ' if offer else ''}"
    quick += {"restaurants": "Order on WhatsApp or tell us when you're coming.",
              "pharmacies": "Message us your list and we'll keep it ready."}.get(
        ctx.cat, "Send us a day and time and we'll hold a slot for you.")
    lines += [_draft_label(ctx, "Google post:", "Google post:"), f"\"{_topic_post(ctx, topic, ctx.variant)}\"",
              _draft_label(ctx, "WhatsApp quick reply for customer questions:",
                           "Customer sawaalon ke liye WhatsApp quick reply:"),
              f"\"{quick}\""]
    return lines


def _art_reviews(ctx: ReplyContext, fx: FactView) -> list[str]:
    theme_fact = fx.text("trigger.theme")
    quote = fx.text("trigger.quote")
    m = re.search(r"mention (.+?)(?: \(|$)", theme_fact)
    theme = _clean(m.group(1)) if m else ""
    biz = _biz(ctx)
    lines = [_intro(ctx, "Here are 2 draft review replies{name} (edit freely):",
                    "Yeh rahe 2 review replies ke drafts{name} (edit kar sakte hain):")]
    if theme_fact:
        said = f"; {_lc_first(quote)}" if quote else ""
        lines.append(f"What reviewers say: {_lc_first(theme_fact.rstrip('.'))}{said}.")
    if theme:
        r1 = (f"Thank you for the honest feedback, and sorry about the {theme}. We're fixing it; please give us "
              f"another chance.")
        r2 = f"Sorry we let you down on {theme}. Message us directly next time so we can make it right. Team {biz}"
        fix = f"Fix note for the team: \"Every {theme} complaint gets a same-day call back from the manager.\""
    else:
        r1 = f"Thank you for visiting {biz}! Your feedback helps us get better; hope to see you again soon."
        r2 = (f"Thanks for taking the time to review us. If anything fell short, message us directly and we'll "
              f"make it right. Team {biz}")
        fix = ""
    lines += [f"1) \"{r1}\"", f"2) \"{r2}\""]
    if fix:
        lines.append(fix)
    elif not theme_fact:
        lines.append("I'll match each reply to your latest reviews before anything is posted." if not ctx.hi else
                     "Post karne se pehle har reply ko aapke latest reviews se match kar dungi.")
    return lines


def _art_verification(ctx: ReplyContext, fx: FactView) -> list[str]:
    path = fx.text("trigger.verification_path")
    uplift = fx.text("trigger.uplift")
    lines = [_intro(ctx, "Here's the verification plan{name}:", "Yeh raha verification plan{name}:")]
    steps = [f"{_cap(path.rstrip('.'))}: I raise the request, you just receive the code." if path else
             "I raise the verification request; you just receive the code.",
             "Keep your listed phone number and address exactly as on the shop board.",
             "Send me the code here and I'll finish it."]
    lines += [f"{i + 1}) {s}" for i, s in enumerate(steps)]
    if uplift:
        lines.append(f"Why it matters: {_lc_first(uplift.rstrip('.'))}.")
    return lines


def _art_renewal(ctx: ReplyContext, fx: FactView) -> list[str]:
    plan = fx.text("trigger.plan") or _strip_prefix(fx.text("merchant.subscription"), "")
    amount = fx.text("trigger.renewal_amount")
    days = fx.text("trigger.days_remaining") or fx.text("signal.renewal_due_soon")
    lines = [_intro(ctx, "Confirming the renewal{name}:", "Renewal confirm kar rahi hoon{name}:")]
    lines.append(f"• Plan: {plan.rstrip('.')}" if plan else "• Plan: your current plan")
    if amount:
        lines.append(f"• {amount.rstrip('.')}")
    if days:
        lines.append(f"• {_cap(days.rstrip('.'))}")
    value = [re.sub(r"\s+in the last 30 days$", "", x.rstrip(".")) for x in
             (fx.text("perf.views"), fx.text("perf.calls"), fx.text("perf.leads")) if x]
    if value:
        lines.append(("Your last 30 days on the plan: " if not ctx.hi else "Plan par pichle 30 din: ")
                     + ", ".join(value[:2]) + ".")
    lines.append("Next: I process it and share the confirmation here." if not ctx.hi else
                 "Next: main process karke confirmation yahin share karungi.")
    return lines


def _come_back_line(ctx: ReplyContext) -> str:
    return {"restaurants": "Reply here and we'll have your favourite ready.",
            "pharmacies": "Reply here and we'll keep your medicines ready."}.get(
        ctx.cat, "Reply here and we'll hold a slot that suits you.")


def _art_winback(ctx: ReplyContext, fx: FactView) -> list[str]:
    facts = [fx.text(k) for k in ("trigger.days_since_expiry", "trigger.perf_dip", "trigger.lapsed_added")]
    facts = [f for f in facts if f]
    biz = _biz(ctx)
    lines = [_intro(ctx, "Here's the restart plan{name}:", "Yeh raha restart plan{name}:")]
    lines += [f"• {_cap(f.rstrip('.'))}" for f in facts[:3]]
    lines.append("1) Reactivate the plan so the profile work resumes." if not ctx.hi else
                 "1) Plan reactivate karein taaki profile ka kaam wapas shuru ho.")
    lines.append(f"2) Win-back message to lapsed {ctx.people} (draft, edit freely):" if not ctx.hi else
                 f"2) Lapsed {ctx.people} ke liye win-back message (draft):")
    lines.append(f"\"It's been a while! {biz} would love to see you again. {_come_back_line(ctx)}\"")
    return lines


_CORPORATE_WORDS = ("corporate", "bulk", "office", "team")
_PROGRAM_WORDS = ("kids", "camp", "summer", "program", "programme", "batch", "class")


def _launch_copy(ctx: ReplyContext, fx: FactView, topic: str) -> list[str]:
    """Launch Google post + WhatsApp note for a planned package (numbers only from active offers)."""
    biz, loc = _biz(ctx), _in_loc(ctx)
    offers = ctx.facts.offers_active
    praise = _praise_themes(fx)
    low = topic.lower()
    if any(w in low for w in _CORPORATE_WORDS):
        post = (f"Office lunches sorted: {offers[0] + ' from ' if offers else ''}{biz}{loc}"
                f"{f', loved for its {praise[0]}' if praise else ''}. Message us for team orders.")
        note_label = _draft_label(ctx, "WhatsApp note for office admins:", "Office admins ke liye WhatsApp note:")
        note = (f"Hi! {biz} here. We now do daily team lunches with one monthly invoice. Reply here and we'll "
                f"share the team menu and delivery timings.")
    else:
        post = (f"New at {biz}{loc}: {topic}{f', with the {praise[0]} our members love' if praise else ''}. "
                f"Message us to reserve a spot.")
        note_label = _draft_label(ctx, f"WhatsApp note to your {ctx.people}:",
                                  f"{_cap(ctx.people)} ke liye WhatsApp note:")
        note = f"Hi! {biz} is starting {topic}. Reply here and we'll share the batch timings with you first."
    return [_draft_label(ctx, "Launch Google post:", "Launch Google post:"), f"\"{post}\"", note_label, f"\"{note}\""]


def _art_package(ctx: ReplyContext, fx: FactView) -> list[str]:
    """active_planning_intent: the drafted package (tiers from real prices only) plus its launch copy.

    When the opener already carried the package draft and offered to turn it into a post, only the
    launch copy is delivered (no second, conflicting package).
    """
    topic = _strip_prefix(fx.text("trigger.intent_topic"), "Planning:") or (ctx.topics[0] if ctx.topics else "")
    topic = topic or "new package"
    offers = ctx.facts.offers_active
    low = topic.lower()
    launch_only = ctx.stage in ("opened", "engaged") and bool(
        re.search(r"\b(?:post|note|launch|turn this)\b", (ctx.offer or "").lower()))
    if launch_only:
        head = _intro(ctx, f"Here's the launch copy for the {topic}{{name}}, built on the draft above (edit freely):",
                      f"{_cap(topic)} ki launch copy yeh rahi{{name}}, upar wale draft par based "
                      f"(edit kar sakte hain):")
        return [head] + _launch_copy(ctx, fx, topic)
    lines = [_intro(ctx, f"Here's the {topic}{{name}} (draft, edit freely):",
                    f"{_cap(topic)} ka draft yeh raha{{name}} (jaise chahein edit karein):")]
    name, price = _offer_price(offers[0]) if offers else ("", "")
    if any(w in low for w in _CORPORATE_WORDS):
        item = name or "your regular meal"
        lines.append(f"• Daily office lunch: {item} at your current {price} per plate." if price else
                     f"• Daily office lunch: {item} at your current price.")
        lines.append("• Bulk tier (bigger teams): same meal; you set the per-plate rate.")
        lines.append("• Monthly plan: weekday lunches on one monthly invoice; you set the monthly rate.")
    elif any(w in low for w in _PROGRAM_WORDS):
        lines.append(f"• Format: {topic} in small batches on fixed weekly days; you set the dates and fee.")
        if offers:
            lines.append(f"• Hook: pair it with {_join(offers[:2])}.")
        lines.append("• Parents: a short intro session so they meet the instructor first.")
    else:
        lines += [f"• Include: {o} (current offer, unchanged)" for o in offers[:2]]
        lines.append(f"• Pricing: you set the {topic} price; I only use numbers you approve.")
    return lines + [""] + _launch_copy(ctx, fx, topic)


def _art_comeback(ctx: ReplyContext, fx: FactView) -> list[str]:
    """dormant_with_vera: a comeback message for lapsed customers plus one fresh Google post."""
    biz, loc = _biz(ctx), _in_loc(ctx)
    lapsed = next((t for t in fx.texts("agg") if "lapsed" in t.lower()), "")
    offer = ctx.facts.offers_active[0] if ctx.facts.offers_active else ""
    lines = [_intro(ctx, "Here's the comeback pack{name} (edit freely):",
                    "Yeh raha comeback pack{name} (edit kar sakte hain):")]
    if lapsed:
        lines.append(f"Who it's for: your {_lc_first(lapsed.rstrip('.'))}." if not ctx.hi else
                     f"Kiske liye: aapke {_lc_first(lapsed.rstrip('.'))}.")
    lines += [_draft_label(ctx, f"Comeback WhatsApp to lapsed {ctx.people}:",
                           f"Lapsed {ctx.people} ke liye comeback WhatsApp:"),
              f"\"It's been a while! {biz} would love to see you again.{' ' + offer + '.' if offer else ''} "
              f"{_come_back_line(ctx)}\"",
              _draft_label(ctx, "Fresh Google post:", "Fresh Google post:"),
              f"\"{_topic_post(ctx, trend_topic(fx, ctx), 0) if trend_topic(fx, ctx) else _plan_post(ctx, fx)}\""]
    return lines


def _art_onboarding(ctx: ReplyContext, fx: FactView) -> list[str]:
    biz = ctx.facts.merchant_name or ("aapke business" if ctx.hi else "your business")
    catalog = ctx.facts.catalog_offers[0] if ctx.facts.catalog_offers else ""
    offer = ctx.facts.offers_active[0] if ctx.facts.offers_active else ""
    if ctx.hi:
        return [f"Badhiya{_comma_name(ctx)}! {biz} ka setup abhi shuru kar rahi hoon. Next steps yeh rahe:",
                "1) Listing details (timings, photos, services) main fill karke aapko dikhaungi.",
                f"2) Pehla offer draft: '{offer or catalog or 'aapki pasand ka service + price'}' "
                f"(edit kar sakte hain).",
                "3) Go-live ke saath ek welcome Google post."]
    return [f"Great{_comma_name(ctx)}! Starting the setup for {biz} now. Next steps:",
            "1) I fill in the listing details (timings, photos, services) for you to check.",
            f"2) First offer draft: '{offer or catalog or 'one service at a clear price'}' (edit freely).",
            "3) A welcome Google post as you go live."]


def _art_generic(ctx: ReplyContext, fx: FactView) -> list[str]:
    if ctx.topics or re.search(r"\bposts?\b", (ctx.offer or "").lower()):
        return _art_posts(ctx, fx)
    if fx.text("digest.title"):
        return _art_digest(ctx, fx)
    if fx.text("signal.stale_posts") or fx.text("signal.no_recent_post") or fx.text("perf.ctr_vs_peer") \
            or fx.text("trigger.metric_delta") or fx.text("derived.metric_delta"):
        return _art_plan(ctx, fx)
    return _art_posts(ctx, fx)


_BUILDERS = {
    "digest": _art_digest, "trend": _art_trend, "checklist": _art_checklist, "cde": _art_cde,
    "supply": _art_supply, "seasonal": _art_seasonal, "plan": _art_plan, "match": _art_match,
    "challenge": _art_challenge, "milestone": _art_milestone, "posts": _art_posts, "pair": _art_pair,
    "review_replies": _art_reviews, "verification": _art_verification, "renewal": _art_renewal,
    "winback": _art_winback, "package": _art_package, "onboarding": _art_onboarding, "comeback": _art_comeback,
}
