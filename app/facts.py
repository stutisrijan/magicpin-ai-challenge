"""Fact extraction: the anti-hallucination core.

build_facts() turns the four contexts (category, merchant, trigger, customer) into a
FactSheet. Everything a message may claim must come from that sheet:

    anchor_facts   why-now facts from the trigger payload / resolved digest item (weight 3)
    support_facts  merchant / customer / category state (weight 2 or 1)
    notes          judgment hints for the writer (never shown verbatim)
    allowed_numbers every number the contexts contain or that we derive, normalised

Fact.text is always merchant-readable English: no snake_case, no field names, no
signal slugs. Placeholder triggers (payload {"placeholder": true}) get honest anchors
derived from real merchant / category data instead of invented specifics.

build_facts() never raises and is deterministic for the same inputs.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

from app.language import customer_language_plan, merchant_language_plan
from app.numbers import extract_number_tokens, normalize_number, number_variants
from app.schemas import SEND_AS_MERCHANT, SEND_AS_VERA, Fact, FactSheet

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- kind tables

CUSTOMER_KINDS = {
    "recall_due", "customer_lapsed_soft", "customer_lapsed_hard", "appointment_tomorrow",
    "chronic_refill_due", "trial_followup", "wedding_package_followup", "unplanned_slot_open",
}
REMINDER_KINDS = {"recall_due", "appointment_tomorrow", "chronic_refill_due", "appointment_reminder",
                  "refill_due", "vaccination_due", "renewal_reminder"}
# consent scopes that explicitly allow reminders (they win over reminder_opt_in: false), as in the planner
REMINDER_SCOPES = {"recall_reminders", "appointment_reminders", "refill_reminders", "renewal_reminders",
                   "recall_alerts"}
OPTED_OUT_STATES = {"opted_out", "unsubscribed", "dnd", "blocked", "do_not_contact"}
LAPSED_KINDS = {"customer_lapsed_soft", "customer_lapsed_hard"}

_KIND_ALIASES = {
    "research_digest_release": "research_digest",
    "category_research_digest_release": "research_digest",
    "compliance_alert": "regulation_change",
    "regulation_update": "regulation_change",
    "bridal_followup": "wedding_package_followup",
    "winback": "winback_eligible",
    "customer_winback": "customer_lapsed_hard",
    "heatwave": "weather_heatwave",
    "trend_movement": "category_trend_movement",
    "curious_ask": "curious_ask_due",
}

# Knowledge kinds: digest item preference by digest.kind; fall back to the first item.
_DIGEST_PREFS = {
    "research_digest": ("research", "tech", "trend"),
    "regulation_change": ("compliance",),
    "cde_opportunity": ("cde",),
    "supply_alert": ("alert", "supply"),
    "category_seasonal": ("seasonal",),
    "category_trend_movement": ("trend",),
}
# Other kinds that may borrow a digest item as supporting evidence (no fallback).
_SOFT_DIGEST_PREFS = {
    "competitor_opened": ("compete",),
    "seasonal_perf_dip": ("seasonal",),
}
_ID_KEYS = ("top_item_id", "digest_item_id", "alert_id", "item_id", "digest_id")
_IGNORED_PAYLOAD_KEYS = {"placeholder", "metric_or_topic", "category", "merchant_id", "customer_id"}

# --------------------------------------------------------------------------- category tables

_CATEGORY_NOUNS = {  # business noun (singular), people noun (plural)
    "dentists": ("dental clinic", "patients"),
    "salons": ("salon", "clients"),
    "restaurants": ("restaurant", "customers"),
    "gyms": ("gym", "members"),
    "pharmacies": ("pharmacy", "customers"),
}

_SERVICE_NOUNS = {
    "recall_due": {
        "dentists": "routine dental check-up and cleaning", "salons": "regular salon appointment",
        "gyms": "next training session", "_yoga": "next yoga class",
        "pharmacies": "routine refill and health check", "restaurants": "next visit", "_any": "next visit",
    },
    "customer_lapsed_soft": {
        "dentists": "dental check-up", "salons": "salon visit", "gyms": "workout session",
        "_yoga": "yoga class", "pharmacies": "refill", "restaurants": "visit", "_any": "visit",
    },
    "appointment_tomorrow": {
        "dentists": "dental appointment", "salons": "salon appointment", "gyms": "training session",
        "_yoga": "yoga session", "pharmacies": "pharmacy appointment", "restaurants": "table booking",
        "_any": "appointment",
    },
    "chronic_refill_due": {
        "pharmacies": "monthly medicine refill", "dentists": "oral-care refill",
        "gyms": "monthly membership renewal", "_yoga": "monthly membership renewal",
        "salons": "hair and skin care product refill", "restaurants": "regular order", "_any": "regular refill",
    },
    "trial_followup": {
        "gyms": "trial class", "_yoga": "trial yoga class", "dentists": "first consultation",
        "salons": "trial session", "pharmacies": "first order", "restaurants": "first visit",
        "_any": "first visit",
    },
    "wedding_package_followup": {"_any": "bridal package"},
    "unplanned_slot_open": {"_any": "open slot"},
}
_SERVICE_NOUNS["customer_lapsed_hard"] = _SERVICE_NOUNS["customer_lapsed_soft"]

_THEMES = {
    "delivery_late": "late delivery", "wait_time": "wait time", "saturday_wait": "Saturday wait times",
    "doctor_manner": "the doctor's manner", "stylist_skill": "stylist skill", "pizza_quality": "pizza quality",
    "thali_quality": "thali quality", "weekend_busy": "weekend crowding", "equipment_quality": "equipment quality",
    "morning_crowd": "morning crowding", "instructor_quality": "instructor quality",
    "small_classes": "small class sizes", "delivery_speed": "delivery speed",
    "medicine_availability": "medicine availability",
}

# known signals: name -> (kind, text with {n}, text without n). kind "fact" or "note".
_SIGNALS = {
    "stale_posts": ("fact", "Last Google post was {n} days ago", "Google posts have gone stale"),
    "renewal_due_soon": ("fact", "Plan renewal due in {n} days", "Plan renewal due soon"),
    "dormant_with_vera": ("fact", "No reply to Vera in {n} days", "No recent reply to Vera"),
    "ctr_below_peer_median": ("fact", "", "CTR is below the peer median"),
    "unverified_gbp": ("fact", "", "Google Business Profile not verified yet"),
    "no_active_offers": ("fact", "", "No active offer on the profile right now"),
    "perf_dip_severe": ("fact", "", "Sharp drop in recent profile performance"),
    "perf_dip_post_expiry": ("fact", "", "Performance dipped after the plan expired"),
    "growing_views_7d": ("fact", "", "Profile views growing over the last 7 days"),
    "above_peer_median_calls": ("fact", "", "Calls above the peer median"),
    "above_peer_ctr": ("fact", "", "CTR above the peer average"),
    "above_peer_calls": ("fact", "", "Calls above the peer average"),
    "no_recent_post": ("fact", "", "No recent Google post"),
    "trial_ending_soon": ("fact", "", "Trial ending soon"),
    "delivery_not_set_up": ("fact", "", "Home delivery not set up yet"),
    "seasonal_dip_apr_may": ("fact", "", "Seasonal Apr-May dip in demand"),
    "high_repeat_rate": ("fact", "", "High repeat-customer rate"),
    "high_retention": ("fact", "", "High member retention"),
    "engaged_in_last_48h": ("note", "", "Merchant engaged with Vera in the last 48 hours: no intro, continue naturally."),
    "engaged_in_last_24h": ("note", "", "Merchant engaged with Vera in the last 24 hours: no intro, continue naturally."),
    "high_engagement": ("note", "", "Highly engaged merchant: can go a little deeper."),
    "high_risk_adult_cohort": ("note", "", "Merchant has a high-risk adult patient cohort (relevant to recall-interval and caries research)."),
    "compliance_aware": ("note", "", "Compliance-aware operator: be precise and cite sources."),
    "boutique_segment": ("note", "", "Boutique studio: emphasise coach quality and community over price."),
    "new_merchant": ("note", "", "New merchant on the platform: keep it simple and helpful."),
    "high_volume": ("note", "", "High-volume operator: talk covers, AOV and throughput."),
    "stable_growth": ("note", "", "Stable growth: frame ideas as incremental upside."),
    "active_planning": ("note", "", "Merchant is actively planning a new programme."),
    "winback_eligible": ("note", "", "Merchant is eligible for a win-back offer."),
    "ipl_eligible_locality": ("note", "", "Locality responds to IPL match-night demand."),
    "no_recent_conversation": ("note", "", "No recent conversation with Vera: a 3-word intro at most."),
}

_AGG_COUNTS = {
    "total_unique_ytd": "{n} unique {people} so far this year",
    "high_risk_adult_count": "{n} high-risk adult patients",
    "delivery_orders_30d": "{n} delivery orders in the last 30 days",
    "dine_in_orders_30d": "{n} dine-in orders in the last 30 days",
    "total_active_members": "{n} active members",
    "chronic_rx_count": "{n} chronic-prescription customers on repeat refills",
}
_AGG_PCTS = {
    "retention_6mo_pct": "{p} 6-month retention",
    "retention_3mo_pct": "{p} 3-month retention",
    "retention_30d_pct": "{p} 30-day retention",
    "repeat_customer_pct": "{p} repeat customers",
    "delivery_share_pct": "{p} of orders via delivery",
    "monthly_churn_pct": "{p} monthly churn",
    "trial_to_paid_pct": "{p} trial-to-paid conversion",
}

_METRIC_LABELS = {
    "calls": "calls", "views": "profile views", "ctr": "CTR", "directions": "direction requests",
    "leads": "leads", "review_count": "reviews", "reviews": "reviews", "rating": "rating",
    "bookings": "bookings", "orders": "orders", "footfall": "footfall",
}

# --------------------------------------------------------------------------- text tables

_SPECIAL_SLUGS = {
    "6_month_cleaning": "6-month cleaning",
    "post_resolution_window_apr_jun": "post-resolution Apr-Jun window",
    "delivery_late": "late delivery",
    "what_service_in_demand_this_week": "which service is most in demand this week",
    "free_for_members": "free for members",
    "postcard_or_phone_call": "postcard or phone call",
    "skin_prep_program_30day": "30-day skin-prep program",
    "cold_cough": "cold & cough",
    "high_risk_adults": "high-risk adults",
    "diabetes_t2": "type 2 diabetes",
    "whatsapp_via_son": "WhatsApp via son",
    "whatsapp_via_parent": "WhatsApp via parent",
}
_ACRONYMS = {
    "rx": "Rx", "pt": "PT", "otc": "OTC", "ors": "ORS", "gbp": "Google profile", "cde": "CDE",
    "ipl": "IPL", "bogo": "BOGO", "gst": "GST", "hiit": "HIIT", "rct": "RCT", "opg": "OPG",
    "iopa": "IOPA", "bbq": "BBQ", "ytd": "YTD", "ctr": "CTR", "aov": "AOV", "yoy": "YoY", "bp": "BP",
    "sr": "SR", "ldl": "LDL", "cad": "CAD", "ivf": "IVF", "sms": "SMS", "ai": "AI", "gro": "GRO",
}
_MONTHS = {m.lower(): i for i, m in enumerate(
    ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]) if m}
_MONTH_ABBR = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_FESTIVE_WORDS = ("festival", "diwali", "holi", "christmas", "new year", "valentine", "eid",
                  "navratri", "wedding", "pongal", "onam", "durga", "ganesh")
_UNIVERSAL_TABOOS = ("guaranteed", "miracle", "100% safe")

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2}))?")
_RELAY_RE = re.compile(
    r"^\s*(?P<about>[^()]+?)\s*\(\s*(?P<rel>parent|guardian|mother|father|mom|mum|dad|son|daughter|"
    r"spouse|wife|husband|family|via|caretaker|carer)\s*[:\-]\s*(?P<to>[^)]+?)\s*\)\s*$",
    re.IGNORECASE,
)
_ANON_RE = re.compile(r"walk-?in|no profile|anonymous|unknown|^\s*\(", re.IGNORECASE)
_HONORIFIC_RE = re.compile(r"^\s*(mr|mrs|ms|miss|shri|shree|smt|sri|dr)\.?\s+", re.IGNORECASE)
_DR_RE = re.compile(r"^\s*(?:doctor\s+|dr\.\s*|dr\s+)", re.IGNORECASE)
_KNOWN_CITIES = ("delhi", "mumbai", "bangalore", "bengaluru", "hyderabad", "chennai", "pune",
                 "chandigarh", "jaipur", "lucknow", "ahmedabad", "kolkata", "noida", "gurgaon")


# --------------------------------------------------------------------------- small helpers


def _d(x: Any) -> dict:
    return x if isinstance(x, dict) else {}


def _l(x: Any) -> list:
    return list(x) if isinstance(x, (list, tuple)) else []


def _s(x: Any) -> str:
    if x is None or isinstance(x, (dict, list, tuple, bool)):
        return ""
    return str(x).strip()


def _num(x: Any) -> float | None:
    """Numeric value of x (numbers or numeric strings), else None. Booleans are not numbers."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, str):
        s = x.strip().replace(",", "").replace("₹", "").replace("%", "").replace("−", "-")
        try:
            v = float(s)
        except ValueError:
            return None
        return v if math.isfinite(v) else None
    return None


def _clean(text: Any) -> str:
    """Final scrub for Fact.text: no URLs, no underscores, single spaces."""
    s = re.sub(r"(https?://|www\.)\S+", "", _s(text)).replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;:!?)])", r"\1", s)
    return s.strip()


def _plural(noun: str) -> str:
    if noun.endswith("y") and noun[-2:-1] not in "aeiou":
        return noun[:-1] + "ies"
    return noun if noun.endswith("s") else noun + "s"


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _join(items: list[str], conj: str = "and") -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} {conj} {items[1]}"
    return ", ".join(items[:-1]) + f" {conj} {items[-1]}"


def _truncate(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:-")
    return cut + "…"


# --------------------------------------------------------------------------- formatting


def _fmt_num(v: float | int) -> str:
    v = float(v)
    if v.is_integer():
        return f"{int(v):,}"
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _pct_value(frac: float) -> float | int:
    v = abs(frac) * 100
    r = round(v)
    return int(r) if abs(v - r) < 0.05 else round(v, 1)


def _fmt_pct(frac: float) -> str:
    return f"{_pct_value(frac)}%"


def _fmt_ctr(frac: float) -> str:
    return f"{frac * 100:.1f}%"


def _fmt_change(frac: float | None) -> str:
    if frac is None:
        return "changed"
    if abs(frac) < 0.005:
        return "flat"
    return f"{'up' if frac > 0 else 'down'} {_fmt_pct(frac)}"


def _fmt_money(v: float) -> str:
    return f"₹{_fmt_num(v)}"


def _parse_local(value: Any) -> datetime | None:
    """Parse an ISO date/datetime keeping its own offset (no UTC conversion)."""
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    s = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None


def _fmt_date(dt: datetime) -> str:
    return f"{dt.day} {_MONTH_ABBR[dt.month]} {dt.year}"


def _fmt_time(dt: datetime) -> str:
    hour = dt.hour % 12 or 12
    suffix = "am" if dt.hour < 12 else "pm"
    return f"{hour}{suffix}" if dt.minute == 0 else f"{hour}:{dt.minute:02d}{suffix}"


def _fmt_datetime(dt: datetime) -> str:
    if dt.hour == 0 and dt.minute == 0:
        return _fmt_date(dt)
    return f"{_fmt_date(dt)}, {_fmt_time(dt)}"


def _days_between(later: datetime, earlier: datetime) -> int:
    return (later.date() - earlier.date()).days


def _consistent_days(payload_days: float, target: datetime | None, ref: datetime | None,
                     *, since: bool = False) -> bool:
    """A payload day-count is usable when we cannot check it, or when it matches the dates we have."""
    if target is None or ref is None:
        return True
    computed = _days_between(ref, target) if since else _days_between(target, ref)
    return abs(computed - payload_days) <= 2


def _humanize_inline_dates(text: str) -> str:
    """'effective 2026-12-15' -> 'effective 15 Dec 2026' (date-only ISO fragments)."""
    def repl(m: re.Match) -> str:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return m.group(0)
        return _fmt_date(dt)
    return re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b(?![T:\d])", repl, text or "")


def humanize(value: Any) -> str:
    """Merchant-readable text for a raw data value (slugs, bools, dates, numbers, lists)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return _fmt_num(value) if math.isfinite(value) else ""
    if isinstance(value, (list, tuple)):
        return _join([humanize(v) for v in value])
    if isinstance(value, dict):
        for k in ("label", "title", "name", "query"):
            if _s(value.get(k)):
                return humanize(value.get(k))
        return "; ".join(f"{_label(k)} {humanize(v)}" for k, v in value.items() if humanize(v))
    s = str(value).strip()
    if not s:
        return ""
    low = s.lower()
    if low in _SPECIAL_SLUGS:
        return _SPECIAL_SLUGS[low]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}([T ].*)?", s):
        dt = _parse_local(s)
        if dt:
            return _fmt_datetime(dt)
    if "_" not in s and not re.search(r"\d(day|month|week|year)s?\b", s, re.I):
        return s
    t = re.sub(r"(\d+)_?(month|day|week|year|hour|minute)s?(?![a-z])", r"\1-\2", s, flags=re.IGNORECASE)
    t = re.sub(r"(\d+)d\b", r"\1 days", t)
    words = [w for w in re.split(r"[_\s]+", t) if w]
    out: list[str] = []
    for w in words:
        lw = w.lower()
        if lw in _ACRONYMS:
            out.append(_ACRONYMS[lw])
        elif lw in _MONTHS:
            if out and out[-1].lower() in _MONTHS:
                out[-1] = f"{out[-1]}-{_MONTH_ABBR[_MONTHS[lw]]}"
            else:
                out.append(_MONTH_ABBR[_MONTHS[lw]])
        else:
            out.append(w)
    # "skin prep program 30-day" -> "30-day skin prep program"
    if len(out) > 1 and re.fullmatch(r"\d+-(day|month|week|year|hour|minute)", out[-1], re.I):
        out = [out[-1]] + out[:-1]
    # "delivery late" -> "late delivery"
    if len(out) > 1 and out[-1].lower() == "late":
        out = ["late"] + out[:-1]
    text = " ".join(out)
    text = re.sub(r"\bpost (\w)", r"post-\1", text)
    return text


def _label(key: str) -> str:
    """Humanised label for an unknown payload / aggregate key."""
    k = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key)).lower()  # camelCase -> snake
    k = re.sub(r"_(iso|pct|id|flag)$", "", k, flags=re.IGNORECASE)
    return _cap(humanize(k if "_" in k else k + "_"))


def _theme_text(theme: Any) -> str:
    t = _s(theme)
    return _THEMES.get(t.lower(), humanize(t)) if t else ""


def _metric_label(metric: str) -> str:
    return _METRIC_LABELS.get(metric.lower(), humanize(metric)) if metric else "performance"


def _window_text(window: Any) -> str:
    w = _s(window).lower()
    if not w:
        return ""
    m = re.fullmatch(r"(\d+)\s*(d|day|days|w|wk|week|weeks|m|mo|month|months)?", w)
    if not m:
        return humanize(w)
    n = int(m.group(1))
    unit = (m.group(2) or "d")[0]
    word = {"d": "day", "w": "week", "m": "month"}[unit]
    return f"{n} {word}{'s' if n != 1 else ''}"


def _category_key(slug: Any) -> str:
    s = _s(slug).lower()
    if "dent" in s:
        return "dentists"
    if any(w in s for w in ("salon", "beauty", "spa", "parlour", "parlor")):
        return "salons"
    if any(w in s for w in ("restaur", "cafe", "food", "dining", "kitchen")):
        return "restaurants"
    if any(w in s for w in ("gym", "fitness", "yoga")):
        return "gyms"
    if any(w in s for w in ("pharm", "chemist", "medic", "drug")):
        return "pharmacies"
    return s


# --------------------------------------------------------------------------- public helpers


def merchant_display_name(merchant: dict, category_slug: str = "") -> str:
    """How to address the merchant owner: "Dr. Meera" for dentists, "Lakshmi" otherwise.

    Never produces "Dr. Dr." (generated owners may already be "Dr. Sameer").
    Falls back to "<merchant name> team" when no owner name is known.
    """
    merchant = _d(merchant)
    identity = _d(merchant.get("identity"))
    slug = category_slug or merchant.get("category_slug") or identity.get("category") or ""
    is_dentist = _category_key(slug) == "dentists"
    owner = _s(identity.get("owner_first_name") or identity.get("owner_name") or identity.get("owner"))
    if owner:
        had_dr = bool(_DR_RE.match(owner))
        bare = _DR_RE.sub("", owner).strip()
        first = bare.split()[0].strip(",.") if bare else ""
        if first:
            if first.islower():
                first = first.capitalize()
            return f"Dr. {first}" if (is_dentist or had_dr) else first
    name = _s(identity.get("name") or merchant.get("name"))
    if name:
        return f"{name} team"
    return "Doc" if is_dentist else "there"


def resolve_digest_item(category: dict | None, trigger: dict) -> dict | None:
    """The category digest item a trigger refers to (a copy), or None.

    Order: inline payload.top_item; id match (top_item_id / digest_item_id / alert_id);
    a payload that is itself an item (title + source/summary); for knowledge kinds the
    best item by digest kind, falling back to the first item. supply_alert payload
    batches / manufacturer / molecule are merged into the returned copy.
    """
    cat = _d(category)
    t = _d(trigger)
    p = _d(t.get("payload"))
    kind = _canonical_kind(t, p)
    digest = [d for d in _l(cat.get("digest")) if isinstance(d, dict)]
    by_id = {str(d.get("id")): d for d in digest if d.get("id")}

    item: dict | None = None
    inline = p.get("top_item") or p.get("digest_item")
    if isinstance(inline, dict) and inline:
        item = {**by_id.get(str(inline.get("id")), {}), **inline}
    if item is None:
        for key in _ID_KEYS:
            ref = p.get(key)
            if isinstance(ref, str) and ref in by_id:
                item = dict(by_id[ref])
                break
    if item is None and _s(p.get("title")) and (_s(p.get("source")) or _s(p.get("summary"))):
        item = {k: v for k, v in p.items() if k not in _IGNORED_PAYLOAD_KEYS}
    if item is None and digest:
        if kind in _DIGEST_PREFS:
            item = _pick_digest(digest, _DIGEST_PREFS[kind]) or dict(digest[0])
        elif kind in _SOFT_DIGEST_PREFS:
            item = _pick_digest(digest, _SOFT_DIGEST_PREFS[kind])
        elif kind == "ipl_match_today":
            item = next((dict(d) for d in digest
                         if "ipl" in f"{d.get('title', '')} {d.get('summary', '')}".lower()), None)
    if item is None:
        return None
    if kind == "supply_alert":
        for src, dst in (("affected_batches", "affected_batches"), ("batches", "affected_batches"),
                         ("manufacturer", "manufacturer"), ("molecule", "molecule")):
            if p.get(src):
                item[dst] = p[src]
    return item


def _pick_digest(digest: list[dict], prefs: tuple[str, ...]) -> dict | None:
    for pref in prefs:
        for d in digest:
            if _s(d.get("kind")).lower() == pref:
                return dict(d)
    return None


def collect_numbers(*objs: Any) -> set[str]:
    """Normalised numeric forms found anywhere in objs (numbers, numbers inside strings, ISO date parts)."""
    out: set[str] = set()

    def walk(o: Any, depth: int) -> None:
        if depth > 12 or o is None or isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            if math.isfinite(o):
                out.update(_value_numbers(o))
        elif isinstance(o, str):
            out.update(_string_numbers(o))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, (list, tuple, set)):
            for v in o:
                walk(v, depth + 1)
        elif isinstance(o, Fact):
            walk(o.text, depth + 1)

    for obj in objs:
        walk(obj, 0)
    return out


@lru_cache(maxsize=16384)
def _value_numbers(v: float | int) -> frozenset[str]:
    return frozenset(number_variants(v))


@lru_cache(maxsize=16384)
def _string_numbers(s: str) -> frozenset[str]:
    """Numbers inside a string (memoised: category and merchant texts repeat across every call)."""
    out: set[str] = set()
    for tok in extract_number_tokens(s):
        n = normalize_number(tok)
        if n is not None:
            out.add(n)
            out.update(number_variants(n))
    for m in _ISO_DATE_RE.finditer(s):
        parts = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
        if m.group(4):
            hh, mm = int(m.group(4)), int(m.group(5))
            parts += [hh, mm, hh % 12 or 12]
        out.update(str(p) for p in parts)
    return frozenset(out)


def _date_parts(dt: datetime) -> set[str]:
    return {str(dt.day), str(dt.month), str(dt.year)}


def _canonical_kind(t: dict, p: dict) -> str:
    kind = _s(t.get("kind")) or _s(p.get("metric_or_topic")) or "generic"
    kind = kind.lower()
    return _KIND_ALIASES.get(kind, kind)


# --------------------------------------------------------------------------- build context


@dataclass
class _Ctx:
    cat: dict
    m: dict
    t: dict
    c: dict | None
    p: dict
    canon: str
    cat_key: str
    sheet: FactSheet
    ref_dt: datetime | None
    placeholder: bool
    customer_facing: bool
    used: set[str] = field(default_factory=set)
    expired_offers: list[tuple[str, dict]] = field(default_factory=list)
    stale_counts: bool = False       # a payload day-count contradicts `now`: use absolute dates only
    extra_numbers: set[str] = field(default_factory=set)
    _seen: set[tuple[str, str]] = field(default_factory=set)

    # -- fact helpers ---------------------------------------------------------------
    def fact(self, key: str, text: Any, *, weight: int, value: Any = None, source: str = "") -> None:
        clean = _clean(text)
        if not clean or (key, clean) in self._seen:
            return
        self._seen.add((key, clean))
        f = Fact(key=key, text=clean, value=value, source=source, weight=weight)
        (self.sheet.anchor_facts if weight >= 3 else self.sheet.support_facts).append(f)

    def anchor(self, key: str, text: Any, value: Any = None, source: str = "trigger") -> None:
        self.fact(key, text, weight=3, value=value, source=source)

    def support(self, key: str, text: Any, value: Any = None, source: str = "merchant", weight: int = 2) -> None:
        self.fact(key, text, weight=min(weight, 2), value=value, source=source)

    def note(self, text: str) -> None:
        text = re.sub(r"\s+", " ", text or "").strip()
        if text and text not in self.sheet.notes:
            self.sheet.notes.append(text)

    def use(self, *keys: str) -> None:
        self.used.update(keys)

    def has(self, key: str) -> bool:
        return any(f.key == key for f in self.sheet.all_facts())

    def has_prefix(self, prefix: str) -> bool:
        return any(f.key.startswith(prefix) for f in self.sheet.anchor_facts)

    # -- data helpers -----------------------------------------------------------------
    @property
    def identity(self) -> dict:
        return _d(self.m.get("identity"))

    @property
    def perf(self) -> dict:
        return _d(self.m.get("performance"))

    @property
    def peer(self) -> dict:
        return _d(self.cat.get("peer_stats"))

    @property
    def agg(self) -> dict:
        return _d(self.m.get("customer_aggregate"))

    @property
    def people(self) -> str:
        return _CATEGORY_NOUNS.get(self.cat_key, ("business", "customers"))[1]

    @property
    def business_noun(self) -> str:
        if self.cat_key == "gyms" and "yoga" in _s(self.identity.get("name")).lower():
            return "yoga studio"
        return _CATEGORY_NOUNS.get(self.cat_key, ("business", "customers"))[0]


# --------------------------------------------------------------------------- build_facts


def build_facts(
    category: dict | None,
    merchant: dict,
    trigger: dict,
    customer: dict | None = None,
    *,
    now: str | None = None,
    conversation_turns: list[dict] | None = None,
) -> FactSheet:
    """Everything a message about this (category, merchant, trigger, customer) may claim."""
    try:
        return _build(category, merchant, trigger, customer, now, conversation_turns)
    except Exception:  # never break composition: fall back to a minimal but valid sheet
        log.exception("build_facts failed; using minimal fact sheet")
        return _minimal_sheet(category, merchant, trigger, customer, now)


def _minimal_sheet(category: Any, merchant: Any, trigger: Any, customer: Any, now: str | None) -> FactSheet:
    sheet = FactSheet()
    try:
        m, t, p = _d(merchant), _d(trigger), _d(_d(trigger).get("payload"))
        identity = _d(m.get("identity"))
        slug = _s(m.get("category_slug")) or _s(_d(category).get("slug"))
        sheet.kind = _s(t.get("kind")) or "generic"
        sheet.category_slug = slug
        sheet.trigger_id = _s(t.get("id"))
        sheet.merchant_id = _s(m.get("merchant_id") or t.get("merchant_id"))
        sheet.customer_id = _s(_d(customer).get("customer_id") or t.get("customer_id")) or None
        sheet.scope = "customer" if _s(t.get("scope")) == "customer" else "merchant"
        sheet.send_as = SEND_AS_MERCHANT if sheet.scope == "customer" else SEND_AS_VERA
        sheet.merchant_name = _s(identity.get("name"))
        sheet.owner_name = sheet.salutation = merchant_display_name(m, slug)
        sheet.locality = _s(identity.get("locality"))
        sheet.city = _s(identity.get("city"))
        sheet.now_iso = now or ""
        sheet.payload_is_placeholder = bool(p.get("placeholder"))
        sheet.consent_ok = sheet.scope != "customer"
        sheet.consent_reason = "" if sheet.consent_ok else "fact extraction failed; consent unverified"
        sheet.allowed_numbers = collect_numbers(category, merchant, trigger, customer)
    except Exception:
        log.exception("minimal fact sheet failed")
    return sheet


def _build(category: Any, merchant: Any, trigger: Any, customer: Any, now: str | None,
           conversation_turns: Any) -> FactSheet:
    cat, m, t = _d(category), _d(merchant), _d(trigger)
    c = customer if isinstance(customer, dict) else None
    p = _d(t.get("payload"))
    identity = _d(m.get("identity"))
    canon = _canonical_kind(t, p)
    slug = _s(m.get("category_slug")) or _s(identity.get("category")) or _s(cat.get("slug")) or _s(p.get("category"))
    cat_key = _category_key(slug)

    trig_scope = _s(t.get("scope")).lower()
    customer_facing = trig_scope == "customer" or (c is not None and canon in CUSTOMER_KINDS)
    try:
        urgency = max(1, min(5, int(float(t.get("urgency") or 1))))
    except (TypeError, ValueError):
        urgency = 1

    sheet = FactSheet(
        kind=_s(t.get("kind")) or _s(p.get("metric_or_topic")) or "generic",
        scope="customer" if customer_facing else "merchant",
        send_as=SEND_AS_MERCHANT if customer_facing else SEND_AS_VERA,
        category_slug=slug,
        trigger_id=_s(t.get("id")),
        merchant_id=_s(m.get("merchant_id")) or _s(t.get("merchant_id")) or _s(p.get("merchant_id")),
        customer_id=(_s(c.get("customer_id")) if c else "") or _s(t.get("customer_id")) or None,
        urgency=urgency,
        now_iso=now or "",
        merchant_name=_s(identity.get("name")),
        locality=_s(identity.get("locality")),
        city=_s(identity.get("city")),
    )
    placeholder = bool(p.get("placeholder")) or not [k for k in p if k not in _IGNORED_PAYLOAD_KEYS]
    sheet.payload_is_placeholder = placeholder

    ctx = _Ctx(cat=cat, m=m, t=t, c=c, p=p, canon=canon, cat_key=cat_key, sheet=sheet,
               ref_dt=_parse_local(now), placeholder=placeholder, customer_facing=customer_facing)
    if ctx.ref_dt:
        ctx.extra_numbers |= _date_parts(ctx.ref_dt)

    history = _history(m, conversation_turns)
    sheet.history = history[-8:]

    for section in (_section_addressing, _section_language, _section_voice, _section_offers):
        _safe(section, ctx, history)
    if canon in _DIGEST_PREFS or any(x in p for x in _ID_KEYS + ("top_item",)):
        _safe(_digest_section, ctx)
        _safe(_payload_facts, ctx)
    else:
        _safe(_payload_facts, ctx)
        _safe(_digest_section, ctx)
    handler = _HANDLERS.get(canon)
    if handler is not None:
        _safe(handler, ctx)
    elif canon not in CUSTOMER_KINDS:
        ctx.note(f"Unfamiliar trigger type ({humanize(canon)}): explain why now using only the facts listed.")
    if customer_facing or canon in CUSTOMER_KINDS:
        _safe(_h_customer, ctx)
    _safe(_generic_payload_facts, ctx)
    if customer_facing:
        _safe(_customer_support, ctx)
    else:
        _safe(_merchant_support, ctx, history)
    _safe(_section_consent, ctx)
    _safe(_section_content_item, ctx)
    _safe(_ensure_anchor, ctx)
    _safe(_general_notes, ctx, history)
    _safe(_section_allowed_numbers, ctx, conversation_turns)
    return sheet


def _safe(fn: Any, *args: Any) -> None:
    """Run one section; a bug in one section must not lose the rest of the sheet."""
    try:
        fn(*args)
    except Exception:
        log.exception("fact section %s failed", getattr(fn, "__name__", fn))


# --------------------------------------------------------------------------- addressing / language / voice


def _section_addressing(ctx: _Ctx, _history: list[dict]) -> None:
    sheet = ctx.sheet
    owner = merchant_display_name(ctx.m, sheet.category_slug)
    sheet.owner_name = owner
    if not ctx.customer_facing:
        sheet.salutation = owner
        sheet.recipient_name = owner
        return

    # customer-facing: sign as the business
    mname = sheet.merchant_name
    if ctx.cat_key == "dentists" or owner.endswith(" team") or owner in ("there", "Doc"):
        sheet.signer = mname or owner
    else:
        sheet.signer = f"{owner} from {mname}" if mname else owner

    c = ctx.c or {}
    ident = _d(c.get("identity"))
    prefs = _d(c.get("preferences"))
    name = _s(ident.get("name"))
    channel = _s(prefs.get("channel")).lower()
    senior = ident.get("senior_citizen") is True or _age_lower(ident.get("age_band")) >= 60
    family_relay = "via_" in channel and "via_parent" not in channel

    if not name or _ANON_RE.search(name):
        sheet.salutation = sheet.recipient_name = sheet.about_name = ""
        ctx.note("No customer name or profile on file: do not address by name.")
        return
    relay = _RELAY_RE.match(name)
    if relay:
        about = relay.group("about").split()[0]
        to = _HONORIFIC_RE.sub("", relay.group("to")).strip()
        to_first = to.split()[0] if to else ""
        sheet.about_name = about
        sheet.recipient_name = to_first
        sheet.salutation = to_first
        rel = relay.group("rel").lower()
        ctx.note(f"Relay: the message goes to {to_first or 'the family'} ({rel}) about {about}; "
                 f"address {to_first or 'them'} and talk about {about} in the third person.")
        return
    bare = _HONORIFIC_RE.sub("", name).strip()
    honorific = _HONORIFIC_RE.match(name)
    if senior or family_relay:
        surname = bare.split()[-1] if bare else ""
        about = f"{surname} ji" if surname else ""
        sheet.about_name = about
        sheet.salutation = about
        sheet.recipient_name = "" if family_relay else about
        if family_relay:
            who = channel.split("via_", 1)[1].replace("_", " ") or "family"
            ctx.note(f"Family relay: this goes to {about}'s {who} on WhatsApp. Greet with 'Namaste', refer to "
                     f"the customer respectfully as {about}, keep it simple and clear.")
        else:
            ctx.note(f"Senior customer: respectful 'Namaste' and '{about}', simple and clear.")
        return
    if honorific and bare:
        title = honorific.group(1).capitalize()
        sheet.salutation = sheet.recipient_name = f"{title}. {bare.split()[-1]}"
        return
    first = bare.split()[0] if bare else ""
    sheet.salutation = sheet.recipient_name = first


def _age_lower(age_band: Any) -> int:
    m = re.search(r"(\d+)", _s(age_band))
    if not m or "child" in _s(age_band).lower():
        return 0
    return int(m.group(1))


def _section_language(ctx: _Ctx, history: list[dict]) -> None:
    if ctx.customer_facing:
        plan = customer_language_plan(ctx.c or {})
        c = ctx.c or {}
        ident = _d(c.get("identity"))
        channel = _s(_d(c.get("preferences")).get("channel")).lower()
        if ident.get("senior_citizen") is True or _age_lower(ident.get("age_band")) >= 60 or (
                "via_" in channel and "via_parent" not in channel):
            plan.greeting = "Namaste"
    else:
        plan = merchant_language_plan(ctx.m, history)
    ctx.sheet.language = plan


def _section_voice(ctx: _Ctx, _history: list[dict]) -> None:
    voice = _d(ctx.cat.get("voice"))
    ctx.sheet.voice_tone = humanize(voice.get("tone"))
    ctx.sheet.voice_register = humanize(voice.get("register"))
    ctx.sheet.vocab_allowed = [_s(v) for v in _l(voice.get("vocab_allowed")) if _s(v)]
    ctx.sheet.tone_examples = [_s(v) for v in _l(voice.get("tone_examples")) if _s(v)]
    taboos: list[str] = []
    raw = _l(voice.get("vocab_taboo")) or _l(voice.get("taboos")) or _l(voice.get("taboo"))
    for item in list(raw) + list(_UNIVERSAL_TABOOS):
        word = _s(item).split("(")[0].strip()
        if word and word.lower() not in (x.lower() for x in taboos):
            taboos.append(word)
    ctx.sheet.taboos = taboos


def _section_offers(ctx: _Ctx, _history: list[dict]) -> None:
    active, expired = [], []
    for o in _l(ctx.m.get("offers")):
        if isinstance(o, str) and o.strip():
            active.append((o.strip(), None))
            continue
        o = _d(o)
        title = _s(o.get("title"))
        if not title:
            continue
        status = _s(o.get("status")).lower()
        if status in ("", "active", "live", "running"):
            active.append((title, o))
        elif status in ("expired", "ended", "inactive"):
            expired.append((title, o))
    ctx.sheet.offers_active = [a for a, _ in active]
    ctx.sheet.catalog_offers = [
        _s(o.get("title")) if isinstance(o, dict) else _s(o)
        for o in _l(ctx.cat.get("offer_catalog")) if (_s(o.get("title")) if isinstance(o, dict) else _s(o))
    ]
    ctx.expired_offers = [(title, _d(o)) for title, o in expired]


def _offer_support(ctx: _Ctx) -> None:
    for title in ctx.sheet.offers_active:
        ctx.support("offer.active", title, value=title, source="merchant")
    if not ctx.customer_facing:
        for title, o in ctx.expired_offers:
            ended = _parse_local(o.get("ended"))
            ctx.support("offer.expired", f"{title} (expired{' ' + _fmt_date(ended) if ended else ''})",
                        value=title, weight=1)


def _history(m: dict, turns: Any) -> list[dict]:
    out: list[dict] = []
    for turn in _l(m.get("conversation_history")) + _l(turns):
        if isinstance(turn, dict):
            role = _s(turn.get("from") or turn.get("role") or turn.get("from_role")).lower()
            body = _s(turn.get("body") or turn.get("message") or turn.get("text"))
            ts = _s(turn.get("ts") or turn.get("received_at"))
            engagement = _s(turn.get("engagement"))
        else:
            role = _s(getattr(turn, "role", "")).lower()
            body = _s(getattr(turn, "body", ""))
            ts = _s(getattr(turn, "ts", ""))
            engagement = ""
        if not body:
            continue
        role = {"bot": "vera", "assistant": "vera", "owner": "merchant"}.get(role, role or "merchant")
        out.append({"from": role, "body": body, "ts": ts, "engagement": engagement})
    return out


# --------------------------------------------------------------------------- trigger payload facts


def _payload_facts(ctx: _Ctx) -> None:
    """Humanise every payload field we understand, regardless of the trigger kind."""
    p, k = ctx.p, ctx.canon
    if "milestone_value" in p or "value_now" in p:
        _milestone_fact(ctx)
    elif "delta_pct" in p:
        _metric_delta_fact(ctx)
    if "likely_driver" in p:
        ctx.use("likely_driver")
        if _s(p.get("likely_driver")):
            ctx.anchor("trigger.likely_driver", f"Likely driver: the {humanize(p['likely_driver'])}")
    if "is_expected_seasonal" in p or "season_note" in p:
        ctx.use("is_expected_seasonal", "season_note")
        note = humanize(p.get("season_note"))
        if note:
            prefix = "Expected seasonal dip" if p.get("is_expected_seasonal") else "Season"
            ctx.anchor("trigger.season_note", f"{prefix}: {note}", value=p.get("season_note"))
    if "theme" in p:
        _review_theme_fact(ctx)
    if "festival" in p or k == "festival_upcoming":
        _festival_facts(ctx)
    if "match" in p or k == "ipl_match_today":
        _match_facts(ctx)
    if any(x in p for x in ("competitor_name", "distance_km", "their_offer", "opened_date")):
        _competitor_facts(ctx)
    _supply_and_refill_facts(ctx)
    _slot_facts(ctx)
    _service_facts(ctx)
    _wedding_and_trial_facts(ctx)
    _season_trend_facts(ctx)
    _subscription_payload_facts(ctx)
    _misc_payload_facts(ctx)


def _metric_delta_fact(ctx: _Ctx) -> None:
    p = ctx.p
    ctx.use("metric", "delta_pct", "window", "vs_baseline")
    metric = _s(p.get("metric")) or "performance"
    delta = _num(p.get("delta_pct"))
    if delta is None:
        return
    label = _metric_label(metric)
    window = _window_text(p.get("window"))
    text = f"{label} {_fmt_change(delta)}" + (f" over the last {window}" if window else "")
    base = _num(p.get("vs_baseline"))
    if base is not None:
        base_txt = _fmt_ctr(base) if metric.lower() == "ctr" and base < 1 else f"{_fmt_num(base)} {label}"
        text += f" vs a baseline of {base_txt}"
    ctx.anchor("trigger.metric_delta", _cap(text),
               value={"metric": metric, "delta_pct": delta, "window": p.get("window"), "vs_baseline": base})


def _milestone_fact(ctx: _Ctx) -> None:
    p = ctx.p
    ctx.use("metric", "value_now", "milestone_value", "is_imminent")
    label = _metric_label(_s(p.get("metric")) or "count")
    now_v, target = _num(p.get("value_now")), _num(p.get("milestone_value"))
    if now_v is not None and target is not None:
        if now_v < target:
            text = f"{_fmt_num(now_v)} {label} now, {_fmt_num(target - now_v)} away from {_fmt_num(target)}"
        else:
            text = f"Crossed {_fmt_num(target)} {label} (now {_fmt_num(now_v)})"
    elif target is not None:
        text = f"Milestone: {_fmt_num(target)} {label}"
    elif now_v is not None:
        text = f"{_fmt_num(now_v)} {label} now"
    else:
        return
    ctx.anchor("trigger.milestone", text, value={"metric": p.get("metric"), "value_now": now_v, "milestone": target})


def _review_theme_fact(ctx: _Ctx) -> None:
    p = ctx.p
    ctx.use("theme", "occurrences_30d", "trend", "common_quote", "sentiment")
    theme = _theme_text(p.get("theme"))
    if not theme:
        return
    occ = _num(p.get("occurrences_30d"))
    trend = _s(p.get("trend")).lower()
    if occ is not None:
        n = int(occ)
        text = f"{n} review{'s' if n != 1 else ''} in the last 30 days mention{'s' if n == 1 else ''} {theme}"
    else:
        text = f"Reviews mention {theme}"
    if trend in ("rising", "up", "increasing", "growing"):
        text += " (rising)"
    ctx.anchor("trigger.theme", text, value=p.get("theme"))
    quote = _s(p.get("common_quote"))
    if not quote:  # borrow the quote from the merchant's own review themes when it matches
        for rt in _l(ctx.m.get("review_themes")):
            rt = _d(rt)
            if _s(rt.get("theme")).lower() == _s(p.get("theme")).lower() and _s(rt.get("common_quote")):
                quote = _s(rt.get("common_quote"))
                break
    if quote:
        ctx.anchor("trigger.quote", f"Customer quote: '{quote}'", value=quote)


def _festival_facts(ctx: _Ctx) -> None:
    p = ctx.p
    ctx.use("festival", "date", "festival_date", "days_until", "category_relevance")
    name = _s(p.get("festival"))
    date = _parse_local(p.get("date") or p.get("festival_date"))
    if name:
        ctx.anchor("trigger.festival", name + (f" on {_fmt_date(date)}" if date else ""),
                   value={"festival": name, "date": p.get("date")})
    elif date:
        ctx.anchor("trigger.festival", f"Festival date: {_fmt_date(date)}", value=p.get("date"))
    days = _num(p.get("days_until"))
    if days is not None and days >= 0:
        if _consistent_days(days, date, ctx.ref_dt):
            ctx.anchor("trigger.days_until", f"{int(days)} days to go", value=int(days))
            if days > 60:
                ctx.note(f"The festival is {int(days)} days away: frame it as early planning, not a rush.")
        else:
            ctx.stale_counts = True
            ctx.note("The payload's day count does not match today's date: cite the festival date, not a day count.")
    relevance = [str(x).lower() for x in _l(p.get("category_relevance"))]
    if relevance and ctx.cat_key and ctx.cat_key not in relevance:
        ctx.note(f"This festival is flagged as most relevant for {_join([humanize(r) for r in relevance])}; "
                 f"keep the {ctx.business_noun} angle light and practical.")


def _match_facts(ctx: _Ctx) -> None:
    p = ctx.p
    ctx.use("match", "venue", "city", "match_time_iso", "is_weeknight", "match_time")
    if _s(p.get("match")):
        ctx.anchor("trigger.match", f"IPL match today: {_s(p.get('match'))}", value=p.get("match"))
    venue, city = _s(p.get("venue")), _s(p.get("city"))
    if venue or city:
        ctx.anchor("trigger.venue", ", ".join(x for x in (venue, city) if x))
    dt = _parse_local(p.get("match_time_iso") or p.get("match_time"))
    if dt:
        ctx.anchor("trigger.match_time", f"Starts {_fmt_time(dt)} on {_fmt_date(dt)}", value=p.get("match_time_iso"))
    wk = p.get("is_weeknight")
    if isinstance(wk, bool):
        ctx.anchor("trigger.weeknight", "Weeknight match" if wk else "Weekend match (not a weeknight)", value=wk)
    if city and ctx.sheet.city and city.lower() != ctx.sheet.city.lower():
        ctx.note(f"The match is in {city}; the merchant is in {ctx.sheet.city}. Frame it as a TV-viewing night.")


def _competitor_facts(ctx: _Ctx) -> None:
    p = ctx.p
    ctx.use("competitor_name", "distance_km", "their_offer", "opened_date")
    if _s(p.get("competitor_name")):
        ctx.anchor("trigger.competitor", f"New competitor nearby: {_s(p['competitor_name'])}", value=p["competitor_name"])
    dist = _num(p.get("distance_km"))
    if dist is not None:
        ctx.anchor("trigger.distance", f"{_fmt_num(dist)} km away", value=dist)
    if _s(p.get("their_offer")):
        ctx.anchor("trigger.their_offer", f"Their offer: {_s(p['their_offer'])}", value=p["their_offer"])
    opened = _parse_local(p.get("opened_date"))
    if opened:
        ctx.anchor("trigger.opened_date", f"Opened on {_fmt_date(opened)}", value=p.get("opened_date"))


def _supply_and_refill_facts(ctx: _Ctx) -> None:
    p = ctx.p
    if _s(p.get("molecule")):
        ctx.use("molecule")
        ctx.anchor("trigger.molecule", f"Molecule: {_s(p['molecule'])}", value=p["molecule"])
    batches = _l(p.get("affected_batches") or p.get("batches"))
    if batches:
        ctx.use("affected_batches", "batches")
        names = [str(b) for b in batches if _s(b)]
        ctx.anchor("trigger.batches",
                   f"{len(names)} affected batch{'es' if len(names) != 1 else ''}: {', '.join(names)}", value=names)
    if _s(p.get("manufacturer")):
        ctx.use("manufacturer")
        ctx.anchor("trigger.manufacturer", f"Manufacturer: {_s(p['manufacturer'])}", value=p["manufacturer"])
    mols = [str(x) for x in _l(p.get("molecule_list") or p.get("medicines")) if _s(x)]
    if mols:
        ctx.use("molecule_list", "medicines")
        ctx.anchor("trigger.molecules", f"{len(mols)} regular medicine{'s' if len(mols) != 1 else ''} due: "
                   f"{', '.join(mols)}", value=mols)
    if "last_refill" in p:
        ctx.use("last_refill")
        dt = _parse_local(p.get("last_refill"))
        if dt:
            ctx.anchor("trigger.last_service_date", f"Last refill on {_fmt_date(dt)}", value=p.get("last_refill"))
    if "stock_runs_out_iso" in p or "runs_out" in p:
        ctx.use("stock_runs_out_iso", "runs_out")
        dt = _parse_local(p.get("stock_runs_out_iso") or p.get("runs_out"))
        if dt:
            ctx.anchor("trigger.stock_runs_out", f"Current stock runs out on {_fmt_date(dt)}",
                       value=p.get("stock_runs_out_iso"))
    if "delivery_address_saved" in p:
        ctx.use("delivery_address_saved")
        if p.get("delivery_address_saved") is True:
            ctx.anchor("trigger.delivery_address", "Delivery address already saved")


def _slot_labels(raw: Any) -> list[str]:
    out = []
    for s in _l(raw):
        if isinstance(s, str) and s.strip():
            out.append(s.strip())
        elif isinstance(s, dict):
            label = _s(s.get("label"))
            if not label:
                dt = _parse_local(s.get("iso") or s.get("start"))
                label = f"{dt.strftime('%a')} {dt.day} {_MONTH_ABBR[dt.month]}, {_fmt_time(dt)}" if dt else ""
            if label:
                out.append(label)
    return out


def _slot_facts(ctx: _Ctx) -> None:
    p = ctx.p
    for key in ("available_slots", "next_session_options", "slots", "open_slots"):
        if key in p:
            ctx.use(key)
            labels = _slot_labels(p.get(key))
            for label in labels:
                if label not in ctx.sheet.slots:
                    ctx.sheet.slots.append(label)
    if ctx.sheet.slots:
        ctx.anchor("trigger.slots", f"Open slots: {_join(ctx.sheet.slots, 'or')}", value=list(ctx.sheet.slots))


def _service_facts(ctx: _Ctx) -> None:
    p = ctx.p
    if _s(p.get("service_due")):
        ctx.use("service_due")
        ctx.anchor("trigger.service_due", f"{_cap(humanize(p['service_due']))} due", value=p["service_due"])
    if "due_date" in p:
        ctx.use("due_date")
        dt = _parse_local(p.get("due_date"))
        if dt:
            ctx.anchor("trigger.due_date", f"Due on {_fmt_date(dt)}", value=p.get("due_date"))
    if "last_service_date" in p:
        ctx.use("last_service_date")
        dt = _parse_local(p.get("last_service_date"))
        if dt:
            ctx.anchor("trigger.last_service_date", f"Last service on {_fmt_date(dt)}", value=p.get("last_service_date"))


def _wedding_and_trial_facts(ctx: _Ctx) -> None:
    p = ctx.p
    wedding = _parse_local(p.get("wedding_date"))
    if "wedding_date" in p:
        ctx.use("wedding_date")
        if wedding:
            ctx.anchor("trigger.wedding_date", f"Wedding on {_fmt_date(wedding)}", value=p.get("wedding_date"))
    if "days_to_wedding" in p:
        ctx.use("days_to_wedding")
        days = _num(p.get("days_to_wedding"))
        if days is not None and days >= 0:
            if _consistent_days(days, wedding, ctx.ref_dt):
                ctx.anchor("trigger.days_to_wedding", f"{int(days)} days to the wedding", value=int(days))
            else:
                ctx.stale_counts = True
                ctx.note("The payload's days-to-wedding does not match today's date: cite the wedding date instead.")
    for key in ("trial_completed", "trial_date"):
        if key in p:
            ctx.use(key)
            dt = _parse_local(p.get(key))
            if dt:
                noun = "Bridal trial" if ctx.canon == "wedding_package_followup" else _cap(
                    _SERVICE_NOUNS["trial_followup"].get("_yoga" if ctx.business_noun == "yoga studio" else ctx.cat_key,
                                                         "trial"))
                ctx.anchor("trigger.trial_date", f"{noun} on {_fmt_date(dt)}", value=p.get(key))
    if _s(p.get("next_step_window_open")):
        ctx.use("next_step_window_open")
        ctx.anchor("trigger.next_step", f"Window now open for the {humanize(p['next_step_window_open'])}",
                   value=p["next_step_window_open"])


def _trend_text(raw: Any) -> str:
    if isinstance(raw, dict):
        q = _s(raw.get("query") or raw.get("name") or raw.get("item"))
        delta = _num(raw.get("delta_yoy") if raw.get("delta_yoy") is not None else raw.get("delta_pct"))
        if not q:
            return ""
        if delta is None:
            return humanize(q)
        frac = delta if abs(delta) < 1.5 else delta / 100
        return f"'{q}' {_fmt_change(frac)}"
    s = _s(raw)
    m = re.match(r"^(?P<name>.+?)_(?:(?P<what>demand|searches|search|sales|orders)_)?(?P<delta>[+\-−]\d+(?:\.\d+)?)%?$", s)
    if not m:
        return humanize(s)
    name = humanize(m.group("name")) if "_" in m.group("name") else _ACRONYMS.get(m.group("name").lower(), m.group("name"))
    what = f" {m.group('what')}" if m.group("what") else ""
    delta = float(m.group("delta").replace("−", "-"))
    return f"{name}{what} {'up' if delta > 0 else 'down'} {_fmt_num(abs(delta))}%"


def _season_trend_facts(ctx: _Ctx) -> None:
    p = ctx.p
    if _s(p.get("season")):
        ctx.use("season")
        ctx.anchor("trigger.season_note", f"Season: {_cap(humanize(p['season']))}", value=p["season"])
    trends = _l(p.get("trends"))
    if trends:
        ctx.use("trends")
        texts = [_trend_text(x) for x in trends]
        ctx.anchor("trigger.trends", _cap(_join([x for x in texts if x])), value=trends)
    if "shelf_action_recommended" in p:
        ctx.use("shelf_action_recommended")
        if p.get("shelf_action_recommended") is True:
            ctx.anchor("trigger.shelf_action", "Shelf rearrangement recommended")
    if _s(p.get("query")):  # category_trend_movement style payload
        ctx.use("query", "delta_yoy", "segment_age")
        delta = _num(p.get("delta_yoy") if p.get("delta_yoy") is not None else p.get("delta_pct"))
        text = f"'{_s(p['query'])}' searches"
        if delta is not None:
            text += f" {_fmt_change(delta if abs(delta) < 1.5 else delta / 100)} YoY"
        seg = _segment_text(p.get("segment_age"))
        ctx.anchor("trigger.trends", text + (f" ({seg})" if seg else ""), value=p.get("query"))
        ctx.use("delta_pct")


def _subscription_payload_facts(ctx: _Ctx) -> None:
    p = ctx.p
    plan = _s(p.get("plan"))
    if "days_remaining" in p:
        ctx.use("days_remaining")
        days = _num(p.get("days_remaining"))
        if days is not None:
            ctx.anchor("trigger.days_remaining",
                       f"{int(days)} days left on the {plan + ' ' if plan else ''}plan", value=int(days))
    if plan:
        ctx.use("plan")
        ctx.anchor("trigger.plan", f"{plan} plan", value=plan)
    amount = _num(p.get("renewal_amount"))
    if "renewal_amount" in p:
        ctx.use("renewal_amount")
        if amount is not None:
            ctx.anchor("trigger.renewal_amount", f"Renewal amount {_fmt_money(amount)}", value=amount)
    if "days_since_expiry" in p:
        ctx.use("days_since_expiry")
        days = _num(p.get("days_since_expiry"))
        if days is not None:
            ctx.anchor("trigger.days_since_expiry", f"Plan expired {int(days)} days ago", value=int(days))
    if "perf_dip_pct" in p:
        ctx.use("perf_dip_pct")
        d = _num(p.get("perf_dip_pct"))
        if d is not None:
            ctx.anchor("trigger.perf_dip", f"Profile performance {_fmt_change(d)} since the plan lapsed", value=d)
    if "lapsed_customers_added_since_expiry" in p:
        ctx.use("lapsed_customers_added_since_expiry")
        n = _num(p.get("lapsed_customers_added_since_expiry"))
        if n is not None:
            ctx.anchor("trigger.lapsed_added", f"{int(n)} more {ctx.people} lapsed since the plan expired", value=int(n))
    if "verification_path" in p or "estimated_uplift_pct" in p or "verified" in p:
        ctx.use("verification_path", "estimated_uplift_pct", "verified")
        if p.get("verified") is False:
            ctx.anchor("trigger.verified", "Google Business Profile not verified yet", value=False)
        if _s(p.get("verification_path")):
            ctx.anchor("trigger.verification_path", f"Verification by {humanize(p['verification_path'])}",
                       value=p["verification_path"])
        up = _num(p.get("estimated_uplift_pct"))
        if up is not None:
            frac = up if abs(up) < 1.5 else up / 100
            ctx.anchor("trigger.uplift", f"Estimated uplift after verification: {_fmt_pct(frac)}", value=up)


def _misc_payload_facts(ctx: _Ctx) -> None:
    p, c = ctx.p, ctx.c or {}
    last_visit = _parse_local(_d(c.get("relationship")).get("last_visit"))
    if _s(p.get("intent_topic")):
        ctx.use("intent_topic")
        ctx.anchor("trigger.intent_topic", f"Planning: {humanize(p['intent_topic'])}", value=p["intent_topic"])
    if _s(p.get("merchant_last_message")):
        ctx.use("merchant_last_message")
        ctx.anchor("trigger.merchant_last_message", f"Merchant's last message: '{_s(p['merchant_last_message'])}'",
                   value=p["merchant_last_message"])
    if "days_since_last_visit" in p:
        ctx.use("days_since_last_visit")
        days = _num(p.get("days_since_last_visit"))
        if days is not None and days >= 0:
            if _consistent_days(days, last_visit, ctx.ref_dt, since=True):
                ctx.anchor("trigger.days_since_last_visit", f"{int(days)} days since the last visit", value=int(days))
            else:
                ctx.stale_counts = True
                ctx.note("The payload's days-since-visit does not match today's date: cite the last-visit date instead.")
    if _s(p.get("previous_focus")):
        ctx.use("previous_focus")
        ctx.anchor("trigger.previous_focus", f"Previous focus: {humanize(p['previous_focus'])}", value=p["previous_focus"])
    if "previous_membership_months" in p:
        ctx.use("previous_membership_months")
        n = _num(p.get("previous_membership_months"))
        if n is not None:
            ctx.anchor("trigger.previous_membership_months", f"Was a member for {int(n)} months", value=int(n))
    if "days_since_last_merchant_message" in p:
        ctx.use("days_since_last_merchant_message")
        n = _num(p.get("days_since_last_merchant_message"))
        if n is not None:
            ctx.anchor("trigger.days_since_last_message", f"{int(n)} days since the merchant last replied", value=int(n))
    if _s(p.get("last_topic")):
        ctx.use("last_topic")
        ctx.anchor("trigger.last_topic", f"Last topic discussed: {humanize(p['last_topic'])}", value=p["last_topic"])
    if "deadline_iso" in p or "deadline" in p:
        ctx.use("deadline_iso", "deadline")
        dt = _parse_local(p.get("deadline_iso") or p.get("deadline"))
        if dt:
            text = f"Deadline: {_fmt_date(dt)}"
            if ctx.ref_dt and not ctx.stale_counts and _days_between(dt, ctx.ref_dt) > 0:
                text += f" ({_days_between(dt, ctx.ref_dt)} days away)"
            ctx.anchor("trigger.deadline", text, value=p.get("deadline_iso") or p.get("deadline"))
    if "credits" in p:
        ctx.use("credits")
        n = _num(p.get("credits"))
        if n is not None:
            ctx.anchor("digest.credits", f"{_fmt_num(n)} CDE credit{'s' if n != 1 else ''}", value=n)
    if "fee" in p:
        ctx.use("fee")
        fee = p.get("fee")
        amount = _num(fee)
        text = _fmt_money(amount) if amount is not None else humanize(fee)
        if text:
            ctx.anchor("digest.fee", f"Fee: {text}", value=fee)
    if "ask_template" in p or "last_ask_at" in p:
        ctx.use("ask_template", "last_ask_at")
        if _s(p.get("ask_template")):
            ctx.anchor("trigger.ask_topic", f"This week's question: {humanize(p['ask_template'])}?", value=p["ask_template"])
    for key in ("temperature_c", "temp_c", "max_temp_c", "temperature"):
        t = _num(p.get(key))
        if t is not None:
            ctx.use(key)
            city = _s(p.get("city")) or ctx.sheet.city
            if city:
                ctx.use("city")
            ctx.anchor("trigger.weather", f"{_fmt_num(t)}°C" + (f" in {city}" if city else ""), value=t)
            break


def _generic_payload_facts(ctx: _Ctx) -> None:
    """Unknown payload keys become trigger.generic.<key> facts with humanised labels and values."""
    for key, val in ctx.p.items():
        if key in ctx.used or key in _IGNORED_PAYLOAD_KEYS:
            continue
        if val is None or val == "" or val == [] or val == {}:
            continue
        label = _label(key)
        text = _generic_value(key, val)
        if not text:
            continue
        ctx.anchor(f"trigger.generic.{key}", f"{label}: {text}", value=val)


def _generic_value(key: str, val: Any) -> str:
    k = key.lower()
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, (int, float)):
        v = float(val)
        pct_hint = any(h in k for h in ("pct", "percent", "delta", "change", "growth", "uplift", "share", "rate", "ratio"))
        if pct_hint and abs(v) < 1.5 and v != 0:
            if any(h in k for h in ("delta", "change", "growth")) or v < 0:
                return _fmt_change(v)
            return _fmt_pct(v)
        if any(h in k for h in ("amount", "price", "fee", "inr", "cost", "value_rs")):
            return _fmt_money(v)
        if k.endswith("_km"):
            return f"{_fmt_num(v)} km"
        if "days" in k:
            return f"{_fmt_num(v)} days"
        return _fmt_num(v)
    if isinstance(val, (list, tuple)):
        return _join([_generic_value(key, v) if not isinstance(v, (dict, str)) else humanize(v) for v in val])
    if isinstance(val, dict):
        return humanize(val)
    return humanize(_humanize_inline_dates(str(val)))


# --------------------------------------------------------------------------- digest


def _digest_section(ctx: _Ctx) -> None:
    ctx.use(*_ID_KEYS, "top_item", "digest_item")
    item = resolve_digest_item(ctx.cat, ctx.t)
    if item is None:
        return
    k = ctx.canon
    if k == "seasonal_perf_dip" and not _digest_matches_season(item, ctx.p):
        return
    ctx.sheet.digest_item = item
    weight = 3 if (k in _DIGEST_PREFS or k == "ipl_match_today" or any(x in ctx.p for x in _ID_KEYS)
                   or "top_item" in ctx.p) else 2
    _digest_facts(ctx, item, weight)


def _digest_matches_season(item: dict, p: dict) -> bool:
    note = humanize(p.get("season_note")).lower()
    text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
    words = [w for w in re.findall(r"[a-z]{5,}", note) if w not in ("window", "season", "seasonal")]
    if any(w in text for w in words):
        return True
    months = _beat_months(note)
    return bool(months & set(_months_in(text)))


def _headline_stat(item: dict) -> str:
    unit = r"\d+(?:\.\d+)?\s?(?:x\b|mSv|km|days|weeks|months|patients|covers)"
    for field_name in ("summary", "title"):
        text = _humanize_inline_dates(_s(item.get(field_name)))
        for sent in re.split(r"(?<=\.)\s+(?=[A-Z'\"])", text):
            if re.search(r"\d\s?%|₹\s?\d|" + unit, sent):
                return sent.strip().rstrip(".")
    return ""


def _digest_facts(ctx: _Ctx, item: dict, weight: int) -> None:
    def add(key: str, text: Any, value: Any = None) -> None:
        ctx.fact(key, text, weight=weight, value=value, source="category")

    title = _humanize_inline_dates(_s(item.get("title")))
    mfr = _s(ctx.p.get("manufacturer"))
    if ctx.canon == "supply_alert" and mfr:
        title = re.sub(r"\bmanufacturer [A-Z]\b", f"manufacturer {mfr}", title)
    add("digest.title", title, item.get("title"))
    add("digest.source", _s(item.get("source")), item.get("source"))
    stat = _headline_stat(item)
    if stat and stat.lower() != title.lower():
        add("digest.stat", stat)
    trial_n = _num(item.get("trial_n"))
    if trial_n:
        noun = "patient" if ctx.cat_key in ("dentists", "pharmacies") else "participant"
        add("digest.trial_n", f"{_fmt_num(trial_n)}-{noun} trial", trial_n)
    segment = item.get("patient_segment") or item.get("segment")
    if _s(segment):
        add("digest.segment", f"Most relevant for {humanize(segment)}", segment)
    dt = _parse_local(item.get("date"))
    if dt:
        add("digest.date", f"On {_fmt_datetime(dt)}", item.get("date"))
    credits = _num(item.get("credits"))
    if credits and "credits" not in ctx.p and not ctx.has("digest.credits"):
        add("digest.credits", f"{_fmt_num(credits)} CDE credit{'s' if credits != 1 else ''}", credits)
    summary = _humanize_inline_dates(_s(item.get("summary")))
    add("digest.summary", summary, item.get("summary"))
    add("digest.actionable", _humanize_inline_dates(_s(item.get("actionable"))), item.get("actionable"))
    if not ctx.has("trigger.batches") and _l(item.get("affected_batches")):
        names = [str(b) for b in _l(item.get("affected_batches"))]
        ctx.anchor("trigger.batches", f"{len(names)} affected batches: {', '.join(names)}", value=names)
    if not ctx.has("trigger.manufacturer") and _s(item.get("manufacturer")):
        ctx.anchor("trigger.manufacturer", f"Manufacturer: {_s(item['manufacturer'])}")

    # link the research segment to the merchant's own cohort
    seg = _s(segment).lower()
    agg = ctx.agg
    if "high_risk" in seg or "high-risk" in seg:
        n = _num(agg.get("high_risk_adult_count"))
        if n:
            ctx.note(f"The item's segment (high-risk adults) matches this merchant's {_fmt_num(n)} high-risk adult "
                     f"patients: lead with that cohort.")
    if _s(item.get("source")):
        ctx.note(f"Cite the source exactly as given: '{_s(item.get('source'))}'.")


# --------------------------------------------------------------------------- kind handlers


def _h_knowledge(ctx: _Ctx) -> None:
    k, agg = ctx.canon, ctx.agg
    if ctx.sheet.digest_item is None and ctx.placeholder:
        ctx.note("No digest item available: do not invent research, regulations or sources.")
    if k == "research_digest":
        tone = "clinical peer" if ctx.cat_key in ("dentists", "pharmacies") else "practical peer"
        ctx.note(f"Research digest: {tone} tone, one takeaway tied to their business, offer to pull the details "
                 f"and draft a customer-facing note.")
    elif k == "regulation_change":
        ctx.note("Compliance change: calm urgency, state the deadline, offer a quick audit checklist for their setup.")
    elif k == "cde_opportunity":
        ctx.note("Learning opportunity: date, credits and fee from the data only; offer to block the calendar / "
                 "register.")
    elif k == "supply_alert":
        rx = _num(agg.get("chronic_rx_count"))
        if rx:
            ctx.note(f"Do not invent how many customers are affected. The pool to check is the {_fmt_num(rx)} "
                     f"chronic-prescription customers: offer to filter them for the affected batches and draft "
                     f"their WhatsApp note plus the replacement workflow.")
        else:
            ctx.note("Do not invent affected-customer counts: offer to filter the repeat-prescription list and draft "
                     "the customer note plus the replacement workflow.")
        ctx.note("Urgent but not alarming: batch numbers and manufacturer exactly as given.")
    elif k == "category_seasonal":
        ctx.note("Seasonal demand shift: turn the trend numbers into one concrete shelf / stock / promo move.")
    elif k == "category_trend_movement":
        ctx.note("Trend movement: connect the search trend to one of the merchant's services or offers.")


def _h_ipl(ctx: _Ctx) -> None:
    item = ctx.sheet.digest_item or {}
    weeknight = ctx.p.get("is_weeknight")
    offers = ctx.sheet.offers_active
    offer_txt = f"'{offers[0]}'" if offers else "a delivery-friendly offer"
    text = _humanize_inline_dates(f"{item.get('summary', '')}")
    sat_sent = next((s for s in re.split(r"(?<=\.)\s+", text) if "saturday" in s.lower() and "%" in s), "")
    wk_sent = next((s for s in re.split(r"(?<=\.)\s+", text) if "weeknight" in s.lower() and "%" in s), "")
    if weeknight is False:
        pct = re.search(r"\d+(?:\.\d+)?%", sat_sent)
        if pct:
            ctx.note(f"Weekend (non-weeknight) match: the restaurants digest says Saturday IPL matches cut restaurant "
                     f"covers {pct.group(0)} as fans watch at home; recommend a delivery push of the existing offer "
                     f"{offer_txt} over a match-night dine-in promo.")
        else:
            ctx.note(f"Weekend match: match-night dine-in promos work best on weeknights; recommend a delivery push "
                     f"of {offer_txt} instead.")
    elif weeknight is True:
        pct = re.search(r"\+?\d+(?:\.\d+)?%", wk_sent)
        combo = next((o for o in ctx.sheet.catalog_offers if "match" in o.lower()), "")
        ctx.note("Weeknight match" + (f": the digest says weeknight matches drive {pct.group(0)} covers" if pct else "")
                 + "; push a match-night combo" + (f" (idea from the category catalog: '{combo}', suggest it, "
                                                   f"it is not an existing offer)" if combo else "") + ".")
    match_dt = _parse_local(ctx.p.get("match_time_iso"))
    for offer in offers:
        days = _offer_days(offer)
        if days and match_dt and match_dt.weekday() not in days:
            ctx.note(f"The active offer '{offer}' only runs on certain weekdays and does not cover today's match; "
                     f"frame it as a delivery push for its valid days or the next weeknight match.")
            break


def _offer_days(title: str) -> set[int]:
    names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    m = re.search(r"\b(mon|tue|wed|thu|fri|sat|sun)[a-z]*\s*-\s*(mon|tue|wed|thu|fri|sat|sun)[a-z]*\b", title.lower())
    if m:
        a, b = names.index(m.group(1)), names.index(m.group(2))
        out, i = set(), a
        while True:
            out.add(i)
            if i == b:
                break
            i = (i + 1) % 7
        return out
    if "weekday" in title.lower():
        return {0, 1, 2, 3, 4}
    return set()


def _h_festival(ctx: _Ctx) -> None:
    if not ctx.placeholder and ctx.has("trigger.festival"):
        date = _parse_local(ctx.p.get("date") or ctx.p.get("festival_date"))
        if date:  # the category's seasonal beat for the festival month is real, relevant evidence
            beats = [_d(b) for b in _l(ctx.cat.get("seasonal_beats")) if _s(_d(b).get("note"))]
            matching = [b for b in beats if date.month in _beat_months(_s(b.get("month_range")))]
            matching.sort(key=lambda b: not _is_festive(b))
            if matching:
                ctx.support("seasonal.note", _beat_text(matching[0]), value=matching[0], source="category")
                ctx.sheet.seasonal_note = _clean(_beat_text(matching[0]))
        ctx.note("Festival hook: tie it to one concrete service/offer and a booking window; no invented demand numbers.")
        return
    ref = ctx.ref_dt or _parse_local(ctx.t.get("expires_at")) or datetime.now(timezone.utc)
    beats = [_d(b) for b in _l(ctx.cat.get("seasonal_beats")) if _s(_d(b).get("note"))]
    current = [b for b in beats if ref.month in _beat_months(_s(b.get("month_range")))]
    festive_now = [b for b in current if _is_festive(b)]
    upcoming = None
    if not festive_now:
        best = None
        for b in beats:
            months = _beat_months(_s(b.get("month_range")))
            if not months or not _is_festive(b):
                continue
            ahead = min(((mm - ref.month) % 12) for mm in months)
            if 0 < ahead <= 6 and (best is None or ahead < best[0]):
                best = (ahead, b)
        if best:
            upcoming = best[1]
            ctx.note(f"The next festive window in this category is {_s(upcoming.get('month_range'))} "
                     f"(about {best[0]} month{'s' if best[0] != 1 else ''} away): plan ahead, no festival name.")
    for b in festive_now + ([upcoming] if upcoming else []) + [b for b in current if b not in festive_now]:
        ctx.anchor("trigger.season_note", _beat_text(b), value=b, source="category")
        if not ctx.sheet.seasonal_note:
            ctx.sheet.seasonal_note = _clean(_beat_text(b))
    ctx.note("No festival name or date in the data: never name a festival; anchor on the seasonal pattern.")


def _is_festive(beat: dict) -> bool:
    text = _s(beat.get("note")).lower()
    return any(w in text for w in _FESTIVE_WORDS)


_FULL_MONTHS = {name.lower(): i for i, name in enumerate(
    ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
     "November", "December"]) if name}


def _months_in(text: str) -> list[int]:
    """Month numbers named in text, by exact token ("Apr", "April"); never substrings like "summary"."""
    out = []
    for tok in re.findall(r"[A-Za-z]+", text or ""):
        low = tok.lower()
        m = _MONTHS.get(low) or _FULL_MONTHS.get(low) or (_MONTHS.get("sep") if low == "sept" else None)
        if m:
            out.append(m)
    return out


def _beat_months(month_range: str) -> set[int]:
    months = _months_in(month_range)
    if not months:
        return set()
    if len(months) >= 2 and "-" in month_range:
        a, b = months[0], months[1]
        out, cur = set(), a
        for _ in range(12):
            out.add(cur)
            if cur == b:
                break
            cur = cur % 12 + 1
        return out
    return set(months)


def _beat_text(beat: dict) -> str:
    rng, note = _s(beat.get("month_range")), _s(beat.get("note"))
    return f"{rng}: {note}" if rng else note


def _h_competitor(ctx: _Ctx) -> None:
    p = ctx.p
    if ctx.placeholder or not _s(p.get("competitor_name")):
        loc = ctx.sheet.locality or ctx.sheet.city
        if not ctx.has("trigger.competitor"):
            ctx.anchor("trigger.competitor", f"A new {ctx.business_noun} listing opened near {loc}" if loc
                       else f"A new {ctx.business_noun} listing opened nearby")
        ctx.note("No competitor name, distance or offer in the data: do not invent any. Defend with the merchant's "
                 "own strengths (offers, reviews, numbers).")
    their = _s(p.get("their_offer"))
    tm = re.search(r"^(.*?)@\s*₹\s?([\d,]+)", their)
    if tm:
        their_svc, their_price = tm.group(1).strip().lower(), _num(tm.group(2))
        for offer in ctx.sheet.offers_active:
            om = re.search(r"^(.*?)@\s*₹\s?([\d,]+)", offer)
            if not om or their_price is None:
                continue
            svc, price = om.group(1).strip().lower(), _num(om.group(2))
            overlap = set(re.findall(r"[a-z]{4,}", svc)) & set(re.findall(r"[a-z]{4,}", their_svc))
            if overlap and price is not None:
                gap = price - their_price
                if gap > 0:
                    ctx.note(f"Their '{their}' undercuts your '{offer}' by {_fmt_money(gap)}: defend on value "
                             f"(experience, reviews, add-ons), not a price war.")
                elif gap < 0:
                    ctx.note(f"Your '{offer}' is already {_fmt_money(-gap)} cheaper than their '{their}': make that visible.")
                break
    if ctx.sheet.digest_item:
        ctx.note("The category digest has a related competition item: use it as supporting context only.")


def _h_perf(ctx: _Ctx) -> None:
    k = ctx.canon
    spike = k == "perf_spike"
    if not ctx.has("trigger.metric_delta"):
        _derive_metric_delta(ctx, want_positive=spike)
    if k == "seasonal_perf_dip":
        season_months = _beat_months(humanize(ctx.p.get("season_note")))
        month = ctx.ref_dt.month if ctx.ref_dt else None
        for b in _l(ctx.cat.get("seasonal_beats")):
            b = _d(b)
            months = _beat_months(_s(b.get("month_range")))
            if months and ((season_months and season_months <= months) or (not season_months and month in months)):
                ctx.support("seasonal.note", _beat_text(b), value=b, source="category")
                ctx.sheet.seasonal_note = _clean(_beat_text(b))
                break
        members = _num(ctx.agg.get("total_active_members"))
        who = f"their {_fmt_num(members)} active members" if members else f"their existing {ctx.people}"
        if ctx.p.get("is_expected_seasonal"):
            ctx.note(f"Expected seasonal dip: reassure, this is the normal seasonal lull for the category, and "
                     f"redirect effort to retaining {who}; do not push acquisition spend now.")
        else:
            ctx.note(f"Seasonal dip: acknowledge the pattern and redirect to retaining {who}.")
    elif spike:
        ctx.note("Positive momentum: credit the likely driver if known, suggest doubling down with one concrete move.")
    else:
        ctx.note("Performance dip: loss framing with their real numbers; name only causes visible in the data "
                 "(stale posts, unverified profile, no active offer, expired plan); never invent causes.")
        sub = _d(ctx.m.get("subscription"))
        exp = _num(sub.get("days_since_expiry"))
        if _s(sub.get("status")).lower() == "expired" and exp:
            ctx.note(f"The plan expired {int(exp)} days ago (profile upkeep paused): a real, visible risk to mention.")


def _derive_metric_delta(ctx: _Ctx, *, want_positive: bool) -> None:
    """Placeholder perf triggers: anchor on real 7-day deltas with the right sign, else a real peer gap."""
    perf, delta = ctx.perf, _d(ctx.perf.get("delta_7d"))
    window = _num(perf.get("window_days")) or 30
    cands = []
    for key, val in delta.items():
        v = _num(val)
        metric = re.sub(r"_pct$", "", str(key))
        if v is None or abs(v) < 0.005 or (v > 0) != want_positive:
            continue
        cands.append((abs(v), metric, v))
    if cands:
        cands.sort(key=lambda x: (-x[0], x[1]))
        _, metric, v = cands[0]
        label = _metric_label(metric)
        text = f"{_cap(label)} {_fmt_change(v)} over the last 7 days"
        count = _num(perf.get(metric))
        if count is not None and metric != "ctr":
            text += f" ({_fmt_num(count)} {label} in the last {_fmt_num(window)} days)"
        ctx.anchor("derived.metric_delta", text, value={"metric": metric, "delta_pct": v, "kind": "delta_7d"},
                   source="derived")
        return
    gaps = _peer_gaps(ctx)
    gaps = [g for g in gaps if (g["gap"] > 0) == want_positive and abs(g["gap"]) >= 0.05]
    if gaps:
        gaps.sort(key=lambda g: (-abs(g["gap"]), g["metric"]))
        g = gaps[0]
        ctx.anchor("derived.metric_delta", g["text"], value={**g, "kind": "peer_gap"}, source="derived")
        ctx.note("No 7-day move in the needed direction: the anchor is a peer comparison; say so honestly, "
                 "don't claim a weekly drop or spike.")
        return
    if not want_positive:
        sub = _d(ctx.m.get("subscription"))
        exp = _num(sub.get("days_since_expiry"))
        if _s(sub.get("status")).lower() == "expired" and exp:
            ctx.anchor("derived.metric_delta", f"Plan expired {int(exp)} days ago, so profile upkeep is paused",
                       value={"kind": "expired_plan", "days_since_expiry": exp}, source="derived")
            ctx.note("The weekly numbers show no drop: frame it as a risk (upkeep paused) rather than a measured dip.")
            return
    views, calls = _num(perf.get("views")), _num(perf.get("calls"))
    if views is not None or calls is not None:
        parts = [f"{_fmt_num(views)} profile views" if views is not None else "",
                 f"{_fmt_num(calls)} calls" if calls is not None else ""]
        ctx.anchor("derived.metric_delta", f"{_join(parts)} in the last {_fmt_num(window)} days",
                   value={"kind": "flat"}, source="derived")
        ctx.note("The data shows no clear move in the expected direction: keep claims neutral and check in instead.")


def _peer_gaps(ctx: _Ctx) -> list[dict]:
    perf, peer = ctx.perf, ctx.peer
    out = []
    for metric, peer_key, label in (("ctr", "avg_ctr", "CTR"), ("views", "avg_views_30d", "profile views"),
                                    ("calls", "avg_calls_30d", "calls"),
                                    ("directions", "avg_directions_30d", "direction requests")):
        mine, avg = _num(perf.get(metric)), _num(peer.get(peer_key))
        if mine is None or not avg:
            continue
        gap = (mine - avg) / avg
        pct = int(round(abs(gap) * 100))
        where = "above" if gap > 0.02 else "below" if gap < -0.02 else "in line with"
        if metric == "ctr":
            head = f"CTR {_fmt_ctr(mine)} vs {_fmt_ctr(avg)} peer average"
        else:
            head = f"{_fmt_num(mine)} {label} vs {_fmt_num(avg)} peer average in 30 days"
        tail = "in line with peers" if where == "in line with" else f"{pct}% {where} peers"
        out.append({"metric": metric, "gap": gap, "text": f"{head} ({tail})", "key": f"perf.{metric}_vs_peer"})
    return out


def _round_threshold(v: float) -> int | None:
    if v < 10:
        return None
    step = 10 if v < 100 else 100 if v < 1000 else 500 if v < 10000 else 1000 if v < 100000 else 10000
    th = int(v // step) * step
    return th if th >= step else None


def _h_milestone(ctx: _Ctx) -> None:
    if ctx.has("trigger.milestone"):
        ctx.note("Milestone: celebrate briefly, then one move that compounds it (e.g. ask happy customers for reviews).")
        return
    perf, agg = ctx.perf, ctx.agg
    window = _num(perf.get("window_days")) or 30
    cands = [
        ("views", _num(perf.get("views")), f"profile views in the last {_fmt_num(window)} days"),
        ("total_unique_ytd", _num(agg.get("total_unique_ytd")), f"unique {ctx.people} this year"),
        ("calls", _num(perf.get("calls")), f"calls in the last {_fmt_num(window)} days"),
        ("directions", _num(perf.get("directions")), f"direction requests in the last {_fmt_num(window)} days"),
    ]
    primary = True
    for metric, value, label in cands:
        if value is None or (metric == "views" and value < 100):
            continue
        th = _round_threshold(value)
        if not th:
            continue
        text = f"Crossed {_fmt_num(th)} {label} ({_fmt_num(value)} now)"
        if primary:
            ctx.anchor("derived.milestone", text, value={"metric": metric, "threshold": th, "actual": value},
                       source="derived")
            primary = False
        elif metric == "total_unique_ytd":
            ctx.support("derived.milestone_customers", text, value={"metric": metric, "threshold": th, "actual": value},
                        source="derived")
        ctx.extra_numbers.add(str(th))
    ctx.note("Milestone derived from the merchant's real numbers (a round threshold below the actual value): "
             "say 'crossed', never a bigger number.")


def _h_review(ctx: _Ctx) -> None:
    if not ctx.has("trigger.theme"):
        themes = [_d(r) for r in _l(ctx.m.get("review_themes")) if _s(_d(r).get("theme"))]
        themes.sort(key=lambda r: (_s(r.get("sentiment")) != "neg", -(_num(r.get("occurrences_30d")) or 0)))
        if themes:
            rt = themes[0]
            theme = _theme_text(rt.get("theme"))
            occ = _num(rt.get("occurrences_30d"))
            text = (f"{int(occ)} review{'s' if occ != 1 else ''} in the last 30 days mention {theme}"
                    if occ is not None else f"Reviews mention {theme}")
            ctx.anchor("trigger.theme", text, value=rt.get("theme"), source="merchant")
            if _s(rt.get("common_quote")):
                ctx.anchor("trigger.quote", f"Customer quote: '{_s(rt['common_quote'])}'", source="merchant")
        else:
            peer = ctx.peer
            rating = _num(peer.get("avg_rating"))
            reviews = _num(peer.get("avg_review_count") or peer.get("avg_reviews"))
            if rating or reviews:
                bits = [f"a {_fmt_num(rating)}★ rating" if rating else "",
                        f"{_fmt_num(reviews)} reviews" if reviews else ""]
                ctx.anchor("derived.review_benchmark",
                           f"Peer {_plural(ctx.business_noun)} in this category average {_join(bits)}",
                           source="category")
            _derive_metric_delta(ctx, want_positive=False)
            ctx.note("No review theme in the data: do not invent one. Offer to pull the latest review summary and "
                     "draft replies to recent reviews.")
    ctx.note("Reviews: acknowledge the pattern, offer a drafted public reply plus one operational fix; never argue "
             "with customers.")


def _h_renewal(ctx: _Ctx) -> None:
    sub = _d(ctx.m.get("subscription"))
    plan = _s(sub.get("plan"))
    status = _s(sub.get("status")).lower()
    if not ctx.has("trigger.days_remaining") and not ctx.has("trigger.days_since_expiry"):
        left, since = _num(sub.get("days_remaining")), _num(sub.get("days_since_expiry"))
        if status == "expired" and since:
            ctx.anchor("trigger.days_since_expiry", f"{plan + ' ' if plan else ''}plan expired {int(since)} days ago",
                       value=int(since), source="merchant")
        elif status == "trial" and left is not None:
            ctx.anchor("trigger.days_remaining", f"Trial ends in {int(left)} days", value=int(left), source="merchant")
        elif left is not None:
            ctx.anchor("trigger.days_remaining", f"{int(left)} days left on the {plan + ' ' if plan else ''}plan",
                       value=int(left), source="merchant")
            if left > 45:
                ctx.note(f"Renewal is {int(left)} days away: frame it as an early value check-in, no urgency claims.")
    if not ctx.has("trigger.renewal_amount"):
        ctx.note("No renewal amount in the data: do not quote a price.")
    ctx.note("Renewal: show value with their own 30-day numbers (views, calls, leads) and what lapses if it expires.")


def _h_winback(ctx: _Ctx) -> None:
    if not ctx.has("trigger.days_since_expiry"):
        _h_renewal(ctx)
    ctx.note("Win-back: loss framing with their real post-expiry numbers; one easy restart step, no guilt.")


def _h_dormant(ctx: _Ctx) -> None:
    if ctx.placeholder or not ctx.has_prefix("trigger."):
        delta = _d(ctx.perf.get("delta_7d"))
        cands = [(abs(v), k, v) for k, v in ((k, _num(v)) for k, v in delta.items()) if v is not None and abs(v) >= 0.005]
        if cands:
            cands.sort(key=lambda x: (-x[0], x[1]))
            _, key, v = cands[0]
            metric = re.sub(r"_pct$", "", key)
            ctx.anchor("derived.metric_delta", f"{_cap(_metric_label(metric))} {_fmt_change(v)} over the last 7 days",
                       value={"metric": metric, "delta_pct": v, "kind": "delta_7d"}, source="derived")
    ctx.note("Dormant merchant: re-open with one fresh number from their own profile and a single easy question; "
             "no guilt about the silence, no long pitch.")


def _h_gbp(ctx: _Ctx) -> None:
    verified = ctx.identity.get("verified")
    if verified is False and not ctx.has("trigger.verified"):
        ctx.anchor("trigger.verified", "Google Business Profile not verified yet", source="merchant")
    ctx.note("Unverified profile: explain the one verification step from the data and offer to walk them through it.")


def _h_curious(ctx: _Ctx) -> None:
    trend = _top_trend(ctx)
    if trend:
        ctx.anchor("trend.top", trend[0], value=trend[1], source="category")
        ctx.sheet.trend_note = _clean(trend[0])
    ctx.note("Curious ask: one easy question about their business this week (optionally a grounded guess from the "
             "trend), and promise a concrete artifact in return: a Google post plus a WhatsApp reply draft.")


def _h_planning(ctx: _Ctx) -> None:
    offers = ctx.sheet.offers_active
    ctx.note("The merchant asked what it would look like: deliver a concrete drafted artifact now (clearly marked as "
             "a draft to edit), then exactly one ask. No qualifying questions.")
    if offers:
        ctx.note(f"Build the draft from their active offer(s): {_join([repr(o) for o in offers])}; any new tier prices "
                 f"must be labelled as suggestions.")
    ctx.note("Do not invent named customers, offices, buildings or partners.")


def _h_weather(ctx: _Ctx) -> None:
    ctx.note("Weather event: connect it to one practical move for this category (e.g. delivery, hydration, timing); "
             "no invented demand numbers.")


def _h_local_news(ctx: _Ctx) -> None:
    ctx.note("Local event: one practical implication for this merchant today; no invented details.")


def _h_customer(ctx: _Ctx) -> None:
    k = ctx.canon
    c = ctx.c or {}
    rel = _d(c.get("relationship"))
    svc = _service_noun(ctx)
    last = _parse_local(rel.get("last_visit"))
    ref = None if ctx.stale_counts else ctx.ref_dt
    if k in ("recall_due", "chronic_refill_due") and not ctx.has("trigger.service_due") \
            and not ctx.has("trigger.molecules"):
        ctx.anchor("trigger.service_due", f"{_cap(svc)} due", value=svc, source="derived")
    if k == "appointment_tomorrow" and not ctx.has("trigger.due_date"):
        tomorrow = ref + timedelta(days=1) if ref else None
        if tomorrow:
            ctx.extra_numbers |= _date_parts(tomorrow)
        ctx.anchor("trigger.due_date", f"{_cap(svc)} scheduled for tomorrow" +
                   (f", {_fmt_date(tomorrow)}" if tomorrow else ""), value=svc, source="derived")
        ctx.note("No appointment time in the data: don't invent one; ask them to confirm or reschedule.")
    if k in LAPSED_KINDS and not ctx.has("trigger.service_due") and not ctx.has("trigger.previous_focus"):
        ctx.anchor("trigger.service_due", f"Due for a {svc}", value=svc, source="derived")
    if k in LAPSED_KINDS and not ctx.has("trigger.days_since_last_visit") and last and ref:
        days = _days_between(ref, last)
        if days > 0:
            ctx.anchor("trigger.days_since_last_visit", f"{days} days since the last visit", value=days, source="derived")
    if k == "trial_followup" and not ctx.has("trigger.trial_date") and not ctx.has("trigger.next_step"):
        ctx.anchor("trigger.next_step", f"Follow-up after the {svc}", value=svc, source="derived")
    if last and not ctx.has("trigger.last_service_date") and (ctx.placeholder or k in LAPSED_KINDS | {"trial_followup"}):
        ago = _ago(last, ref)
        ctx.anchor("trigger.last_service_date", f"Last visit on {_fmt_date(last)}{ago}", value=rel.get("last_visit"),
                   source="customer")

    # judgment hints
    if ctx.customer_facing:
        ctx.note(f"Customer-facing: speak as the business ({ctx.sheet.signer or ctx.sheet.merchant_name}); never "
                 f"mention Vera or magicpin; no medical claims or guarantees.")
    offers = ctx.sheet.offers_active
    if offers:
        ctx.note(f"Quote prices only from the merchant's active offers: {_join([repr(o) for o in offers])}.")
    else:
        ctx.note("The merchant has no active offer: do not quote any price, discount or freebie.")
    if ctx.sheet.slots:
        ctx.note(f"Offer exactly these slots: {_join(ctx.sheet.slots, 'or')}.")
    elif k not in ("chronic_refill_due", "appointment_tomorrow"):
        ctx.note("No open slots in the data: ask for a convenient time; don't invent slots.")
    pref = _s(_d(c.get("preferences")).get("preferred_slots"))
    if pref:
        ctx.note(f"Customer prefers {_slot_pref_text(pref)}: honour it.")
    if k in LAPSED_KINDS or (_s(c.get("state")).startswith("lapsed")
                             and k in ("recall_due", "trial_followup", "chronic_refill_due")):
        ctx.note("Lapsed customer: warm, no guilt-tripping about the gap, one easy way back.")
    if k == "chronic_refill_due" and ctx.cat_key != "pharmacies":
        ctx.note(f"Refill trigger for a {ctx.business_noun} customer: keep it to a gentle '{svc}' check-in; no "
                 f"medicine or product names.")
    if k == "recall_due" and ctx.cat_key not in ("dentists",):
        ctx.note(f"Recall at a {ctx.business_noun}: frame it as time for the {svc}, not a medical recall.")
    if k == "wedding_package_followup":
        ctx.note("Bridal follow-up: count down to the wedding date from the data; quote no package price unless it "
                 "is an active offer.")
    if ctx.placeholder:
        ctx.note(f"Service wording ('{svc}') is inferred from the business category: keep it generic.")


def _service_noun(ctx: _Ctx) -> str:
    table = _SERVICE_NOUNS.get(ctx.canon) or {"_any": "next visit"}
    if ctx.business_noun == "yoga studio" and "_yoga" in table:
        return table["_yoga"]
    return table.get(ctx.cat_key) or table.get("_any") or "next visit"


def _ago(dt: datetime, ref: datetime | None) -> str:
    if ref is None:
        return ""
    days = _days_between(ref, dt)
    if days <= 0:
        return ""
    if days < 60:
        return f" ({days} days ago)"
    return f" (about {days // 30} months ago)"


def _slot_pref_text(pref: str) -> str:
    words = [w for w in re.split(r"[_\s]+", pref.lower()) if w]
    days = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri", "sat": "Sat", "sun": "Sun",
            "monday": "Monday", "tuesday": "Tuesday", "wednesday": "Wednesday", "thursday": "Thursday",
            "friday": "Friday", "saturday": "Saturday", "sunday": "Sunday", "weekday": "weekday", "weekend": "weekend"}
    times = {"morning": "mornings", "evening": "evenings", "afternoon": "afternoons", "night": "nights"}
    out: list[str] = []
    for i, w in enumerate(words):
        last = i == len(words) - 1
        if w in days:
            if out and out[-1] in days.values():
                out[-1] = f"{out[-1]}/{days[w]}"
            else:
                out.append(days[w])
        elif w in times:
            out.append(times[w] if last else w)
        else:
            out.append(w)
    if len(out) == 1 and out[0] in days.values() and out[0] not in ("weekday", "weekend"):
        out[0] += "s"                                   # "saturday" -> "Saturdays"
    text = " ".join(out)
    text = re.sub(r"^(weekday|weekend) (?=\d)", r"\1s at ", text)
    text = re.sub(r"\bweekday after\b", "weekdays after", text)
    text = re.sub(r"^(morning|evening|afternoon|night)s? (\d+(?:am|pm))$", r"\1s around \2", text)
    return text


_HANDLERS = {
    "research_digest": _h_knowledge, "regulation_change": _h_knowledge, "cde_opportunity": _h_knowledge,
    "supply_alert": _h_knowledge, "category_seasonal": _h_knowledge, "category_trend_movement": _h_knowledge,
    "festival_upcoming": _h_festival, "ipl_match_today": _h_ipl, "weather_heatwave": _h_weather,
    "local_news_event": _h_local_news, "competitor_opened": _h_competitor, "perf_dip": _h_perf,
    "perf_spike": _h_perf, "seasonal_perf_dip": _h_perf, "milestone_reached": _h_milestone,
    "review_theme_emerged": _h_review, "renewal_due": _h_renewal, "winback_eligible": _h_winback,
    "dormant_with_vera": _h_dormant, "gbp_unverified": _h_gbp, "curious_ask_due": _h_curious,
    "scheduled_recurring": _h_curious, "active_planning_intent": _h_planning,
}


# --------------------------------------------------------------------------- support facts


def _merchant_support(ctx: _Ctx, history: list[dict]) -> None:
    perf, delta = ctx.perf, _d(ctx.perf.get("delta_7d"))
    window = _fmt_num(_num(perf.get("window_days")) or 30)
    for metric, label in (("views", "profile views"), ("calls", "calls"), ("directions", "direction requests"),
                          ("leads", "leads")):
        v = _num(perf.get(metric))
        if v is not None:
            ctx.support(f"perf.{metric}", f"{_fmt_num(v)} {label} in the last {window} days", value=v)
    ctr = _num(perf.get("ctr"))
    if ctr is not None:
        ctx.support("perf.ctr", f"CTR {_fmt_ctr(ctr)} in the last {window} days", value=ctr)
    for key, label in (("views_pct", "Profile views"), ("calls_pct", "Calls"), ("ctr_pct", "CTR"),
                       ("directions_pct", "Direction requests"), ("leads_pct", "Leads")):
        v = _num(delta.get(key))
        if v is not None:
            name = "perf.delta_" + key.replace("_pct", "")
            ctx.support(name, f"{label} {_fmt_change(v)} over the last 7 days", value=v)
    for g in _peer_gaps(ctx):
        ctx.support(g["key"], g["text"], value=g["gap"])

    _subscription_support(ctx)
    verified = ctx.identity.get("verified")
    if isinstance(verified, bool):
        ctx.support("merchant.verified", "Google Business Profile verified" if verified
                    else "Google Business Profile not verified yet", value=verified, weight=1 if verified else 2)
    year = _num(ctx.identity.get("established_year"))
    if year:
        ctx.support("merchant.established", f"In business since {int(year)}", value=int(year), weight=1)

    _aggregate_support(ctx)
    _offer_support(ctx)
    _review_support(ctx)
    _history_support(ctx, history)
    _signal_support(ctx)

    if ctx.ref_dt and not ctx.sheet.seasonal_note:
        for b in _l(ctx.cat.get("seasonal_beats")):
            b = _d(b)
            if _s(b.get("note")) and ctx.ref_dt.month in _beat_months(_s(b.get("month_range"))):
                ctx.support("seasonal.note", _beat_text(b), value=b, source="category", weight=1)
                ctx.sheet.seasonal_note = _clean(_beat_text(b))
                break
    trend = _top_trend(ctx)
    if trend:
        ctx.support("trend.top", trend[0], value=trend[1], source="category", weight=1)
        if not ctx.sheet.trend_note:
            ctx.sheet.trend_note = _clean(trend[0])
    peer = ctx.peer
    rating = _num(peer.get("avg_rating"))
    if rating:
        ctx.support("peer.avg_rating", f"Peer average rating {_fmt_num(rating)}★", value=rating, source="category", weight=1)
    reviews = _num(peer.get("avg_review_count") or peer.get("avg_reviews"))
    if reviews:
        ctx.support("peer.avg_review_count", f"Peers average {_fmt_num(reviews)} reviews", value=reviews,
                    source="category", weight=1)
    freq = _num(peer.get("avg_post_freq_days"))
    if freq:
        ctx.support("peer.avg_post_freq_days", f"Peers post on Google every {_fmt_num(freq)} days on average",
                    value=freq, source="category", weight=1)
    photos = _num(peer.get("avg_photos"))
    if photos:
        ctx.support("peer.avg_photos", f"Peers average {_fmt_num(photos)} profile photos", value=photos,
                    source="category", weight=1)


def _subscription_support(ctx: _Ctx) -> None:
    sub = _d(ctx.m.get("subscription"))
    status = _s(sub.get("status")).lower()
    plan = _s(sub.get("plan"))
    left, since = _num(sub.get("days_remaining")), _num(sub.get("days_since_expiry"))
    plan_txt = f"{plan} plan" if plan and plan.lower() != "trial" else "plan"
    if status == "active":
        text = f"{plan_txt} active" + (f", {int(left)} days left" if left is not None else "")
    elif status == "expired":
        text = f"{plan_txt} expired" + (f" {int(since)} days ago" if since else "")
    elif status == "trial":
        text = "On a trial plan" + (f", {int(left)} days left" if left is not None else "")
    elif status:
        text = f"Subscription {humanize(status)}"
    else:
        return
    ctx.support("merchant.subscription", _cap(text), value=sub, weight=2)


def _aggregate_support(ctx: _Ctx) -> None:
    peer = ctx.peer
    for key, val in ctx.agg.items():
        v = _num(val)
        if v is None:
            continue
        key_l = str(key).lower()
        if key_l in _AGG_COUNTS:
            if v <= 0:
                continue
            text = _AGG_COUNTS[key_l].format(n=_fmt_num(v), people=ctx.people)
        elif key_l in _AGG_PCTS or (key_l.endswith("_pct") and abs(v) <= 1.5):
            template = _AGG_PCTS.get(key_l) or (_label(key_l) + ": {p}")
            text = template.format(p=_fmt_pct(v))
            peer_v = _num(peer.get(key_l))
            if peer_v is not None:
                where = "above" if v > peer_v + 0.005 else "below" if v < peer_v - 0.005 else "in line with"
                text += f" vs {_fmt_pct(peer_v)} peer average ({where} peers)" if where != "in line with" \
                    else f" vs {_fmt_pct(peer_v)} peer average (in line)"
        else:
            m = re.fullmatch(r"lapsed_(\d+)d_plus", key_l)
            if m:
                if v <= 0:
                    continue
                text = f"{_fmt_num(v)} {ctx.people} lapsed {m.group(1)}+ days"
            else:
                if v == 0:
                    continue
                text = f"{_label(key_l)}: {_fmt_num(v)}"
        ctx.support(f"agg.{key_l}", _cap(text) if text[:1].isalpha() else text, value=val)


def _review_support(ctx: _Ctx) -> None:
    for rt in _l(ctx.m.get("review_themes")):
        rt = _d(rt)
        theme_raw = _s(rt.get("theme"))
        if not theme_raw:
            continue
        theme = _theme_text(theme_raw)
        occ = _num(rt.get("occurrences_30d"))
        positive = _s(rt.get("sentiment")).lower().startswith("pos")
        verb = "praise" if positive else "mention"
        if occ is not None:
            n = int(occ)
            text = f"{n} review{'s' if n != 1 else ''} in the last 30 days {verb}{'s' if n == 1 else ''} {theme}"
        else:
            text = f"Reviews {verb} {theme}"
        quote = _s(rt.get("common_quote"))
        if quote:
            text += f": '{quote}'"
        ctx.support(f"review.{theme_raw.lower()}", text, value=rt)


def _history_support(ctx: _Ctx, history: list[dict]) -> None:
    last_m = next((h for h in reversed(history) if h["from"] == "merchant"), None)
    last_v = next((h for h in reversed(history) if h["from"] == "vera"), None)
    if last_m:
        when = _parse_local(last_m.get("ts"))
        ctx.support("history.last_merchant",
                    f"Merchant's last message{' (' + _fmt_date(when) + ')' if when else ''}: "
                    f"'{_truncate(last_m['body'])}'", value=last_m, source="merchant")
    if last_v:
        when = _parse_local(last_v.get("ts"))
        eng = last_v.get("engagement", "")
        status = " (no reply)" if eng == "merchant_no_reply" else ""
        ctx.support("history.last_vera",
                    f"Vera's last message{' (' + _fmt_date(when) + ')' if when else ''}{status}: "
                    f"'{_truncate(last_v['body'])}'", value=last_v, source="merchant")
    if last_v and last_v.get("engagement") == "merchant_no_reply" and (
            not last_m or history.index(last_v) > history.index(last_m)):
        ctx.note("The last Vera message got no reply: use a different angle, never repeat it.")
    if last_m and _s(last_m.get("engagement")).startswith("intent"):
        ctx.note(f"The merchant's last message ('{_truncate(last_m['body'], 90)}') showed intent: acknowledge it "
                 f"and move to action.")


def _signal_support(ctx: _Ctx) -> None:
    for raw in _l(ctx.m.get("signals")):
        s = _s(raw).lower()
        if not s:
            continue
        m = re.match(r"^([a-z_]+?)[:_](\d+)d$", s)
        name, n = (m.group(1), int(m.group(2))) if m else (s, None)
        spec = _SIGNALS.get(name)
        if spec is None:
            continue  # unknown signal: skip rather than leak a slug
        kind, with_n, without_n = spec
        text = with_n.format(n=n) if (n is not None and with_n) else without_n
        if kind == "note":
            ctx.note(text)
        else:
            ctx.support(f"signal.{name}", text, value=raw, weight=1 if name.startswith(("above", "growing", "high")) else 2)


_TREND_STOP = {"near", "offer", "offers", "price", "cost", "free", "with", "class", "classes", "trial", "month",
               "first", "your", "best", "program", "service", "services", "care", "clinic", "studio", "salon"}


def _merchant_words(ctx: _Ctx) -> set[str]:
    parts = list(ctx.sheet.offers_active) + [_s(ctx.identity.get("name"))]
    parts += [_s(_d(r).get("theme")).replace("_", " ") for r in _l(ctx.m.get("review_themes"))]
    parts += [h["body"] for h in ctx.sheet.history if h.get("from") == "merchant"]
    words = set()
    for w in re.findall(r"[a-z]{4,}", " ".join(parts).lower()):
        if w not in _TREND_STOP:
            words.add(w)
            words.add(w.rstrip("s"))
    return words


def _top_trend(ctx: _Ctx) -> tuple[str, dict] | None:
    """Most relevant trend signal: overlaps the merchant's own offers/themes/history, then local, then growth."""
    city = (ctx.sheet.city or "").lower()
    mwords = _merchant_words(ctx)
    best = None
    for tr in _l(ctx.cat.get("trend_signals")):
        tr = _d(tr)
        q = _s(tr.get("query"))
        d = _num(tr.get("delta_yoy"))
        if not q or d is None:
            continue
        other_city = any(ci in q.lower() for ci in _KNOWN_CITIES if ci not in city)
        if other_city and city not in q.lower():
            continue
        local = bool(city) and city in q.lower()
        qwords = {w for w in re.findall(r"[a-z]{4,}", q.lower()) if w not in _TREND_STOP}
        overlap = len({w for w in qwords if w in mwords or w.rstrip("s") in mwords})
        score = (overlap, local, d)
        if best is None or score > best[0]:
            best = (score, tr)
    if best is None:
        return None
    tr = best[1]
    text = f"'{_s(tr.get('query'))}' searches {_fmt_change(_num(tr.get('delta_yoy')))} YoY"
    seg = _segment_text(tr.get("segment_age"))
    return (f"{text} ({seg})" if seg else text), tr


def _segment_text(raw: Any) -> str:
    """'28-45' -> 'age 28-45'; 'office_25-45' -> 'office-goers aged 25-45'; 'all' -> ''."""
    seg = _s(raw).lower()
    if not seg or seg == "all":
        return ""
    if re.fullmatch(r"\d+\s*-\s*\d+", seg):
        return f"age {seg}"
    m = re.fullmatch(r"([a-z]+)[_\s]+(\d+\s*-\s*\d+)", seg)
    if m:
        who = {"office": "office-goers"}.get(m.group(1), m.group(1))
        return f"{who} aged {m.group(2)}"
    return humanize(seg)


def _customer_support(ctx: _Ctx) -> None:
    c = ctx.c or {}
    rel, prefs, ident = _d(c.get("relationship")), _d(c.get("preferences")), _d(c.get("identity"))
    ref = None if ctx.stale_counts else ctx.ref_dt
    _offer_support(ctx)
    last, first = _parse_local(rel.get("last_visit")), _parse_local(rel.get("first_visit"))
    if last:
        ctx.support("customer.last_visit", f"Last visit on {_fmt_date(last)}{_ago(last, ref)}",
                    value=rel.get("last_visit"), source="customer")
    visits = _num(rel.get("visits_total"))
    if visits:
        ctx.support("customer.visits", f"{int(visits)} visit{'s' if visits != 1 else ''}"
                    + (f" since {_fmt_date(first)}" if first else " so far"), value=int(visits), source="customer")
    services: dict[str, int] = {}
    for svc in _l(rel.get("services_received")):
        name = humanize(svc)
        if not name or set(name) <= {"."}:
            continue
        services[name] = services.get(name, 0) + 1
    if services:
        parts = [f"{n} ({k} times)" if k > 1 else n for n, k in list(services.items())[:5]]
        ctx.support("customer.services", f"Past services: {', '.join(parts)}", value=list(services), source="customer")
    state = _s(c.get("state")).lower()
    state_text = {"new": "New customer", "active": "Active customer", "lapsed_soft": "Recently lapsed customer",
                  "lapsed_hard": "Lapsed for a while", "churned": "Has not returned in a long time"}.get(state)
    if state_text:
        ctx.support("customer.state", state_text, value=state, source="customer", weight=1)
    if _s(prefs.get("preferred_slots")):
        ctx.support("customer.preferred_slots", f"Prefers {_slot_pref_text(_s(prefs['preferred_slots']))}",
                    value=prefs["preferred_slots"], source="customer")
    for key, label in (("training_focus", "Training focus"), ("health_focus", "Health focus")):
        if _s(prefs.get(key)):
            ctx.support("customer.focus", f"{label}: {humanize(prefs[key])}", value=prefs[key], source="customer")
    if _s(rel.get("favourite_dish")):
        ctx.support("customer.favourite", f"Favourite dish: {_s(rel['favourite_dish'])}", source="customer")
    if _s(prefs.get("preferred_stylist")):
        ctx.support("customer.stylist", f"Preferred stylist: {_s(prefs['preferred_stylist'])}", source="customer")
    wedding = _parse_local(prefs.get("wedding_date"))
    if wedding:
        ctx.support("customer.wedding_date", f"Wedding on {_fmt_date(wedding)}", value=prefs.get("wedding_date"),
                    source="customer")
    band = _s(ident.get("age_band"))
    if band and band.lower() != "unknown":
        low = band.lower()
        if low.startswith("child"):
            rest = humanize(low.replace("child", "").strip("_ -"))
            text = f"Child ({'aged ' if re.match(r'\d', rest) else ''}{rest})" if rest else "Child"
        else:
            text = f"Age {band}"
        ctx.support("customer.age_band", text, value=band, source="customer", weight=1)
    if ident.get("senior_citizen") is True:
        ctx.support("customer.senior", "Senior citizen", source="customer", weight=1)
    if _s(prefs.get("delivery_address")).lower() == "saved" and not ctx.has("trigger.delivery_address"):
        ctx.support("customer.delivery", "Delivery address saved", source="customer", weight=1)
    size = _num(prefs.get("family_size") or prefs.get("household_size"))
    if size:
        ctx.support("customer.household", f"Household of {int(size)}", value=int(size), source="customer", weight=1)
    if ctx.sheet.about_name and ctx.sheet.about_name != ctx.sheet.recipient_name:
        ctx.support("customer.about", f"Message is about {ctx.sheet.about_name}", source="customer", weight=1)
    if ref:
        for b in _l(ctx.cat.get("seasonal_beats")):
            b = _d(b)
            if _s(b.get("note")) and ref.month in _beat_months(_s(b.get("month_range"))):
                ctx.sheet.seasonal_note = ctx.sheet.seasonal_note or _clean(_beat_text(b))
                break


# --------------------------------------------------------------------------- consent / content / anchors


def _section_consent(ctx: _Ctx) -> None:
    sheet = ctx.sheet
    if not ctx.customer_facing:
        sheet.consent_ok, sheet.consent_reason = True, ""
        return
    c = ctx.c
    reasons: list[str] = []
    if c is None:
        reasons.append("customer context missing")
    else:
        ident, prefs, consent = _d(c.get("identity")), _d(c.get("preferences")), _d(c.get("consent"))
        name = _s(ident.get("name"))
        if not name or _ANON_RE.search(name):
            reasons.append("walk-in with no customer profile")
        if "phone_redacted" in ident and ident.get("phone_redacted") in (None, ""):
            reasons.append("no phone number on file")
        if not consent.get("opted_in_at"):
            reasons.append("no opt-in recorded")
        if consent.get("opted_out_at") or consent.get("revoked_at"):
            reasons.append("consent revoked")
        raw_scope = consent.get("scope")
        scopes = {str(x).lower() for x in _l(raw_scope)} or ({_s(raw_scope).lower()} if _s(raw_scope) else set())
        if not scopes:
            reasons.append("empty consent scope")
        if _s(c.get("state")).lower() in OPTED_OUT_STATES:
            reasons.append("customer opted out")
        channel = _s(prefs.get("channel")).lower()
        if channel.startswith("none") or channel in ("no_channel", "unknown"):
            reasons.append("no contact channel recorded")
        if ctx.canon in REMINDER_KINDS and prefs.get("reminder_opt_in") is False and not (scopes & REMINDER_SCOPES):
            reasons.append("customer opted out of reminders")
    sheet.consent_ok = not reasons
    if reasons:
        sheet.consent_reason = "; ".join(reasons)
        ctx.note("Consent is not usable for this customer: do not send a customer message.")
    else:
        consent = _d(c.get("consent"))
        when = _parse_local(consent.get("opted_in_at"))
        scope = _join([humanize(s) for s in _l(consent.get("scope"))])
        sheet.consent_reason = f"opted in{' on ' + _fmt_date(when) if when else ''} for {scope}"


def _section_content_item(ctx: _Ctx) -> None:
    library = [_d(x) for x in _l(ctx.cat.get("patient_content_library")) if _s(_d(x).get("title"))]
    if not library:
        return
    stop = {"your", "what", "with", "that", "this", "from", "have", "they", "their", "about", "which", "when",
            "into", "more", "most", "than", "only", "also", "been", "will", "each", "every", "there"}
    parts = [ctx.canon.replace("_", " ")]
    item = ctx.sheet.digest_item or {}
    parts += [_s(item.get("title")), _s(item.get("summary"))]
    c = ctx.c or {}
    parts += [humanize(x) for x in _l(_d(c.get("relationship")).get("services_received"))]
    parts += [humanize(_d(c.get("identity")).get("age_band")), humanize(ctx.p.get("service_due"))]
    parts += [humanize(x) for x in _l(ctx.p.get("trends"))]
    words = {w for w in re.findall(r"[a-z]{4,}", " ".join(parts).lower()) if w not in stop}
    best = None
    for entry in library:
        text = f"{entry.get('title', '')} {entry.get('body', '')}".lower()
        cwords = {w for w in re.findall(r"[a-z]{4,}", text) if w not in stop}
        score = len(words & cwords)
        if score >= 2 and (best is None or score > best[0]):
            best = (score, entry)
    if best:
        ctx.sheet.content_item = dict(best[1])


def _ensure_anchor(ctx: _Ctx) -> None:
    """Every sheet gets at least one why-now anchor built from real data."""
    if ctx.sheet.anchor_facts:
        return
    if not ctx.customer_facing:
        _derive_metric_delta(ctx, want_positive=True)
        if not ctx.sheet.anchor_facts:
            _derive_metric_delta(ctx, want_positive=False)
    if ctx.sheet.anchor_facts:
        return
    for prefix in ("customer.last_visit", "perf.views", "trend.top", "seasonal.note", "offer.active"):
        for f in ctx.sheet.support_facts:
            if f.key.startswith(prefix):      # promote the support fact itself (it is already de-duplicated)
                ctx.sheet.support_facts.remove(f)
                f.weight = 3
                ctx.sheet.anchor_facts.append(f)
                return
    trend = _top_trend(ctx)
    if trend:
        ctx.anchor("trend.top", trend[0], value=trend[1], source="category")
        return
    # A real (novel) kind name is still a usable why-now; placeholder kinds are not.
    if ctx.canon and ctx.canon.lower() not in {"unknown", "generic", "none", "null", "trigger"}:
        ctx.anchor("trigger.kind", _cap(humanize(ctx.canon)), value=ctx.canon, source="trigger")


def _general_notes(ctx: _Ctx, history: list[dict]) -> None:
    if ctx.customer_facing:
        return
    if not history and not any("3-word intro" in n for n in ctx.sheet.notes):
        ctx.note("First conversation with this merchant: at most a 3-word intro, then straight to the point.")
    if not ctx.sheet.offers_active:
        idea = next((o for o in ctx.sheet.catalog_offers if "@" in o), ctx.sheet.catalog_offers[0]
                    if ctx.sheet.catalog_offers else "")
        if idea:
            ctx.note(f"No active offers: you may suggest a catalog idea such as '{idea}', clearly as a suggestion.")
    if ctx.placeholder:
        ctx.note("The trigger carries no specific data: build the message only from the facts listed; "
                 "invent nothing.")


def _section_allowed_numbers(ctx: _Ctx, conversation_turns: Any) -> None:
    sheet = ctx.sheet
    nums = collect_numbers(ctx.cat, ctx.m, ctx.t, ctx.c)
    nums |= ctx.extra_numbers
    texts: list[Any] = [f.text for f in sheet.all_facts()]
    texts += list(sheet.notes) + list(sheet.slots) + list(sheet.offers_active) + list(sheet.catalog_offers)
    texts += [sheet.seasonal_note, sheet.trend_note, sheet.salutation, sheet.signer, sheet.merchant_name,
              sheet.owner_name, sheet.locality]
    nums |= collect_numbers(texts, sheet.digest_item, sheet.content_item, conversation_turns)
    sheet.allowed_numbers = nums
