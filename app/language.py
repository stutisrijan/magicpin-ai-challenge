"""Language planning: which language and register a message should use.

    merchant_language_plan()  merchant-facing default (mirror their latest message, else identity.languages)
    customer_language_plan()  customer-facing default (from identity.language_pref)
    detect_message_language() classify one inbound message
    plan_for_reply()          mirror the language of an inbound reply, per turn

Everything stays romanised for WhatsApp except when the other side writes in
Devanagari. Regional languages (Tamil, Telugu, Kannada, Marathi) get English plus a
warm regional greeting: we never attempt long regional-language sentences.
"""

from __future__ import annotations

import re
from typing import Any

from app.schemas import LanguagePlan

REGIONAL_GREETINGS = {"ta": "Vanakkam", "te": "Namaskaram", "kn": "Namaskara", "mr": "Namaskar"}
_REGIONAL_NAMES = {"ta": "Tamil", "te": "Telugu", "kn": "Kannada", "mr": "Marathi"}
_LANGUAGE_ALIASES = {
    "english": "en", "hindi": "hi", "tamil": "ta", "telugu": "te", "kannada": "kn", "marathi": "mr",
}

_INSTR_EN = "Write in clear, simple English."
_INSTR_HINGLISH = (
    "Write natural Hinglish: Roman-script Hindi-English code-mix, the way Indian business owners text "
    "(e.g. 'aapke liye', 'bas YES reply karein', 'chalega?'). Keep facts, numbers, names, offer titles "
    "and technical terms in English. No Devanagari."
)
_INSTR_HI_ROMAN = (
    "Write in simple Roman-script Hindi (Hindi-dominant, respectful 'aap' form). Keep medicine, "
    "service and product names, offer titles and all numbers in English/digits. No Devanagari."
)
_INSTR_HI_DEVANAGARI = (
    "Reply in simple Hindi in Devanagari script (respectful 'aap' form). Keep numbers as digits and "
    "service, medicine and product names in English."
)

# Unicode blocks for script detection. Marathi shares Devanagari with Hindi.
_SCRIPT_RANGES = {
    "deva": (0x0900, 0x097F),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF),
}
_MARATHI_MARKERS = {
    "आहे", "आहेत", "नाही", "काय", "मला", "तुम्ही", "आम्ही", "होय", "करा", "झाले", "नको",
    "पाहिजे", "कसे", "तुमचा", "तुमची", "माझा", "माझी", "आणि", "पण", "केले",
}
_HINDI_MARKERS = {
    "है", "हैं", "नहीं", "क्या", "मुझे", "आप", "करो", "करें", "हूँ", "हूं", "था", "की", "का",
    "के", "में", "और", "लेकिन", "चाहिए", "ठीक",
}

# Romanised Hindi words that rarely occur in English text. Deliberately excludes
# ambiguous tokens ("me", "to", "main", "do", "par", "band", names like "diya").
_HINGLISH_WORDS = {
    "haan", "haa", "haanji", "nahi", "nahin", "nhi", "nai", "mat", "karo", "karna", "karein",
    "kar", "kardo", "karte", "karta", "karti", "karenge", "karunga", "karungi",
    "kya", "kyu", "kyun", "kyon", "hai", "hain", "hoon", "hun", "hu", "ho", "tha", "thi", "mujhe",
    "muje", "humein", "hame", "hamein", "aap", "aapka", "aapki", "aapke", "apka", "apki", "apke",
    "chahiye", "chahie", "theek", "thik", "thek", "bhej", "bhejo", "bhejna", "bhejiye", "bhejdo",
    "abhi", "kal", "baad", "mein", "bhai", "ji", "accha", "acha", "achha", "achcha", "dekho",
    "dekh", "dekhte", "batao", "bata", "bataiye", "batayein", "kitna", "kitne", "kitni", "kaise",
    "kab", "kaun", "kahan", "kidhar", "wala", "wali", "wale", "sab", "bahut", "bohot", "bahot",
    "zaroor", "jaroor", "jaldi", "samajh", "samjha", "sahi", "chalega", "chalo", "bilkul", "lekin",
    "aur", "hum", "hamara", "hamari", "hamare", "mera", "meri", "mere", "tum", "tumhara", "yeh",
    "ye", "woh", "wo", "raha", "rahi", "rahe", "hoga", "hogi", "liya", "karke", "dijiye", "kijiye",
    "shukriya", "dhanyavad", "dhanyawad", "namaste", "matlab", "pehle", "phir", "thoda", "zyada",
    "jyada", "paisa", "paise", "kaam", "dukaan", "dukan", "grahak", "agar", "isko",
    "usko", "iska", "uska", "apna", "apni", "kuch", "koi", "sirf", "ke", "ki", "ka", "liye", "tak",
    "deti", "deta", "sabhi", "baat", "bolo", "boliye", "suniye", "jaankari", "jankari", "madad",
    "judna", "judrna", "jodna", "chahta", "chahti", "sakte", "sakta", "sakti", "milega", "milegi",
}
_WORD_RE = re.compile(r"[a-z]+")


# --------------------------------------------------------------------------- plans


def plan_for_code(code: str) -> LanguagePlan:
    """Canonical LanguagePlan for a code: en, hinglish, hi, hi-deva, ta-en, te-en, kn-en, mr-en."""
    code = (code or "en").strip().lower()
    if code == "hinglish":
        return LanguagePlan(code="hinglish", instruction=_INSTR_HINGLISH, greeting="Hi", script="latin")
    if code == "hi":
        return LanguagePlan(code="hi", instruction=_INSTR_HI_ROMAN, greeting="Namaste", script="latin")
    if code in ("hi-deva", "devanagari"):
        return LanguagePlan(code="hi", instruction=_INSTR_HI_DEVANAGARI, greeting="Namaste", script="devanagari")
    base = code[:2]
    if base in REGIONAL_GREETINGS:
        return _regional_plan(base)
    return LanguagePlan(code="en", instruction=_INSTR_EN, greeting="Hi", script="latin")


def _regional_plan(code2: str) -> LanguagePlan:
    greet = REGIONAL_GREETINGS[code2]
    name = _REGIONAL_NAMES[code2]
    instruction = (
        f"Write in clear, simple English. Open with the {name} greeting '{greet}'; at most one short, "
        f"common {name} word elsewhere. Never write full sentences in {name}."
    )
    return LanguagePlan(code=f"{code2}-en", instruction=instruction, greeting=greet, script="latin")


def merchant_language_plan(merchant: dict, history: list[dict] | None = None) -> LanguagePlan:
    """Merchant-facing plan.

    1. Mirror the language of the merchant's latest message (history, else merchant.conversation_history).
    2. Else 'hi' in identity.languages -> Hinglish.
    3. Else the first non-English language, if Tamil/Telugu/Kannada/Marathi -> English + regional greeting.
    4. Else English.
    """
    merchant = merchant if isinstance(merchant, dict) else {}
    turns = history if history is not None else merchant.get("conversation_history")
    latest = _latest_inbound_text(turns)
    if latest:
        lang = detect_message_language(latest)
        if lang != "en" or len(_WORD_RE.findall(latest.lower())) >= 3:
            return _plan_for_detected(lang)
        # a bare "ok" / "yes" says nothing about language: fall through to identity defaults

    identity = merchant.get("identity") if isinstance(merchant.get("identity"), dict) else {}
    langs = [_norm_lang(x) for x in (identity.get("languages") or []) if isinstance(x, str)]
    if "hi" in langs:
        return plan_for_code("hinglish")
    for lang in langs:
        if lang == "en":
            continue
        if lang in REGIONAL_GREETINGS:
            return _regional_plan(lang)
        break
    return plan_for_code("en")


def customer_language_plan(customer: dict) -> LanguagePlan:
    """Customer-facing plan from identity.language_pref ("hi-en mix", "hi", "english", "ta-en mix", ...)."""
    customer = customer if isinstance(customer, dict) else {}
    identity = customer.get("identity") if isinstance(customer.get("identity"), dict) else {}
    pref = str(identity.get("language_pref") or customer.get("language_pref") or "").strip().lower()
    if not pref:
        return plan_for_code("en")
    has_hi = re.search(r"\b(hi|hindi)\b", pref) is not None
    mixed = "mix" in pref or re.search(r"\b(en|english)\b", pref) is not None or "hinglish" in pref
    if "hinglish" in pref or (has_hi and mixed):
        return plan_for_code("hinglish")
    for code, name in _REGIONAL_NAMES.items():
        if re.search(rf"\b({code}|{name.lower()})\b", pref):
            return _regional_plan(code)
    if has_hi:
        return plan_for_code("hi")
    return plan_for_code("en")


def plan_for_reply(base: LanguagePlan, inbound_text: str) -> LanguagePlan:
    """Mirror the language of an inbound message; keep `base` when the message is too short to tell."""
    base = base if isinstance(base, LanguagePlan) else plan_for_code("en")
    text = inbound_text if isinstance(inbound_text, str) else ""
    if not text.strip():
        return base
    lang = detect_message_language(text)
    if lang == "en":
        if len(_WORD_RE.findall(text.lower())) < 3:
            return base                      # "ok", "yes", "1": no language signal
        if base.code == "en" or (base.code.endswith("-en") and base.code[:2] in REGIONAL_GREETINGS):
            return base                      # already English (regional greeting kept)
        return plan_for_code("en")
    if lang == "hinglish" and base.code == "hi" and base.script == "latin":
        return base                          # Roman Hindi customer writing Roman Hindi: keep their register
    return _plan_for_detected(lang)


# --------------------------------------------------------------------------- detection


def detect_message_language(text: str) -> str:
    """One of "en", "hinglish", "hi", "ta", "te", "kn", "mr"."""
    if not isinstance(text, str) or not text.strip():
        return "en"
    counts = {k: 0 for k in _SCRIPT_RANGES}
    latin = 0
    for ch in text:
        if ch.isascii():
            latin += ch.isalpha()
            continue
        cp = ord(ch)
        for script, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[script] += 1
                break
    script, n = max(counts.items(), key=lambda kv: kv[1])
    if n and n >= 0.25 * (n + latin):
        if script == "deva":
            return "mr" if _is_marathi(text) else "hi"
        return script

    words = _WORD_RE.findall(text.lower())
    if not words:
        return "en"
    hits = sum(1 for w in words if w in _HINGLISH_WORDS)
    if hits and (hits / len(words) >= 0.25 or hits >= 3):
        return "hinglish"
    return "en"


def _is_marathi(text: str) -> bool:
    tokens = re.findall(r"[ऀ-ॿ]+", text)
    mr = sum(1 for t in tokens if t in _MARATHI_MARKERS)
    hi = sum(1 for t in tokens if t in _HINDI_MARKERS)
    return mr > hi


def _plan_for_detected(lang: str) -> LanguagePlan:
    if lang == "hinglish":
        return plan_for_code("hinglish")
    if lang == "hi":
        return plan_for_code("hi-deva")
    if lang in REGIONAL_GREETINGS:
        return _regional_plan(lang)
    return plan_for_code("en")


def _norm_lang(value: str) -> str:
    v = value.strip().lower()
    return _LANGUAGE_ALIASES.get(v, v[:2] if len(v) > 2 and v[2] in "-_" else v)


def _latest_inbound_text(turns: Any) -> str:
    """Body of the most recent merchant-authored turn (supports {from|role|from_role, body|message|text})."""
    if not isinstance(turns, (list, tuple)):
        return ""
    for turn in reversed(turns):
        if isinstance(turn, dict):
            role = turn.get("from") or turn.get("role") or turn.get("from_role")
            body = turn.get("body") or turn.get("message") or turn.get("text")
        else:
            role = getattr(turn, "role", None)
            body = getattr(turn, "body", None)
        if str(role or "").lower() in ("merchant", "owner") and isinstance(body, str) and body.strip():
            return body
    return ""
