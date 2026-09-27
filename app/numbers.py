"""Shared numeric-token helpers.

facts.py uses these to build FactSheet.allowed_numbers; validator.py uses the same
functions to check that every number in a message is grounded in the contexts.
One implementation keeps both sides in agreement.
"""

from __future__ import annotations

import re

# ₹1,499 | Rs. 499 | 2,100 | 38% | 2.1 | -0.5 | 1.5x | 3.0
_NUM_RE = re.compile(
    r"(?<![A-Za-z0-9_])"              # not glued to a word on the left (skips "W17", "d_2026")
    r"(?<![A-Za-z0-9]-)"              # nor the tail of an id like "AT2024-1102"
    r"(?:₹|rs\.?\s?|inr\s?)?"          # optional currency
    r"[-+−]?\d[\d,]*(?:\.\d+)?"       # the number (commas allowed)
    r"(?:\s?%|x(?![a-z]))?",          # optional percent or multiplier
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b", re.IGNORECASE)


def normalize_number(tok: str) -> str | None:
    """Canonical form of a numeric token: no currency, commas, sign, %, x; no trailing zeros.

    "₹1,499" -> "1499", "38%" -> "38", "2.10" -> "2.1", "-0.5" -> "0.5", "3.0" -> "3".
    Returns None when no digits are present.
    """
    if tok is None:
        return None
    s = str(tok).strip().lower()
    s = s.replace("₹", "").replace("inr", "").replace("rs.", "").replace("rs", "")
    s = s.replace(",", "").replace("%", "").replace("−", "-").strip()
    if s.endswith("x"):
        s = s[:-1]
    s = s.lstrip("+-").strip()
    if not s or not any(ch.isdigit() for ch in s):
        return None
    try:
        val = float(s)
    except ValueError:
        return None
    if val == int(val):
        return str(int(val))
    return f"{val:.4f}".rstrip("0").rstrip(".")


def extract_number_tokens(text: str) -> list[str]:
    """Raw numeric tokens as they appear in text (clock times like '6pm' are excluded)."""
    if not text:
        return []
    masked = _CLOCK_RE.sub(" ", text)
    return [m.group(0).strip() for m in _NUM_RE.finditer(masked)]


def clock_times(text: str) -> list[str]:
    return [m.group(0) for m in _CLOCK_RE.finditer(text or "")]


def number_variants(value: float | int | str) -> set[str]:
    """All normalised forms under which a raw data value may legitimately appear in copy.

    Fractions in (-1, 1) are also expressed as percentages (0.021 -> 2.1; -0.5 -> 50),
    and every value appears rounded to 0 and 1 decimals.
    """
    out: set[str] = set()
    try:
        v = float(str(value).replace(",", "").replace("₹", "").replace("%", ""))
    except (TypeError, ValueError):
        return out
    cands = [v]
    if -1 < v < 1 and v != 0:
        cands.append(v * 100)
    for c in cands:
        for form in (c, round(c, 1), round(c), abs(c), round(abs(c), 1), round(abs(c))):
            n = normalize_number(str(form))
            if n is not None:
                out.add(n)
    return out
