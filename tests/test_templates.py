"""Tests for app/templates.py: the deterministic writer that must score well on its own.

Every seed trigger and every trigger of the expanded dataset (regenerated from the seeds
into a temp dir) is rendered under three "now" values and checked against the message
rules: salutation up front, exactly one ask as the last sentence, no URL / taboo /
underscore / "Dr. Dr.", every number grounded in the facts, customer copy never naming
Vera or magicpin, exact slot labels, prices only from active offers, determinism and
language handling. Targeted cases cover relays, placeholders and judgment calls.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from app.facts import build_facts
from app.numbers import clock_times, extract_number_tokens, normalize_number
from app.playbooks import get_playbook
from app.schemas import (
    CTA_BINARY_CONFIRM,
    CTA_BINARY_YES_NO,
    CTA_MULTI_CHOICE_SLOT,
    CTA_OPEN_ENDED,
    CTA_VALUES,
    SEND_AS_MERCHANT,
    Draft,
    FactSheet,
    Playbook,
)
from app.templates import render

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
NOWS = [None, "2026-04-26T10:30:00Z", "2026-09-26T10:30:00Z"]

HINGLISH_MARKERS = re.compile(r"\b(?:aapke|aapka|aapki|hai|hain|kar doon|karein|mein|ke liye|bhej doon|abhi)\b",
                              re.IGNORECASE)
_URL = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|in|org|net|io|co|ly|app|me)\b", re.IGNORECASE)
_DIRECTIVE = re.compile(
    r"\b(?:reply|type|text|send|say|message|likhiye|likhein|likho)\s+(?:with\s+|us\s+)?[\"'*]*"
    r"(?:yes|haan|ok|okay|go|confirm|y|\d)(?![\w'])"
    r"|(?<![\w'])(?:yes|haan|ok|go|confirm|\d)\s+(?:reply|likh|bhej|type)\w*",
    re.IGNORECASE,
)
_TIME_UNIT = re.compile(r"^\s?-?\s?(?:mins?|minutes?|hrs?|hours?|h|ghante|ghanta|secs?|seconds?)\b", re.IGNORECASE)
_MASKS = [re.compile(r"\b24\s?[x×/]\s?7\b", re.I),
          re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:-|–|—|to)\s?\d{1,2}(?::\d{2})?\s?(?:am|pm)\b", re.I),
          re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b")]


# --------------------------------------------------------------------------- data fixtures


def _load_dir(path: Path, key: str) -> dict[str, dict]:
    return {obj[key]: obj for obj in (json.loads(f.read_text()) for f in sorted(path.glob("*.json")))}


@pytest.fixture(scope="session")
def seeds() -> dict:
    return {
        "categories": _load_dir(DATASET / "categories", "slug"),
        "merchants": {m["merchant_id"]: m for m in json.loads((DATASET / "merchants_seed.json").read_text())["merchants"]},
        "customers": {c["customer_id"]: c for c in json.loads((DATASET / "customers_seed.json").read_text())["customers"]},
        "triggers": {t["id"]: t for t in json.loads((DATASET / "triggers_seed.json").read_text())["triggers"]},
    }


@pytest.fixture(scope="session")
def expanded(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("expanded_templates")
    subprocess.run([sys.executable, str(DATASET / "generate_dataset.py"), "--seed-dir", str(DATASET), "--out", str(out)],
                   check=True, capture_output=True, cwd=str(ROOT))
    data = {
        "categories": _load_dir(out / "categories", "slug"),
        "merchants": _load_dir(out / "merchants", "merchant_id"),
        "customers": _load_dir(out / "customers", "customer_id"),
        "triggers": _load_dir(out / "triggers", "id"),
        "pairs": json.loads((out / "test_pairs.json").read_text())["pairs"],
    }
    assert len(data["triggers"]) == 100
    return data


def compose(data: dict, trigger: dict, customer_id: str | None = None, now: str | None = None,
            merchant: dict | None = None) -> tuple[FactSheet, Playbook, Draft]:
    m = merchant or data["merchants"][trigger.get("merchant_id") or trigger["payload"]["merchant_id"]]
    cid = customer_id or trigger.get("customer_id")
    c = data["customers"].get(cid) if cid else None
    facts = build_facts(data["categories"].get(m.get("category_slug")), m, trigger, c, now=now)
    pb = get_playbook(facts.kind, facts.scope, facts.category_slug)
    return facts, pb, render(facts, pb)


def by_id(data: dict, tid: str, **kw) -> tuple[FactSheet, Playbook, Draft]:
    return compose(data, data["triggers"][tid], **kw)


# --------------------------------------------------------------------------- the rule checks


def ungrounded(body: str, allowed: set[str]) -> list[str]:
    """Spec section 10: a token must be in allowed_numbers unless it is a small plain count or a common duration."""
    text = body
    for rx in _MASKS:
        text = rx.sub(lambda m: " " * len(m.group(0)), text)
    for clock in clock_times(text):
        text = text.replace(clock, " " * len(clock))
    bad, pos = [], 0
    for tok in extract_number_tokens(text):
        m = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(tok)).search(text, pos)
        after = text[m.end(): m.end() + 12] if m else ""
        pos = m.end() if m else pos
        norm = normalize_number(tok)
        if norm is None or norm in allowed:
            continue
        low = tok.lower()
        marked = "%" in low or "₹" in low or low.startswith(("rs", "inr")) or low.rstrip().endswith("x")
        if not marked and norm.isdigit() and int(norm) <= 10:
            continue
        if not marked and norm in {"15", "20", "24", "30", "45", "48", "60", "90"} and _TIME_UNIT.match(after):
            continue
        bad.append(tok)
    return bad


def check_draft(facts: FactSheet, pb: Playbook, draft: Draft, label: str) -> None:
    body = draft.body
    customer = facts.send_as == SEND_AS_MERCHANT
    assert isinstance(draft, Draft) and body and body == body.strip(), label
    assert draft.source == "template"
    # length (spec rule 11, with headroom for the drafted-artifact planning kind)
    limit = 800 if facts.kind == "active_planning_intent" else (460 if customer else 560)
    assert 60 <= len(body) <= limit, f"{label}: {len(body)} chars"
    # salutation / addressee up front
    if facts.recipient_name:
        name = facts.salutation or facts.recipient_name
        assert name in body[:60], f"{label}: '{name}' not in the first 60 chars: {body[:80]}"
    elif facts.about_name:
        assert facts.about_name in body[:120], f"{label}: relay name missing: {body[:120]}"
    elif facts.salutation and not customer:
        assert facts.salutation in body[:60], label
    # exactly one ask, and it is the last sentence
    params = draft.template_params
    assert isinstance(params, list) and len(params) == 3 and all(isinstance(p, str) for p in params), label
    ask = params[2]
    assert ask and body.endswith(ask), f"{label}: CTA param is not the last sentence"
    rest = body[: len(body) - len(ask)]
    assert "?" not in rest, f"{label}: question before the CTA: {rest}"
    assert not _DIRECTIVE.search(rest), f"{label}: reply directive before the CTA"
    assert ask.endswith("?") or _DIRECTIVE.search(ask), f"{label}: last sentence is not an ask: {ask}"
    assert len(_DIRECTIVE.findall(ask)) <= 1 and ask.count("?") <= 1, f"{label}: several asks: {ask}"
    assert draft.cta in CTA_VALUES
    if draft.cta == CTA_OPEN_ENDED:
        assert ask.endswith("?")
    # hygiene
    assert not _URL.search(body), f"{label}: URL"
    assert "_" not in body and "_" not in draft.rationale and "_" not in draft.offer, label
    assert not re.search(r"\bDr\.?\s*Dr\b", body), f"{label}: Dr. Dr."
    assert not re.search(r"\b(?:payload|placeholder|trigger|null|None|undefined)\b|\{\{", body), label
    low = body.lower()
    for taboo in facts.taboos:
        word = taboo.split("(")[0].strip().lower()
        assert not re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", low), f"{label}: taboo '{word}'"
    bad = ungrounded(body, facts.allowed_numbers)
    assert not bad, f"{label}: ungrounded numbers {bad} in: {body}"
    # customer-facing: speak as the business, exact slots, prices only from active offers
    if customer:
        assert not re.search(r"\b(?:vera|magic\s?pin)\b", low), f"{label}: brand mention"
        offer_nums = {normalize_number(t) for o in facts.offers_active for t in extract_number_tokens(o)}
        for price in re.findall(r"₹\s?[\d,]+", body):
            assert normalize_number(price) in offer_nums, f"{label}: price {price} not from an active offer"
        for slot in facts.slots[:3]:
            assert slot in body, f"{label}: slot '{slot}' missing"
        if len(facts.slots) >= 2:
            assert draft.cta == CTA_MULTI_CHOICE_SLOT
    else:
        assert pb.template_name.startswith("vera_")
    # rationale and offer
    assert draft.rationale.startswith("Why now:") and len(draft.rationale) < 600
    assert draft.offer
    # determinism
    again = render(facts, pb)
    assert again == draft, f"{label}: not deterministic"


# --------------------------------------------------------------------------- every trigger


@pytest.mark.parametrize("now", NOWS)
def test_every_seed_trigger(seeds, now):
    for tid, trigger in seeds["triggers"].items():
        facts, pb, draft = compose(seeds, trigger, now=now)
        check_draft(facts, pb, draft, f"{tid}@{now}")


@pytest.mark.parametrize("now", NOWS)
def test_every_expanded_trigger(expanded, now):
    for tid, trigger in expanded["triggers"].items():
        facts, pb, draft = compose(expanded, trigger, now=now)
        check_draft(facts, pb, draft, f"{tid}@{now}")


def test_official_test_pairs(expanded):
    assert len(expanded["pairs"]) == 30
    for pair in expanded["pairs"]:
        facts, pb, draft = compose(expanded, expanded["triggers"][pair["trigger_id"]],
                                   customer_id=pair.get("customer_id"))
        check_draft(facts, pb, draft, pair["test_id"])
        assert facts.merchant_id == pair["merchant_id"]


def test_every_seed_customer_with_every_customer_kind(seeds):
    slots = [{"iso": "2026-05-02T11:00:00+05:30", "label": "Sat 2 May, 11am"},
             {"iso": "2026-05-03T17:00:00+05:30", "label": "Sun 3 May, 5pm"}]
    kinds = {"recall_due": {"available_slots": slots}, "customer_lapsed_soft": {}, "customer_lapsed_hard": {},
             "appointment_tomorrow": {}, "chronic_refill_due": {}, "trial_followup": {},
             "wedding_package_followup": {}, "unplanned_slot_open": {"open_slots": slots[:1]}}
    for cid, customer in seeds["customers"].items():
        for kind, payload in kinds.items():
            trigger = {"id": f"t_{kind}_{cid}", "scope": "customer", "kind": kind, "merchant_id": customer["merchant_id"],
                       "customer_id": cid, "payload": payload, "urgency": 3}
            for now in (None, "2026-04-26T10:30:00Z"):
                facts, pb, draft = compose(seeds, trigger, now=now)
                check_draft(facts, pb, draft, f"{cid}:{kind}@{now}")


# --------------------------------------------------------------------------- language


def test_hinglish_merchant_gets_hinglish(seeds):
    m = copy.deepcopy(seeds["merchants"]["m_001_drmeera_dentist_delhi"])
    m["conversation_history"] = []          # no English reply to mirror: identity languages (hi) win
    facts, _, draft = compose(seeds, seeds["triggers"]["trg_001_research_digest_dentists"], merchant=m)
    assert facts.language.code == "hinglish"
    assert len(HINGLISH_MARKERS.findall(draft.body)) >= 3, draft.body
    assert "JIDA Oct 2026, p.14" in draft.body and "38%" in draft.body      # facts stay in English


def test_hinglish_seed_merchants(seeds):
    for tid in ("trg_004_perf_dip_bharat", "trg_006_festival_diwali", "trg_010_ipl_match_delhi",
                "trg_021_unverified_gbp_sunrise"):
        facts, _, draft = by_id(seeds, tid)
        assert facts.language.code == "hinglish", tid
        assert len(HINGLISH_MARKERS.findall(draft.body)) >= 3, draft.body


def test_english_merchant_and_customer_stay_english(seeds):
    facts, _, draft = by_id(seeds, "trg_001_research_digest_dentists")    # Dr. Meera last wrote in English
    assert facts.language.code == "en"
    facts, _, draft = by_id(seeds, "trg_015_winback_rashmi")               # c_010, language_pref english
    assert facts.language.code == "en" and facts.customer_id == "c_010_rashmi_for_m007"
    assert not HINGLISH_MARKERS.search(draft.body), draft.body
    assert draft.body.startswith("Hi Rashmi,")


def test_hindi_customer_gets_roman_hindi_and_namaste(seeds):
    facts, _, draft = by_id(seeds, "trg_019_chronic_refill_grandfather")
    assert facts.language.code == "hi"
    assert draft.body.startswith("Namaste")
    assert "Sharma ji" in draft.body[:120] and "dawaiyan" in draft.body
    assert "metformin, atorvastatin, telmisartan" in draft.body and "28 Apr 2026" in draft.body
    assert draft.cta == CTA_BINARY_CONFIRM and "CONFIRM" in draft.template_params[2]
    assert not re.search(r"[ऀ-ॿ]", draft.body)                   # Roman script only


def test_regional_greeting(seeds):
    facts, _, draft = by_id(seeds, "trg_017_kids_yoga_trial_followup_karthik")
    assert facts.language.code == "ta-en"
    assert draft.body.startswith("Vanakkam Sumitra,")
    assert "Karthik" in draft.body and "Sat 3 May, 8am" in draft.body


# --------------------------------------------------------------------------- judgment and specificity


def test_ipl_weekend_recommends_delivery_not_promo(seeds):
    _, _, draft = by_id(seeds, "trg_010_ipl_match_delhi")
    body = draft.body
    assert "12%" in body and "delivery" in body.lower() and "skip" in body.lower()
    assert "Buy 1 Pizza Get 1 Free (Tue-Thu)" in body and "weeknight" in body


def test_ipl_weeknight_pushes_match_night(seeds):
    t = copy.deepcopy(seeds["triggers"]["trg_010_ipl_match_delhi"])
    t["payload"].update(is_weeknight=True, match_time_iso="2026-04-28T19:30:00+05:30")
    _, _, draft = compose(seeds, t)
    assert "+18%" in draft.body and "skip" not in draft.body.lower()


def test_research_digest_cites_source_and_cohort(seeds):
    _, _, draft = by_id(seeds, "trg_001_research_digest_dentists")
    for bit in ("Dr. Meera", "JIDA Oct 2026, p.14", "38%", "2,100", "124 high-risk adult patients"):
        assert bit in draft.body, bit
    assert "124" in draft.rationale


def test_supply_alert_batches_and_pool_without_invented_count(seeds):
    _, _, draft = by_id(seeds, "trg_018_supply_atorvastatin_recall")
    body = draft.body
    assert "AT2024-1102" in body and "AT2024-1108" in body and "MfrZ" in body
    assert "240 chronic-prescription customers" in body
    assert "22 of" not in body


def test_competitor_real_and_placeholder(seeds, expanded):
    _, _, draft = by_id(seeds, "trg_023_competitor_opened_dentist")
    assert "Smile Studio" in draft.body and "1.3 km" in draft.body and "₹199" in draft.body
    assert "Dental Cleaning @ ₹299" in draft.body and "price war" in draft.body
    facts, _, draft = by_id(expanded, "trg_056_competitor_opened_m_006_southindiancaf")
    assert facts.payload_is_placeholder
    assert "Indiranagar" in draft.body and "Smile Studio" not in draft.body
    assert "22 reviews" in draft.body                                       # their own review strength


def test_seasonal_dip_reassures_and_redirects_to_retention(seeds):
    _, _, draft = by_id(seeds, "trg_014_seasonal_acquisition_dip_powerhouse")
    body = draft.body.lower()
    assert "30%" in body and "245 active members" in body and "galti nahi" in body
    assert "pause acquisition spend" in body


def test_planning_delivers_a_draft_not_questions(seeds):
    _, _, draft = by_id(seeds, "trg_013_corporate_thali_planning")
    body = draft.body
    assert body.count("\n•") >= 3 and "₹149" in body and "18 orders/day" in body
    assert body.count("?") == 1 and "would you" not in body.lower()
    for invented in ("Embassy", "RMZ", "Sigma"):
        assert invented not in body
    _, _, draft = by_id(seeds, "trg_016_kids_yoga_program_drafting")
    assert "₹2,499" in draft.body and "95 active members" in draft.body


def test_recall_with_slots(seeds):
    facts, pb, draft = by_id(seeds, "trg_003_recall_due_priya")
    assert draft.cta == CTA_MULTI_CHOICE_SLOT and pb.template_name.startswith("merchant_")
    assert "Wed 5 Nov, 6pm" in draft.body and "Thu 6 Nov, 5pm" in draft.body
    assert "₹299" in draft.body and "12 Nov 2026" in draft.body
    assert re.search(r"\b1\b", draft.template_params[2]) and re.search(r"\b2\b", draft.template_params[2])


def test_customer_offer_is_never_a_catalog_price(expanded):
    # generated merchants have no active offers: customer copy must not quote any price
    for tid in ("trg_066_recall_due_m_008_zenyoga_gym_ch", "trg_071_customer_lapsed_soft_m_014_dr_asha_dentis",
                "trg_077_appointment_tomorrow_m_020_renu_salon_luc", "trg_085_chronic_refill_due_m_030_sandeep_restau"):
        facts, _, draft = by_id(expanded, tid)
        if not facts.offers_active:
            assert "₹" not in draft.body, draft.body


def test_chronic_refill_for_a_dentist_is_a_check_in(expanded):
    facts, pb, draft = by_id(expanded, "trg_081_chronic_refill_due_m_011_dr_sameer_dent")
    assert facts.category_slug == "dentists"
    assert draft.cta == CTA_BINARY_YES_NO
    assert "oral-care" in draft.body
    for word in ("metformin", "medicine", "dose", "prescription"):
        assert word not in draft.body.lower()


def test_placeholder_milestone_is_a_real_threshold(expanded):
    facts, _, draft = by_id(expanded, "trg_041_milestone_reached_m_032_mukesh_restaur")
    val = next(f.value for f in facts.anchor_facts if f.key == "derived.milestone")
    assert f"{val['threshold']:,}" in draft.body and f"{int(val['actual']):,}" in draft.body
    assert val["threshold"] <= val["actual"]


def test_placeholder_perf_uses_real_numbers(expanded):
    facts, _, draft = by_id(expanded, "trg_032_perf_dip_m_020_renu_salon_luc")
    assert "18%" in draft.body and "33 calls" in draft.body      # the merchant's own 7-day delta and volume
    assert "Haircut @ ₹99" in draft.body                          # a catalog idea, framed as a suggestion ("jaisa")


def test_festival_placeholder_never_names_a_festival(expanded):
    for tid in ("trg_061_festival_upcoming_m_037_pooja_gym_bang", "trg_062_festival_upcoming_m_015_dr_priya_denti",
                "trg_064_festival_upcoming_m_010_sunrisepharm_p"):
        _, _, draft = by_id(expanded, tid)
        assert not re.search(r"\b(?:Diwali|Holi|Eid|Christmas|Navratri|Pongal|Onam)\b", draft.body), draft.body
    _, _, draft = by_id(expanded, "trg_062_festival_upcoming_m_015_dr_priya_denti")
    assert "festive window" not in draft.body                      # a school-holiday beat is not "festive"


def test_review_placeholder_invents_no_theme(expanded):
    _, _, draft = by_id(expanded, "trg_051_review_theme_emerged_m_020_renu_salon_luc")
    assert "4.5★" in draft.body and "88 reviews" in draft.body
    assert "mention" not in draft.body                             # no invented complaint theme


def test_dr_prefix_never_doubled(expanded):
    for tid, t in expanded["triggers"].items():
        m = expanded["merchants"][t["merchant_id"]]
        if m["identity"].get("owner_first_name", "").startswith("Dr."):
            facts, _, draft = compose(expanded, t)
            if facts.scope == "merchant":
                assert draft.body.startswith(facts.salutation) and facts.salutation.startswith("Dr. ")
            assert "Dr. Dr" not in draft.body


# --------------------------------------------------------------------------- relays and consent edge cases


def test_parent_relay_addresses_parent_about_child(seeds):
    trigger = {"id": "t_relay", "scope": "customer", "kind": "customer_lapsed_soft",
               "merchant_id": "m_001_drmeera_dentist_delhi", "customer_id": "c_003_aanya_for_m001", "payload": {}}
    facts, _, draft = compose(seeds, trigger)
    assert draft.body.startswith("Hi Sneha,") and "Aanya" in draft.body[:120]
    assert "Aanya ki last visit" in draft.body


def test_family_relay_speaks_respectfully_about_sharma_ji(seeds):
    trigger = {"id": "t_relay2", "scope": "customer", "kind": "appointment_tomorrow",
               "merchant_id": "m_009_apollo_pharmacy_jaipur", "customer_id": "c_013_grandfather_for_m009", "payload": {}}
    _, _, draft = compose(seeds, trigger)
    assert draft.body.startswith("Namaste") and "Sharma ji" in draft.body[:120]
    assert not draft.body.startswith("Hi")


def test_walk_in_without_profile_is_not_addressed_by_name(seeds):
    trigger = {"id": "t_walkin", "scope": "customer", "kind": "customer_lapsed_soft",
               "merchant_id": "m_010_sunrisepharm_pharmacy_lucknow", "customer_id": "c_015_anonymous_for_m010",
               "payload": {}}
    facts, _, draft = compose(seeds, trigger)
    assert not facts.consent_ok
    assert "walk-in" not in draft.body.lower() and "(" not in draft.body.split(".")[0]


# --------------------------------------------------------------------------- kinds absent from the dataset


@pytest.mark.parametrize("kind,payload", [
    ("weather_heatwave", {"temperature_c": 44, "city": "Delhi"}),
    ("local_news_event", {"event": "Ring road closed near Sant Nagar for 3 hours", "city": "Delhi"}),
    ("category_trend_movement", {"query": "match night offer", "delta_yoy": 0.65, "segment_age": "20-40"}),
    ("scheduled_recurring", {}),
    ("stock_low", {"item": "cheese", "units_left": 4}),
    ("winback_eligible", {"placeholder": True}),
    ("gbp_unverified", {"placeholder": True}),
    ("ipl_match_today", {"placeholder": True}),
    ("active_planning_intent", {"placeholder": True}),
])
def test_synthetic_merchant_kinds(seeds, kind, payload):
    trigger = {"id": f"t_{kind}", "scope": "merchant", "kind": kind, "merchant_id": "m_005_pizzajunction_restaurant_delhi",
               "payload": payload, "urgency": 2}
    facts, pb, draft = compose(seeds, trigger)
    check_draft(facts, pb, draft, kind)
    if kind == "weather_heatwave":
        assert "44°C" in draft.body
    if kind == "local_news_event":
        assert "Ring road closed" in draft.body and "Event:" not in draft.body
    if kind == "ipl_match_today":
        assert "+18%" in draft.body and "12%" in draft.body         # season-level advice, no invented fixture


def test_customer_kind_without_customer_goes_to_the_merchant(seeds):
    trigger = {"id": "t_x", "scope": "merchant", "kind": "recall_due", "merchant_id": "m_001_drmeera_dentist_delhi",
               "payload": {"service_due": "6_month_cleaning"}}
    facts, pb, draft = compose(seeds, trigger)
    check_draft(facts, pb, draft, "recall-to-merchant")
    assert "behalf" in draft.template_params[2] and "6-month cleaning" in draft.body


def test_unplanned_slot_open(seeds):
    trigger = {"id": "t_slot", "scope": "customer", "kind": "unplanned_slot_open",
               "merchant_id": "m_003_studio11_salon_hyderabad", "customer_id": "c_004_sneha_for_m003",
               "payload": {"open_slots": [{"iso": "2026-05-02T15:00:00+05:30", "label": "Sat 2 May, 3pm"}]}}
    facts, pb, draft = compose(seeds, trigger)
    check_draft(facts, pb, draft, "slot-open")
    assert "Sat 2 May, 3pm" in draft.body and draft.body.startswith("Namaskaram Sneha,")


# --------------------------------------------------------------------------- robustness and variety


def test_render_never_raises_on_empty_or_odd_input():
    for facts in (FactSheet(), FactSheet(kind="perf_dip"), FactSheet(kind="recall_due", scope="customer"),
                  FactSheet(kind="???", salutation="Dr. Dr. X")):
        draft = render(facts, get_playbook(facts.kind, facts.scope))
        assert isinstance(draft, Draft) and draft.body and "Dr. Dr." not in draft.body
    draft = render(None, None)  # type: ignore[arg-type]
    assert isinstance(draft, Draft) and draft.body


def test_minimal_contexts_render(seeds):
    facts = build_facts(None, {"merchant_id": "m_x"}, {"id": "t_x", "kind": "perf_dip", "payload": {}})
    draft = render(facts, get_playbook(facts.kind, facts.scope, facts.category_slug))
    assert draft.body and draft.cta in CTA_VALUES


def test_phrasing_varies_across_merchants(expanded):
    """Same kind, different merchants: the skeletons are not all identical (stable hash variation)."""
    groups: dict[str, set[str]] = {}
    for tid, t in expanded["triggers"].items():
        facts, _, draft = compose(expanded, t)
        skeleton = re.sub(r"[\d,.%₹]+", "#", draft.body)
        for name in {facts.salutation, facts.recipient_name, facts.merchant_name, facts.signer, facts.locality} - {""}:
            skeleton = skeleton.replace(name, "<n>")
        groups.setdefault(facts.kind, set()).add(skeleton.split(". ")[0][:60])
    varied = [k for k, v in groups.items() if len(v) > 1]
    assert len(varied) >= 8, groups


def test_same_input_same_output_across_calls(expanded):
    t = expanded["triggers"]["trg_096_curious_ask_due_m_006_southindiancaf"]
    first = compose(expanded, t)[2]
    for _ in range(3):
        assert compose(expanded, t)[2] == first
