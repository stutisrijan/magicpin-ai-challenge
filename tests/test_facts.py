"""Tests for app/facts.py: the anti-hallucination fact layer.

Covers every seed trigger and every trigger in the expanded dataset (regenerated into a
temp dir from the seeds), plus targeted checks for addressing, consent, digest
resolution, placeholder anchors and number grounding.
"""

from __future__ import annotations

import copy
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

from app.facts import (
    build_facts,
    collect_numbers,
    humanize,
    merchant_display_name,
    resolve_digest_item,
)
from app.numbers import extract_number_tokens, normalize_number
from app.schemas import SEND_AS_MERCHANT, SEND_AS_VERA, FactSheet

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
NOWS = [None, "2026-04-26T10:30:00Z", "2026-09-26T10:30:00Z"]


# --------------------------------------------------------------------------- data fixtures


def _load_dir(path: Path, key: str) -> dict[str, dict]:
    out = {}
    for f in sorted(path.glob("*.json")):
        obj = json.loads(f.read_text())
        out[obj[key]] = obj
    return out


@pytest.fixture(scope="session")
def seeds() -> dict:
    cats = _load_dir(DATASET / "categories", "slug")
    merchants = {m["merchant_id"]: m for m in json.loads((DATASET / "merchants_seed.json").read_text())["merchants"]}
    customers = {c["customer_id"]: c for c in json.loads((DATASET / "customers_seed.json").read_text())["customers"]}
    triggers = {t["id"]: t for t in json.loads((DATASET / "triggers_seed.json").read_text())["triggers"]}
    return {"categories": cats, "merchants": merchants, "customers": customers, "triggers": triggers}


@pytest.fixture(scope="session")
def expanded(tmp_path_factory) -> dict:
    out = tmp_path_factory.mktemp("expanded")
    subprocess.run(
        [sys.executable, str(DATASET / "generate_dataset.py"), "--seed-dir", str(DATASET), "--out", str(out)],
        check=True, capture_output=True, cwd=str(ROOT),
    )
    data = {
        "categories": _load_dir(out / "categories", "slug"),
        "merchants": _load_dir(out / "merchants", "merchant_id"),
        "customers": _load_dir(out / "customers", "customer_id"),
        "triggers": _load_dir(out / "triggers", "id"),
    }
    assert len(data["triggers"]) == 100 and len(data["merchants"]) == 50
    return data


def facts_for(data: dict, trigger_id: str, now: str | None = None, **kw) -> FactSheet:
    t = data["triggers"][trigger_id]
    m = data["merchants"][t["merchant_id"]]
    c = data["customers"].get(t.get("customer_id") or "")
    return build_facts(data["categories"].get(m["category_slug"]), m, t, c, now=now, **kw)


def texts(fs: FactSheet) -> list[str]:
    return [f.text for f in fs.all_facts()]


def fact(fs: FactSheet, key: str):
    return next((f for f in fs.all_facts() if f.key == key), None)


# --------------------------------------------------------------------------- invariants on every trigger


def _check_invariants(fs: FactSheet, data: dict, trigger_id: str) -> None:
    t = data["triggers"][trigger_id]
    m = data["merchants"][t["merchant_id"]]
    assert isinstance(fs, FactSheet)
    assert fs.kind == t["kind"]
    assert fs.trigger_id == trigger_id and fs.merchant_id == t["merchant_id"]
    for name in (fs.salutation, fs.owner_name, fs.signer, fs.recipient_name):
        assert not re.search(r"\bDr\.?\s*Dr\b", name), name
    assert fs.anchor_facts, f"{trigger_id}: no anchor facts"
    if (t.get("payload") or {}).get("placeholder"):
        assert fs.payload_is_placeholder
    slugs = [s for s in (m.get("signals") or []) if isinstance(s, str)]
    for f in fs.all_facts():
        assert f.text and f.text == f.text.strip(), f"{trigger_id}: empty/unstripped fact {f.key}"
        assert "_" not in f.text, f"{trigger_id}: underscore in {f.key}: {f.text}"
        for slug in slugs:
            assert slug not in f.text, f"{trigger_id}: raw signal slug in {f.text}"
        for field_name in ("payload", "placeholder", "metric_or_topic", "delta_7d", "views_pct"):
            assert field_name not in f.text
        for tok in extract_number_tokens(f.text):
            assert normalize_number(tok) in fs.allowed_numbers, f"{trigger_id}: '{tok}' in '{f.text}' ungrounded"
        assert f.weight in (1, 2, 3)
    assert all(f.weight == 3 for f in fs.anchor_facts)
    assert all(f.weight in (1, 2) for f in fs.support_facts)
    for note in fs.notes:
        assert "_" not in note, f"{trigger_id}: underscore in note: {note}"
        for tok in extract_number_tokens(note):
            assert normalize_number(tok) in fs.allowed_numbers, f"{trigger_id}: '{tok}' in note ungrounded"
    for slot in fs.slots:
        for tok in extract_number_tokens(slot):
            assert normalize_number(tok) in fs.allowed_numbers
    expected_send_as = SEND_AS_MERCHANT if t.get("scope") == "customer" else SEND_AS_VERA
    assert fs.send_as == expected_send_as
    assert fs.urgency == t["urgency"]


@pytest.mark.parametrize("now", NOWS)
def test_all_seed_triggers(seeds, now):
    assert len(seeds["triggers"]) == 25
    for tid in seeds["triggers"]:
        fs = facts_for(seeds, tid, now=now)
        _check_invariants(fs, seeds, tid)


@pytest.mark.parametrize("now", NOWS)
def test_all_expanded_triggers(expanded, now):
    for tid in expanded["triggers"]:
        fs = facts_for(expanded, tid, now=now)
        _check_invariants(fs, expanded, tid)


def test_build_facts_is_deterministic(expanded):
    for tid in list(expanded["triggers"])[::7]:
        a = facts_for(expanded, tid, now="2026-04-26T10:30:00Z")
        b = facts_for(expanded, tid, now="2026-04-26T10:30:00Z")
        assert [(f.key, f.text) for f in a.all_facts()] == [(f.key, f.text) for f in b.all_facts()]
        assert a.notes == b.notes and a.allowed_numbers == b.allowed_numbers


def test_build_facts_does_not_mutate_inputs(seeds):
    before = copy.deepcopy(seeds)
    for tid in seeds["triggers"]:
        facts_for(seeds, tid, now="2026-04-26T10:30:00Z")
    assert seeds == before


# --------------------------------------------------------------------------- addressing


@pytest.mark.parametrize("owner, slug, expected", [
    ("Meera", "dentists", "Dr. Meera"),
    ("Dr. Sameer", "dentists", "Dr. Sameer"),
    ("dr sameer", "dentists", "Dr. Sameer"),
    ("Dr.Asha", "dentists", "Dr. Asha"),
    ("Lakshmi", "salons", "Lakshmi"),
    ("Suresh Kumar", "restaurants", "Suresh"),
    ("Dr. Nisha", "gyms", "Dr. Nisha"),
])
def test_merchant_display_name(owner, slug, expected):
    m = {"category_slug": slug, "identity": {"owner_first_name": owner, "name": "X Clinic"}}
    assert merchant_display_name(m, slug) == expected


def test_merchant_display_name_fallbacks():
    assert merchant_display_name({"identity": {"name": "Studio11 Family Salon"}}, "salons") == "Studio11 Family Salon team"
    assert merchant_display_name({}, "dentists") == "Doc"
    assert merchant_display_name({}, "salons") == "there"
    assert merchant_display_name(None, "") == "there"  # type: ignore[arg-type]


def test_generated_dr_owner_never_doubled(expanded):
    fs = facts_for(expanded, "trg_081_chronic_refill_due_m_011_dr_sameer_dent")
    assert fs.owner_name == "Dr. Sameer"
    assert fs.signer == "Bright Smile Dental"          # dentists sign as the clinic
    fs2 = facts_for(expanded, "trg_026_research_digest_m_017_dr_rajan_denti")
    assert fs2.salutation == "Dr. Rajan"


def test_customer_facing_signer_non_dentist(seeds):
    fs = facts_for(seeds, "trg_015_winback_rashmi")
    assert fs.signer == "Karthik from PowerHouse Fitness"
    assert fs.salutation == "Rashmi"


def test_relay_parent_c003(seeds):
    t = copy.deepcopy(seeds["triggers"]["trg_003_recall_due_priya"])
    t["customer_id"] = "c_003_aanya_for_m001"
    t["payload"] = {"placeholder": True, "metric_or_topic": "recall_due"}
    fs = build_facts(seeds["categories"]["dentists"], seeds["merchants"]["m_001_drmeera_dentist_delhi"], t,
                     seeds["customers"]["c_003_aanya_for_m001"], now="2026-04-26T10:30:00Z")
    assert fs.salutation == "Sneha" and fs.recipient_name == "Sneha" and fs.about_name == "Aanya"
    assert fs.language.code == "hinglish"
    assert fs.consent_ok
    assert any("Sneha" in n and "Aanya" in n for n in fs.notes)
    assert fs.signer == "Dr. Meera's Dental Clinic"


def test_relay_parent_c012(seeds):
    fs = facts_for(seeds, "trg_017_kids_yoga_trial_followup_karthik", now="2026-04-26T10:30:00Z")
    assert fs.salutation == "Sumitra" and fs.about_name == "Karthik"
    assert fs.language.code == "ta-en" and fs.language.greeting == "Vanakkam"
    assert fs.slots == ["Sat 3 May, 8am"]
    assert fs.signer == "Padma from Zen Yoga Studio"


def test_relay_senior_via_son_c013(seeds):
    fs = facts_for(seeds, "trg_019_chronic_refill_grandfather", now="2026-04-26T10:30:00Z")
    assert fs.about_name == "Sharma ji"
    assert fs.salutation == "Sharma ji"
    assert fs.recipient_name == ""                    # the son reads it; unnamed
    assert fs.language.greeting == "Namaste" and fs.language.code == "hi"
    assert any("son" in n and "Namaste" in n for n in fs.notes)
    assert fact(fs, "trigger.molecules").text == "3 regular medicines due: metformin, atorvastatin, telmisartan"
    assert fact(fs, "trigger.stock_runs_out").text == "Current stock runs out on 28 Apr 2026"
    assert "Senior Citizen 15% OFF" in fs.offers_active


def test_senior_direct_customer_gets_ji():
    c = {"customer_id": "c_x", "identity": {"name": "Mrs. Kapoor", "language_pref": "english", "age_band": "65-75"},
         "preferences": {"channel": "whatsapp", "reminder_opt_in": True},
         "consent": {"opted_in_at": "2025-01-01", "scope": ["refill_reminders"]}}
    t = {"id": "t", "scope": "customer", "kind": "chronic_refill_due", "payload": {"placeholder": True}}
    fs = build_facts({}, {"merchant_id": "m", "category_slug": "pharmacies", "identity": {"name": "P"}}, t, c)
    assert fs.salutation == "Kapoor ji" and fs.recipient_name == "Kapoor ji"
    assert fs.language.greeting == "Namaste"


# --------------------------------------------------------------------------- consent


def test_consent_walk_in_c015(seeds):
    t = {"id": "trg_x", "scope": "customer", "kind": "recall_due", "merchant_id": "m_010_sunrisepharm_pharmacy_lucknow",
         "customer_id": "c_015_anonymous_for_m010", "payload": {"placeholder": True}, "urgency": 2}
    fs = build_facts(seeds["categories"]["pharmacies"], seeds["merchants"]["m_010_sunrisepharm_pharmacy_lucknow"], t,
                     seeds["customers"]["c_015_anonymous_for_m010"])
    assert fs.consent_ok is False
    for part in ("walk-in", "no opt-in", "empty consent scope", "no contact channel", "opted out of reminders"):
        assert part in fs.consent_reason
    assert fs.salutation == "" and fs.recipient_name == ""
    assert fs.send_as == SEND_AS_MERCHANT


def test_consent_reminder_opt_out_only_for_reminder_kinds(expanded):
    assert facts_for(expanded, "trg_070_recall_due_m_045_vinod_pharmaci").consent_ok is False
    assert facts_for(expanded, "trg_079_appointment_tomorrow_m_017_dr_rajan_denti").consent_ok is False
    # same opt-out flag, but a trial follow-up is not a reminder
    assert facts_for(expanded, "trg_089_trial_followup_m_043_komal_pharmaci").consent_ok is True


def test_consent_matches_planner_rules(seeds):
    base = copy.deepcopy(seeds["customers"]["c_001_priya_for_m001"])
    t = seeds["triggers"]["trg_003_recall_due_priya"]
    cat, m = seeds["categories"]["dentists"], seeds["merchants"]["m_001_drmeera_dentist_delhi"]
    # an explicit reminder scope wins over reminder_opt_in: false
    c = copy.deepcopy(base)
    c["preferences"]["reminder_opt_in"] = False
    assert build_facts(cat, m, t, c).consent_ok is True
    c["consent"]["scope"] = ["promotional_offers"]
    assert build_facts(cat, m, t, c).consent_ok is False
    revoked = copy.deepcopy(base)
    revoked["consent"]["revoked_at"] = "2026-04-01"
    assert "consent revoked" in build_facts(cat, m, t, revoked).consent_reason
    dnd = copy.deepcopy(base)
    dnd["state"] = "dnd"
    assert build_facts(cat, m, t, dnd).consent_ok is False


def test_consent_ok_reason_for_opted_in(seeds):
    fs = facts_for(seeds, "trg_003_recall_due_priya")
    assert fs.consent_ok and "4 Nov 2025" in fs.consent_reason and "recall reminders" in fs.consent_reason


def test_merchant_facing_consent_always_ok(seeds):
    fs = facts_for(seeds, "trg_001_research_digest_dentists")
    assert fs.consent_ok and fs.consent_reason == ""


def test_customer_scope_without_customer_context(seeds):
    t = seeds["triggers"]["trg_003_recall_due_priya"]
    fs = build_facts(seeds["categories"]["dentists"], seeds["merchants"]["m_001_drmeera_dentist_delhi"], t, None)
    assert fs.send_as == SEND_AS_MERCHANT and fs.consent_ok is False
    assert "customer context missing" in fs.consent_reason


def test_customer_kind_with_customer_but_merchant_scope_is_on_behalf(seeds):
    t = copy.deepcopy(seeds["triggers"]["trg_003_recall_due_priya"])
    t["scope"] = "merchant"
    fs = build_facts(seeds["categories"]["dentists"], seeds["merchants"]["m_001_drmeera_dentist_delhi"], t,
                     seeds["customers"]["c_001_priya_for_m001"])
    assert fs.send_as == SEND_AS_MERCHANT and fs.scope == "customer"


# --------------------------------------------------------------------------- digest resolution


def test_digest_resolution_seed_triggers(seeds):
    cases = {
        "trg_001_research_digest_dentists": "d_2026W17_jida_fluoride",
        "trg_002_compliance_dci_radiograph": "d_2026W17_dci_radiograph",
        "trg_018_supply_atorvastatin_recall": "d_2026W17_atorvastatin_recall",
        "trg_022_cde_webinar_dentists": "d_2026W17_ida_webinar",
        "trg_010_ipl_match_delhi": "d_2026W17_ipl_window",
    }
    for tid, item_id in cases.items():
        fs = facts_for(seeds, tid)
        assert fs.digest_item and fs.digest_item["id"] == item_id, tid
        m = seeds["merchants"][seeds["triggers"][tid]["merchant_id"]]
        assert resolve_digest_item(seeds["categories"][m["category_slug"]], seeds["triggers"][tid])["id"] == item_id


def test_research_digest_facts(seeds):
    fs = facts_for(seeds, "trg_001_research_digest_dentists")
    assert fact(fs, "digest.source").text == "JIDA Oct 2026, p.14"
    assert fact(fs, "digest.trial_n").text == "2,100-patient trial"
    assert "38%" in fact(fs, "digest.stat").text
    assert fact(fs, "digest.segment").text == "Most relevant for high-risk adults"
    assert fs.anchor_facts[0].key == "digest.title"
    assert any("124 high-risk adult patients" in n for n in fs.notes)
    assert fact(fs, "agg.high_risk_adult_count").text == "124 high-risk adult patients"
    assert not any(f.key.startswith("trigger.generic") for f in fs.all_facts())


def test_regulation_facts_and_deadline(seeds):
    fs = facts_for(seeds, "trg_002_compliance_dci_radiograph", now="2026-04-26T10:30:00Z")
    assert fact(fs, "digest.title").text == "DCI revised radiograph dose limits effective 15 Dec 2026"
    assert fact(fs, "digest.source").text == "Dental Council of India circular 2026-11-04"
    assert fact(fs, "trigger.deadline").text == "Deadline: 15 Dec 2026 (233 days away)"
    assert "1.5 mSv" in fact(fs, "digest.stat").text
    assert {"233", "15", "12", "2026"} <= fs.allowed_numbers


def test_supply_alert_merges_payload(seeds):
    fs = facts_for(seeds, "trg_018_supply_atorvastatin_recall")
    assert fs.digest_item["affected_batches"] == ["AT2024-1102", "AT2024-1108"]
    assert fs.digest_item["manufacturer"] == "MfrZ"
    assert fact(fs, "trigger.batches").text == "2 affected batches: AT2024-1102, AT2024-1108"
    assert fact(fs, "digest.title").text.endswith("by manufacturer MfrZ")
    assert any("240 chronic-prescription customers" in n and "Do not invent" in n for n in fs.notes)


def test_cde_payload_overrides(seeds):
    fs = facts_for(seeds, "trg_022_cde_webinar_dentists")
    assert fact(fs, "digest.credits").text == "2 CDE credits"
    assert fact(fs, "digest.fee").text == "Fee: free for members"
    assert fact(fs, "digest.date").text == "On 2 May 2026, 7pm"


def test_resolve_digest_by_kind_for_placeholders(seeds):
    cats = seeds["categories"]

    def resolve(slug: str, kind: str):
        item = resolve_digest_item(cats[slug], {"kind": kind, "payload": {"placeholder": True}})
        return item["id"] if item else None

    assert resolve("dentists", "research_digest") == "d_2026W17_jida_fluoride"
    assert resolve("gyms", "research_digest") == "d_2026W17_creatine_safety_bulletin"
    assert resolve("salons", "research_digest") == "d_2026W17_olaplex_no9"          # no research item: tech
    assert resolve("restaurants", "regulation_change") == "d_2026W17_packaged_food_gst"
    assert resolve("pharmacies", "supply_alert") == "d_2026W17_atorvastatin_recall"
    assert resolve("pharmacies", "category_seasonal") == "d_2026W17_summer_demand"
    assert resolve("dentists", "category_trend_movement") == "d_2026W17_aligner_trend"
    assert resolve("dentists", "cde_opportunity") == "d_2026W17_ida_webinar"
    assert resolve("gyms", "cde_opportunity") == "d_2026W17_resolution_window"      # no cde item: first item
    assert resolve("dentists", "perf_dip") is None
    assert resolve_digest_item({}, {"kind": "research_digest", "payload": {}}) is None
    assert resolve_digest_item(None, {"kind": "research_digest"}) is None


def test_resolve_digest_inline_and_unknown_id(seeds):
    cat = seeds["categories"]["dentists"]
    inline = {"kind": "research_digest", "payload": {"top_item": {"title": "New item", "source": "JIDA Nov 2026"}}}
    assert resolve_digest_item(cat, inline)["source"] == "JIDA Nov 2026"
    unknown = {"kind": "regulation_change", "payload": {"top_item_id": "does_not_exist"}}
    assert resolve_digest_item(cat, unknown)["id"] == "d_2026W17_dci_radiograph"
    payload_item = {"kind": "regulation_change", "payload": {"title": "New DCI rule", "source": "DCI 2026"}}
    assert resolve_digest_item(cat, payload_item)["title"] == "New DCI rule"


# --------------------------------------------------------------------------- judgment notes


def test_ipl_saturday_note(seeds):
    fs = facts_for(seeds, "trg_010_ipl_match_delhi", now="2026-04-26T10:30:00Z")
    assert fact(fs, "trigger.weeknight").text == "Weekend match (not a weeknight)"
    assert fact(fs, "trigger.match").text == "IPL match today: DC vs MI"
    assert fact(fs, "trigger.match_time").text == "Starts 7:30pm on 26 Apr 2026"
    assert any("Saturday" in n and "12%" in n and "delivery" in n and "Buy 1 Pizza Get 1 Free" in n
               for n in fs.notes)
    assert any("does not cover today's match" in n for n in fs.notes)   # Tue-Thu offer vs weekend match


def test_ipl_weeknight_note(seeds):
    t = copy.deepcopy(seeds["triggers"]["trg_010_ipl_match_delhi"])
    t["payload"]["is_weeknight"] = True
    t["payload"]["match_time_iso"] = "2026-04-28T19:30:00+05:30"
    fs = build_facts(seeds["categories"]["restaurants"], seeds["merchants"]["m_005_pizzajunction_restaurant_delhi"], t)
    assert any(n.startswith("Weeknight match") and "+18%" in n and "Match-night Combo" in n for n in fs.notes)


def test_seasonal_dip_reassurance(seeds):
    fs = facts_for(seeds, "trg_014_seasonal_acquisition_dip_powerhouse", now="2026-04-26T10:30:00Z")
    assert fact(fs, "trigger.season_note").text == "Expected seasonal dip: post-resolution Apr-Jun window"
    assert fact(fs, "seasonal.note").text.startswith("Apr-Jun: lowest acquisition window")
    assert any("245 active members" in n and "reassure" in n for n in fs.notes)


def test_competitor_price_gap_note(seeds):
    fs = facts_for(seeds, "trg_023_competitor_opened_dentist")
    assert fact(fs, "trigger.competitor").text == "New competitor nearby: Smile Studio"
    assert fact(fs, "trigger.distance").text == "1.3 km away"
    assert any("undercuts" in n and "₹100" in n for n in fs.notes)
    assert "100" in fs.allowed_numbers


def test_planning_intent_notes(seeds):
    fs = facts_for(seeds, "trg_013_corporate_thali_planning")
    assert fact(fs, "trigger.intent_topic").text == "Planning: corporate bulk thali package"
    assert any("drafted artifact" in n for n in fs.notes)
    assert any("Weekday Lunch Thali @ ₹149" in n for n in fs.notes)


# --------------------------------------------------------------------------- placeholder anchors


def test_placeholder_milestone_threshold(expanded):
    t = copy.deepcopy(expanded["triggers"]["trg_041_milestone_reached_m_032_mukesh_restaur"])
    m = expanded["merchants"]["m_011_dr_sameer_dentist_bangalore"]                 # views 4792
    fs = build_facts(expanded["categories"]["dentists"], m, t)
    ms = fact(fs, "derived.milestone")
    assert ms.text == "Crossed 4,500 profile views in the last 30 days (4,792 now)"
    assert "4500" in fs.allowed_numbers and ms.source == "derived"


def test_placeholder_perf_dip_matches_sign(expanded):
    fs = facts_for(expanded, "trg_032_perf_dip_m_020_renu_salon_luc")
    assert fact(fs, "derived.metric_delta").text.startswith("Calls down 18% over the last 7 days")
    # m_023 has only positive 7-day deltas: fall back to an honest peer gap, never a fake drop
    fs2 = facts_for(expanded, "trg_031_perf_dip_m_023_sushma_salon_p")
    delta = fact(fs2, "derived.metric_delta")
    assert "below peers" in delta.text and "down" not in delta.text


def test_placeholder_perf_spike_matches_sign(expanded):
    fs = facts_for(expanded, "trg_040_perf_spike_m_019_karim_salon_lu")
    assert fact(fs, "derived.metric_delta").text.startswith("Profile views up 26%")


def test_placeholder_competitor_has_no_invented_name(expanded):
    fs = facts_for(expanded, "trg_056_competitor_opened_m_006_southindiancaf")
    assert fact(fs, "trigger.competitor").text == "A new restaurant listing opened near Indiranagar"
    assert any("do not invent" in n for n in fs.notes)


def test_placeholder_festival_uses_seasonal_beats(expanded):
    sept = facts_for(expanded, "trg_061_festival_upcoming_m_037_pooja_gym_bang", now="2026-09-26T10:30:00Z")
    assert fact(sept, "trigger.season_note").text.startswith("Aug-Oct: wedding-prep + festival window")
    # no `now`: falls back to expires_at (June) -> current beat plus the next festive window
    june = facts_for(expanded, "trg_061_festival_upcoming_m_037_pooja_gym_bang")
    notes_and_facts = " ".join(texts(june) + june.notes)
    assert "Aug-Oct" in notes_and_facts and "Apr-Jun" in notes_and_facts
    for fs in (sept, june):
        assert not any(name in " ".join(texts(fs)) for name in ("Diwali", "Holi", "Christmas", "Eid"))
        assert any("never name a festival" in n for n in fs.notes)


def test_real_festival_gets_matching_beat(seeds):
    fs = facts_for(seeds, "trg_006_festival_diwali", now="2026-04-26T10:30:00Z")
    assert fact(fs, "trigger.festival").text == "Diwali on 31 Oct 2026"
    assert fact(fs, "trigger.days_until").text == "188 days to go"
    assert fact(fs, "seasonal.note").text.startswith("Oct-Dec: primary wedding/festival season")


def test_stale_day_counts_are_dropped(seeds):
    # local simulator sends wall-clock time; payload counts written for 26 Apr no longer match
    fs = facts_for(seeds, "trg_006_festival_diwali", now="2026-09-26T10:30:00Z")
    assert fact(fs, "trigger.days_until") is None
    lapsed = facts_for(seeds, "trg_015_winback_rashmi", now="2026-09-26T10:30:00Z")
    assert fact(lapsed, "trigger.days_since_last_visit") is None
    assert fact(lapsed, "customer.last_visit").text == "Last visit on 28 Feb 2026"
    ok = facts_for(seeds, "trg_015_winback_rashmi", now="2026-04-26T10:30:00Z")
    assert fact(ok, "trigger.days_since_last_visit").text == "57 days since the last visit"


def test_placeholder_review_theme_not_invented(expanded):
    fs = facts_for(expanded, "trg_051_review_theme_emerged_m_020_renu_salon_luc")
    assert fact(fs, "trigger.theme") is None
    assert fact(fs, "derived.review_benchmark").text == "Peer salons in this category average a 4.5★ rating and 88 reviews"
    assert any("do not invent one" in n for n in fs.notes)


def test_placeholder_review_uses_real_merchant_theme(seeds):
    t = {"id": "t", "scope": "merchant", "kind": "review_theme_emerged", "urgency": 3,
         "payload": {"placeholder": True, "metric_or_topic": "review_theme_emerged"}}
    fs = build_facts(seeds["categories"]["dentists"], seeds["merchants"]["m_001_drmeera_dentist_delhi"], t)
    assert fact(fs, "trigger.theme").text == "3 reviews in the last 30 days mention wait time"
    assert fact(fs, "trigger.quote").text == "Customer quote: 'had to wait 30 min on Sunday afternoon'"


def test_placeholder_customer_kinds_adapt_to_category(expanded):
    refill = facts_for(expanded, "trg_081_chronic_refill_due_m_011_dr_sameer_dent", now="2026-04-26T10:30:00Z")
    assert fact(refill, "trigger.service_due").text == "Oral-care refill due"
    assert any("no medicine or product names" in n for n in refill.notes)
    yoga = facts_for(expanded, "trg_066_recall_due_m_008_zenyoga_gym_ch", now="2026-04-26T10:30:00Z")
    assert fact(yoga, "trigger.service_due").text == "Next yoga class due"
    assert fact(yoga, "trigger.last_service_date").text == "Last visit on 1 Apr 2026 (25 days ago)"
    appt = facts_for(expanded, "trg_076_appointment_tomorrow_m_019_karim_salon_lu", now="2026-04-26T10:30:00Z")
    assert fact(appt, "trigger.due_date").text == "Salon appointment scheduled for tomorrow, 27 Apr 2026"
    assert "27" in appt.allowed_numbers
    assert any("don't invent one" in n for n in appt.notes)


def test_placeholder_renewal_from_subscription(expanded):
    trial = facts_for(expanded, "trg_091_renewal_due_m_048_vinod_pharmaci")
    assert fact(trial, "trigger.days_remaining").text == "Trial ends in 5 days"
    expired = facts_for(expanded, "trg_094_renewal_due_m_021_paras_salon_ch")
    assert fact(expired, "trigger.days_since_expiry").text == "Pro plan expired 73 days ago"
    far = facts_for(expanded, "trg_092_renewal_due_m_033_anand_restaura")
    assert any("early value check-in" in n for n in far.notes)
    assert all("price" in n or "amount" not in n for n in far.notes)


def test_curious_ask_uses_relevant_trend(expanded):
    fs = facts_for(expanded, "trg_096_curious_ask_due_m_006_southindiancaf")
    assert fact(fs, "trend.top").text.startswith("'weekday lunch thali' searches up 34% YoY")
    assert fs.trend_note


# --------------------------------------------------------------------------- humanising & support facts


@pytest.mark.parametrize("raw, expected", [
    ("6_month_cleaning", "6-month cleaning"),
    ("post_resolution_window_apr_jun", "post-resolution Apr-Jun window"),
    ("delivery_late", "late delivery"),
    ("kids_yoga_post", "kids yoga post"),
    ("corporate_bulk_thali_package", "corporate bulk thali package"),
    ("skin_prep_program_30day", "30-day skin-prep program"),
    ("pickup_late", "late pickup"),
    ("window_mar_may", "window Mar-May"),
    ("12_week_program", "12-week program"),
    ("chronic_rx_metformin", "chronic Rx metformin"),
    ("2026-11-12", "12 Nov 2026"),
    ("2026-05-02T19:00:00+05:30", "2 May 2026, 7pm"),
    (True, "yes"),
    (False, "no"),
    (["gold_tier", "silver_tier", "bronze_tier"], "gold tier, silver tier and bronze tier"),
    (1499, "1,499"),
    ("Dental Cleaning @ ₹299", "Dental Cleaning @ ₹299"),
    (None, ""),
])
def test_humanize(raw, expected):
    assert humanize(raw) == expected


def test_generic_unknown_kind_humanises_payload():
    t = {"id": "t_new", "scope": "merchant", "kind": "loyalty_program_launch", "urgency": 2,
         "payload": {"expected_change_pct": -0.5, "launch_date": "2026-11-12", "tiers": ["gold_tier", "silver_tier"],
                     "is_live": True, "reward_amount": 250, "skip_me": None, "notes_text": "starts_after_diwali"}}
    fs = build_facts({"slug": "salons"}, {"merchant_id": "m", "category_slug": "salons",
                                          "identity": {"owner_first_name": "Renu", "name": "Beauty Lounge"}}, t)
    got = {f.key: f.text for f in fs.anchor_facts}
    assert got["trigger.generic.expected_change_pct"] == "Expected change: down 50%"
    assert got["trigger.generic.launch_date"] == "Launch date: 12 Nov 2026"
    assert got["trigger.generic.tiers"] == "Tiers: gold tier and silver tier"
    assert got["trigger.generic.is_live"] == "Is live: yes"
    assert got["trigger.generic.reward_amount"] == "Reward amount: ₹250"
    assert got["trigger.generic.notes_text"] == "Notes text: starts after diwali"
    assert "trigger.generic.skip_me" not in got
    assert all("_" not in text for text in got.values())
    assert {"50", "250", "12", "11", "2026"} <= fs.allowed_numbers


def test_generic_labels_camel_case_and_urls_are_scrubbed():
    t = {"id": "t", "scope": "merchant", "kind": "local_news_event", "urgency": 2,
         "payload": {"roadClosureHours": 3, "headline": "Expressway closed, details at https://example.com/x"}}
    fs = build_facts({}, {"merchant_id": "m", "identity": {"owner_first_name": "Asha"}}, t)
    got = {f.key: f.text for f in fs.anchor_facts}
    assert got["trigger.generic.roadClosureHours"] == "Road closure hours: 3"
    assert "http" not in got["trigger.generic.headline"] and got["trigger.generic.headline"].startswith("Headline:")


def test_trend_movement_payload(seeds):
    t = {"id": "t", "scope": "merchant", "kind": "category_trend_movement", "urgency": 2,
         "payload": {"query": "clear aligners delhi", "delta_yoy": 0.62, "segment_age": "28-45"}}
    fs = build_facts(seeds["categories"]["dentists"], seeds["merchants"]["m_001_drmeera_dentist_delhi"], t)
    assert fact(fs, "trigger.trends").text == "'clear aligners delhi' searches up 62% YoY (age 28-45)"
    assert fs.digest_item["id"] == "d_2026W17_aligner_trend"


def test_known_kind_extra_keys_go_generic_and_ids_do_not(seeds):
    t = copy.deepcopy(seeds["triggers"]["trg_001_research_digest_dentists"])
    t["payload"]["audience_size_hint"] = 1200
    fs = build_facts(seeds["categories"]["dentists"], seeds["merchants"]["m_001_drmeera_dentist_delhi"], t)
    keys = [f.key for f in fs.all_facts()]
    assert "trigger.generic.audience_size_hint" in keys
    assert "trigger.generic.top_item_id" not in keys


def test_perf_and_peer_support_facts(seeds):
    fs = facts_for(seeds, "trg_001_research_digest_dentists")
    assert fact(fs, "perf.views").text == "2,410 profile views in the last 30 days"
    assert fact(fs, "perf.ctr_vs_peer").text == "CTR 2.1% vs 3.0% peer average (30% below peers)"
    assert fact(fs, "perf.calls_vs_peer").text == "18 calls vs 12 peer average in 30 days (50% above peers)"
    assert fact(fs, "perf.delta_views").text == "Profile views up 18% over the last 7 days"
    assert fact(fs, "agg.retention_6mo_pct").text == "38% 6-month retention vs 42% peer average (below peers)"
    assert fact(fs, "agg.lapsed_180d_plus").text == "78 patients lapsed 180+ days"
    assert fact(fs, "merchant.subscription").text == "Pro plan active, 82 days left"
    assert [f.text for f in fs.support_facts if f.key == "offer.active"] == ["Dental Cleaning @ ₹299"]
    assert fact(fs, "offer.expired").text == "Deep Cleaning @ ₹499 (expired 28 Feb 2026)"
    assert fact(fs, "review.wait_time").text == ("3 reviews in the last 30 days mention wait time: "
                                                 "'had to wait 30 min on Sunday afternoon'")
    assert fact(fs, "history.last_merchant").text.endswith("'Yes please, focus on whitening and aligners'")
    assert fact(fs, "trend.top").text == "'clear aligners delhi' searches up 62% YoY (age 28-45)"


def test_signals_humanised_and_unknown_skipped(seeds):
    m = copy.deepcopy(seeds["merchants"]["m_002_bharat_dentist_mumbai"])
    m["signals"].append("some_future_signal:9d")
    fs = build_facts(seeds["categories"]["dentists"], m, seeds["triggers"]["trg_004_perf_dip_bharat"])
    got = {f.key: f.text for f in fs.support_facts if f.key.startswith("signal.")}
    assert got["signal.renewal_due_soon"] == "Plan renewal due in 12 days"
    assert got["signal.dormant_with_vera"] == "No reply to Vera in 14 days"
    assert got["signal.unverified_gbp"] == "Google Business Profile not verified yet"
    assert not any("future" in k for k in got)
    fs1 = facts_for(seeds, "trg_001_research_digest_dentists")
    assert fact(fs1, "signal.stale_posts").text == "Last Google post was 22 days ago"
    assert any("last 48 hours" in n for n in fs1.notes)          # engagement signals become writer notes


def test_customer_support_facts(seeds):
    fs = facts_for(seeds, "trg_003_recall_due_priya", now="2026-04-26T10:30:00Z")
    assert fact(fs, "customer.services").text == "Past services: cleaning (3 times), whitening"
    assert fact(fs, "customer.preferred_slots").text == "Prefers weekday evenings"
    assert fact(fs, "customer.last_visit").text == "Last visit on 12 May 2026"   # after `now`: no duration
    assert fact(fs, "trigger.service_due").text == "6-month cleaning due"
    assert fs.slots == ["Wed 5 Nov, 6pm", "Thu 6 Nov, 5pm"]
    assert fs.language.code == "hinglish"
    # customer-facing sheets never carry merchant dashboard numbers
    assert not any(f.key.startswith(("perf.", "agg.", "signal.", "history.")) for f in fs.all_facts())


def test_merchant_language_on_sheet(seeds):
    assert facts_for(seeds, "trg_001_research_digest_dentists").language.code == "en"
    assert facts_for(seeds, "trg_004_perf_dip_bharat").language.code == "hinglish"
    fs = facts_for(seeds, "trg_001_research_digest_dentists",
                   conversation_turns=[{"role": "merchant", "body": "haan bhej do abhi"}])
    assert fs.language.code == "hinglish"
    assert fact(fs, "history.last_merchant").text.endswith("'haan bhej do abhi'")


def test_wedding_followup(seeds):
    fs = facts_for(seeds, "trg_007_bridal_followup_kavya", now="2026-04-26T10:30:00Z")
    assert fact(fs, "trigger.days_to_wedding").text == "196 days to the wedding"
    assert fact(fs, "trigger.next_step").text == "Window now open for the 30-day skin-prep program"
    assert fact(fs, "trigger.trial_date").text == "Bridal trial on 22 Mar 2026"
    assert fs.signer == "Lakshmi from Studio11 Family Salon"


def test_category_seasonal_trends(seeds):
    fs = facts_for(seeds, "trg_020_summer_demand_shift")
    assert fact(fs, "trigger.trends").text == ("ORS demand up 40%, sunscreen demand up 38%, antifungal demand up 45% "
                                               "and cold & cough demand down 60%")
    assert fs.content_item and fs.content_item["id"] == "pc_summer_basics"


# --------------------------------------------------------------------------- numbers


def test_collect_numbers():
    nums = collect_numbers({"a": 0.021, "b": "Dental Cleaning @ ₹1,499", "c": ["38%", "2026-11-12"], "d": True})
    assert {"0.021", "2.1", "1499", "38", "2026", "11", "12"} <= nums
    assert "1" not in nums                                        # booleans are not numbers
    assert collect_numbers(None, [], {}) == set()


def test_allowed_numbers_cover_contexts_and_derivations(seeds):
    fs = facts_for(seeds, "trg_004_perf_dip_bharat")
    for n in ("50", "0.5", "12", "980", "1.8", "4999", "1499", "40", "67"):   # 67: calls peer gap (4 vs 12)
        assert n in fs.allowed_numbers, n


# --------------------------------------------------------------------------- robustness


@pytest.mark.parametrize("args", [
    (None, {}, {}, None),
    ({}, None, None, None),
    (None, {"identity": None, "performance": "junk", "offers": "x"}, {"payload": None}, None),
    ([], [], [], []),
    ({"digest": [None, 3, "x"], "peer_stats": None, "seasonal_beats": [{"month_range": None}]},
     {"merchant_id": "m", "performance": {"views": "1,234", "delta_7d": {"views_pct": "oops"}},
      "customer_aggregate": {"retention_6mo_pct": "0.4"}, "signals": [None, 5, "stale_posts:xd"],
      "conversation_history": [None, {"from": "merchant"}], "review_themes": [{"theme": None}]},
     {"kind": "perf_dip", "scope": "customer", "urgency": "high", "payload": {"placeholder": True}},
     {"identity": {"name": None}, "consent": None, "relationship": {"last_visit": "not-a-date"}}),
])
def test_never_raises_on_garbage(args):
    fs = build_facts(*args)
    assert isinstance(fs, FactSheet)
    for f in fs.all_facts():
        assert "_" not in f.text


def test_never_raises_on_randomly_damaged_contexts(seeds):
    rnd = random.Random(20260926)
    tids = sorted(seeds["triggers"])
    for i in range(300):
        tid = tids[i % len(tids)]
        t = copy.deepcopy(seeds["triggers"][tid])
        m = copy.deepcopy(seeds["merchants"][t["merchant_id"]])
        cat = copy.deepcopy(seeds["categories"][m["category_slug"]])
        c = copy.deepcopy(seeds["customers"].get(t.get("customer_id") or ""))
        for obj in (t, m, cat, c, t.get("payload"), m.get("identity"), m.get("performance")):
            if isinstance(obj, dict) and obj:
                key = rnd.choice(sorted(obj))
                obj[key] = rnd.choice([None, "", [], {}, 0, -1, "junk", True])
        fs = build_facts(cat, m, t, c, now=rnd.choice(NOWS))
        assert isinstance(fs, FactSheet)
        assert fs.anchor_facts
        for f in fs.all_facts():
            assert "_" not in f.text
            for tok in extract_number_tokens(f.text):
                assert normalize_number(tok) in fs.allowed_numbers
