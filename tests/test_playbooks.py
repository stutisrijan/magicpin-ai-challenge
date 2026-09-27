"""Tests for app/playbooks.py: one playbook per trigger kind, specialised by scope and category."""

from __future__ import annotations

import re

import pytest

from app.playbooks import CUSTOMER_KINDS, KIND_ALIASES, KNOWN_KINDS, canonical_kind, category_key, get_playbook
from app.schemas import (
    CTA_BINARY_CONFIRM,
    CTA_BINARY_YES_NO,
    CTA_MULTI_CHOICE_SLOT,
    CTA_OPEN_ENDED,
    CTA_VALUES,
    Playbook,
)

# every kind named in the design spec (seed kinds, generated kinds, brief kinds)
SPEC_KINDS = {
    "research_digest", "regulation_change", "cde_opportunity", "supply_alert", "category_seasonal",
    "category_trend_movement", "festival_upcoming", "ipl_match_today", "weather_heatwave", "local_news_event",
    "competitor_opened", "perf_dip", "perf_spike", "seasonal_perf_dip", "milestone_reached", "review_theme_emerged",
    "renewal_due", "winback_eligible", "dormant_with_vera", "gbp_unverified", "curious_ask_due",
    "scheduled_recurring", "active_planning_intent", "recall_due", "customer_lapsed_soft", "customer_lapsed_hard",
    "appointment_tomorrow", "chronic_refill_due", "trial_followup", "wedding_package_followup",
    "unplanned_slot_open",
}
CATEGORIES = [None, "dentists", "salons", "restaurants", "gyms", "pharmacies", "yoga_studio", "unknown_vertical"]
_SNAKE = re.compile(r"\b[a-z]+_[a-z_]+\b")


def _default_scope(kind: str) -> str:
    return "customer" if kind in CUSTOMER_KINDS else "merchant"


def test_known_kinds_cover_the_spec():
    assert SPEC_KINDS <= KNOWN_KINDS
    assert CUSTOMER_KINDS <= KNOWN_KINDS


@pytest.mark.parametrize("kind", sorted(SPEC_KINDS))
@pytest.mark.parametrize("category", CATEGORIES)
def test_every_kind_has_a_complete_playbook(kind, category):
    pb = get_playbook(kind, _default_scope(kind), category)
    assert isinstance(pb, Playbook)
    assert pb.kind == kind
    assert pb.goal and pb.framing and pb.offer and pb.cta_hint
    assert pb.levers and all(isinstance(x, str) and x for x in pb.levers)
    assert pb.cta in CTA_VALUES
    assert pb.family in {"knowledge", "compliance", "performance", "event", "lifecycle", "planning", "curiosity",
                         "customer"}
    assert pb.customer_facing == (kind in CUSTOMER_KINDS)
    prefix = "merchant_" if pb.customer_facing else "vera_"
    assert pb.template_name.startswith(prefix) and re.fullmatch(r"[a-z0-9_]+_v\d+", pb.template_name)
    # the offer is shown back to merchants in action mode: plain words only
    assert not _SNAKE.search(pb.offer) and "_" not in pb.offer
    assert "invent" in pb.framing.lower() or "guess" in pb.framing.lower()   # no-fabrication clause
    assert len(pb.framing) > 120                                               # real writer instructions


def test_customer_rules_in_customer_framing():
    for kind in CUSTOMER_KINDS:
        pb = get_playbook(kind, "customer", "dentists")
        low = pb.framing.lower()
        assert "vera" in low and "magicpin" in low          # the instruction to never mention them
        assert "active offers" in low
        assert "medical claims" in low


def test_cta_shapes_follow_the_kind():
    assert get_playbook("curious_ask_due").cta == CTA_OPEN_ENDED
    assert get_playbook("recall_due", "customer", "dentists").cta == CTA_MULTI_CHOICE_SLOT
    assert get_playbook("appointment_tomorrow", "customer", "salons").cta == CTA_BINARY_CONFIRM
    assert get_playbook("chronic_refill_due", "customer", "pharmacies").cta == CTA_BINARY_CONFIRM
    assert get_playbook("research_digest", "merchant", "dentists").cta == CTA_BINARY_YES_NO


def test_judgment_calls_are_in_the_framing():
    ipl = get_playbook("ipl_match_today", "merchant", "restaurants").framing.lower()
    assert "weeknight" in ipl and "delivery" in ipl and "promo" in ipl
    dip = get_playbook("seasonal_perf_dip", "merchant", "gyms").framing.lower()
    assert "retention" in dip or "retaining" in dip
    supply = get_playbook("supply_alert", "merchant", "pharmacies").framing.lower()
    assert "batch" in supply and "affected count" in supply
    comp = get_playbook("competitor_opened", "merchant", "dentists").framing.lower()
    assert "price war" in comp and "invent" in comp
    plan = get_playbook("active_planning_intent", "merchant", "restaurants").framing.lower()
    assert "qualifying" in plan and "draft" in plan
    milestone = get_playbook("milestone_reached").framing.lower()
    assert "round" in milestone
    festival = get_playbook("festival_upcoming").framing.lower()
    assert "never name one" in festival


def test_recall_specialised_per_category():
    dent = get_playbook("recall_due", "customer", "dentists")
    gym = get_playbook("recall_due", "customer", "gyms")
    salon = get_playbook("recall_due", "customer", "salons")
    assert len({dent.framing, gym.framing, salon.framing}) == 3
    assert len({dent.offer, gym.offer, salon.offer}) == 3
    assert "not a medical recall" in gym.framing.lower()


def test_chronic_refill_outside_pharmacies_is_a_check_in():
    pharm = get_playbook("chronic_refill_due", "customer", "pharmacies")
    dent = get_playbook("chronic_refill_due", "customer", "dentists")
    gym = get_playbook("chronic_refill_due", "customer", "gyms")
    assert pharm.cta == CTA_BINARY_CONFIRM and "dispatch" in pharm.offer
    assert dent.cta == CTA_BINARY_YES_NO and "check" in dent.framing.lower()
    assert "medicine" not in dent.offer.lower()
    assert "monthly plan" in gym.offer and "membership" in gym.framing
    assert dent.template_name != pharm.template_name


def test_unknown_kind_gets_generic_playbooks():
    m = get_playbook("stock_low", "merchant", "pharmacies")
    c = get_playbook("stock_low", "customer", "pharmacies")
    assert m.kind == "stock_low" and not m.customer_facing and m.template_name.startswith("vera_")
    assert c.customer_facing and c.template_name.startswith("merchant_")
    assert "stock low" in m.framing and "_" not in m.framing.split("'")[0]
    for pb in (m, c):
        assert pb.cta in CTA_VALUES and pb.offer


def test_scope_mismatch_is_adapted():
    # a customer kind delivered to the merchant: offer to send it on their behalf
    pb = get_playbook("recall_due", "merchant", "dentists")
    assert not pb.customer_facing and pb.template_name.startswith("vera_")
    assert "behalf" in pb.offer and "_" not in pb.offer
    # a merchant kind with a customer scope: speak as the business
    pb = get_playbook("festival_upcoming", "customer", "salons")
    assert pb.customer_facing and pb.template_name.startswith("merchant_")


def test_aliases_and_normalisation():
    assert canonical_kind("research_digest_release") == "research_digest"
    assert canonical_kind("bridal_followup") == "wedding_package_followup"
    assert canonical_kind("  Recall_Due ") == "recall_due"
    assert canonical_kind(None) == "generic"
    for alias, target in KIND_ALIASES.items():
        assert target in KNOWN_KINDS, alias
        assert get_playbook(alias).kind == target
    assert category_key("Dental Clinic") == "dentists"
    assert category_key("yoga_studio") == "gyms"
    assert category_key("Beauty Parlour") == "salons"
    assert category_key("") == ""


def test_returns_independent_copies():
    a = get_playbook("perf_dip", "merchant", "salons")
    a.levers.append("mutated")
    a.framing = "changed"
    b = get_playbook("perf_dip", "merchant", "salons")
    assert "mutated" not in b.levers and b.framing != "changed"


def test_never_raises_on_odd_input():
    for kind in (None, "", "   ", "💥", "x" * 500):
        pb = get_playbook(kind, None, None)  # type: ignore[arg-type]
        assert isinstance(pb, Playbook) and pb.cta in CTA_VALUES
