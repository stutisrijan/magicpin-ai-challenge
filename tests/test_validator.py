"""Tests for app/validator.py: the last gate before any body leaves the bot.

Each issue type is exercised with a failing and a passing example, the number-grounding
rule (spec section 10) is checked token by token, and the real template drafts for every
seed and expanded trigger must pass untouched (skipped until templates.render exists).
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.numbers import number_variants
from app.schemas import SEND_AS_MERCHANT, SEND_AS_VERA, Fact, FactSheet
from app.validator import (
    MAX_CHARS,
    extract_number_tokens,
    issue_code,
    normalize_for_compare,
    normalize_number,
    ungrounded_numbers,
    validate_body,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


# --------------------------------------------------------------------------- helpers


def _allowed(*values) -> set[str]:
    out: set[str] = set()
    for v in values:
        out |= number_variants(v)
    return out


def _sheet(**overrides) -> FactSheet:
    """A dentist FactSheet in the shape facts.py builds (numbers from the Dr. Meera seed)."""
    base = dict(
        kind="research_digest",
        category_slug="dentists",
        trigger_id="trg_001",
        merchant_id="m_001",
        salutation="Dr. Meera",
        recipient_name="Dr. Meera",
        owner_name="Dr. Meera",
        merchant_name="Dr. Meera's Dental Clinic",
        locality="Lajpat Nagar",
        city="Delhi",
        anchor_facts=[
            Fact("digest.source", "JIDA Oct 2026, p.14", weight=3),
            Fact("digest.stat", "38% lower caries recurrence with 3-month vs 6-month recall", weight=3),
            Fact("digest.trial_n", "2,100-patient trial", weight=3),
        ],
        support_facts=[
            Fact("perf.ctr_vs_peer", "CTR 2.1% vs 3.0% peer average (30% below peers)", weight=2),
            Fact("agg.high_risk_adult_count", "124 high-risk adult patients", weight=2),
        ],
        offers_active=["Dental Cleaning @ ₹299"],
        catalog_offers=["Teeth Whitening @ ₹1,499"],
        taboos=["guaranteed", "100% safe", "best in city", "FDA-approved (use only when actually applicable)"],
        allowed_numbers=_allowed(2026, 14, 38, 3, 6, 2100, 2.1, 3.0, 30, 124, 299, 1499),
    )
    base.update(overrides)
    return FactSheet(**base)


GOOD = ("Dr. Meera, JIDA Oct 2026, p.14 has a finding for your 124 high-risk adult patients: 38% lower "
        "caries recurrence with 3-month recall (2,100-patient trial). Want me to draft a patient note on it?")


def _codes(issues: list[str]) -> set[str]:
    return {issue_code(i) for i in issues}


def _check(body: str, sheet: FactSheet | None = None, **kw) -> set[str]:
    return _codes(validate_body(body, sheet or _sheet(), **kw))


# --------------------------------------------------------------------------- clean copy


def test_clean_body_passes():
    assert validate_body(GOOD, _sheet()) == []


def test_clean_body_passes_in_every_mode():
    reply = "Done, Dr. Meera: here is the draft note for your 124 high-risk adult patients. Confirm and I'll send it."
    assert validate_body(reply, _sheet(), mode="reply") == []
    assert validate_body(reply, _sheet(), mode="reply_action") == []


def test_hinglish_body_with_slots_and_offer_passes_customer_facing():
    sheet = _sheet(
        kind="recall_due", scope="customer", send_as=SEND_AS_MERCHANT, salutation="Priya", recipient_name="Priya",
        slots=["Wed 5 Nov, 6pm", "Thu 6 Nov, 5pm"], allowed_numbers=_allowed(5, 6, 299, 12, 2026),
    )
    body = ("Hi Priya, Dr. Meera's Dental Clinic se reminder: aapka 6-month cleaning due hai (12 Nov 2026 tak). "
            "Slots: Wed 5 Nov, 6pm ya Thu 6 Nov, 5pm. Dental Cleaning @ ₹299 apply hoga. "
            "Wed ke liye 1, Thu ke liye 2 reply karein.")
    assert validate_body(body, sheet, customer_facing=True) == []


# --------------------------------------------------------------------------- length


def test_empty_and_whitespace_bodies():
    assert _check("") == {"empty"}
    assert _check("   \n ") == {"empty"}
    assert _check(None) == {"empty"}          # type: ignore[arg-type]


def test_too_short():
    assert "too_short" in _check("Dr. Meera, quick one?")


def test_too_long_limits_depend_on_mode():
    filler = " Your clinic keeps doing well with patients in Lajpat Nagar."
    body = "Dr. Meera," + filler * 17                     # ~1,000 chars
    assert 900 < len(body) < 1100
    assert "too_long" in _check(body, mode="compose")
    assert "too_long" in _check(body, mode="reply")
    assert "too_long" not in _check(body, mode="reply_action")
    assert MAX_CHARS == {"compose": 900, "reply": 700, "reply_action": 1100}


# --------------------------------------------------------------------------- links


@pytest.mark.parametrize("link", [
    "https://magicpin.in/deal", "http://bit.ly/x", "www.meeraclinic.com", "magicpin.in", "book at mysalon.com today",
    "wa.me/919999999999", "write to help@clinic.co.in",
])
def test_urls_and_web_addresses_are_flagged(link):
    assert "url" in _check(f"Dr. Meera, 38% lower caries recurrence per JIDA Oct 2026, p.14: {link}. Want the note?")


@pytest.mark.parametrize("text", [
    "JIDA Oct 2026, p.14", "Dr. Meera", "Olaplex No.3 treatment", "e.g. scaling", "a 3.0 rating", "done.In short",
])
def test_non_urls_are_not_flagged(text):
    assert "url" not in _check(f"Dr. Meera, a quick note on {text} for your clinic this week. Want me to draft it?")


# --------------------------------------------------------------------------- taboo


def test_taboo_words_case_insensitive():
    assert "taboo" in _check(GOOD.replace("Want me", "Results are Guaranteed. Want me"))


def test_taboo_parenthetical_note_is_stripped():
    issues = validate_body(GOOD.replace("Want me", "It is FDA-approved. Want me"), _sheet())
    assert any(issue_code(i) == "taboo" and "FDA-approved" in i for i in issues)


def test_taboo_multiword_and_word_boundaries():
    assert "taboo" in _check(GOOD.replace("Want me", "Rated best  in city. Want me"))
    # a taboo inside a longer word is not a hit
    assert "taboo" not in _check(GOOD.replace("Want me", "No unguaranteedness here. Want me"))
    sheet = _sheet(taboos=["cure"])
    assert "taboo" not in _check(GOOD.replace("Want me", "Keep records secure. Want me"), sheet)
    assert "taboo" in _check(GOOD.replace("Want me", "It will cure decay. Want me"), sheet)


def test_empty_taboo_entries_are_ignored():
    assert validate_body(GOOD, _sheet(taboos=["", "   ", "(note only)"])) == []


# --------------------------------------------------------------------------- numbers


def test_ungrounded_percentage_and_price_are_flagged():
    issues = validate_body(GOOD.replace("38%", "47%"), _sheet())
    assert any(issue_code(i) == "ungrounded_number" and "47%" in i for i in issues)
    assert "ungrounded_number" in _check(GOOD.replace("Want me", "Whitening now ₹999. Want me"))
    assert "ungrounded_number" not in _check(GOOD.replace("Want me", "Cleaning is ₹299. Want me"))


def test_grounded_number_formats_normalise():
    allowed = _allowed(2100, 2.1, 1499, 299)
    assert ungrounded_numbers("2,100 patients, CTR 2.10%, ₹1,499 and Rs. 299", allowed) == []


@pytest.mark.parametrize("text", ["3 quick steps", "0 cost", "10 slots", "2 options"])
def test_small_plain_integers_are_allowed(text):
    assert ungrounded_numbers(text, set()) == []


@pytest.mark.parametrize("text", ["3%", "₹5", "2x", "Rs 8"])
def test_small_marked_numbers_must_be_grounded(text):
    assert ungrounded_numbers(text, set()) != []


@pytest.mark.parametrize("text", ["in 15 min", "for 20 minutes", "within 24 hours", "30 mins", "45 minutes",
                                  "48 hrs", "60 seconds", "90 min", "15-min call", "24 ghante"])
def test_common_durations_with_time_units_are_allowed(text):
    assert ungrounded_numbers(text, set()) == []


@pytest.mark.parametrize("text", ["15 days", "17 min", "30 patients", "45%", "99 hours"])
def test_other_numbers_need_grounding(text):
    assert ungrounded_numbers(text, set()) != []


@pytest.mark.parametrize("text", ["at 6pm", "7:30pm tonight", "open 24x7", "open 24/7", "7-9pm window", "at 18:00",
                                  "10 am slot"])
def test_clock_times_are_not_number_checked(text):
    assert ungrounded_numbers(text, set()) == []


def test_ids_glued_to_letters_are_skipped():
    assert ungrounded_numbers("batch AT2024-1102 and slot W17", set()) == []


def test_number_helpers_are_reexported():
    assert extract_number_tokens("₹1,499 and 38%") == ["₹1,499", "38%"]
    assert normalize_number("₹1,499") == "1499"
    assert normalize_number("2.10") == "2.1"


# --------------------------------------------------------------------------- jargon


@pytest.mark.parametrize("term", ["ctr_below_peer_median", "stale_posts", "research_digest", "perf.ctr",
                                  "customer.last_visit", "the payload", "context_id", "suppression rules",
                                  "this trigger", "Triggers fired"])
def test_internal_jargon_is_flagged(term):
    assert "jargon" in _check(GOOD.replace("Want me", f"Note: {term} applies. Want me"))


def test_trigger_as_a_verb_is_fine():
    assert "jargon" not in _check(GOOD.replace("Want me", "Sugary drinks can trigger decay. Want me"))


def test_unfilled_template_placeholders_are_flagged():
    assert "unfilled_placeholder" in _check(GOOD.replace("Want me", "Generic for {molecule} is out. Want me"))
    assert "unfilled_placeholder" in _check(GOOD.replace("Want me", "Price: undefined. Want me"))


# --------------------------------------------------------------------------- openers


@pytest.mark.parametrize("opener", [
    "Hope you're doing well, Dr. Meera!", "Dr. Meera, I hope all is fine.", "Dr. Meera, I am reaching out because",
    "Dr. Meera, I'm reaching out since", "Greetings Dr. Meera,", "Dr. Meera, warm greetings from us.",
    "Dr. Meera, hope all is well.",
])
def test_preamble_openers_are_flagged(opener):
    assert "preamble" in _check(f"{opener} JIDA Oct 2026, p.14 reports 38% lower caries recurrence. Want the note?")


def test_hope_late_in_the_message_is_not_a_preamble():
    body = GOOD[:-1] + " I hope it is useful for your recall planning this month?"
    assert len(GOOD) > 140
    assert "preamble" not in _check(body)


@pytest.mark.parametrize("intro", ["I'm Vera", "I am Vera", "This is Vera", "Vera here"])
def test_self_intro_flagged_only_in_reply_modes(intro):
    body = f"Dr. Meera, {intro}: JIDA Oct 2026, p.14 reports 38% lower caries recurrence. Here's the draft note."
    assert "self_intro" in _check(body, mode="reply")
    assert "self_intro" in _check(body, mode="reply_action")
    assert "self_intro" not in _check(body, mode="compose")


@pytest.mark.parametrize("brand", ["Vera", "magicpin", "MagicPin", "magic pin"])
def test_customer_facing_brand_mentions(brand):
    sheet = _sheet(scope="customer", send_as=SEND_AS_MERCHANT, salutation="Priya", recipient_name="Priya")
    body = f"Hi Priya, Dr. Meera's Dental Clinic here via {brand}. Your cleaning is due soon. Reply YES to book."
    assert "brand_mention" in _check(body, sheet, customer_facing=True)
    assert "brand_mention" not in _check(body, sheet, customer_facing=False)


def test_dr_dr_is_flagged():
    assert "dr_dr" in _check(GOOD.replace("Dr. Meera", "Dr. Dr. Meera"))
    assert "dr_dr" in _check(GOOD.replace("Dr. Meera", "Dr Dr Meera"))
    assert "dr_dr" not in _check(GOOD.replace("Dr. Meera", "Dr. Drishti"), _sheet(salutation="Dr. Drishti"))


# --------------------------------------------------------------------------- addressee


def test_missing_name_in_compose_mode():
    body = "JIDA Oct 2026, p.14 reports 38% lower caries recurrence with 3-month recall. Want me to draft a note?"
    assert "missing_name" in _check(body)
    assert "missing_name" not in _check(body, mode="reply")


def test_name_must_appear_early():
    late = ("JIDA Oct 2026, p.14 reports 38% lower caries recurrence with 3-month recall, and Dr. Meera, "
            "that is your cohort. Want me to draft a note?")
    assert late.index("Meera") > 80
    assert "missing_name" in _check(late)
    assert "missing_name" not in _check("Quick one for your calendar, Dr. Meera: 38% lower caries recurrence "
                                        "with 3-month recall. Want the note?")


def test_first_name_alone_satisfies_the_name_check():
    assert "missing_name" not in _check("Meera ji, 38% lower caries recurrence with 3-month recall per "
                                        "JIDA Oct 2026, p.14. Want the note?")


def test_no_name_on_file_skips_the_name_check():
    sheet = _sheet(salutation="", recipient_name="", owner_name="", about_name="")
    assert "missing_name" not in _check("Namaste, 38% lower caries recurrence with 3-month recall is worth a look. "
                                        "Want the note?", sheet)
    placeholder = _sheet(salutation="there", recipient_name="there", owner_name="there")
    assert "missing_name" not in _check("Quick update: 38% lower caries recurrence with 3-month recall is worth "
                                        "a look. Want the note?", placeholder)


def test_relay_customer_can_be_addressed_by_either_name():
    sheet = _sheet(scope="customer", send_as=SEND_AS_MERCHANT, salutation="Sneha", recipient_name="Sneha",
                   about_name="Aanya", allowed_numbers=_allowed(6))
    to_parent = "Hi Sneha, Aanya's 6-month check-up at Dr. Meera's Dental Clinic is due. Reply YES to book a time."
    assert "missing_name" not in _check(to_parent, sheet, customer_facing=True)
    assert "missing_name" in _check("Hi, a check-up at Dr. Meera's Dental Clinic is due soon for your family. "
                                    "Reply YES to book a time.", sheet, customer_facing=True)


# --------------------------------------------------------------------------- asks


def test_multiple_reply_directives_are_flagged():
    body = GOOD[:-1] + ". Reply YES for the note. Reply YES again for the checklist."
    assert "multiple_ctas" in _check(body)
    mixed = GOOD[:-1] + ". Reply YES for the note, or reply 1 for the checklist."
    assert "multiple_ctas" in _check(mixed)
    hinglish = GOOD[:-1] + ". Bas YES reply karein. Ya reply YES for the checklist."
    assert "multiple_ctas" in _check(hinglish)


def test_single_directive_and_slot_lists_are_one_ask():
    assert "multiple_ctas" not in _check(GOOD[:-1] + ". Reply YES and I'll send it.")
    sheet = _sheet(scope="customer", send_as=SEND_AS_MERCHANT, salutation="Priya", recipient_name="Priya",
                   slots=["Wed 5 Nov, 6pm", "Thu 6 Nov, 5pm"], allowed_numbers=_allowed(5, 6))
    slots = "Hi Priya, your cleaning is due. Reply 1 for Wed 5 Nov, 6pm or 2 for Thu 6 Nov, 5pm."
    assert validate_body(slots, sheet, customer_facing=True) == []
    # STOP / NO are opt-out paths, not asks
    assert "multiple_ctas" not in _check(GOOD[:-1] + ". Reply YES to get it, or reply NO to skip.")


def test_question_marks_limit():
    assert "too_many_questions" not in _check(GOOD.replace("Want me", "Seen it? Want me"))
    assert "too_many_questions" in _check(GOOD.replace("Want me", "Seen it? Useful? Want me"))


# --------------------------------------------------------------------------- repetition


def test_verbatim_repeat_is_case_and_whitespace_insensitive():
    prior = GOOD.upper().replace(" ", "   ")
    assert "repeat" in _check(GOOD, prior_bodies=[prior])
    assert "repeat" in _check(GOOD, prior_bodies=("different text", GOOD.rstrip("?") + " ?"))
    assert "repeat" not in _check(GOOD, prior_bodies=[GOOD.replace("patient note", "patient WhatsApp")])
    assert normalize_for_compare("Hi,  Dr. Meera!") == normalize_for_compare("hi dr meera")


# --------------------------------------------------------------------------- action mode


@pytest.mark.parametrize("phrase", ["would you", "do you", "can you tell", "what if", "how about"])
def test_reply_action_forbids_qualifying_phrases(phrase):
    body = f"Done, Dr. Meera: here is the draft note. {phrase.capitalize()} want it sent today to all patients?"
    assert "qualifying" in _check(body, mode="reply_action")
    assert "qualifying" not in _check(body, mode="reply")


def test_reply_action_catches_substrings_like_the_judge():
    # the local judge does a plain substring match, so "do your" counts as "do you"
    body = "Here is the draft, Dr. Meera. I will do your recall list next and confirm by message."
    assert "qualifying" in _check(body, mode="reply_action")


def test_reply_action_needs_an_action_word():
    body = "Dr. Meera, the patient note covers the 38% finding and the 3-month recall for your adults."
    assert "no_action_word" in _check(body, mode="reply_action")
    assert "no_action_word" not in _check(body + " Sending it now.", mode="reply_action")


# --------------------------------------------------------------------------- robustness


def test_validator_never_raises_on_odd_inputs():
    assert isinstance(validate_body(GOOD, None), list)                     # type: ignore[arg-type]
    assert isinstance(validate_body(GOOD, FactSheet(), prior_bodies=[None, 3]), list)  # type: ignore[list-item]
    assert isinstance(validate_body(GOOD, _sheet(), mode="weird"), list)
    assert validate_body(123, _sheet()) == ["empty: the message body is empty"]       # type: ignore[arg-type]


def test_issue_strings_carry_a_code_and_a_hint():
    for issue in validate_body("Hope you are well! Visit www.x.com for 99% off!!", _sheet()):
        code, _, detail = issue.partition(":")
        assert code and detail.strip()


# --------------------------------------------------------------------------- real template drafts


def _load_dir(path: Path, key: str) -> dict[str, dict]:
    return {obj[key]: obj for obj in (json.loads(f.read_text()) for f in sorted(path.glob("*.json")))}


def _real_modules():
    templates = importlib.import_module("app.templates")
    playbooks = importlib.import_module("app.playbooks")
    if not hasattr(templates, "render") or not hasattr(playbooks, "get_playbook"):
        pytest.skip("templates.render / playbooks.get_playbook not available yet")
    return templates, playbooks


def _drafts(cats: dict, merchants: dict, customers: dict, triggers: dict, now: str):
    from app.facts import build_facts

    templates, playbooks = _real_modules()
    for tid, trg in sorted(triggers.items()):
        payload = trg.get("payload") if isinstance(trg.get("payload"), dict) else {}
        merchant = merchants.get(trg.get("merchant_id") or payload.get("merchant_id"))
        if merchant is None:
            continue
        customer = customers.get(trg.get("customer_id")) if trg.get("customer_id") else None
        facts = build_facts(cats.get(merchant.get("category_slug")), merchant, trg, customer, now=now)
        pb = playbooks.get_playbook(facts.kind, facts.scope, facts.category_slug)
        yield tid, facts, templates.render(facts, pb)


@pytest.fixture(scope="module")
def seed_world() -> tuple[dict, dict, dict, dict]:
    cats = _load_dir(DATASET / "categories", "slug")
    merchants = {m["merchant_id"]: m for m in json.loads((DATASET / "merchants_seed.json").read_text())["merchants"]}
    customers = {c["customer_id"]: c for c in json.loads((DATASET / "customers_seed.json").read_text())["customers"]}
    triggers = {t["id"]: t for t in json.loads((DATASET / "triggers_seed.json").read_text())["triggers"]}
    return cats, merchants, customers, triggers


@pytest.fixture(scope="module")
def expanded_world(tmp_path_factory) -> tuple[dict, dict, dict, dict]:
    out = tmp_path_factory.mktemp("expanded_validator")
    subprocess.run([sys.executable, str(DATASET / "generate_dataset.py"), "--seed-dir", str(DATASET),
                    "--out", str(out)], check=True, capture_output=True, cwd=str(ROOT))
    return (_load_dir(out / "categories", "slug"), _load_dir(out / "merchants", "merchant_id"),
            _load_dir(out / "customers", "customer_id"), _load_dir(out / "triggers", "id"))


@pytest.mark.parametrize("now", ["2026-04-26T10:30:00Z", "2026-09-26T10:30:00Z"])
def test_seed_template_drafts_pass_the_validator(seed_world, now):
    failures = {}
    for tid, facts, draft in _drafts(*seed_world, now=now):
        issues = validate_body(draft.body, facts, customer_facing=facts.send_as == SEND_AS_MERCHANT)
        if issues:
            failures[tid] = (issues, draft.body)
    assert failures == {}


def test_expanded_template_drafts_pass_the_validator(expanded_world):
    failures, seen = {}, 0
    for tid, facts, draft in _drafts(*expanded_world, now="2026-04-26T10:30:00Z"):
        seen += 1
        assert facts.send_as in (SEND_AS_VERA, SEND_AS_MERCHANT)
        issues = validate_body(draft.body, facts, customer_facing=facts.send_as == SEND_AS_MERCHANT)
        if issues:
            failures[tid] = (issues, draft.body)
    assert seen >= 90
    assert failures == {}
