"""Tests for app/language.py: detection, merchant/customer plans, per-turn mirroring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.language import (
    customer_language_plan,
    detect_message_language,
    merchant_language_plan,
    plan_for_code,
    plan_for_reply,
)
from app.schemas import LanguagePlan

DATASET = Path(__file__).resolve().parent.parent / "dataset"


@pytest.fixture(scope="module")
def merchants() -> dict[str, dict]:
    data = json.loads((DATASET / "merchants_seed.json").read_text())
    return {m["merchant_id"]: m for m in data["merchants"]}


@pytest.fixture(scope="module")
def customers() -> dict[str, dict]:
    data = json.loads((DATASET / "customers_seed.json").read_text())
    return {c["customer_id"]: c for c in data["customers"]}


# --------------------------------------------------------------------------- detection


@pytest.mark.parametrize("text, expected", [
    ("Yes please send the abstract", "en"),
    ("Not interested. Stop messaging me.", "en"),
    ("Btw can you also help me with my GST filing this month?", "en"),
    ("Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly.", "en"),
    ("haan bhej do", "hinglish"),
    ("Mujhe magicpin judrna hai", "hinglish"),
    ("ok karo", "hinglish"),
    ("Theek hai, kal baat karte hain", "hinglish"),
    ("Aapki jaankari ke liye bahut-bahut shukriya. Main aapki yeh sabhi baatein aur sujhaav hamari "
     "team tak pahuncha deti hoon.", "hinglish"),
    ("abhi nahi, baad mein", "hinglish"),
    ("ठीक है, भेज दीजिए", "hi"),
    ("ok ठीक है", "hi"),
    ("मला हे हवे आहे, धन्यवाद", "mr"),
    ("சரி, நன்றி", "ta"),
    ("సరే ధన్యవాదాలు", "te"),
    ("ಸರಿ ಧನ್ಯವಾದಗಳು", "kn"),
    ("", "en"),
    ("   ", "en"),
    ("ok", "en"),
    ("1", "en"),
    ("👍", "en"),
])
def test_detect_message_language(text, expected):
    assert detect_message_language(text) == expected


def test_detect_handles_non_strings():
    assert detect_message_language(None) == "en"  # type: ignore[arg-type]
    assert detect_message_language(123) == "en"   # type: ignore[arg-type]


def test_english_words_that_look_hindi_do_not_flip():
    # "tab", "jab", "band", "main", "do" are common English words and must not count as Hinglish
    assert detect_message_language("Please do check the main tab and the band schedule") == "en"


# --------------------------------------------------------------------------- merchant plans


def test_merchant_mirrors_latest_english_message(merchants):
    # Dr. Meera speaks en+hi, but her latest message is English
    plan = merchant_language_plan(merchants["m_001_drmeera_dentist_delhi"])
    assert plan.code == "en"


def test_merchant_without_replies_and_hindi_gets_hinglish(merchants):
    plan = merchant_language_plan(merchants["m_002_bharat_dentist_mumbai"])  # only Vera turns in history
    assert plan.code == "hinglish"
    assert plan.script == "latin"
    assert "Hinglish" in plan.instruction


def test_merchant_history_param_overrides_stored_history(merchants):
    m = merchants["m_001_drmeera_dentist_delhi"]
    plan = merchant_language_plan(m, [{"from": "merchant", "body": "haan theek hai, bhej do"}])
    assert plan.code == "hinglish"
    plan = merchant_language_plan(m, [{"role": "merchant", "body": "ठीक है भेज दीजिए"}])
    assert plan.code == "hi" and plan.script == "devanagari"


def test_merchant_short_reply_falls_back_to_identity(merchants):
    m = merchants["m_002_bharat_dentist_mumbai"]
    plan = merchant_language_plan(m, [{"from": "merchant", "body": "ok"}])
    assert plan.code == "hinglish"


def test_merchant_ignores_bot_turns():
    m = {"identity": {"languages": ["en"]}}
    plan = merchant_language_plan(m, [{"role": "bot", "body": "haan ji, bhej diya hai aapko"}])
    assert plan.code == "en"


@pytest.mark.parametrize("langs, code, greeting", [
    (["en", "ta"], "ta-en", "Vanakkam"),
    (["en", "te"], "te-en", "Namaskaram"),
    (["en", "kn"], "kn-en", "Namaskara"),
    (["en", "mr"], "mr-en", "Namaskar"),
    (["en", "hi", "ta"], "hinglish", "Hi"),
    (["en"], "en", "Hi"),
    ([], "en", "Hi"),
    (["English", "Tamil"], "ta-en", "Vanakkam"),
])
def test_merchant_identity_languages(langs, code, greeting):
    plan = merchant_language_plan({"identity": {"languages": langs}, "conversation_history": []})
    assert plan.code == code
    assert plan.greeting == greeting
    if code.endswith("-en"):
        assert "Never write full sentences" in plan.instruction


def test_merchant_plan_is_defensive():
    assert merchant_language_plan(None).code == "en"  # type: ignore[arg-type]
    assert merchant_language_plan({"identity": None, "conversation_history": "junk"}).code == "en"


# --------------------------------------------------------------------------- customer plans


@pytest.mark.parametrize("pref, code, greeting", [
    ("hi-en mix", "hinglish", "Hi"),
    ("hi", "hi", "Namaste"),
    ("Hindi", "hi", "Namaste"),
    ("english", "en", "Hi"),
    ("en", "en", "Hi"),
    ("ta-en mix", "ta-en", "Vanakkam"),
    ("te-en mix", "te-en", "Namaskaram"),
    ("kn-en mix", "kn-en", "Namaskara"),
    ("mr-en mix", "mr-en", "Namaskar"),
    ("Tamil", "ta-en", "Vanakkam"),
    ("", "en", "Hi"),
    ("klingon", "en", "Hi"),
])
def test_customer_language_plan(pref, code, greeting):
    plan = customer_language_plan({"identity": {"language_pref": pref}})
    assert plan.code == code
    assert plan.greeting == greeting
    assert plan.script == "latin"


def test_customer_plan_seed_customers(customers):
    assert customer_language_plan(customers["c_001_priya_for_m001"]).code == "hinglish"
    assert customer_language_plan(customers["c_004_sneha_for_m003"]).code == "te-en"
    assert customer_language_plan(customers["c_012_karthik_jr_for_m008"]).code == "ta-en"
    hi_plan = customer_language_plan(customers["c_013_grandfather_for_m009"])
    assert hi_plan.code == "hi" and "Roman-script Hindi" in hi_plan.instruction
    assert customer_language_plan(customers["c_005_kavya_for_m003"]).code == "en"


def test_customer_plan_is_defensive():
    assert customer_language_plan(None).code == "en"  # type: ignore[arg-type]
    assert customer_language_plan({"identity": "junk"}).code == "en"


# --------------------------------------------------------------------------- per-turn mirroring


def test_plan_for_reply_switches_to_english():
    base = plan_for_code("hinglish")
    assert plan_for_reply(base, "Please send me the details now").code == "en"


def test_plan_for_reply_switches_to_hinglish():
    assert plan_for_reply(plan_for_code("en"), "haan theek hai bhej do").code == "hinglish"


def test_plan_for_reply_keeps_base_on_short_replies():
    base = plan_for_code("hinglish")
    for text in ("ok", "yes", "1", "", "👍"):
        assert plan_for_reply(base, text) is base


def test_plan_for_reply_devanagari_and_regional():
    deva = plan_for_reply(plan_for_code("hinglish"), "ठीक है, भेज दीजिए")
    assert deva.code == "hi" and deva.script == "devanagari"
    tamil = plan_for_reply(plan_for_code("en"), "சரி, நன்றி")
    assert tamil.code == "ta-en" and tamil.greeting == "Vanakkam"


def test_plan_for_reply_keeps_regional_english_base():
    base = plan_for_code("ta-en")
    assert plan_for_reply(base, "Sounds good, please go ahead with it") is base


def test_plan_for_reply_keeps_roman_hindi_customer_register():
    base = plan_for_code("hi")
    assert plan_for_reply(base, "haan ji theek hai") is base


def test_plan_for_reply_defensive():
    assert isinstance(plan_for_reply(None, "hello there friend"), LanguagePlan)  # type: ignore[arg-type]
    base = plan_for_code("en")
    assert plan_for_reply(base, None) is base  # type: ignore[arg-type]


def test_plan_for_code_unknown_is_english():
    assert plan_for_code("xx").code == "en"
    assert plan_for_code("").code == "en"
