"""Shared data contracts between modules.

    facts.py      builds  FactSheet (+ LanguagePlan via language.py)
    playbooks.py  returns Playbook
    templates.py  renders Draft from (FactSheet, Playbook)   -- deterministic, no LLM
    composer.py   turns (FactSheet, Playbook, Draft) into the final action dict (LLM-refined or Draft)
    validator.py  checks any body against the FactSheet

Every field has a default so partially-populated contexts never crash a module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# CTA vocabulary used across the bot (mirrors the testing brief examples).
CTA_BINARY_YES_NO = "binary_yes_no"
CTA_BINARY_CONFIRM = "binary_confirm_cancel"
CTA_OPEN_ENDED = "open_ended"
CTA_MULTI_CHOICE_SLOT = "multi_choice_slot"
CTA_NONE = "none"
CTA_VALUES = {CTA_BINARY_YES_NO, CTA_BINARY_CONFIRM, CTA_OPEN_ENDED, CTA_MULTI_CHOICE_SLOT, CTA_NONE}

SEND_AS_VERA = "vera"
SEND_AS_MERCHANT = "merchant_on_behalf"


@dataclass
class Fact:
    """One verifiable statement the message may use.

    `text` is merchant-readable (never snake_case / internal signal names).
    """

    key: str                 # stable id, e.g. "digest.title", "perf.ctr_vs_peer"
    text: str                # e.g. "CTR 2.1% vs 3.0% peer average"
    value: Any = None        # raw value when useful
    source: str = ""         # "trigger" | "merchant" | "category" | "customer" | "derived"
    weight: int = 1          # 3 = anchor / why-now, 2 = strong support, 1 = colour


@dataclass
class LanguagePlan:
    code: str = "en"                 # "en" | "hinglish" | "hi" | "ta-en" | "te-en" | "kn-en" | "mr-en"
    instruction: str = "Write in clear, simple English."
    greeting: str = "Hi"             # opener word that fits the language ("Hi", "Namaste", "Vanakkam", ...)
    script: str = "latin"            # always romanised for WhatsApp unless explicitly "devanagari"


@dataclass
class FactSheet:
    kind: str = ""
    scope: str = "merchant"          # "merchant" | "customer"
    send_as: str = SEND_AS_VERA
    category_slug: str = ""
    trigger_id: str = ""
    merchant_id: str = ""
    customer_id: str | None = None
    urgency: int = 1
    now_iso: str = ""                # simulated "now" if known

    # who / how to address
    salutation: str = ""             # "Dr. Meera", "Lakshmi", "Sneha" (addressee name as written in greeting)
    recipient_name: str = ""         # person reading the message
    about_name: str = ""             # for parent/son relays: the patient/member the message is about
    owner_name: str = ""             # merchant owner display, e.g. "Dr. Meera", "Karthik"
    merchant_name: str = ""          # "Dr. Meera's Dental Clinic"
    locality: str = ""
    city: str = ""
    signer: str = ""                 # customer-facing sign-off, e.g. "Dr. Meera's Dental Clinic"

    # facts
    anchor_facts: list[Fact] = field(default_factory=list)    # why-now facts from the trigger (+ resolved digest)
    support_facts: list[Fact] = field(default_factory=list)   # merchant/customer/category state facts
    offers_active: list[str] = field(default_factory=list)    # merchant's own active offers (quotable)
    catalog_offers: list[str] = field(default_factory=list)   # category catalog (only as suggestions)
    digest_item: dict | None = None
    content_item: dict | None = None                          # patient_content_library item, if relevant
    slots: list[str] = field(default_factory=list)            # human labels, e.g. "Wed 5 Nov, 6pm"
    seasonal_note: str = ""                                   # matching seasonal beat text, if any
    trend_note: str = ""                                      # most relevant trend signal text, if any
    history: list[dict] = field(default_factory=list)         # recent conversation turns {from, body, ts, engagement}
    notes: list[str] = field(default_factory=list)            # judgment hints for the writer (never shown verbatim)

    # style / safety
    language: LanguagePlan = field(default_factory=LanguagePlan)
    voice_tone: str = ""
    voice_register: str = ""
    vocab_allowed: list[str] = field(default_factory=list)
    taboos: list[str] = field(default_factory=list)
    tone_examples: list[str] = field(default_factory=list)
    allowed_numbers: set[str] = field(default_factory=set)    # normalised numeric tokens found in / derived from contexts
    payload_is_placeholder: bool = False
    consent_ok: bool = True                                   # customer-facing: has usable consent
    consent_reason: str = ""

    def all_facts(self) -> list[Fact]:
        return list(self.anchor_facts) + list(self.support_facts)


@dataclass
class Playbook:
    kind: str = "generic"
    family: str = "generic"          # knowledge | compliance | performance | event | lifecycle | planning | curiosity | customer
    customer_facing: bool = False
    goal: str = ""                   # what this message must achieve
    framing: str = ""                # writer instructions: hook, angle, judgment calls
    levers: list[str] = field(default_factory=list)   # compulsion levers to use
    cta: str = CTA_OPEN_ENDED
    cta_hint: str = ""               # shape of the closing ask
    offer: str = ""                  # concrete next thing Vera offers to do (drives action mode in replies)
    template_name: str = "vera_generic_v1"


@dataclass
class Draft:
    """A complete message candidate (deterministic template or LLM output)."""

    body: str = ""
    cta: str = CTA_OPEN_ENDED
    rationale: str = ""
    template_params: list[str] = field(default_factory=list)
    offer: str = ""                  # what we promised to do if they say yes
    source: str = "template"         # "template" | "llm"
