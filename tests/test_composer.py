"""Tests for app/composer.py and app/prompts.py.

Playbooks, templates and the LLM are replaced by fakes in sys.modules (the composer
imports them at call time), so these tests pass offline and independently of those
modules. facts.py is real. The integration tests at the end use the real playbooks and
templates and are skipped until templates.render exists.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

from app import composer
from app.facts import build_facts
from app.prompts import COMPOSE_SCHEMA, MAX_PROMPT_CHARS, compose_prompt, repair_prompt
from app.schemas import (
    CTA_BINARY_YES_NO,
    CTA_OPEN_ENDED,
    CTA_VALUES,
    SEND_AS_MERCHANT,
    SEND_AS_VERA,
    Draft,
    FactSheet,
    Playbook,
)
from app.validator import issue_code, validate_body

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
NOW = "2026-04-26T10:30:00Z"

MEERA = "m_001_drmeera_dentist_delhi"
BHARAT = "m_002_bharat_dentist_mumbai"
T_RESEARCH = "trg_001_research_digest_dentists"
T_RECALL = "trg_003_recall_due_priya"
T_PERF_DIP = "trg_004_perf_dip_bharat"
PRIYA = "c_001_priya_for_m001"

CONTRACT_KEYS = {"body", "cta", "send_as", "suppression_key", "rationale", "template_name", "template_params", "meta"}
META_KEYS = {"offer", "kind", "source", "language", "facts", "trigger_id", "merchant_id", "customer_id"}

# A strong, fully grounded rewrite of the research digest for Dr. Meera.
LLM_RESEARCH = ("Dr. Meera, JIDA Oct 2026, p.14 reports 38% lower caries recurrence with 3-month fluoride varnish "
                "recall in high-risk adults (2,100-patient trial). That is exactly your 124 high-risk adult patients. "
                "Want me to draft a patient WhatsApp note on it?")


# --------------------------------------------------------------------------- data


@pytest.fixture(scope="module")
def seeds() -> dict:
    cats = {json.loads(f.read_text())["slug"]: json.loads(f.read_text())
            for f in sorted((DATASET / "categories").glob("*.json"))}
    merchants = {m["merchant_id"]: m for m in json.loads((DATASET / "merchants_seed.json").read_text())["merchants"]}
    customers = {c["customer_id"]: c for c in json.loads((DATASET / "customers_seed.json").read_text())["customers"]}
    triggers = {t["id"]: t for t in json.loads((DATASET / "triggers_seed.json").read_text())["triggers"]}
    return {"categories": cats, "merchants": merchants, "customers": customers, "triggers": triggers}


def _args(seeds: dict, trigger_id: str, customer_id: str | None = None) -> tuple:
    trg = copy.deepcopy(seeds["triggers"][trigger_id])
    merchant = copy.deepcopy(seeds["merchants"][trg["merchant_id"]])
    category = copy.deepcopy(seeds["categories"][merchant["category_slug"]])
    cid = customer_id or trg.get("customer_id")
    customer = copy.deepcopy(seeds["customers"][cid]) if cid else None
    return category, merchant, trg, customer


# --------------------------------------------------------------------------- fakes


class Fakes:
    """Fake playbooks / templates modules; the knobs let a test change their behaviour."""

    def __init__(self) -> None:
        self.template_name: str | None = None      # force a playbook template_name
        self.pb_cta = CTA_BINARY_YES_NO
        self.render_raises = False
        self.render_calls = 0

    def playbooks(self) -> types.ModuleType:
        mod = types.ModuleType("app.playbooks")

        def get_playbook(kind: str, scope: str = "merchant", category_slug: str | None = None) -> Playbook:
            customer = scope == "customer"
            return Playbook(
                kind=kind or "generic", family="customer" if customer else "knowledge", customer_facing=customer,
                goal="Share the one reason for writing now.", framing="Lead with the anchor fact.",
                levers=["specificity", "effort_externalization"], cta=self.pb_cta,
                cta_hint="Want me to draft it?", offer="draft the next step",
                template_name=self.template_name or f"{'merchant' if customer else 'vera'}_{kind}_v1",
            )

        mod.get_playbook = get_playbook
        mod.KNOWN_KINDS = set()
        return mod

    def templates(self) -> types.ModuleType:
        mod = types.ModuleType("app.templates")

        def render(facts: FactSheet, pb: Playbook) -> Draft:
            self.render_calls += 1
            if self.render_raises:
                raise RuntimeError("template bug")
            name = facts.salutation or "there"
            lead = "; ".join(f.text for f in facts.anchor_facts[:2]) or "a quick update on your listing"
            hinglish = facts.language.code in ("hinglish", "hi")
            if facts.send_as == SEND_AS_MERCHANT:
                ask = ("Bas YES reply karein, hum aapke liye time book kar denge." if hinglish
                       else "Reply YES and we'll book a time that suits you.")
                body = f"Hi {name}, {facts.signer} here. {lead}. {ask}"
            else:
                ask = ("Kya main aapke liye next step draft kar doon?" if hinglish
                       else "Want me to draft the next step for you?")
                body = f"{name}, {lead}. {ask}"
            return Draft(body=body, cta=pb.cta, rationale="Template: anchored on the why-now fact.",
                         template_params=[name, lead, ask], offer=pb.offer)

        mod.render = render
        return mod


class FakeLLM:
    """Scripted LLM: each call pops the next response (the last one repeats); callables get the call index."""

    def __init__(self, responses: list[Any] | None = None, *, delay: float = 0.0, available: bool = True) -> None:
        self.responses = list(responses or [])
        self.delay = delay
        self._available = available
        self.calls: list[dict] = []

    def available(self) -> bool:
        return self._available

    def describe(self) -> str:
        return "fake-llm"

    async def complete_json(self, system: str, user: str, *, schema: dict | None = None, timeout: float | None = None,
                            max_tokens: int = 800, temperature: float = 0.0) -> Any:
        idx = len(self.calls)
        self.calls.append({"system": system, "user": user, "schema": schema, "timeout": timeout,
                           "temperature": temperature, "max_tokens": max_tokens})
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.responses:
            return None
        resp = self.responses[min(idx, len(self.responses) - 1)]
        return resp(idx) if callable(resp) else copy.deepcopy(resp)


def _llm_module(llm: FakeLLM) -> types.ModuleType:
    mod = types.ModuleType("app.llm")

    def extract_json(text: Any) -> dict | None:
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    mod.get_llm = lambda: llm
    mod.set_llm = lambda client: None
    mod.extract_json = extract_json
    return mod


@pytest.fixture(autouse=True)
def _clean_cache():
    composer.clear_cache()
    yield
    composer.clear_cache()


@pytest.fixture
def fakes(monkeypatch) -> Fakes:
    f = Fakes()
    monkeypatch.setitem(sys.modules, "app.playbooks", f.playbooks())
    monkeypatch.setitem(sys.modules, "app.templates", f.templates())
    monkeypatch.setitem(sys.modules, "app.llm", _llm_module(FakeLLM(available=False)))
    return f


@pytest.fixture
def use_llm(monkeypatch, fakes):
    """Install a FakeLLM with the given responses: llm = use_llm([...], delay=...)."""

    def install(responses: list[Any] | None = None, **kw) -> FakeLLM:
        llm = FakeLLM(responses, **kw)
        monkeypatch.setitem(sys.modules, "app.llm", _llm_module(llm))
        return llm

    return install


def _run(coro):
    return asyncio.run(coro)


def _resp(body: str, **extra) -> dict:
    out = {"body": body, "cta": CTA_BINARY_YES_NO, "rationale": "LLM: research digest, 38% finding, draft offer.",
           "used_facts": ["digest.source", "digest.stat"]}
    out.update(extra)
    return out


def _facts(seeds: dict, trigger_id: str) -> FactSheet:
    category, merchant, trg, customer = _args(seeds, trigger_id)
    return build_facts(category, merchant, trg, customer, now=NOW)


def assert_contract(out: dict) -> None:
    assert set(out) >= CONTRACT_KEYS
    assert isinstance(out["body"], str) and out["body"].strip()
    assert out["cta"] in CTA_VALUES
    assert out["send_as"] in (SEND_AS_VERA, SEND_AS_MERCHANT)
    assert isinstance(out["suppression_key"], str) and out["suppression_key"]
    assert isinstance(out["rationale"], str) and out["rationale"].strip()
    assert isinstance(out["template_name"], str) and out["template_name"]
    params = out["template_params"]
    assert isinstance(params, list) and params and all(isinstance(p, str) and p.strip() for p in params)
    assert set(out["meta"]) >= META_KEYS
    assert out["meta"]["source"] in ("llm", "template")
    assert isinstance(out["meta"]["facts"], list)
    if out["send_as"] == SEND_AS_MERCHANT:
        assert out["template_name"].startswith("merchant_")
    else:
        assert not out["template_name"].startswith("merchant_")


# --------------------------------------------------------------------------- template path


def test_template_path_output_contract(seeds, fakes):
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW, use_llm=False))
    assert_contract(out)
    assert out["meta"]["source"] == "template"
    assert out["meta"]["draft_issues"] == []                    # the fake draft is itself valid
    assert out["body"].startswith("Dr. Meera, ")
    assert out["send_as"] == SEND_AS_VERA
    assert out["suppression_key"] == "research:dentists:2026-W17"
    assert out["template_name"] == "vera_research_digest_v1"
    assert out["rationale"] == "Template: anchored on the why-now fact."
    assert out["meta"]["trigger_id"] == T_RESEARCH
    assert out["meta"]["merchant_id"] == MEERA
    assert out["meta"]["customer_id"] is None
    assert out["meta"]["kind"] == "research_digest"
    assert out["meta"]["language"] == "en"


def test_llm_unavailable_means_template_and_no_calls(seeds, use_llm):
    llm = use_llm([_resp(LLM_RESEARCH)], available=False)
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW, use_llm=True))
    assert out["meta"]["source"] == "template" and llm.calls == []


def test_use_llm_false_never_calls_the_llm(seeds, use_llm):
    llm = use_llm([_resp(LLM_RESEARCH)])
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW, use_llm=False))
    assert out["meta"]["source"] == "template" and llm.calls == []


# --------------------------------------------------------------------------- LLM path


def test_valid_llm_body_is_used(seeds, use_llm):
    assert validate_body(LLM_RESEARCH, _facts(seeds, T_RESEARCH)) == []       # precondition: grounded rewrite
    llm = use_llm([_resp(LLM_RESEARCH, template_params=["Dr. Meera", "JIDA Oct 2026, p.14 reports 38% lower"])])
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))
    assert_contract(out)
    assert out["meta"]["source"] == "llm" and out["meta"]["llm_attempts"] == 1
    assert out["body"] == LLM_RESEARCH
    assert out["rationale"].startswith("LLM:")
    assert out["template_params"] == ["Dr. Meera", "JIDA Oct 2026, p.14 reports 38% lower"]
    assert "JIDA Oct 2026, p.14" in out["meta"]["facts"]
    assert len(llm.calls) == 1
    call = llm.calls[0]
    assert call["schema"] is COMPOSE_SCHEMA and call["temperature"] == 0.0
    assert 0 < call["timeout"] <= 7.0
    assert "BASELINE DRAFT" in call["user"]


def test_llm_template_params_not_in_body_are_rederived(seeds, use_llm):
    use_llm([_resp(LLM_RESEARCH, template_params=["something the body never says"])])
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))
    assert out["meta"]["source"] == "llm"
    assert out["template_params"][0] == "Dr. Meera"
    assert all(p in out["body"] for p in out["template_params"])


def test_invalid_llm_cta_falls_back_to_playbook_cta(seeds, use_llm):
    use_llm([_resp(LLM_RESEARCH, cta="reply_now")])
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))
    assert out["meta"]["source"] == "llm" and out["cta"] == CTA_BINARY_YES_NO


def test_missing_llm_rationale_uses_draft_rationale(seeds, use_llm):
    use_llm([{"body": LLM_RESEARCH, "cta": CTA_OPEN_ENDED}])
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))
    assert out["meta"]["source"] == "llm"
    assert out["cta"] == CTA_OPEN_ENDED
    assert out["rationale"] == "Template: anchored on the why-now fact."


@pytest.mark.parametrize("bad", [None, "not json {", {"message": "no body key"}, {"body": 42}, {"body": "   "}, [1, 2]])
def test_unusable_llm_output_falls_back_to_draft(seeds, use_llm, bad):
    llm = use_llm([bad])
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))
    assert out["meta"]["source"] == "template"
    assert out["body"].startswith("Dr. Meera, ")
    assert len(llm.calls) == 1                      # nothing to repair: no second call


def test_llm_exception_falls_back_to_draft(seeds, use_llm):
    def boom(_i):
        raise RuntimeError("provider exploded")
    use_llm([boom])
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))
    assert out["meta"]["source"] == "template"


def test_invented_number_is_rejected_and_repair_that_fails_falls_back(seeds, use_llm):
    invented = LLM_RESEARCH.replace("38%", "47%")
    llm = use_llm([_resp(invented), _resp(invented)])
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))
    assert out["meta"]["source"] == "template"
    assert "47%" not in out["body"]
    assert len(llm.calls) == 2
    assert out["meta"]["llm_attempts"] == 2                       # why the LLM lost is kept for debugging
    assert any(issue_code(i) == "ungrounded_number" for i in out["meta"]["rejected_issues"])
    repair_user = llm.calls[1]["user"]
    assert "YOUR PREVIOUS ATTEMPT WAS REJECTED" in repair_user
    assert "ungrounded_number" in repair_user and "47%" in repair_user


@pytest.mark.parametrize("bad_body, code", [
    (LLM_RESEARCH.replace("Want me", "Results are guaranteed. Want me"), "taboo"),
    (LLM_RESEARCH.replace("(2,100-patient trial)", "(2,300-patient trial)"), "ungrounded_number"),
    (LLM_RESEARCH.replace("Want me", "Details at www.jida.in. Want me"), "url"),
    (LLM_RESEARCH.replace("That is exactly", "Your ctr_below_peer_median shows that is exactly"), "jargon"),
    (LLM_RESEARCH.replace("Dr. Meera, ", "Hope you are well! "), "missing_name"),
])
def test_rejected_twice_falls_back_to_the_draft(seeds, use_llm, bad_body, code):
    llm = use_llm([_resp(bad_body)])
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))
    assert out["meta"]["source"] == "template"
    assert out["body"].startswith("Dr. Meera, ") and out["body"] != bad_body
    assert len(llm.calls) == 2
    assert code in {issue_code(i) for i in out["meta"]["rejected_issues"]}


def test_taboo_is_rejected_and_a_clean_repair_is_used(seeds, use_llm):
    taboo = LLM_RESEARCH.replace("Want me", "Results are guaranteed. Want me")
    llm = use_llm([_resp(taboo), _resp(LLM_RESEARCH)])
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))
    assert out["meta"]["source"] == "llm"
    assert out["body"] == LLM_RESEARCH
    assert out["meta"]["llm_attempts"] == 2
    assert any(issue_code(i) == "taboo" for i in out["meta"]["rejected_issues"])
    assert "taboo" in llm.calls[1]["user"]


def test_repair_is_skipped_when_too_little_time_remains(seeds, use_llm):
    invented = LLM_RESEARCH.replace("38%", "47%")
    llm = use_llm([_resp(invented), _resp(LLM_RESEARCH)], delay=1.0)
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW, timeout=3.0))
    assert out["meta"]["source"] == "template"
    assert len(llm.calls) == 1                      # ~2.0s left after the first call: below the 2.5s repair floor


def test_slow_llm_respects_the_timeout(seeds, use_llm):
    llm = use_llm([_resp(LLM_RESEARCH)], delay=5.0)
    started = time.monotonic()
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW, timeout=2.0))
    elapsed = time.monotonic() - started
    assert elapsed < 2.6
    assert out["meta"]["source"] == "template"
    assert llm.calls and llm.calls[0]["timeout"] <= 2.0


def test_no_llm_call_when_budget_is_tiny(seeds, use_llm):
    llm = use_llm([_resp(LLM_RESEARCH)])
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW, timeout=0.5))
    assert out["meta"]["source"] == "template" and llm.calls == []


# --------------------------------------------------------------------------- regression guards


def test_dropping_the_source_citation_is_rejected(seeds, use_llm):
    uncited = LLM_RESEARCH.replace("JIDA Oct 2026, p.14 reports", "A new trial reports")
    assert validate_body(uncited, _facts(seeds, T_RESEARCH)) == []            # the validator alone accepts it
    use_llm([_resp(uncited), _resp(LLM_RESEARCH)])
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))
    assert out["body"] == LLM_RESEARCH
    assert any(issue_code(i) == "missing_source" for i in out["meta"]["rejected_issues"])


def test_vague_rewrite_is_rejected(seeds, use_llm):
    vague = ("Dr. Meera, a new study on recall intervals looks relevant for your high-risk adult patients. "
             "Want me to draft a patient note on it?")
    use_llm([_resp(vague), _resp(vague)])
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))
    assert out["meta"]["source"] == "template"


def test_invented_clock_time_is_rejected_for_customer_slots(seeds, use_llm):
    facts = _facts(seeds, T_RECALL)
    assert facts.slots                                                          # seed recall has real slots
    invented = (f"Hi Priya, {facts.signer} here. Your 6-month cleaning is due. We can see you Sat at 10am. "
                "Reply YES to book.")
    llm = use_llm([_resp(invented), _resp(invented)])
    out = _run(composer.compose_async(*_args(seeds, T_RECALL), now=NOW))
    assert out["meta"]["source"] == "template"
    assert "ungrounded_time" in llm.calls[1]["user"]


def test_slot_time_from_the_facts_is_accepted(seeds, use_llm):
    facts = _facts(seeds, T_RECALL)
    body = (f"Hi Priya, {facts.signer} here 🦷 aapka 6-month cleaning due hai. {facts.slots[0]} ke liye 1, "
            f"{facts.slots[1]} ke liye 2 reply karein, ya apna time batayein.")
    issues = validate_body(body, facts, customer_facing=True)
    assert issues == []
    use_llm([_resp(body, cta="multi_choice_slot")])
    out = _run(composer.compose_async(*_args(seeds, T_RECALL), now=NOW))
    assert out["meta"]["source"] == "llm" and out["cta"] == "multi_choice_slot"


def test_wrong_language_is_rejected(seeds, use_llm):
    facts = _facts(seeds, T_PERF_DIP)
    assert facts.language.code == "hinglish"
    english = ("Dr. Bharat, calls are down 50% over the last 7 days against a baseline of 12. "
               "Want me to draft a fresh post and an offer line for your approval?")
    assert validate_body(english, facts) == []
    llm = use_llm([_resp(english), _resp(english)])
    out = _run(composer.compose_async(*_args(seeds, T_PERF_DIP), now=NOW))
    assert out["meta"]["source"] == "template"
    assert "language" in llm.calls[1]["user"]


def test_ask_must_close_the_message(seeds, use_llm):
    trailing = LLM_RESEARCH.replace("Want me to draft a patient WhatsApp note on it?",
                                    "Want me to draft a patient WhatsApp note on it? It could help your recall list.")
    assert validate_body(trailing, _facts(seeds, T_RESEARCH)) == []
    use_llm([_resp(trailing), _resp(LLM_RESEARCH)])
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))
    assert out["body"] == LLM_RESEARCH
    assert any(issue_code(i) == "cta_not_last" for i in out["meta"]["rejected_issues"])


def test_devanagari_is_rejected_for_roman_script_plans(seeds, use_llm):
    deva = LLM_RESEARCH.replace("Want me to", "क्या मैं आपके लिए draft करूँ? Want me to")
    llm = use_llm([_resp(deva), _resp(deva)])
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))
    assert out["meta"]["source"] == "template"
    assert "script" in llm.calls[1]["user"]


def test_customer_facing_llm_body_mentioning_vera_is_rejected(seeds, use_llm):
    facts = _facts(seeds, T_RECALL)
    body = f"Hi Priya, Vera here for {facts.signer}. Your 6-month cleaning is due. Reply YES to book."
    use_llm([_resp(body), _resp(body)])
    out = _run(composer.compose_async(*_args(seeds, T_RECALL), now=NOW))
    assert out["meta"]["source"] == "template"
    assert "vera" not in out["body"].lower() and "magicpin" not in out["body"].lower()


# --------------------------------------------------------------------------- determinism / cache


def test_repeat_calls_return_the_cached_result(seeds, use_llm):
    other = LLM_RESEARCH.replace("Want me to draft", "Shall I draft")
    llm = use_llm([lambda i: _resp(LLM_RESEARCH if i == 0 else other)])
    first = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))
    second = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))
    assert first == second and len(llm.calls) == 1
    second["body"] = "mutated"                                   # callers get copies, not the cached object
    assert _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))["body"] == LLM_RESEARCH


def test_template_path_is_deterministic_without_the_cache(seeds, fakes):
    first = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW, use_llm=False))
    composer.clear_cache()
    second = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW, use_llm=False))
    assert first == second


def test_cached_llm_result_is_served_to_template_only_callers(seeds, use_llm):
    use_llm([_resp(LLM_RESEARCH)])
    _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW))
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW, use_llm=False))
    assert out["meta"]["source"] == "llm"


def test_input_fingerprint(seeds):
    category, merchant, trg, customer = _args(seeds, T_RESEARCH)
    fp = composer.input_fingerprint(category, merchant, trg, customer)
    assert len(fp) == 64 and fp == composer.input_fingerprint(category, merchant, trg, customer)
    reordered = dict(reversed(list(merchant.items())))
    assert composer.input_fingerprint(category, reordered, trg, customer) == fp
    changed = copy.deepcopy(merchant)
    changed["performance"]["views"] = 1
    assert composer.input_fingerprint(category, changed, trg, customer) != fp
    # the 4-argument form (planner / precompose) ignores now / prior bodies
    assert composer.input_fingerprint(category, merchant, trg, customer, now=None, prior_bodies=()) == fp
    day1 = composer.input_fingerprint(category, merchant, trg, customer, now="2026-04-26T09:00:00Z")
    assert day1 == composer.input_fingerprint(category, merchant, trg, customer, now="2026-04-26T18:00:00Z")
    assert day1 != composer.input_fingerprint(category, merchant, trg, customer, now="2026-04-27T09:00:00Z")
    assert day1 != composer.input_fingerprint(category, merchant, trg, customer, now="2026-04-26T09:00:00Z",
                                              prior_bodies=["x"])


def test_compose_cache_basics():
    cache = composer.ComposeCache(max_entries=2)
    cache.put("a", {"body": "1"})
    cache.put("b", {"body": "2"})
    got = cache.get("a")
    got["body"] = "changed"
    assert cache.get("a") == {"body": "1"}
    cache.put("c", {"body": "3"})                                # evicts the least recently used ("b")
    assert "b" not in cache and "a" in cache and len(cache) == 2
    assert cache.get("missing") is None
    cache.put("", {"body": "x"})
    cache.put("d", "not a dict")                                 # type: ignore[arg-type]
    assert "d" not in cache
    cache.clear()
    assert len(cache) == 0


# --------------------------------------------------------------------------- send_as / suppression / names


def test_customer_facing_send_as_and_template_name(seeds, fakes):
    fakes.template_name = "vera_recall_due_v1"                   # a sloppy playbook name gets the right prefix
    out = _run(composer.compose_async(*_args(seeds, T_RECALL), now=NOW, use_llm=False))
    assert_contract(out)
    assert out["send_as"] == SEND_AS_MERCHANT
    assert out["template_name"] == "merchant_recall_due_v1"
    assert out["meta"]["customer_id"] == PRIYA
    assert out["suppression_key"] == "recall:c_001_priya_for_m001:6mo"
    assert "vera" not in out["body"].lower()


def test_merchant_facing_never_gets_a_merchant_template_name(seeds, fakes):
    fakes.template_name = "merchant_research_digest_v1"
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW, use_llm=False))
    assert out["send_as"] == SEND_AS_VERA
    assert out["template_name"] == "vera_research_digest_v1"


def test_suppression_key_fallback(seeds, fakes):
    category, merchant, trg, customer = _args(seeds, T_RESEARCH)
    trg.pop("suppression_key")
    out = _run(composer.compose_async(category, merchant, trg, customer, now=NOW, use_llm=False))
    assert out["suppression_key"] == f"research_digest:{MEERA}:{T_RESEARCH}"
    trg["suppression_key"] = "   "
    trg.pop("merchant_id")
    trg["payload"]["merchant_id"] = MEERA
    composer.clear_cache()
    out = _run(composer.compose_async(category, merchant, trg, customer, now=NOW, use_llm=False))
    assert out["suppression_key"] == f"research_digest:{MEERA}:{T_RESEARCH}"


def test_placeholder_trigger_and_dr_prefixed_owner(seeds, fakes):
    category, merchant, _trg, _c = _args(seeds, T_RESEARCH)
    merchant["identity"]["owner_first_name"] = "Dr. Meera"
    trg = {"id": "trg_900_perf_dip_x", "scope": "merchant", "kind": "perf_dip", "merchant_id": MEERA,
           "payload": {"placeholder": True, "metric_or_topic": "perf_dip"}, "urgency": 2}
    out = _run(composer.compose_async(category, merchant, trg, None, now=NOW, use_llm=False))
    assert_contract(out)
    assert "Dr. Dr." not in out["body"] and out["body"].startswith("Dr. Meera")
    assert out["suppression_key"] == f"perf_dip:{MEERA}:trg_900_perf_dip_x"


# --------------------------------------------------------------------------- anti-repetition


def test_template_body_repeating_a_prior_body_is_varied(seeds, fakes):
    base = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW, use_llm=False))
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW, use_llm=False,
                                      prior_bodies=[base["body"]]))
    assert out["body"] != base["body"]
    assert out["body"].startswith("Dr. Meera, ")
    facts = _facts(seeds, T_RESEARCH)
    assert validate_body(out["body"], facts, prior_bodies=[base["body"]]) == []
    assert all(p in out["body"] for p in out["template_params"])


def test_hinglish_variation_line_stays_hinglish(seeds, fakes):
    base = _run(composer.compose_async(*_args(seeds, T_PERF_DIP), now=NOW, use_llm=False))
    out = _run(composer.compose_async(*_args(seeds, T_PERF_DIP), now=NOW, use_llm=False,
                                      prior_bodies=[base["body"]]))
    added = out["body"].replace(base["body"].rsplit(". ", 1)[0], "")
    assert out["body"] != base["body"]
    assert any(w in added.lower() for w in ("yeh", "phir", "dobara", "baar"))


def test_llm_body_repeating_a_prior_body_is_not_used(seeds, use_llm):
    use_llm([_resp(LLM_RESEARCH)])
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW, prior_bodies=[LLM_RESEARCH]))
    assert out["body"] != LLM_RESEARCH
    assert out["meta"]["source"] == "template"


# --------------------------------------------------------------------------- sync wrapper


def test_compose_sync_outside_a_loop(seeds, fakes):
    out = composer.compose(*_args(seeds, T_RESEARCH), now=NOW, use_llm=False)
    assert_contract(out)


def test_compose_sync_uses_llm_when_available_by_default(seeds, use_llm):
    use_llm([_resp(LLM_RESEARCH)])
    out = composer.compose(*_args(seeds, T_RESEARCH), now=NOW)
    assert out["meta"]["source"] == "llm"


def test_compose_sync_inside_a_running_loop(seeds, use_llm):
    use_llm([_resp(LLM_RESEARCH)])

    async def main() -> dict:
        return composer.compose(*_args(seeds, T_RESEARCH), now=NOW)

    out = asyncio.run(main())
    assert out["meta"]["source"] == "llm" and out["body"] == LLM_RESEARCH


# --------------------------------------------------------------------------- never raises


def test_template_failure_uses_the_generic_grounded_draft(seeds, fakes):
    fakes.render_raises = True
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW, use_llm=False))
    assert_contract(out)
    assert out["body"].startswith("Dr. Meera")
    assert validate_body(out["body"], _facts(seeds, T_RESEARCH)) == []


def test_missing_sibling_modules_fall_back(seeds, monkeypatch):
    monkeypatch.setitem(sys.modules, "app.playbooks", None)
    monkeypatch.setitem(sys.modules, "app.templates", None)
    monkeypatch.setitem(sys.modules, "app.llm", None)
    out = _run(composer.compose_async(*_args(seeds, T_RECALL), now=NOW))
    assert_contract(out)
    assert out["send_as"] == SEND_AS_MERCHANT and out["meta"]["source"] == "template"


def test_facts_failure_still_returns_a_safe_composition(seeds, fakes, monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("facts bug")
    monkeypatch.setattr(composer, "build_facts", broken)
    out = _run(composer.compose_async(*_args(seeds, T_RESEARCH), now=NOW, use_llm=False))
    assert_contract(out)
    assert out["suppression_key"] == "research:dentists:2026-W17"
    sync = composer.compose(*_args(seeds, T_RECALL), now=NOW, use_llm=False)
    assert_contract(sync)
    assert sync["send_as"] == SEND_AS_MERCHANT


@pytest.mark.parametrize("args", [(None, None, None), ({}, {}, {}), ("x", 3, ["y"]), ({}, {"merchant_id": "m"}, {"kind": 5})])
def test_garbage_inputs_never_raise(fakes, args):
    out = composer.compose(*args, use_llm=False)
    assert_contract(out)


# --------------------------------------------------------------------------- prompts


def test_compose_prompt_sections_and_rules(seeds, fakes):
    facts = _facts(seeds, T_RESEARCH)
    pb = fakes.playbooks().get_playbook(facts.kind, facts.scope, facts.category_slug)
    draft = fakes.templates().render(facts, pb)
    system, user = compose_prompt(facts, pb, draft)
    for needle in ("NON-NEGOTIABLE RULES", "HOW THE MESSAGE IS JUDGED", "No URLs", "JSON", "used_facts",
                   "template_params", "Grounding", "one call to action"):
        assert needle in system
    order = ["RECIPIENT & SENDER", "SEND AS", "LANGUAGE", "CATEGORY VOICE", "TRIGGER / WHY NOW", "ANCHOR FACTS",
             "SUPPORT FACTS", "MERCHANT'S ACTIVE OFFERS", "CATALOG IDEAS", "RECENT CONVERSATION",
             "JUDGMENT NOTES", "BASELINE DRAFT", "OUTPUT FORMAT"]
    positions = [user.index(section) for section in order]
    assert positions == sorted(positions)
    assert "[digest.source] JIDA Oct 2026, p.14" in user
    assert "'Dr. Meera'" in user and "Dr. Dr." not in user.replace("never 'Dr. Dr.'", "")
    assert draft.body in user
    assert "TABOO (never write):" in user and "guaranteed" in user
    ideas = user.split("CATALOG IDEAS", 1)[1].split("\n\n", 1)[0]
    assert "Dental Cleaning @ ₹299" not in ideas                  # the live offer is not repeated as an idea
    assert "Dental Cleaning @ ₹299" in user.split("MERCHANT'S ACTIVE OFFERS", 1)[1].split("\n\n", 1)[0]
    assert len(system) + len(user) <= MAX_PROMPT_CHARS


def test_customer_prompt_hides_merchant_intelligence(seeds, fakes):
    facts = _facts(seeds, T_RECALL)
    pb = fakes.playbooks().get_playbook(facts.kind, facts.scope, facts.category_slug)
    system, user = compose_prompt(facts, pb, fakes.templates().render(facts, pb))
    assert "to its customer" in system
    assert "merchant_on_behalf" in user and "never mention Vera or magicpin" in user
    assert "CATALOG IDEAS" not in user and "[perf." not in user and "[trend.top]" not in user
    assert "SLOTS (offer these exact labels)" in user
    for slot in facts.slots[:2]:
        assert slot in user


def test_prompt_mentions_prior_bodies_and_no_name_customers(seeds, fakes):
    facts = _facts(seeds, T_RESEARCH)
    pb = fakes.playbooks().get_playbook(facts.kind, facts.scope, facts.category_slug)
    _s, user = compose_prompt(facts, pb, Draft(body="x"), prior_bodies=["An earlier message we sent."])
    assert "already sent: 'An earlier message we sent.'" in user
    anon = FactSheet(kind="recall_due", scope="customer", send_as=SEND_AS_MERCHANT, signer="Sunrise Medicos")
    _s, user = compose_prompt(anon, Playbook(customer_facing=True), Draft())
    assert "do not address by name" in user and "no customer name is on file" in user


def test_repair_prompt_lists_issues_and_the_attempt(seeds, fakes):
    facts = _facts(seeds, T_RESEARCH)
    pb = fakes.playbooks().get_playbook(facts.kind, facts.scope, facts.category_slug)
    system, user = repair_prompt(facts, pb, Draft(body="draft"), "bad attempt 47%",
                                 ["ungrounded_number: 47% not found", "taboo: never use 'guaranteed'"])
    assert "YOUR PREVIOUS ATTEMPT WAS REJECTED" in user
    assert "bad attempt 47%" in user and "- ungrounded_number: 47% not found" in user and "taboo" in user
    assert system == compose_prompt(facts, pb, Draft(body="draft"))[0]


def test_compose_schema_contract():
    assert COMPOSE_SCHEMA["type"] == "object"
    assert set(COMPOSE_SCHEMA["required"]) >= {"body", "cta", "rationale"}
    props = COMPOSE_SCHEMA["properties"]
    assert set(props) == {"body", "cta", "rationale", "template_params", "used_facts"}
    assert props["template_params"] == {"type": "array", "items": {"type": "string"},
                                        "description": props["template_params"]["description"]}
    assert props["used_facts"]["items"] == {"type": "string"}
    assert set(props["cta"]["enum"]) == CTA_VALUES


def test_prompts_never_crash_on_empty_inputs():
    system, user = compose_prompt(FactSheet(), Playbook(), Draft())
    assert system and user and "OUTPUT FORMAT" in user
    system, user = compose_prompt(None, None, None)              # type: ignore[arg-type]
    assert system and user


def test_prompt_budget_holds_for_every_seed_trigger(seeds, fakes):
    for tid, trg in seeds["triggers"].items():
        merchant = seeds["merchants"].get(trg.get("merchant_id"))
        customer = seeds["customers"].get(trg.get("customer_id")) if trg.get("customer_id") else None
        facts = build_facts(seeds["categories"].get(merchant["category_slug"]), merchant, trg, customer, now=NOW)
        pb = fakes.playbooks().get_playbook(facts.kind, facts.scope, facts.category_slug)
        pb.framing = "x" * 2000                                  # a very long playbook framing is clipped
        system, user = compose_prompt(facts, pb, fakes.templates().render(facts, pb))
        assert len(system) + len(user) <= MAX_PROMPT_CHARS, tid


# --------------------------------------------------------------------------- integration (real modules)


def _real_modules_or_skip():
    templates = importlib.import_module("app.templates")
    playbooks = importlib.import_module("app.playbooks")
    if not hasattr(templates, "render") or not hasattr(playbooks, "get_playbook"):
        pytest.skip("templates.render / playbooks.get_playbook not available yet")


def _check_real(category, merchant, trg, customer) -> None:
    out = composer.compose(category, merchant, trg, customer, now=NOW, use_llm=False)
    assert_contract(out)
    facts = build_facts(category, merchant, trg, customer, now=NOW)
    customer_facing = out["send_as"] == SEND_AS_MERCHANT
    assert customer_facing == (facts.send_as == SEND_AS_MERCHANT)
    assert validate_body(out["body"], facts, customer_facing=customer_facing) == [], (trg["id"], out["body"])
    if trg.get("suppression_key"):
        assert out["suppression_key"] == trg["suppression_key"]
    assert "Dr. Dr." not in out["body"]
    if customer_facing:
        assert "vera" not in out["body"].lower() and "magicpin" not in out["body"].lower()
    system, user = compose_prompt(facts, importlib.import_module("app.playbooks").get_playbook(
        facts.kind, facts.scope, facts.category_slug), Draft(body=out["body"]))
    assert len(system) + len(user) <= MAX_PROMPT_CHARS


def test_real_modules_compose_every_seed_trigger(seeds, monkeypatch):
    _real_modules_or_skip()
    monkeypatch.setitem(sys.modules, "app.llm", _llm_module(FakeLLM(available=False)))
    for tid, trg in sorted(seeds["triggers"].items()):
        merchant = seeds["merchants"].get(trg.get("merchant_id"))
        customer = seeds["customers"].get(trg.get("customer_id")) if trg.get("customer_id") else None
        _check_real(seeds["categories"].get(merchant["category_slug"]), merchant, trg, customer)


def test_real_modules_compose_the_official_test_pairs(tmp_path_factory, monkeypatch):
    _real_modules_or_skip()
    monkeypatch.setitem(sys.modules, "app.llm", _llm_module(FakeLLM(available=False)))
    out = tmp_path_factory.mktemp("expanded_composer")
    subprocess.run([sys.executable, str(DATASET / "generate_dataset.py"), "--seed-dir", str(DATASET),
                    "--out", str(out)], check=True, capture_output=True, cwd=str(ROOT))

    def load(sub: str, id_: str | None) -> dict | None:
        path = out / sub / f"{id_}.json"
        return json.loads(path.read_text()) if id_ and path.exists() else None

    pairs = json.loads((out / "test_pairs.json").read_text())["pairs"]
    assert len(pairs) >= 25
    for pair in pairs:
        merchant = load("merchants", pair["merchant_id"])
        trg = load("triggers", pair["trigger_id"])
        customer = load("customers", pair.get("customer_id"))
        category = load("categories", merchant["category_slug"])
        _check_real(category, merchant, trg, customer)
