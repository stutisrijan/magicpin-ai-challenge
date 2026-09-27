"""Tick planner tests (app/planner.py).

The composer is replaced by a fake module in sys.modules, so these tests pass before
(and independently of) the real app/composer.py. Offline, no API keys.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from app import config, planner
from app.state import ContextStore, ConversationStore

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
REQUIRED_ACTION_KEYS = {"conversation_id", "merchant_id", "customer_id", "send_as", "trigger_id", "template_name",
                        "template_params", "body", "cta", "suppression_key", "rationale"}


# --------------------------------------------------------------------------- fakes + data


class FakeComposer:
    """Stands in for app.composer. mode: ok | raise | hang | llm_fails (template path works)."""

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.calls: list[dict] = []
        self.fixed_body: str | None = None

    def module(self) -> types.ModuleType:
        mod = types.ModuleType("app.composer")
        mod.compose_async = self.compose_async
        mod.compose = self.compose
        mod.input_fingerprint = self.input_fingerprint
        mod.ComposeCache = FakeCache
        return mod

    @staticmethod
    def input_fingerprint(category, merchant, trigger, customer) -> str:
        blob = json.dumps([category, merchant, trigger, customer], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _build(self, merchant, trigger, customer) -> dict:
        kind = trigger.get("kind", "generic")
        owner = (merchant.get("identity") or {}).get("owner_first_name") or "there"
        who = ((customer or {}).get("identity") or {}).get("name")
        body = self.fixed_body or (f"Hi {who or owner}, a quick note about {kind.replace('_', ' ')} "
                                   f"for {merchant['identity']['name']} ({trigger['id']}). Want me to set it up?")
        mid = merchant.get("merchant_id")
        return {
            "body": body,
            "cta": "binary_yes_no",
            "send_as": "merchant_on_behalf" if customer else "vera",
            "suppression_key": trigger.get("suppression_key") or f"{kind}:{mid}:{trigger.get('id')}",
            "rationale": f"{kind} is live for this merchant; anchored on their own numbers.",
            "template_name": ("merchant_" if customer else "vera_") + f"{kind}_v1",
            "template_params": [who or owner, kind, "Want me to set it up?"],
            "meta": {"offer": "set it up", "kind": kind, "source": "template", "language": "en",
                     "facts": [kind], "trigger_id": trigger.get("id"), "merchant_id": mid,
                     "customer_id": (customer or {}).get("customer_id")},
        }

    async def compose_async(self, category, merchant, trigger, customer=None, *, now=None, prior_bodies=(),
                            use_llm=True, timeout=None) -> dict:
        self.calls.append({"trigger_id": trigger.get("id"), "use_llm": use_llm, "timeout": timeout,
                           "now": now, "prior_bodies": list(prior_bodies)})
        if self.mode == "raise" or (self.mode == "llm_fails" and use_llm):
            raise RuntimeError("composer exploded")
        if self.mode == "hang" or (self.mode == "llm_hangs" and use_llm):
            await asyncio.sleep(3600)
        return self._build(merchant, trigger, customer)

    def compose(self, category, merchant, trigger, customer=None, *, now=None, use_llm=None) -> dict:
        return self._build(merchant, trigger, customer)


class FakeCache:
    def __init__(self) -> None:
        self.data: dict[str, dict] = {}

    def get(self, fp):
        return self.data.get(fp)

    def put(self, fp, value):
        self.data[fp] = value


class FakeEngine:
    def __init__(self) -> None:
        self.opened: list[tuple[dict, dict, str | None]] = []

    def open_from_action(self, action, composition, now):
        self.opened.append((action, composition, now))


def load_seed() -> dict:
    cats = {}
    for f in sorted((DATASET / "categories").glob("*.json")):
        data = json.loads(f.read_text())
        cats[data["slug"]] = data
    return {
        "category": cats,
        "merchant": {m["merchant_id"]: m for m in json.loads((DATASET / "merchants_seed.json").read_text())["merchants"]},
        "customer": {c["customer_id"]: c for c in json.loads((DATASET / "customers_seed.json").read_text())["customers"]},
        "trigger": {t["id"]: t for t in json.loads((DATASET / "triggers_seed.json").read_text())["triggers"]},
    }


SEED = load_seed()


@pytest.fixture
def fake(monkeypatch):
    comp = FakeComposer()
    monkeypatch.setitem(sys.modules, "app.composer", comp.module())
    return comp


@pytest.fixture
def world():
    """ContextStore loaded with every seed context, a fresh ConversationStore and engine."""
    contexts, convs = ContextStore(), ConversationStore()
    for scope in ("category", "merchant", "customer", "trigger"):
        for cid, payload in SEED[scope].items():
            contexts.put(scope, cid, 1, copy.deepcopy(payload))
    yield contexts, convs, FakeEngine()
    planner.reset(convs)


def tick(world, ids, now="2026-04-26T10:30:00Z", **kw):
    contexts, convs, engine = world
    return planner.plan_tick_sync(now, ids, contexts=contexts, convs=convs, engine=engine, **kw)


# --------------------------------------------------------------------------- basics


def test_no_triggers_returns_empty(fake, world):
    assert tick(world, []) == []
    assert fake.calls == []


def test_unknown_and_malformed_ids_ignored(fake, world):
    assert tick(world, ["trg_does_not_exist", None, 42, "", {"x": 1}]) == []
    assert planner.deferred_ids(world[1]) == []


def test_action_has_all_required_fields(fake, world):
    actions = tick(world, ["trg_001_research_digest_dentists", "trg_003_recall_due_priya"])
    assert len(actions) == 2
    for a in actions:
        assert set(a) == REQUIRED_ACTION_KEYS
        assert a["body"] and a["rationale"] and a["template_name"] and a["suppression_key"]
        assert isinstance(a["template_params"], list) and all(isinstance(p, str) for p in a["template_params"])
        assert a["conversation_id"].startswith("conv_m001_drmeera_")
    by_trigger = {a["trigger_id"]: a for a in actions}
    research = by_trigger["trg_001_research_digest_dentists"]
    recall = by_trigger["trg_003_recall_due_priya"]
    assert research["send_as"] == "vera" and research["customer_id"] is None
    assert research["suppression_key"] == "research:dentists:2026-W17"
    assert recall["send_as"] == "merchant_on_behalf" and recall["customer_id"] == "c_001_priya_for_m001"
    assert "c001_priya" in recall["conversation_id"]


def test_conversation_ids_unique_and_never_reused(fake, world):
    contexts, convs, _ = world
    ids = [planner.new_conversation_id(convs, "m_001_drmeera_dentist_delhi", "research_digest") for _ in range(50)]
    assert len(set(ids)) == 50
    assert all(i.startswith("conv_m001_drmeera_research_digest_") for i in ids)
    convs.create("conv_x_1")
    assert planner.new_conversation_id(convs, None, "") != "conv_x_1"


def test_short_id_shapes():
    assert planner._short_id("m_001_drmeera_dentist_delhi") == "m001_drmeera"
    assert planner._short_id("c_075_aditya_for_m_019_karim_salon_lucknow") == "c075_aditya"
    assert planner._short_id("Weird ID!!") == "weird_id"
    assert planner._short_id(None) == ""


# --------------------------------------------------------------------------- selection + deferral


def test_one_merchant_facing_action_per_merchant_and_rest_deferred(fake, world):
    meera = ["trg_001_research_digest_dentists", "trg_002_compliance_dci_radiograph",
             "trg_022_cde_webinar_dentists", "trg_023_competitor_opened_dentist"]
    first = tick(world, meera)
    assert [a["trigger_id"] for a in first] == ["trg_002_compliance_dci_radiograph"]   # urgency 4 wins
    assert set(planner.deferred_ids(world[1])) == set(meera) - {"trg_002_compliance_dci_radiograph"}

    # Deferred triggers are retried on later ticks even when the judge no longer lists them.
    second = tick(world, [])
    assert [a["trigger_id"] for a in second] == ["trg_001_research_digest_dentists"]   # urgency 2, earlier order
    third = tick(world, [])
    fourth = tick(world, [])
    sent = {a["trigger_id"] for a in first + second + third + fourth}
    assert sent == set(meera)
    assert tick(world, meera) == []           # all suppressed now


def test_merchant_and_customer_facing_can_share_a_tick(fake, world):
    actions = tick(world, ["trg_001_research_digest_dentists", "trg_003_recall_due_priya"])
    assert {a["send_as"] for a in actions} == {"vera", "merchant_on_behalf"}


def test_real_payload_beats_placeholder_at_equal_urgency(fake, world):
    contexts, convs, _ = world
    placeholder = {"id": "trg_ph", "scope": "merchant", "kind": "perf_spike", "merchant_id": "m_001_drmeera_dentist_delhi",
                   "customer_id": None, "payload": {"placeholder": True, "metric_or_topic": "perf_spike"},
                   "urgency": 2, "suppression_key": "ph:1"}
    contexts.put("trigger", "trg_ph", 1, placeholder)
    actions = tick(world, ["trg_ph", "trg_023_competitor_opened_dentist"])
    assert [a["trigger_id"] for a in actions] == ["trg_023_competitor_opened_dentist"]


def test_max_actions_per_tick_cap(fake, world, monkeypatch):
    monkeypatch.setattr(config.settings, "max_actions_per_tick", 2)
    all_ids = list(SEED["trigger"])
    actions = tick(world, all_ids)
    assert len(actions) == 2
    assert all(a["trigger_id"] in all_ids for a in actions)


def test_every_seed_trigger_is_eventually_handled(fake, world):
    contexts, convs, _ = world
    all_ids = list(SEED["trigger"])
    sent: list[dict] = []
    for _ in range(8):
        sent += tick(world, all_ids)
    sent_ids = [a["trigger_id"] for a in sent]
    assert len(sent_ids) == len(set(sent_ids)) == len(all_ids)   # every seed trigger once, never twice
    assert len({a["conversation_id"] for a in sent}) == len(sent)


# --------------------------------------------------------------------------- suppression + consent + merchant state


def test_suppression_prevents_resend(fake, world):
    first = tick(world, ["trg_001_research_digest_dentists"])
    assert len(first) == 1
    assert tick(world, ["trg_001_research_digest_dentists"]) == []
    assert world[1].skipped_reason("trg_001_research_digest_dentists") == "suppressed"


def test_shared_suppression_key_sends_only_once(fake, world):
    contexts, convs, _ = world
    twin = dict(SEED["trigger"]["trg_004_perf_dip_bharat"], id="trg_twin", merchant_id="m_010_sunrisepharm_pharmacy_lucknow")
    contexts.put("trigger", "trg_twin", 1, twin)
    actions = tick(world, ["trg_004_perf_dip_bharat", "trg_twin"])
    assert len(actions) == 1
    assert tick(world, ["trg_twin"]) == []


def test_consentless_customer_trigger_skipped(fake, world):
    contexts, convs, _ = world
    trg = {"id": "trg_walkin", "scope": "customer", "kind": "customer_lapsed_soft",
           "merchant_id": "m_010_sunrisepharm_pharmacy_lucknow", "customer_id": "c_015_anonymous_for_m010",
           "payload": {}, "urgency": 3, "suppression_key": "walkin:1"}
    contexts.put("trigger", "trg_walkin", 1, trg)
    assert tick(world, ["trg_walkin"]) == []
    assert convs.skipped_reason("trg_walkin") == "no_consent"
    assert "trg_walkin" not in planner.deferred_ids(convs)
    assert fake.calls == []


def test_reminder_opt_out_blocks_reminders_only():
    cust = {"consent": {"opted_in_at": "2026-01-01", "scope": ["promotional_offers"]},
            "preferences": {"channel": "whatsapp", "reminder_opt_in": False}}
    assert planner.consent_problem(cust, "recall_due") == "reminders_opted_out"
    assert planner.consent_problem(cust, "appointment_tomorrow") == "reminders_opted_out"
    assert planner.consent_problem(cust, "customer_lapsed_soft") is None
    cust["consent"]["scope"].append("recall_reminders")           # explicit scope wins
    assert planner.consent_problem(cust, "recall_due") is None


def test_consent_problem_edge_cases():
    seed = SEED["customer"]
    assert planner.consent_problem(seed["c_013_grandfather_for_m009"], "chronic_refill_due") is None   # via son is fine
    assert planner.consent_problem(seed["c_012_karthik_jr_for_m008"], "trial_followup") is None      # via parent
    assert planner.consent_problem(seed["c_015_anonymous_for_m010"], "recall_due") == "no_consent"
    assert planner.consent_problem({"consent": {"opted_in_at": "2026-01-01", "scope": []}}, "x") == "no_consent_scope"
    assert planner.consent_problem({"consent": {"opted_in_at": "2026-01-01", "scope": ["a"]},
                                    "preferences": {"channel": "none_recorded"}}, "x") == "no_channel"
    assert planner.consent_problem({"consent": {"opted_in_at": "2026-01-01", "scope": ["a"], "opted_out_at": "x"}},
                                   "x") == "consent_revoked"
    assert planner.consent_problem(None, "x") == "no_customer"
    assert planner.consent_problem({"consent": None, "preferences": None}, "x") == "no_consent"


def test_customer_trigger_waits_for_customer_context(fake, world):
    contexts, convs, _ = world
    contexts.clear()
    for scope in ("category", "merchant"):
        for cid, payload in SEED[scope].items():
            contexts.put(scope, cid, 1, payload)
    contexts.put("trigger", "trg_003_recall_due_priya", 1, SEED["trigger"]["trg_003_recall_due_priya"])
    assert tick(world, ["trg_003_recall_due_priya"]) == []
    assert "trg_003_recall_due_priya" in planner.deferred_ids(convs)
    contexts.put("customer", "c_001_priya_for_m001", 1, SEED["customer"]["c_001_priya_for_m001"])
    actions = tick(world, [])
    assert [a["trigger_id"] for a in actions] == ["trg_003_recall_due_priya"]


def test_missing_merchant_deferred_and_missing_category_allowed(fake, world):
    contexts, convs, _ = world
    trg = {"id": "trg_new_m", "scope": "merchant", "kind": "perf_dip", "merchant_id": "m_999_new",
           "payload": {"metric": "calls"}, "urgency": 2}
    contexts.put("trigger", "trg_new_m", 1, trg)
    assert tick(world, ["trg_new_m"]) == []
    assert "trg_new_m" in planner.deferred_ids(convs)
    merchant = dict(SEED["merchant"]["m_001_drmeera_dentist_delhi"], merchant_id="m_999_new", category_slug="florists")
    contexts.put("merchant", "m_999_new", 1, merchant)
    actions = tick(world, ["trg_new_m"])
    assert len(actions) == 1
    assert actions[0]["suppression_key"] == "perf_dip:m_999_new:trg_new_m"


def test_trigger_without_merchant_id_skipped(fake, world):
    contexts, convs, _ = world
    contexts.put("trigger", "trg_orphan", 1, {"id": "trg_orphan", "kind": "research_digest", "payload": {}})
    assert tick(world, ["trg_orphan"]) == []
    assert convs.skipped_reason("trg_orphan") == "no_merchant_id"


def test_opted_out_merchant_deferred(fake, world):
    contexts, convs, _ = world
    convs.merchant("m_002_bharat_dentist_mumbai").opted_out = True
    assert tick(world, ["trg_004_perf_dip_bharat", "trg_005_renewal_due_bharat"]) == []
    assert set(planner.deferred_ids(convs)) == {"trg_004_perf_dip_bharat", "trg_005_renewal_due_bharat"}
    convs.merchant("m_002_bharat_dentist_mumbai").opted_out = False
    assert len(tick(world, [])) == 1


def test_cooldown_defers_low_urgency_but_urgent_bypasses(fake, world):
    contexts, convs, _ = world
    convs.merchant("m_001_drmeera_dentist_delhi").cooldown_until = "2026-04-26T12:00:00Z"
    actions = tick(world, ["trg_001_research_digest_dentists", "trg_002_compliance_dci_radiograph"])
    assert [a["trigger_id"] for a in actions] == ["trg_002_compliance_dci_radiograph"]   # urgency 4 >= bypass
    assert tick(world, [], now="2026-04-26T11:00:00Z") == []                           # still cooling down
    later = tick(world, [], now="2026-04-26T12:05:00Z")
    assert [a["trigger_id"] for a in later] == ["trg_001_research_digest_dentists"]


def test_cooldown_does_not_hold_back_customer_facing(fake, world):
    contexts, convs, _ = world
    convs.merchant("m_001_drmeera_dentist_delhi").cooldown_until = "2026-04-26T12:00:00Z"
    actions = tick(world, ["trg_001_research_digest_dentists", "trg_003_recall_due_priya"])
    assert [a["trigger_id"] for a in actions] == ["trg_003_recall_due_priya"]
    assert planner.deferred_ids(convs) == ["trg_001_research_digest_dentists"]


def test_older_deferred_trigger_wins_ties(fake, world):
    contexts, convs, _ = world
    # Tick 1: trg_001 and trg_023 (both urgency 2, same merchant) -> trg_001 sent, trg_023 deferred.
    assert [a["trigger_id"] for a in tick(world, ["trg_001_research_digest_dentists",
                                                  "trg_023_competitor_opened_dentist"])] == ["trg_001_research_digest_dentists"]
    # Tick 2: a fresh urgency-2 trigger for the same merchant is listed first, but the deferred one is older.
    fresh = dict(SEED["trigger"]["trg_023_competitor_opened_dentist"], id="trg_fresh", suppression_key="fresh:1")
    contexts.put("trigger", "trg_fresh", 1, fresh)
    assert [a["trigger_id"] for a in tick(world, ["trg_fresh"])] == ["trg_023_competitor_opened_dentist"]
    assert [a["trigger_id"] for a in tick(world, [])] == ["trg_fresh"]


def test_merchant_explicit_wait_defers_merchant_facing(fake, world):
    contexts, convs, _ = world
    conv = convs.create("conv_prev", merchant_id="m_001_drmeera_dentist_delhi")
    conv.status, conv.wait_until = "waiting", "2026-04-26T11:00:00Z"
    actions = tick(world, ["trg_001_research_digest_dentists", "trg_003_recall_due_priya"])
    assert [a["trigger_id"] for a in actions] == ["trg_003_recall_due_priya"]   # customer-facing unaffected
    assert [a["trigger_id"] for a in tick(world, [], now="2026-04-26T11:05:00Z")] == ["trg_001_research_digest_dentists"]


def test_state_recorded_after_send(fake, world):
    contexts, convs, engine = world
    actions = tick(world, ["trg_001_research_digest_dentists"])
    a = actions[0]
    st = convs.merchant("m_001_drmeera_dentist_delhi")
    assert convs.suppression_used("research:dentists:2026-W17")
    assert st.sent_bodies == [a["body"]] and st.last_sent_at == "2026-04-26T10:30:00Z"
    assert engine.opened and engine.opened[0][0] is a and engine.opened[0][2] == "2026-04-26T10:30:00Z"
    assert engine.opened[0][1]["meta"]["kind"] == "research_digest"
    assert convs.exists(a["conversation_id"])            # guaranteed even though the fake engine stores nothing


def test_open_from_action_failure_still_registers_conversation(fake, world):
    contexts, convs, _ = world

    class Broken:
        def open_from_action(self, *a):
            raise RuntimeError("boom")

    actions = planner.plan_tick_sync("2026-04-26T10:30:00Z", ["trg_001_research_digest_dentists"],
                                     contexts=contexts, convs=convs, engine=Broken())
    assert len(actions) == 1
    conv = convs.get(actions[0]["conversation_id"])
    assert conv is not None and conv.bot_bodies() == [actions[0]["body"]] and conv.kind == "research_digest"


def test_no_engine_is_tolerated(fake, world):
    contexts, convs, _ = world
    actions = planner.plan_tick_sync(None, ["trg_001_research_digest_dentists"], contexts=contexts, convs=convs,
                                     engine=None)
    assert len(actions) == 1


def test_prior_bodies_passed_and_duplicate_body_skipped(fake, world):
    contexts, convs, _ = world
    fake.fixed_body = "Same body every time."
    first = tick(world, ["trg_001_research_digest_dentists"])
    assert len(first) == 1
    second = tick(world, ["trg_023_competitor_opened_dentist"])
    assert second == []
    assert convs.skipped_reason("trg_023_competitor_opened_dentist") == "duplicate_body"
    assert fake.calls[-1]["prior_bodies"] == ["Same body every time."]


def test_same_tick_body_collision_is_deferred_not_dropped(fake, world):
    contexts, convs, _ = world
    fake.fixed_body = "Namaste, a gentle reminder from the clinic."
    twin = dict(SEED["trigger"]["trg_003_recall_due_priya"], id="trg_twin_recall", customer_id="c_002_rohit_for_m001",
                suppression_key="recall:twin")
    contexts.put("trigger", "trg_twin_recall", 1, twin)
    cust = contexts.get("customer", "c_002_rohit_for_m001")
    if cust is None or planner.consent_problem(cust, "recall_due"):
        pytest.skip("seed customer shape changed")
    first = tick(world, ["trg_003_recall_due_priya", "trg_twin_recall"])
    assert len(first) == 1
    other = ({"trg_003_recall_due_priya", "trg_twin_recall"} - {first[0]["trigger_id"]}).pop()
    assert other in planner.deferred_ids(convs) and convs.skipped_reason(other) is None
    fake.fixed_body = None                        # next tick the composer sees the prior body and varies
    assert [a["trigger_id"] for a in tick(world, [])] == [other]


# --------------------------------------------------------------------------- composer failures + deadline + cache


def test_llm_failure_falls_back_to_template(monkeypatch, world):
    comp = FakeComposer(mode="llm_fails")
    monkeypatch.setitem(sys.modules, "app.composer", comp.module())
    actions = tick(world, ["trg_001_research_digest_dentists"])
    assert len(actions) == 1
    assert [c["use_llm"] for c in comp.calls] == [True, False]


def test_composer_that_raises_yields_no_actions_and_retries_later(monkeypatch, world):
    comp = FakeComposer(mode="raise")
    monkeypatch.setitem(sys.modules, "app.composer", comp.module())
    assert tick(world, ["trg_001_research_digest_dentists"]) == []
    assert "trg_001_research_digest_dentists" in planner.deferred_ids(world[1])
    assert not world[1].suppression_used("research:dentists:2026-W17")
    comp.mode = "ok"
    assert len(tick(world, [])) == 1


def test_hanging_llm_returns_template_within_deadline(monkeypatch, world):
    comp = FakeComposer(mode="llm_hangs")
    monkeypatch.setitem(sys.modules, "app.composer", comp.module())
    start = time.perf_counter()
    actions = tick(world, ["trg_001_research_digest_dentists", "trg_004_perf_dip_bharat"], deadline_s=2.0)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.2
    assert len(actions) == 2


def test_hanging_composer_returns_within_deadline(monkeypatch, world):
    comp = FakeComposer(mode="hang")
    monkeypatch.setitem(sys.modules, "app.composer", comp.module())
    start = time.perf_counter()
    actions = tick(world, list(SEED["trigger"]), deadline_s=1.0)
    assert time.perf_counter() - start < 1.3
    assert actions == []
    assert not world[1]._sent_suppression          # nothing half-recorded


def test_cache_hit_skips_composition(fake, world):
    contexts, convs, _ = world
    cache = FakeCache()
    trg = SEED["trigger"]["trg_001_research_digest_dentists"]
    merchant = contexts.get("merchant", "m_001_drmeera_dentist_delhi")
    fp = FakeComposer.input_fingerprint(contexts.get("category", "dentists"), merchant, contexts.get("trigger", trg["id"]), None)
    cached = fake._build(merchant, trg, None)
    cached["body"] = "Dr. Meera, precomposed body from cache."
    cache.put(fp, cached)
    actions = tick(world, [trg["id"]], cache=cache)
    assert actions[0]["body"] == "Dr. Meera, precomposed body from cache."
    assert fake.calls == []


def test_cache_entry_for_another_date_is_recomposed(fake, world):
    contexts, convs, _ = world
    cache = FakeCache()
    trg = contexts.get("trigger", "trg_001_research_digest_dentists")
    merchant = contexts.get("merchant", "m_001_drmeera_dentist_delhi")
    fp = FakeComposer.input_fingerprint(contexts.get("category", "dentists"), merchant, trg, None)
    cached = {**fake._build(merchant, trg, None), "body": "Dr. Meera, composed for yesterday.",
              planner.COMPOSED_FOR_KEY: "2026-04-25"}
    cache.put(fp, cached)
    actions = tick(world, [trg["id"]], now="2026-04-26T10:30:00Z", cache=cache)
    assert actions[0]["body"] != "Dr. Meera, composed for yesterday." and len(fake.calls) == 1
    assert planner.COMPOSED_FOR_KEY not in actions[0]


def test_cache_entry_for_same_date_is_used(fake, world):
    contexts, convs, engine = world
    cache = FakeCache()
    trg = contexts.get("trigger", "trg_001_research_digest_dentists")
    merchant = contexts.get("merchant", "m_001_drmeera_dentist_delhi")
    fp = FakeComposer.input_fingerprint(contexts.get("category", "dentists"), merchant, trg, None)
    cache.put(fp, {**fake._build(merchant, trg, None), "body": "Dr. Meera, composed earlier today.",
                   planner.COMPOSED_FOR_KEY: "2026-04-26"})
    actions = tick(world, [trg["id"]], now="2026-04-26T10:30:00Z", cache=cache)
    assert actions[0]["body"] == "Dr. Meera, composed earlier today." and fake.calls == []
    assert planner.COMPOSED_FOR_KEY not in engine.opened[0][1]      # internal tag never leaks


def test_composed_for_helpers():
    assert planner.now_date("2026-04-26T23:30:00+05:30") == "2026-04-26"
    assert planner.now_date(None) == "" and planner.now_date("garbage") == ""
    tagged = {planner.COMPOSED_FOR_KEY: "2026-04-26"}
    assert planner.composed_for_ok(tagged, "2026-04-26T10:00:00Z")
    assert not planner.composed_for_ok(tagged, "2026-04-27T10:00:00Z")
    assert planner.composed_for_ok(tagged, None)                    # tick without a usable now
    assert planner.composed_for_ok({"body": "x"}, "2026-04-27T10:00:00Z")   # untagged


def test_inflight_precompose_is_awaited(fake, world):
    contexts, convs, engine = world
    trg = contexts.get("trigger", "trg_001_research_digest_dentists")
    merchant = contexts.get("merchant", "m_001_drmeera_dentist_delhi")
    fp = FakeComposer.input_fingerprint(contexts.get("category", "dentists"), merchant, trg, None)

    async def scenario():
        async def slow():
            await asyncio.sleep(0.2)
            return {**fake._build(merchant, trg, None), "body": "Dr. Meera, from the in-flight precompose."}
        task = asyncio.ensure_future(slow())
        return await planner.plan_tick(None, [trg["id"]], contexts=contexts, convs=convs, engine=engine,
                                       inflight={fp: task})

    actions = asyncio.run(scenario())
    assert actions[0]["body"] == "Dr. Meera, from the in-flight precompose."
    assert fake.calls == []


def test_bad_composition_fields_are_normalised(monkeypatch, world):
    comp = FakeComposer()

    async def sloppy(category, merchant, trigger, customer=None, **kw):
        return {"body": "  Dr. Meera, sloppy output.  ", "cta": "weird", "template_params": None}

    mod = comp.module()
    mod.compose_async = sloppy
    monkeypatch.setitem(sys.modules, "app.composer", mod)
    a = tick(world, ["trg_001_research_digest_dentists"])[0]
    assert set(a) == REQUIRED_ACTION_KEYS
    assert a["body"] == "Dr. Meera, sloppy output." and a["cta"] == "open_ended"
    assert a["template_name"] == "vera_research_digest_v1" and a["template_params"] == [a["body"]]
    assert a["suppression_key"] == "research:dentists:2026-W17" and a["rationale"]


def test_empty_body_composition_is_not_sent(monkeypatch, world):
    comp = FakeComposer()
    comp.fixed_body = "   "
    monkeypatch.setitem(sys.modules, "app.composer", comp.module())
    assert tick(world, ["trg_001_research_digest_dentists"]) == []


# --------------------------------------------------------------------------- expanded dataset (generated into tmp_path)


def test_expanded_dataset_invariants(fake, tmp_path):
    out = tmp_path / "expanded"
    subprocess.run([sys.executable, str(DATASET / "generate_dataset.py"), "--seed-dir", str(DATASET), "--out", str(out)],
                   check=True, capture_output=True)
    contexts, convs, engine = ContextStore(), ConversationStore(), FakeEngine()
    keys = {"categories": ("category", "slug"), "merchants": ("merchant", "merchant_id"),
            "customers": ("customer", "customer_id"), "triggers": ("trigger", "id")}
    for folder, (scope, key) in keys.items():
        for f in sorted((out / folder).glob("*.json")):
            payload = json.loads(f.read_text())
            contexts.put(scope, payload[key], 1, payload)
    triggers = contexts.ids("trigger")
    assert len(triggers) == 100

    seen_triggers: set[str] = set()
    for n in range(12):
        actions = planner.plan_tick_sync(f"2026-04-26T10:{n * 5:02d}:00Z", triggers, contexts=contexts,
                                         convs=convs, engine=engine)
        vera = [a["merchant_id"] for a in actions if a["send_as"] == "vera"]
        custs = [a["customer_id"] for a in actions if a["send_as"] == "merchant_on_behalf"]
        assert len(vera) == len(set(vera)) and len(custs) == len(set(custs))
        assert len(actions) <= config.settings.max_actions_per_tick
        for a in actions:
            assert set(a) == REQUIRED_ACTION_KEYS and a["trigger_id"] not in seen_triggers
            seen_triggers.add(a["trigger_id"])
            if a["customer_id"]:
                cust = contexts.get("customer", a["customer_id"])
                trg = contexts.get("trigger", a["trigger_id"])
                assert planner.consent_problem(cust, trg["kind"]) is None
    skipped = {t for t in triggers if convs.skipped_reason(t) not in (None, "suppressed")}
    assert seen_triggers.isdisjoint(skipped)
    assert all(convs.skipped_reason(t) == "suppressed" for t in seen_triggers)   # re-listed after sending
    assert len(seen_triggers) + len(skipped) + len(planner.deferred_ids(convs)) >= 95
    assert len(seen_triggers) >= 80
    planner.reset(convs)
