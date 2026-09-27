"""HTTP contract tests (app/main.py) with fake composer / conversation engine / LLM modules.

Offline, no API keys. The fakes are installed in sys.modules, and app.main resolves those
modules lazily, so these tests do not depend on the real implementations.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config, main

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
REQUIRED_ACTION_KEYS = {"conversation_id", "merchant_id", "customer_id", "send_as", "trigger_id", "template_name",
                        "template_params", "body", "cta", "suppression_key", "rationale"}

MERCHANTS = {m["merchant_id"]: m for m in json.loads((DATASET / "merchants_seed.json").read_text())["merchants"]}
CUSTOMERS = {c["customer_id"]: c for c in json.loads((DATASET / "customers_seed.json").read_text())["customers"]}
TRIGGERS = {t["id"]: t for t in json.loads((DATASET / "triggers_seed.json").read_text())["triggers"]}
DENTISTS = json.loads((DATASET / "categories" / "dentists.json").read_text())
MEERA = "m_001_drmeera_dentist_delhi"


# --------------------------------------------------------------------------- fakes


class Fakes:
    def __init__(self) -> None:
        self.compose_mode = "ok"            # ok | raise | hang
        self.compose_source = "template"    # what meta.source the fake reports
        self.compose_delay = 0.0
        self.compose_calls: list[str] = []
        self.engine_mode = "ok"             # ok | raise | hang | empty_send | junk
        self.engine_reqs: list[dict] = []
        self.opened: list[dict] = []
        self.llm_available = False
        self.llm_closed = False

    # -- app.composer
    def composer_module(self) -> types.ModuleType:
        fakes = self
        mod = types.ModuleType("app.composer")

        def input_fingerprint(category, merchant, trigger, customer):
            blob = json.dumps([category, merchant, trigger, customer], sort_keys=True, default=str)
            return hashlib.sha256(blob.encode()).hexdigest()[:16]

        def build(merchant, trigger, customer):
            kind = trigger.get("kind", "generic")
            return {
                "body": f"Dr. {merchant['identity'].get('owner_first_name', 'there')}, {kind.replace('_', ' ')} "
                        f"update ({trigger.get('id')}). Want me to draft it?",
                "cta": "binary_yes_no", "send_as": "merchant_on_behalf" if customer else "vera",
                "suppression_key": trigger.get("suppression_key") or f"{kind}:x:{trigger.get('id')}",
                "rationale": f"{kind} is live; anchored on merchant data.",
                "template_name": f"vera_{kind}_v1", "template_params": ["a", "b"],
                "meta": {"source": fakes.compose_source, "kind": kind, "offer": "draft it", "facts": []},
            }

        async def compose_async(category, merchant, trigger, customer=None, *, now=None, prior_bodies=(),
                                use_llm=True, timeout=None):
            fakes.compose_calls.append(trigger.get("id", "?"))
            if fakes.compose_mode == "raise":
                raise RuntimeError("composer down")
            if fakes.compose_mode == "hang":
                await asyncio.sleep(3600)
            if fakes.compose_delay:
                await asyncio.sleep(fakes.compose_delay)
            return build(merchant, trigger, customer)

        class ComposeCache:
            def __init__(self):
                self.data = {}

            def get(self, fp):
                return self.data.get(fp)

            def put(self, fp, value):
                self.data[fp] = value

        mod.input_fingerprint = input_fingerprint
        mod.compose_async = compose_async
        mod.compose = lambda c, m, t, cu=None, **kw: build(m, t, cu)
        mod.ComposeCache = ComposeCache
        return mod

    # -- app.conversation
    def conversation_module(self) -> types.ModuleType:
        fakes = self
        mod = types.ModuleType("app.conversation")

        class ConversationEngine:
            def __init__(self, contexts, convs, llm):
                self.contexts, self.convs, self.llm = contexts, convs, llm

            async def handle_reply(self, req):
                fakes.engine_reqs.append(req)
                if fakes.engine_mode == "raise":
                    raise RuntimeError("engine down")
                if fakes.engine_mode == "hang":
                    await asyncio.sleep(3600)
                if fakes.engine_mode == "empty_send":
                    return {"action": "send", "body": "  ", "cta": "open_ended", "rationale": "x"}
                if fakes.engine_mode == "junk":
                    return {"action": "send", "body": "Here is the draft.", "cta": "whatever"}
                return {"action": "send", "body": f"Echo: {req['message']}", "cta": "binary_confirm_cancel",
                        "rationale": "delegated"}

            def open_from_action(self, action, composition, now):
                fakes.opened.append(action)

        mod.ConversationEngine = ConversationEngine
        return mod

    # -- app.llm
    def llm_module(self) -> types.ModuleType:
        fakes = self
        mod = types.ModuleType("app.llm")

        class FakeLLM:
            def available(self):
                return fakes.llm_available

            def describe(self):
                return "fake-llm (tests)"

            async def aclose(self):
                fakes.llm_closed = True

        client = FakeLLM()
        mod.get_llm = lambda: client
        mod.set_llm = lambda c: None
        return mod


@pytest.fixture
def fakes(monkeypatch) -> Fakes:
    f = Fakes()
    monkeypatch.setitem(sys.modules, "app.composer", f.composer_module())
    monkeypatch.setitem(sys.modules, "app.conversation", f.conversation_module())
    monkeypatch.setitem(sys.modules, "app.llm", f.llm_module())
    main.reset_state()
    yield f
    main.reset_state()


@pytest.fixture
def client(fakes):
    with TestClient(main.app) as c:
        yield c


def push(client, scope, cid, version, payload):
    return client.post("/v1/context", json={"scope": scope, "context_id": cid, "version": version,
                                            "payload": payload, "delivered_at": "2026-04-26T10:00:00Z"})


def push_meera_world(client, *trigger_ids):
    assert push(client, "category", "dentists", 1, DENTISTS).status_code == 200
    assert push(client, "merchant", MEERA, 1, MERCHANTS[MEERA]).status_code == 200
    assert push(client, "customer", "c_001_priya_for_m001", 1, CUSTOMERS["c_001_priya_for_m001"]).status_code == 200
    for tid in trigger_ids:
        assert push(client, "trigger", tid, 1, TRIGGERS[tid]).status_code == 200


# --------------------------------------------------------------------------- healthz / metadata / root


def test_healthz_shape_and_counts(client):
    r = client.get("/v1/healthz")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok" and isinstance(data["uptime_seconds"], int)
    assert data["contexts_loaded"] == {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    push_meera_world(client, "trg_001_research_digest_dentists")
    push(client, "merchant", MEERA, 1, MERCHANTS[MEERA])          # stale re-push does not double count
    assert client.get("/v1/healthz").json()["contexts_loaded"] == {"category": 1, "merchant": 1, "customer": 1,
                                                                    "trigger": 1}


def test_metadata_fields(client):
    data = client.get("/v1/metadata").json()
    for key in ("team_name", "team_members", "model", "approach", "contact_email", "version", "submitted_at"):
        assert key in data
    assert data["model"] == "fake-llm (tests)"
    assert isinstance(data["team_members"], list) and data["approach"]


def test_root_and_unknown_route_are_json(client):
    assert client.get("/").json()["health"] == "/v1/healthz"
    r = client.get("/v1/nope")
    assert r.status_code == 404 and r.headers["content-type"].startswith("application/json")


def test_lifespan_closes_llm_client(fakes):
    with TestClient(main.app) as c:
        c.get("/v1/healthz")
    assert fakes.llm_closed


# --------------------------------------------------------------------------- context


def test_context_accept_stale_and_replace(client):
    r1 = push(client, "merchant", MEERA, 1, MERCHANTS[MEERA])
    assert r1.status_code == 200
    body = r1.json()
    assert body["accepted"] is True and body["ack_id"] == f"ack_{MEERA}_v1" and body["stored_at"].endswith("Z")

    r2 = push(client, "merchant", MEERA, 1, MERCHANTS[MEERA])
    assert r2.status_code == 409
    assert r2.json() == {"accepted": False, "reason": "stale_version", "current_version": 1}

    updated = json.loads(json.dumps(MERCHANTS[MEERA]))
    updated["performance"]["views"] = 2580
    r3 = push(client, "merchant", MEERA, 2, updated)
    assert r3.status_code == 200 and r3.json()["ack_id"] == f"ack_{MEERA}_v2"
    assert main.STATE.contexts.get("merchant", MEERA)["performance"]["views"] == 2580

    r4 = push(client, "merchant", MEERA, 1, MERCHANTS[MEERA])   # lower version after the bump
    assert r4.status_code == 409 and r4.json()["current_version"] == 2


def test_context_invalid_scope(client):
    r = push(client, "planet", "earth", 1, {})
    assert r.status_code == 400
    assert r.json()["accepted"] is False and r.json()["reason"] == "invalid_scope" and r.json()["details"]


@pytest.mark.parametrize("body", [
    {"scope": "merchant", "version": 1, "payload": {}},                       # no context_id
    {"scope": "merchant", "context_id": "m1", "payload": {}},                 # no version
    {"scope": "merchant", "context_id": "m1", "version": 1},                  # no payload
    {"scope": "merchant", "context_id": "", "version": 1, "payload": {}},
    {"scope": "merchant", "context_id": "m1", "version": "abc", "payload": {}},
    {"scope": "merchant", "context_id": "m1", "version": 1.5, "payload": {}},
    {"scope": "merchant", "context_id": "m1", "version": True, "payload": {}},
    {"scope": "merchant", "context_id": "m1", "version": 1, "payload": ["not", "a", "dict"]},
])
def test_context_missing_or_bad_fields_is_400_not_422(client, body):
    r = client.post("/v1/context", json=body)
    assert r.status_code == 400
    assert r.json()["accepted"] is False and r.json()["reason"] == "invalid_payload" and r.json()["details"]


def test_context_malformed_json_and_non_object(client):
    r = client.post("/v1/context", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["reason"] == "invalid_payload"
    r = client.post("/v1/context", json=[1, 2, 3])
    assert r.status_code == 400 and r.json()["reason"] == "invalid_payload"
    r = client.post("/v1/context", content=b"")
    assert r.status_code == 400


def test_context_numeric_string_version_accepted(client):
    r = push(client, "category", "dentists", "3", DENTISTS)
    assert r.status_code == 200 and r.json()["ack_id"] == "ack_dentists_v3"
    assert push(client, "category", "dentists", 3, DENTISTS).status_code == 409
    assert push(client, "category", "dentists", 4.0, DENTISTS).status_code == 200


def test_context_oversized_body_rejected(client):
    big = {"slug": "x", "blob": "a" * (main.MAX_CONTEXT_BYTES + 10)}
    r = push(client, "category", "x", 1, big)
    assert r.status_code == 400 and r.json()["reason"] == "invalid_payload" and "KB" in r.json()["details"]


def test_context_is_fast_even_if_composer_hangs(client, fakes):
    fakes.llm_available = True
    fakes.compose_mode = "hang"
    push_meera_world(client)
    start = time.perf_counter()
    r = push(client, "trigger", "trg_001_research_digest_dentists", 1, TRIGGERS["trg_001_research_digest_dentists"])
    assert r.status_code == 200 and time.perf_counter() - start < 0.5


# --------------------------------------------------------------------------- teardown


def test_teardown_wipes_everything(client):
    push_meera_world(client, "trg_001_research_digest_dentists")
    assert client.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z",
                                         "available_triggers": ["trg_001_research_digest_dentists"]}).json()["actions"]
    r = client.post("/v1/teardown")
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert client.get("/v1/healthz").json()["contexts_loaded"] == {"category": 0, "merchant": 0, "customer": 0,
                                                                    "trigger": 0}
    assert main.STATE.convs.all() == []
    # After teardown the same context can be pushed again from version 1.
    assert push(client, "merchant", MEERA, 1, MERCHANTS[MEERA]).status_code == 200


# --------------------------------------------------------------------------- tick


def test_tick_with_no_triggers(client):
    assert client.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z", "available_triggers": []}).json() == {"actions": []}
    assert client.post("/v1/tick", json={}).json() == {"actions": []}
    assert client.post("/v1/tick", content=b"garbage").json() == {"actions": []}


def test_tick_end_to_end_actions_are_complete(client, fakes):
    ids = ["trg_001_research_digest_dentists", "trg_003_recall_due_priya", "trg_023_competitor_opened_dentist"]
    push_meera_world(client, *ids)
    r = client.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z", "available_triggers": ids + ["trg_unknown"]})
    assert r.status_code == 200
    actions = r.json()["actions"]
    assert len(actions) == 2          # one merchant-facing (Meera) + one customer-facing (Priya); one deferred
    for a in actions:
        assert set(a) == REQUIRED_ACTION_KEYS and a["body"] and a["rationale"]
    assert {a["send_as"] for a in actions} == {"vera", "merchant_on_behalf"}
    assert len(fakes.opened) == 2
    nxt = client.post("/v1/tick", json={"now": "2026-04-26T10:35:00Z", "available_triggers": ids}).json()["actions"]
    assert [a["trigger_id"] for a in nxt] == ["trg_023_competitor_opened_dentist"]
    assert client.post("/v1/tick", json={"now": "2026-04-26T10:40:00Z", "available_triggers": ids}).json() == {"actions": []}


def test_tick_lenient_body_shapes(client):
    push_meera_world(client, "trg_001_research_digest_dentists")
    r = client.post("/v1/tick", json={"available_triggers": "trg_001_research_digest_dentists"})
    assert r.status_code == 200 and len(r.json()["actions"]) == 1


def test_tick_survives_composer_that_raises(client, fakes):
    fakes.compose_mode = "raise"
    push_meera_world(client, "trg_001_research_digest_dentists")
    r = client.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z",
                                      "available_triggers": ["trg_001_research_digest_dentists"]})
    assert r.status_code == 200 and r.json() == {"actions": []}


def test_tick_survives_composer_that_hangs(client, fakes, monkeypatch):
    monkeypatch.setattr(config.settings, "tick_deadline_s", 1.0)
    fakes.compose_mode = "hang"
    push_meera_world(client, "trg_001_research_digest_dentists", "trg_003_recall_due_priya")
    start = time.perf_counter()
    r = client.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z",
                                      "available_triggers": ["trg_001_research_digest_dentists", "trg_003_recall_due_priya"]})
    assert r.status_code == 200 and r.json() == {"actions": []}
    assert time.perf_counter() - start < 2.5


def test_tick_survives_planner_crash(client, monkeypatch):
    async def boom(*a, **kw):
        raise RuntimeError("planner bug")

    monkeypatch.setattr(main.planner, "plan_tick", boom)
    r = client.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z", "available_triggers": ["x"]})
    assert r.status_code == 200 and r.json() == {"actions": []}


def test_tick_works_without_conversation_module(client, monkeypatch):
    broken = types.ModuleType("app.conversation")          # no ConversationEngine attribute
    monkeypatch.setitem(sys.modules, "app.conversation", broken)
    main.reset_state()
    push_meera_world(client, "trg_001_research_digest_dentists")
    actions = client.post("/v1/tick", json={"available_triggers": ["trg_001_research_digest_dentists"]}).json()["actions"]
    assert len(actions) == 1 and main.STATE.convs.exists(actions[0]["conversation_id"])


# --------------------------------------------------------------------------- precompose


def _wait_until(pred, timeout=2.0):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def test_precompose_fills_cache_and_tick_uses_it(client, fakes):
    fakes.llm_available = True
    fakes.compose_source = "llm"
    push_meera_world(client)
    push(client, "trigger", "trg_001_research_digest_dentists", 1, TRIGGERS["trg_001_research_digest_dentists"])
    assert _wait_until(lambda: fakes.compose_calls == ["trg_001_research_digest_dentists"]
                       and not main.STATE.precomposer.inflight)
    assert len(main.STATE.cache.data) == 1
    actions = client.post("/v1/tick", json={"available_triggers": ["trg_001_research_digest_dentists"]}).json()["actions"]
    assert len(actions) == 1
    assert fakes.compose_calls == ["trg_001_research_digest_dentists"]      # served from cache, not recomposed


def test_precompose_reschedules_on_merchant_bump(client, fakes):
    fakes.llm_available = True
    fakes.compose_source = "llm"
    push_meera_world(client, "trg_001_research_digest_dentists")
    assert _wait_until(lambda: len(fakes.compose_calls) == 1 and not main.STATE.precomposer.inflight)
    updated = json.loads(json.dumps(MERCHANTS[MEERA]))
    updated["performance"]["views"] = 9999
    push(client, "merchant", MEERA, 2, updated)
    assert _wait_until(lambda: len(fakes.compose_calls) == 2 and not main.STATE.precomposer.inflight)
    assert len(main.STATE.cache.data) == 2


def test_precompose_waits_for_customer_context(client, fakes):
    fakes.llm_available = True
    fakes.compose_source = "llm"
    push(client, "category", "dentists", 1, DENTISTS)
    push(client, "merchant", MEERA, 1, MERCHANTS[MEERA])
    push(client, "trigger", "trg_003_recall_due_priya", 1, TRIGGERS["trg_003_recall_due_priya"])
    time.sleep(0.1)
    assert fakes.compose_calls == []                          # no customer context yet: nothing to compose
    push(client, "customer", "c_001_priya_for_m001", 1, CUSTOMERS["c_001_priya_for_m001"])
    assert _wait_until(lambda: fakes.compose_calls == ["trg_003_recall_due_priya"])


def test_precompose_uses_context_id_when_payload_id_differs(client, fakes):
    fakes.llm_available = True
    fakes.compose_source = "llm"
    push_meera_world(client)
    trg = dict(TRIGGERS["trg_001_research_digest_dentists"])
    trg.pop("id")
    push(client, "trigger", "trg_custom_ctx_id", 1, trg)
    assert _wait_until(lambda: len(fakes.compose_calls) == 1 and not main.STATE.precomposer.inflight)
    assert len(main.STATE.cache.data) == 1
    actions = client.post("/v1/tick", json={"available_triggers": ["trg_custom_ctx_id"]}).json()["actions"]
    assert [a["trigger_id"] for a in actions] == ["trg_custom_ctx_id"] and len(fakes.compose_calls) == 1


def test_precompose_skipped_when_llm_unavailable(client, fakes):
    fakes.llm_available = False
    push_meera_world(client, "trg_001_research_digest_dentists")
    time.sleep(0.1)
    assert fakes.compose_calls == [] and not main.STATE.precomposer.inflight


def test_precompose_skipped_when_disabled(client, fakes, monkeypatch):
    monkeypatch.setattr(config.settings, "precompose", False)
    fakes.llm_available = True
    push_meera_world(client, "trg_001_research_digest_dentists")
    time.sleep(0.1)
    assert fakes.compose_calls == []


def test_precompose_does_not_cache_template_results(client, fakes):
    fakes.llm_available = True
    fakes.compose_source = "template"
    push_meera_world(client, "trg_001_research_digest_dentists")
    assert _wait_until(lambda: len(fakes.compose_calls) == 1 and not main.STATE.precomposer.inflight)
    assert main.STATE.cache.data == {}


def test_tick_awaits_inflight_precompose(client, fakes):
    fakes.llm_available = True
    fakes.compose_source = "llm"
    fakes.compose_delay = 0.3
    push_meera_world(client, "trg_001_research_digest_dentists")
    actions = client.post("/v1/tick", json={"available_triggers": ["trg_001_research_digest_dentists"]}).json()["actions"]
    assert len(actions) == 1
    assert fakes.compose_calls == ["trg_001_research_digest_dentists"]   # the tick reused the in-flight task


def test_precompose_tags_date_and_tick_recomposes_for_other_date(client, fakes):
    fakes.llm_available = True
    fakes.compose_source = "llm"
    push_meera_world(client, "trg_001_research_digest_dentists")      # delivered_at 2026-04-26
    assert _wait_until(lambda: len(fakes.compose_calls) == 1 and not main.STATE.precomposer.inflight)
    cached = next(iter(main.STATE.cache.data.values()))
    assert cached[main.planner.COMPOSED_FOR_KEY] == "2026-04-26"
    r = client.post("/v1/tick", json={"now": "2026-04-27T09:00:00Z",
                                      "available_triggers": ["trg_001_research_digest_dentists"]})
    assert len(r.json()["actions"]) == 1 and len(fakes.compose_calls) == 2     # stale date: composed again
    assert main.planner.COMPOSED_FOR_KEY not in r.json()["actions"][0]


def test_precompose_same_date_is_served_from_cache(client, fakes):
    fakes.llm_available = True
    fakes.compose_source = "llm"
    push_meera_world(client, "trg_001_research_digest_dentists")
    assert _wait_until(lambda: len(fakes.compose_calls) == 1 and not main.STATE.precomposer.inflight)
    r = client.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z",
                                      "available_triggers": ["trg_001_research_digest_dentists"]})
    assert len(r.json()["actions"]) == 1 and len(fakes.compose_calls) == 1


def test_precompose_skips_opted_out_merchant(client, fakes):
    fakes.llm_available = True
    fakes.compose_source = "llm"
    push_meera_world(client)
    main.STATE.convs.merchant(MEERA).opted_out = True
    push(client, "trigger", "trg_001_research_digest_dentists", 1, TRIGGERS["trg_001_research_digest_dentists"])
    time.sleep(0.1)
    assert fakes.compose_calls == [] and not main.STATE.precomposer.inflight


def test_inflight_view_claims_queued_tasks():
    pre = main.Precomposer(main.BotState())

    async def scenario():
        running = asyncio.ensure_future(asyncio.sleep(10))
        queued = asyncio.ensure_future(asyncio.sleep(10))
        pre.inflight.update({"fp_run": running, "fp_queue": queued})
        pre.started.add("fp_run")
        view = pre.view()
        assert bool(view) and list(view) == ["fp_run"]
        assert view.get("fp_run") is running
        assert view.get("fp_queue") is None                  # claimed: cancelled and forgotten
        assert view.get("fp_missing") is None
        await asyncio.sleep(0)
        assert queued.cancelled() and "fp_queue" not in pre.inflight and not running.done()
        running.cancel()

    asyncio.run(scenario())


def test_tick_claims_queued_precompose_instead_of_waiting(client, fakes, monkeypatch):
    monkeypatch.setattr(config.settings, "precompose_concurrency", 1)
    monkeypatch.setattr(config.settings, "tick_deadline_s", 3.0)
    main.reset_state()
    fakes.llm_available = True
    fakes.compose_source = "llm"
    fakes.compose_delay = 1.0
    bharat = "m_002_bharat_dentist_mumbai"
    push_meera_world(client)
    push(client, "merchant", bharat, 1, MERCHANTS[bharat])
    ids = ["trg_001_research_digest_dentists", "trg_004_perf_dip_bharat"]
    for tid in ids:
        push(client, "trigger", tid, 1, TRIGGERS[tid])
    start = time.perf_counter()
    actions = client.post("/v1/tick", json={"now": "2026-04-26T10:30:00Z", "available_triggers": ids}).json()["actions"]
    elapsed = time.perf_counter() - start
    assert sorted(a["trigger_id"] for a in actions) == sorted(ids)
    assert elapsed < 1.9                  # waiting in line behind the first precompose would take ~2s
    assert sorted(fakes.compose_calls) == sorted(ids)      # the claimed task never composed


# --------------------------------------------------------------------------- reply


def reply(client, message, conv="conv_x", **extra):
    body = {"conversation_id": conv, "merchant_id": MEERA, "customer_id": None, "from_role": "merchant",
            "message": message, "received_at": "2026-04-26T10:42:00Z", "turn_number": 2, **extra}
    return client.post("/v1/reply", json=body)


def test_reply_delegates_to_engine(client, fakes):
    r = reply(client, "Yes please send the abstract")
    assert r.status_code == 200
    assert r.json() == {"action": "send", "body": "Echo: Yes please send the abstract", "cta": "binary_confirm_cancel",
                        "rationale": "delegated"}
    req = fakes.engine_reqs[-1]
    assert req["conversation_id"] == "conv_x" and req["merchant_id"] == MEERA and req["turn_number"] == 2
    assert req["from_role"] == "merchant" and req["received_at"] == "2026-04-26T10:42:00Z"


def test_reply_survives_engine_that_raises(client, fakes):
    fakes.engine_mode = "raise"
    r = reply(client, "Tell me more about this")
    assert r.status_code == 200
    data = r.json()
    assert data["action"] == "wait" and data["wait_seconds"] == 1800 and data["rationale"]
    r = reply(client, "Not interested. Stop messaging me.")
    assert r.json()["action"] == "end"
    data = reply(client, "Ok lets do it. Whats next?").json()
    assert data["action"] == "send" and data["cta"] == "binary_confirm_cancel" and "draft" in data["body"]
    assert not any(q in data["body"].lower() for q in ("would you", "do you", "can you tell", "what if", "how about"))


def test_reply_survives_engine_that_hangs(client, fakes, monkeypatch):
    monkeypatch.setattr(config.settings, "reply_deadline_s", 0.2)
    fakes.engine_mode = "hang"
    start = time.perf_counter()
    r = reply(client, "Hello?")
    assert r.status_code == 200 and r.json()["action"] == "wait"
    assert time.perf_counter() - start < 2.0


def test_reply_never_returns_empty_send(client, fakes):
    fakes.engine_mode = "empty_send"
    data = reply(client, "hmm, tell me more").json()
    assert data["action"] in {"wait", "end"}
    data = reply(client, "ok").json()
    assert data["action"] != "send" or data["body"].strip()
    fakes.engine_mode = "junk"
    data = reply(client, "ok").json()
    assert data["action"] == "send" and data["cta"] == "open_ended" and data["rationale"]


def test_reply_lenient_body(client, fakes):
    r = client.post("/v1/reply", json={"message": "hi"})
    assert r.status_code == 200 and r.json()["action"] == "send"
    assert fakes.engine_reqs[-1]["conversation_id"] == "conv_unknown"
    r = client.post("/v1/reply", content=b"nonsense")
    assert r.status_code == 200 and r.json()["action"] in {"send", "wait", "end"}


def test_reply_without_engine_uses_safe_fallback(client, monkeypatch):
    monkeypatch.setitem(sys.modules, "app.conversation", types.ModuleType("app.conversation"))
    main.reset_state()
    assert reply(client, "What is this about?").json()["action"] == "wait"
    assert reply(client, "stop sending these, band karo").json()["action"] == "end"


# --------------------------------------------------------------------------- bot.py


def test_bot_module_exposes_app_and_compose(fakes):
    import bot

    assert bot.app is main.app
    out = bot.compose(DENTISTS, MERCHANTS[MEERA], TRIGGERS["trg_001_research_digest_dentists"])
    assert set(out) == {"body", "cta", "send_as", "suppression_key", "rationale"}
    assert out["send_as"] == "vera" and out["suppression_key"] == "research:dentists:2026-W17"
