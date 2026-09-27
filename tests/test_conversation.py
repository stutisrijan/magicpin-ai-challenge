"""Conversation engine tests (app/conversation.py, app/reply_templates.py, conversation_handlers.py).

Offline, no API keys: the engine runs with llm=None (deterministic reply templates) unless a
test injects a fake LLM. Openers are registered with hand-written compositions so these tests
do not depend on the composer.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

import conversation_handlers
from app import config
from app import reply_templates as rt
from app.conversation import (
    ConversationEngine,
    check_body,
    classify_inbound,
    extract_ask,
    parse_wait_seconds,
)
from app.schemas import CTA_VALUES
from app.state import ContextStore, Conversation, ConversationStore, MerchantState

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
NOW = "2026-04-26T10:30:00Z"
MEERA = "m_001_drmeera_dentist_delhi"
BHARAT = "m_002_bharat_dentist_mumbai"
QUALIFYING = ["would you", "do you", "can you tell", "what if", "how about"]
ACTIONING = ["done", "sending", "draft", "here", "confirm", "proceed", "next"]
AUTO_MSG = "Thank you for contacting us! Our team will respond shortly."


# --------------------------------------------------------------------------- fixtures / helpers


def _load_seed() -> dict:
    cats = [json.loads(p.read_text()) for p in sorted((DATASET / "categories").glob("*.json"))]
    merchants = json.loads((DATASET / "merchants_seed.json").read_text())["merchants"]
    customers = json.loads((DATASET / "customers_seed.json").read_text())["customers"]
    triggers = json.loads((DATASET / "triggers_seed.json").read_text())["triggers"]
    return {"categories": cats, "merchants": merchants, "customers": customers, "triggers": triggers}


SEED = _load_seed()
TRIGGERS = {t["id"]: t for t in SEED["triggers"]}
MERCHANTS = {m["merchant_id"]: m for m in SEED["merchants"]}


def _store(data: dict) -> ContextStore:
    cs = ContextStore()
    for c in data["categories"]:
        cs.put("category", c["slug"], 1, c)
    for m in data["merchants"]:
        cs.put("merchant", m["merchant_id"], 1, m)
    for c in data["customers"]:
        cs.put("customer", c["customer_id"], 1, c)
    for t in data["triggers"]:
        cs.put("trigger", t["id"], 1, t)
    return cs


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("LLM_DISABLED", "1")
    config.reload_settings()
    yield
    config.reload_settings()


@pytest.fixture
def world():
    contexts = _store(SEED)
    convs = ConversationStore()
    return contexts, convs, ConversationEngine(contexts, convs, None)


def reply(engine, cid, message, *, merchant_id=MEERA, customer_id=None, role="merchant", turn=2, at=NOW) -> dict:
    return asyncio.run(engine.handle_reply({
        "conversation_id": cid, "merchant_id": merchant_id, "customer_id": customer_id, "from_role": role,
        "message": message, "received_at": at, "turn_number": turn,
    }))


def open_conv(engine, trigger_id, body, *, offer="", cid=None, cta="binary_yes_no") -> str:
    trg = TRIGGERS[trigger_id]
    cid = cid or f"conv_{trigger_id}"
    customer_facing = bool(trg.get("customer_id"))
    action = {
        "conversation_id": cid, "merchant_id": trg["merchant_id"], "customer_id": trg.get("customer_id"),
        "send_as": "merchant_on_behalf" if customer_facing else "vera", "trigger_id": trigger_id,
        "template_name": f"vera_{trg['kind']}_v1", "template_params": [body], "body": body, "cta": cta,
        "suppression_key": trg.get("suppression_key") or trigger_id, "rationale": "test opener",
    }
    composition = {"body": body, "cta": cta, "meta": {"offer": offer, "kind": trg["kind"], "facts": [],
                                                      "language": "en", "trigger_id": trigger_id}}
    engine.open_from_action(action, composition, NOW)
    return cid


def assert_shape(res: dict) -> None:
    assert isinstance(res, dict)
    assert res.get("action") in ("send", "wait", "end")
    assert isinstance(res.get("rationale"), str) and res["rationale"].strip()
    if res["action"] == "send":
        assert isinstance(res.get("body"), str) and res["body"].strip()
        assert res.get("cta") in CTA_VALUES
        assert "http" not in res["body"].lower()
        assert "dr. dr" not in res["body"].lower()
    elif res["action"] == "wait":
        assert isinstance(res.get("wait_seconds"), int) and res["wait_seconds"] > 0


def assert_action_mode(res: dict) -> None:
    assert res["action"] == "send", res
    low = res["body"].lower()
    assert not any(q in low for q in QUALIFYING), res["body"]
    assert any(w in low for w in ACTIONING), res["body"]
    assert res["cta"] == "binary_confirm_cancel"


RESEARCH_OPENER = ("Dr. Meera, JIDA Oct 2026, p.14 has one for your 124 high-risk adult patients: 3-month fluoride "
                   "recall cut caries recurrence 38% vs 6-month (2,100-patient trial). Want me to pull the key points "
                   "and draft a patient WhatsApp you can forward?")
CDE_OPENER = ("Dr. Meera, IDA Delhi's session on digital impressions is on 2 May 2026, 7pm (2 CDE credits, free for "
              "IDA members). Want me to block the slot and send the details on the day?")
RECALL_OPENER = ("Hi Priya, Dr. Meera's Dental Clinic here. Your 6-month cleaning is due; slots: Wed 5 Nov, 6pm or "
                 "Thu 6 Nov, 5pm. Dental Cleaning @ ₹299. Reply 1 for Wed or 2 for Thu.")
RENEWAL_OPENER = ("Dr. Bharat, 12 days left on your Pro plan (renewal ₹4,999). Want me to set up the renewal for your "
                  "confirmation?")


# --------------------------------------------------------------------------- judge_simulator scenarios


def test_judge_auto_reply_hell_across_conversations(world):
    contexts, convs, engine = world
    results = [reply(engine, f"conv_auto_{i}", AUTO_MSG, turn=i + 1) for i in range(1, 5)]
    for r in results:
        assert_shape(r)
    assert [r["action"] for r in results] == ["send", "wait", "end", "end"]
    assert results[0]["cta"] == "binary_yes_no"
    assert "auto-reply" in results[0]["body"].lower()
    assert results[1]["wait_seconds"] == 86400
    assert convs.merchant(MEERA).cooldown_until is not None


def test_judge_auto_reply_hell_unknown_merchant(world):
    _, _, engine = world
    actions = [reply(engine, f"conv_auto_{i}", AUTO_MSG, merchant_id="m_not_pushed", turn=i + 1)["action"]
               for i in range(1, 5)]
    assert actions[:3] == ["send", "wait", "end"]


def test_judge_intent_transition(world):
    _, _, engine = world
    res = reply(engine, "conv_intent_1", "Ok lets do it. Whats next?")
    assert_shape(res)
    assert_action_mode(res)


def test_judge_hostile(world):
    _, convs, engine = world
    res = reply(engine, "conv_hostile", "Stop messaging me. This is useless spam.")
    assert res["action"] == "end"
    assert convs.merchant(MEERA).opted_out is True


# --------------------------------------------------------------------------- api-call-examples replays


def test_replay_auto_reply_same_conversation(world):
    _, _, engine = world
    cid = open_conv(engine, "trg_022_cde_webinar_dentists", CDE_OPENER)
    canned = "Thank you for contacting Dr. Meera's Dental Clinic! Our team will respond shortly."
    results = [reply(engine, cid, canned, turn=t) for t in (2, 3, 4)]
    assert [r["action"] for r in results] == ["send", "wait", "end"]
    assert results[0]["cta"] == "binary_yes_no"
    assert "block the slot" in results[0]["body"]          # the nudge references the thread's own offer
    assert results[1]["wait_seconds"] == 86400


def test_replay_intent_transition_after_qualifying(world):
    _, convs, engine = world
    cid = open_conv(engine, "trg_001_research_digest_dentists", RESEARCH_OPENER)
    for msg, turn in (("Which patients does it apply to?", 2), ("Mostly diabetics and smokers in my practice", 4)):
        assert reply(engine, cid, msg, turn=turn)["action"] == "send"
    res = reply(engine, cid, "Ok, let's do it. What's next?", turn=6)
    assert_action_mode(res)
    assert "JIDA Oct 2026, p.14" in res["body"]              # the artifact is the research summary itself
    assert convs.get(cid).stage == "action"


def test_replay_hostile_then_off_topic(world):
    _, convs, engine = world
    cid = open_conv(engine, "trg_001_research_digest_dentists", RESEARCH_OPENER)
    res = reply(engine, cid, "Why are you bothering me. This is useless. Stop sending these.")
    assert res["action"] == "end"
    res = reply(engine, cid, "Btw can you also help me with my GST filing this month?", turn=4)
    assert_shape(res)
    assert res["action"] == "send" and "CA" in res["body"]
    assert convs.merchant(MEERA).opted_out is True            # answering politely does not re-subscribe


def test_gst_curveball_redirects_to_offer(world):
    _, _, engine = world
    cid = open_conv(engine, "trg_001_research_digest_dentists", RESEARCH_OPENER)
    res = reply(engine, cid, "Btw can you also help me with my GST filing this month?")
    assert_shape(res)
    assert res["action"] == "send" and res["cta"] == "open_ended"
    assert "CA" in res["body"] and "JIDA" in res["body"]


def test_abuse_without_stop_apologises_once_then_ends(world):
    _, _, engine = world
    cid = open_conv(engine, "trg_001_research_digest_dentists", RESEARCH_OPENER)
    first = reply(engine, cid, "You people are useless, total nonsense.")
    assert first["action"] == "send" and first["cta"] == "none"
    assert any(w in first["body"].lower() for w in ("sorry", "apolog", "won't"))
    off = reply(engine, cid, "Can you help me get a loan?", turn=4)
    assert off["action"] == "send" and "bank" in off["body"].lower()
    assert reply(engine, cid, "Idiots. Pathetic service.", turn=6)["action"] == "end"


# --------------------------------------------------------------------------- waits


@pytest.mark.parametrize("text,secs", [
    ("tomorrow", 86400), ("Call me tomorrow", 86400), ("in 2 hours", 7200), ("busy", 1800),
    ("I'm in a meeting", 1800), ("next week", 604800), ("kal", 86400), ("2 ghante baad", 7200),
    ("after 30 mins", 1800), ("half an hour", 1800), ("day after tomorrow", 172800),
])
def test_wait_parsing(text, secs):
    assert parse_wait_seconds(text) == secs


def test_wait_request_sets_waiting_state(world):
    _, convs, engine = world
    cid = open_conv(engine, "trg_001_research_digest_dentists", RESEARCH_OPENER)
    res = reply(engine, cid, "Busy right now, call me in 2 hours")
    assert res == {**res, "action": "wait", "wait_seconds": 7200}
    conv = convs.get(cid)
    assert conv.status == "waiting" and conv.wait_until.startswith("2026-04-26T12:30")


# --------------------------------------------------------------------------- Hinglish / Hindi


def test_hinglish_haan_karo_is_action(world):
    _, _, engine = world
    cid = open_conv(engine, "trg_004_perf_dip_bharat", "Dr. Bharat, calls down 50% this week. Post daal doon?",
                    offer="draft a fresh Google post plus an offer line for their profile")
    res = reply(engine, cid, "haan karo", merchant_id=BHARAT)
    assert_action_mode(res)
    assert "CONFIRM reply karein" in res["body"]            # mirrors the merchant's Hinglish


def test_hinglish_later_is_wait(world):
    _, _, engine = world
    cid = open_conv(engine, "trg_004_perf_dip_bharat", "Dr. Bharat, calls down 50% this week. Post daal doon?")
    res = reply(engine, cid, "abhi nahi, kal baat karte hain", merchant_id=BHARAT)
    assert res["action"] == "wait" and res["wait_seconds"] == 86400


def test_hinglish_join_intent_goes_to_action_not_qualifying(world):
    _, _, engine = world
    res = reply(engine, "conv_join_1", "Mujhe magicpin judna hai", merchant_id=BHARAT)
    assert_action_mode(res)


def test_devanagari_intents(world):
    conv = Conversation("x", kind="research_digest")
    st = MerchantState("m")
    assert classify_inbound("हाँ, कर दो", conv, st).label == "accept"
    assert classify_inbound("मत भेजो", conv, st).label == "opt_out"
    assert classify_inbound("बाद में बात करते हैं", conv, st).label == "wait_request"
    assert classify_inbound("कल बात करेंगे", conv, st).details["wait_seconds"] == 86400


# --------------------------------------------------------------------------- customers


def test_customer_slot_choice_confirms_exact_label(world):
    _, convs, engine = world
    trg = TRIGGERS["trg_003_recall_due_priya"]
    cid = open_conv(engine, trg["id"], RECALL_OPENER, cta="multi_choice_slot")
    res = reply(engine, cid, "1", merchant_id=trg["merchant_id"], customer_id=trg["customer_id"], role="customer")
    assert_shape(res)
    assert res["action"] == "send" and "Wed 5 Nov, 6pm" in res["body"]
    assert "vera" not in res["body"].lower() and "magicpin" not in res["body"].lower()
    assert convs.get(cid).stage == "done"


def test_customer_day_name_and_yes_flows(world):
    _, _, engine = world
    trg = TRIGGERS["trg_003_recall_due_priya"]
    cid = open_conv(engine, trg["id"], RECALL_OPENER, cid="conv_priya_a")
    res = reply(engine, cid, "Thursday works for me", merchant_id=trg["merchant_id"],
                customer_id=trg["customer_id"], role="customer")
    assert "Thu 6 Nov, 5pm" in res["body"]
    cid = open_conv(engine, trg["id"], RECALL_OPENER, cid="conv_priya_b")
    # plain "Yes" to a two-slot offer: hold the first slot, mention the alternative, ask to confirm
    res = reply(engine, cid, "Yes", merchant_id=trg["merchant_id"], customer_id=trg["customer_id"], role="customer")
    assert res["cta"] == "binary_confirm_cancel" and "Wed 5 Nov, 6pm" in res["body"] and "Thu 6 Nov, 5pm" in res["body"]
    res = reply(engine, cid, "CONFIRM", merchant_id=trg["merchant_id"], customer_id=trg["customer_id"],
                role="customer", turn=3)
    assert res["action"] == "send" and "Wed 5 Nov, 6pm" in res["body"] and "Thu 6 Nov" not in res["body"]


def test_customer_opt_out_does_not_opt_out_merchant(world):
    _, convs, engine = world
    trg = TRIGGERS["trg_003_recall_due_priya"]
    cid = open_conv(engine, trg["id"], RECALL_OPENER)
    res = reply(engine, cid, "Please stop sending me these", merchant_id=trg["merchant_id"],
                customer_id=trg["customer_id"], role="customer")
    assert res["action"] == "end"
    assert convs.get(cid).meta.get("customer_opted_out") is True
    assert convs.merchant(trg["merchant_id"]).opted_out is False


def test_relay_refill_customer_is_addressed_about_the_senior(world):
    _, _, engine = world
    trg = TRIGGERS["trg_019_chronic_refill_grandfather"]
    cid = open_conv(engine, trg["id"], "Namaste, Sharma ji ki dawaiyan 28 Apr ko khatam ho rahi hain. Bhej dein?")
    res = reply(engine, cid, "Haan bhej dijiye", merchant_id=trg["merchant_id"], customer_id=trg["customer_id"],
                role="customer")
    assert res["action"] == "send" and "Sharma ji" in res["body"]
    assert "metformin" in res["body"]


# --------------------------------------------------------------------------- flow rules


def test_anti_repetition_across_turns(world):
    _, convs, engine = world
    cid = open_conv(engine, "trg_001_research_digest_dentists", RESEARCH_OPENER)
    for turn, msg in enumerate(["Thank you!", "Thank you!", "What's the price?", "What's the price?"], start=2):
        res = reply(engine, cid, msg, turn=turn)
        assert_shape(res)
    bodies = convs.get(cid).bot_bodies()
    assert len(bodies) == len(set(bodies))


def test_accept_then_confirm_then_thanks_closes(world):
    _, convs, engine = world
    cid = open_conv(engine, "trg_001_research_digest_dentists", RESEARCH_OPENER)
    assert_action_mode(reply(engine, cid, "Yes please send the abstract. Also draft the patient WhatsApp."))
    done = reply(engine, cid, "CONFIRM", turn=4)
    assert done["action"] == "send" and done["cta"] == "none" and "done" in done["body"].lower()
    assert convs.get(cid).stage == "done"
    assert reply(engine, cid, "Thanks!", turn=6)["action"] == "end"


def test_edit_request_in_action_stage(world):
    _, _, engine = world
    cid = open_conv(engine, "trg_001_research_digest_dentists", RESEARCH_OPENER)
    reply(engine, cid, "yes")
    res = reply(engine, cid, "Make the patient note shorter please", turn=4)
    assert res["action"] == "send" and res["cta"] == "binary_confirm_cancel" and "shorter" in res["body"]


def test_yes_but_question_answers_then_acts(world):
    _, _, engine = world
    cid = open_conv(engine, "trg_005_renewal_due_bharat", RENEWAL_OPENER,
                    offer="set up the plan renewal for their confirmation")
    res = reply(engine, cid, "Yes but what's the price?", merchant_id=BHARAT)
    assert_action_mode(res)
    assert "₹4,999" in res["body"]


def test_decline_ends_politely(world):
    _, convs, engine = world
    cid = open_conv(engine, "trg_001_research_digest_dentists", RESEARCH_OPENER)
    res = reply(engine, cid, "No thanks, not needed")
    assert res["action"] == "end"
    assert convs.merchant(MEERA).opted_out is False


def test_reopen_after_opt_out_on_genuine_question(world):
    _, _, engine = world
    cid = open_conv(engine, "trg_022_cde_webinar_dentists", CDE_OPENER)
    assert reply(engine, cid, "Not interested. Stop messaging me.")["action"] == "end"
    res = reply(engine, cid, "Actually, when is the webinar?", turn=4)
    assert_shape(res)
    assert res["action"] == "send" and "2 May 2026" in res["body"] and res["cta"] == "none"
    assert reply(engine, cid, "ok stop now", turn=6)["action"] == "end"
    assert reply(engine, cid, "hmm", turn=8)["action"] == "end"


def test_owner_replies_after_auto_reply_end_reopens(world):
    _, _, engine = world
    cid = open_conv(engine, "trg_022_cde_webinar_dentists", CDE_OPENER)
    for turn in (2, 3, 4):
        reply(engine, cid, AUTO_MSG, turn=turn)
    res = reply(engine, cid, "Sorry, was busy. Yes, go ahead and block it", turn=5)
    assert_action_mode(res)


def test_unknown_merchant_and_conversation(world):
    _, _, engine = world
    for i, msg in enumerate(["Yes let's do it", "What does this cost?", "hmm ok", "who are you?"]):
        res = reply(engine, f"conv_ghost_{i}", msg, merchant_id="m_999_ghost")
        assert_shape(res)
        assert res["action"] == "send"
    res = reply(engine, "conv_ghost_x", "stop", merchant_id=None)
    assert res["action"] == "end"


def test_unknown_conversation_inherits_merchants_recent_thread(world):
    _, _, engine = world
    open_conv(engine, "trg_001_research_digest_dentists", RESEARCH_OPENER)
    res = reply(engine, "conv_replay_new", "ok let's do it")
    assert_action_mode(res)
    assert "JIDA" in res["body"]


@pytest.mark.parametrize("message", [
    "", "   ", "👍", "?", "1", "ok", "Yes", "no", "Thank you!", "stop", "STOP", "later", "what?", "₹₹₹",
    "a" * 3000, "हाँ", "Hi Vera", "Send me the details now", "Kya hai yeh?", "gst", "Tell me more",
])
def test_every_response_shape_is_valid(world, message):
    _, _, engine = world
    cid = open_conv(engine, "trg_001_research_digest_dentists", RESEARCH_OPENER, cid=f"conv_shape_{hash(message)}")
    assert_shape(reply(engine, cid, message))


def test_garbage_requests_never_raise(world):
    _, _, engine = world
    for req in ({}, {"message": None}, {"conversation_id": 123, "message": 42}, {"received_at": "not-a-date"}):
        assert_shape(asyncio.run(engine.handle_reply(req)))
    assert_shape(asyncio.run(engine.handle_reply(None)))   # type: ignore[arg-type]


def test_open_from_action_is_idempotent(world):
    _, convs, engine = world
    cid = open_conv(engine, "trg_001_research_digest_dentists", RESEARCH_OPENER)
    open_conv(engine, "trg_001_research_digest_dentists", RESEARCH_OPENER)
    conv = convs.get(cid)
    assert len(conv.bot_bodies()) == 1
    assert conv.kind == "research_digest" and conv.trigger_id == "trg_001_research_digest_dentists"
    assert conv.meta["ask"].startswith("pull the key points")


def test_extract_ask():
    assert extract_ask("Dr. X, big news. Want me to draft 3 posts you can review?") == "draft 3 posts you can review"
    assert extract_ask("Kya main post draft kar doon?") == ""


# --------------------------------------------------------------------------- classifier edge cases


@pytest.mark.parametrize("text,stage,label", [
    ("Thank you!", "opened", "engaged"),
    ("Thank you for contacting us! Our team will respond shortly.", "opened", "auto_reply"),
    ("Aapki jaankari ke liye bahut-bahut shukriya. Main aapki yeh sabhi baatein hamari team tak pahuncha deti hoon.",
     "opened", "auto_reply"),
    ("Aapki madad ke liye shukriya, lekin main ek automated assistant hoon", "opened", "auto_reply"),
    ("no problem, go ahead", "opened", "accept"),
    ("stop by tomorrow", "opened", "wait_request"),
    ("yes but what's the price?", "opened", "question"),
    ("ok", "opened", "accept"),
    ("ok", "action", "accept"),
    ("ok thanks", "action", "thanks_close"),
    ("Not interested. Stop messaging me.", "opened", "opt_out"),
    ("Please don't message me again", "opened", "opt_out"),
    ("haan kyun nahi", "opened", "accept"),
    ("No, I already have someone for this", "opened", "decline"),
    ("Is the renewal inclusive of GST?", "opened", "question"),
    ("Can you file my GST return?", "opened", "off_topic"),
    ("Sure, when do we start?", "opened", "accept"),
    ("Let me think about it", "opened", "wait_request"),
    ("change the headline to mention whitening", "action", "edit"),
    ("focus on whitening and aligners", "opened", "accept"),
])
def test_classifier(text, stage, label):
    conv = Conversation("x", kind="research_digest", stage=stage)
    assert classify_inbound(text, conv, MerchantState("m")).label == label


def test_repeated_genuine_message_is_not_auto_reply(world):
    _, _, engine = world
    for i in range(3):
        cid = open_conv(engine, "trg_001_research_digest_dentists", RESEARCH_OPENER, cid=f"conv_rep_{i}")
        assert reply(engine, cid, "abhi nahi, kal baat karte hain")["action"] == "wait"


def test_repeated_unknown_canned_text_becomes_auto_reply(world):
    _, _, engine = world
    canned = "Namaste! Welcome to Bharat Dental Care. Timings 10 to 8, Monday to Saturday."
    actions = [reply(engine, f"conv_canned_{i}", canned, merchant_id=BHARAT)["action"] for i in range(4)]
    assert actions[-1] == "end" and "wait" in actions


# --------------------------------------------------------------------------- grounding of template replies


@pytest.mark.parametrize("trigger_id", sorted(TRIGGERS))
def test_template_replies_are_grounded_for_every_seed_trigger(world, trigger_id):
    _, convs, engine = world
    trg = TRIGGERS[trigger_id]
    customer_facing = bool(trg.get("customer_id"))
    cid = open_conv(engine, trigger_id, f"Opener for {trg['kind'].replace('_', ' ')}. Want me to set it up?")
    role = "customer" if customer_facing else "merchant"
    for turn, msg in enumerate(["Yes please", "What's the price?", "haan karo"], start=2):
        res = reply(engine, cid, msg, merchant_id=trg["merchant_id"], customer_id=trg.get("customer_id"),
                    role=role, turn=turn)
        assert_shape(res)
        if res["action"] != "send":
            continue
        conv = convs.get(cid)
        facts = engine._facts(conv, customer_facing, NOW)
        move = conv.turns[-1].meta.get("move")
        mode = "reply_action" if move in ("action", "answer_action", "edit") else "reply"
        issues = [i for i in check_body(res["body"], facts, prior_bodies=conv.bot_bodies()[:-1], mode=mode,
                                        customer_facing=customer_facing) if not i.startswith("too_short")]
        assert issues == [], (msg, res["body"], issues)


def test_expanded_dataset_test_pairs(tmp_path):
    """Regenerate the expanded dataset and run the official test pairs through a commit + a question."""
    out = tmp_path / "expanded"
    proc = subprocess.run([sys.executable, str(DATASET / "generate_dataset.py"), "--seed-dir", str(DATASET),
                           "--out", str(out)], capture_output=True, text=True, cwd=ROOT, timeout=120)
    if proc.returncode != 0 or not (out / "test_pairs.json").exists():
        pytest.skip(f"dataset generator unavailable: {proc.stderr[-300:]}")

    def load(sub: str) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted((out / sub).glob("*.json"))]

    data = {"categories": load("categories"), "merchants": load("merchants"), "customers": load("customers"),
            "triggers": load("triggers")}
    contexts, convs = _store(data), ConversationStore()
    engine = ConversationEngine(contexts, convs, None)
    pairs = json.loads((out / "test_pairs.json").read_text())["pairs"]
    for p in pairs:
        trg = contexts.get("trigger", p["trigger_id"])
        customer_facing = bool(p.get("customer_id"))
        cid = f"conv_{p['test_id']}"
        engine.open_from_action({"conversation_id": cid, "merchant_id": p["merchant_id"],
                                 "customer_id": p.get("customer_id"), "trigger_id": p["trigger_id"],
                                 "send_as": "merchant_on_behalf" if customer_facing else "vera",
                                 "body": "Opener. Want me to go ahead?", "cta": "binary_yes_no"},
                                {"meta": {"kind": trg["kind"], "offer": ""}}, NOW)
        role = "customer" if customer_facing else "merchant"
        for turn, msg in enumerate(["Ok lets do it. Whats next?", "kitna lagega?"], start=2):
            res = asyncio.run(engine.handle_reply({
                "conversation_id": cid, "merchant_id": p["merchant_id"], "customer_id": p.get("customer_id"),
                "from_role": role, "message": msg, "received_at": NOW, "turn_number": turn}))
            assert_shape(res)
            if res["action"] == "send" and turn == 2 and not customer_facing:
                assert_action_mode(res)
            if res["action"] == "send":
                conv = convs.get(cid)
                facts = engine._facts(conv, customer_facing, NOW)
                issues = [i for i in check_body(res["body"], facts, mode="reply", customer_facing=customer_facing)
                          if i.split(":")[0] in ("ungrounded_number", "url", "jargon", "taboo", "brand_mention",
                                                 "unapproved_price", "dr_dr")]
                assert issues == [], (p["test_id"], res["body"], issues)


# --------------------------------------------------------------------------- LLM path


class FakeLLM:
    def __init__(self, body: str | None) -> None:
        self.body = body
        self.calls: list[tuple[str, str]] = []

    def available(self) -> bool:
        return True

    async def complete_json(self, system, user, *, schema=None, timeout=None, max_tokens=800, temperature=0.0):
        self.calls.append((system, user))
        return None if self.body is None else {"body": self.body, "rationale": "fake"}


def test_llm_body_used_when_valid(world):
    contexts, convs, _ = world
    good = ("Dr. Meera, here's the summary: JIDA Oct 2026, p.14 found 38% lower caries recurrence with a 3-month "
            "recall in high-risk adults. Draft patient note is ready below.\n\n"
            "\"Hi! Quick update from our clinic.\"\n\n"
            "Reply CONFIRM and I'll send it to your patient list.")
    llm = FakeLLM(good)
    engine = ConversationEngine(contexts, convs, llm)
    cid = open_conv(engine, "trg_001_research_digest_dentists", RESEARCH_OPENER)
    res = reply(engine, cid, "Yes please")
    assert llm.calls and "REQUIRED MOVE" in llm.calls[0][1]
    assert res["body"] == good and "LLM" in res["rationale"]


@pytest.mark.parametrize("bad", [
    "Great! Would you like me to check which patients are diabetic first? Reply CONFIRM.",   # qualifying
    "Done, here's the draft: 73% of your patients will return. Reply CONFIRM to send.",       # ungrounded number
    None,                                                                                     # provider failed
])
def test_llm_body_rejected_falls_back_to_template(world, bad):
    contexts, convs, _ = world
    engine = ConversationEngine(contexts, convs, FakeLLM(bad))
    cid = open_conv(engine, "trg_001_research_digest_dentists", RESEARCH_OPENER)
    res = reply(engine, cid, "Yes please")
    assert_action_mode(res)
    assert res["body"] != bad and "JIDA Oct 2026, p.14" in res["body"]


def test_llm_not_used_for_exact_moves(world):
    contexts, convs, _ = world
    llm = FakeLLM("anything")
    engine = ConversationEngine(contexts, convs, llm)
    reply(engine, "conv_auto_x", AUTO_MSG)
    trg = TRIGGERS["trg_003_recall_due_priya"]
    cid = open_conv(engine, trg["id"], RECALL_OPENER)
    reply(engine, cid, "2", merchant_id=trg["merchant_id"], customer_id=trg["customer_id"], role="customer")
    assert llm.calls == []


# --------------------------------------------------------------------------- templates


def test_offer_phrase_is_second_person_and_short():
    offer = "pull the key points of the study and draft a customer-education WhatsApp note they can forward"
    assert "they" not in rt.offer_phrase(offer, "research_digest").split()
    assert rt.offer_phrase("hold a convenient slot for the customer", "recall_due").startswith("hold")
    assert len(rt.offer_phrase(offer, "research_digest", short=True)) <= 60


def test_first_sentence_keeps_abbreviations():
    assert rt.first_sentence("Speaker: Dr. R. Mehta. Covers Primescan 2.") == "Speaker: Dr. R. Mehta"


# --------------------------------------------------------------------------- conversation_handlers.respond


def _state(trigger_id: str, turns: list[dict], **extra) -> dict:
    trg = TRIGGERS[trigger_id]
    merchant = MERCHANTS[trg["merchant_id"]]
    category = next(c for c in SEED["categories"] if c["slug"] == merchant["category_slug"])
    customer = next((c for c in SEED["customers"] if c["customer_id"] == trg.get("customer_id")), None)
    return {"conversation_id": f"conv_state_{trigger_id}", "merchant": merchant, "category": category,
            "trigger": trg, "customer": customer, "turns": turns, "use_llm": False, "now": NOW, **extra}


def test_respond_action_mode_from_state():
    state = _state("trg_001_research_digest_dentists", [{"from": "vera", "body": RESEARCH_OPENER}])
    res = conversation_handlers.respond(state, "Ok lets do it. Whats next?")
    assert_shape(res)
    assert_action_mode(res)


def test_respond_auto_reply_history_escalates():
    turns = [{"from": "vera", "body": CDE_OPENER}, {"from": "merchant", "body": AUTO_MSG},
             {"from": "vera", "body": "Looks like an auto-reply. When the owner sees this, just reply YES."},
             {"from": "merchant", "body": AUTO_MSG}]
    res = conversation_handlers.respond(_state("trg_022_cde_webinar_dentists", turns), AUTO_MSG)
    assert res["action"] == "end"


def test_respond_customer_slot_and_garbage_state():
    turns = [{"role": "bot", "body": RECALL_OPENER}]
    res = conversation_handlers.respond(_state("trg_003_recall_due_priya", turns), "2")
    assert res["action"] == "send" and "Thu 6 Nov, 5pm" in res["body"]
    for bad in (None, {}, {"turns": "nope"}, {"merchant": "x", "trigger": 5}):
        assert_shape(conversation_handlers.respond(bad, "Stop messaging me"))
        assert_shape(conversation_handlers.respond(bad, "yes"))


def test_respond_works_inside_running_event_loop():
    async def inner() -> dict:
        return conversation_handlers.respond(_state("trg_001_research_digest_dentists", []), "haan karo")
    assert_shape(asyncio.run(inner()))
