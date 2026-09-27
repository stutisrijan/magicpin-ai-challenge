"""Full-scale local harness: the whole expanded dataset through a running bot.

    .venv/bin/uvicorn bot:app --port 8765 --workers 1
    BOT_URL=http://localhost:8765 .venv/bin/python scripts/local_harness.py

Warmup (5 categories, 50 merchants, 200 customers), then 10 ticks that release all
100 triggers in batches, then up to 4 scripted reply turns per conversation. Every bot
message is re-checked with the bot's own validator against freshly built facts, plus
contract invariants. Exits non-zero if any invariant fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.facts import build_facts  # noqa: E402
from app.validator import validate_body  # noqa: E402

BOT = os.environ.get("BOT_URL", "http://localhost:8765").rstrip("/")
START = datetime(2026, 4, 26, 10, 0, tzinfo=timezone.utc)

PERSONAS = [
    ["Yes please, go ahead", "CONFIRM", "thanks"],
    ["Thank you for contacting us! Our team will respond shortly."] * 3,
    ["Not interested. Stop messaging me."],
    ["Btw can you help me with my GST filing?", "ok fine, lets do it"],
    ["How much will this cost me?", "ok"],
    ["haan karo", "theek hai, confirm"],
    ["Busy right now, call me tomorrow"],
    ["Why are you bothering me, this is useless"],
]
ACTION_KEYS = {"conversation_id", "merchant_id", "customer_id", "send_as", "trigger_id", "template_name",
               "template_params", "body", "cta", "suppression_key", "rationale"}


def iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def load_expanded() -> dict:
    out = Path(tempfile.mkdtemp(prefix="vera_expanded_"))
    subprocess.run([sys.executable, str(ROOT / "dataset" / "generate_dataset.py"), "--seed-dir",
                    str(ROOT / "dataset"), "--out", str(out)], check=True, capture_output=True)
    data = {}
    for scope, sub, idf in [("category", "categories", "slug"), ("merchant", "merchants", "merchant_id"),
                            ("customer", "customers", "customer_id"), ("trigger", "triggers", "id")]:
        data[scope] = {}
        for f in sorted((out / sub).glob("*.json")):
            item = json.loads(f.read_text())
            data[scope][item[idf]] = item
    return data


def main() -> int:
    client = httpx.Client(timeout=35)
    problems: list[str] = []
    latencies: dict[str, list[float]] = {"context": [], "tick": [], "reply": []}
    outcomes: Counter = Counter()

    def call(kind: str, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        t0 = time.perf_counter()
        r = client.request(method, f"{BOT}{path}", json=body)
        latencies.setdefault(kind, []).append(time.perf_counter() - t0)
        try:
            return r.status_code, r.json()
        except ValueError:
            problems.append(f"{path}: non-JSON response ({r.status_code})")
            return r.status_code, {}

    call("context", "POST", "/v1/teardown", {})
    data = load_expanded()
    for scope in ("category", "merchant", "customer"):
        for cid, payload in data[scope].items():
            code, res = call("context", "POST", "/v1/context", {"scope": scope, "context_id": cid, "version": 1,
                                                                "payload": payload, "delivered_at": iso(START)})
            if code != 200 or not res.get("accepted"):
                problems.append(f"context {scope}/{cid}: {code} {res}")
    _, health = call("healthz", "GET", "/v1/healthz")
    loaded = health.get("contexts_loaded", {})
    print(f"warmup loaded: {loaded}")
    if (loaded.get("category"), loaded.get("merchant"), loaded.get("customer")) != (5, 50, 200):
        problems.append(f"warmup counts wrong: {loaded}")

    trigger_ids = list(data["trigger"])
    active: list[str] = []
    all_actions: list[dict] = []
    for tick_no in range(10):
        now = START + timedelta(minutes=5 * tick_no)
        for tid in trigger_ids[tick_no * 10:(tick_no + 1) * 10]:
            call("context", "POST", "/v1/context", {"scope": "trigger", "context_id": tid, "version": 1,
                                                    "payload": data["trigger"][tid], "delivered_at": iso(now)})
            active.append(tid)
        code, res = call("tick", "POST", "/v1/tick", {"now": iso(now), "available_triggers": list(active)})
        actions = res.get("actions", [])
        if code != 200 or not isinstance(actions, list):
            problems.append(f"tick {tick_no}: bad response {code}")
            continue
        per_merchant = Counter(a["merchant_id"] for a in actions if not a.get("customer_id"))
        for mid, n in per_merchant.items():
            if n > 1:
                problems.append(f"tick {tick_no}: {n} merchant-facing actions to {mid}")
        for a in actions:
            missing = ACTION_KEYS - set(a)
            if missing:
                problems.append(f"{a.get('trigger_id')}: missing keys {sorted(missing)}")
            all_actions.append({**a, "_now": iso(now)})

    sent_triggers = {a["trigger_id"] for a in all_actions}
    print(f"ticks: {len(all_actions)} proactive messages for {len(sent_triggers)}/{len(trigger_ids)} triggers")

    seen_conv: set[str] = set()
    for a in all_actions:
        if a["conversation_id"] in seen_conv:
            problems.append(f"conversation id reused: {a['conversation_id']}")
        seen_conv.add(a["conversation_id"])
        trig = data["trigger"][a["trigger_id"]]
        merchant = data["merchant"][a["merchant_id"]]
        category = data["category"].get(merchant.get("category_slug"), {})
        customer = data["customer"].get(a["customer_id"]) if a.get("customer_id") else None
        facts = build_facts(category, merchant, trig, customer, now=a["_now"])
        customer_facing = bool(customer)
        issues = validate_body(a["body"], facts, customer_facing=customer_facing)
        if issues:
            problems.append(f"{a['trigger_id']}: opener issues {issues}")
        if customer_facing != (a["send_as"] == "merchant_on_behalf"):
            problems.append(f"{a['trigger_id']}: send_as {a['send_as']} mismatch")

        persona = PERSONAS[int(hashlib.sha1(a["conversation_id"].encode()).hexdigest(), 16) % len(PERSONAS)]
        role = "customer" if customer_facing else "merchant"
        bodies = [a["body"]]
        for turn, msg in enumerate(persona, start=2):
            code, res = call("reply", "POST", "/v1/reply", {
                "conversation_id": a["conversation_id"], "merchant_id": a["merchant_id"],
                "customer_id": a.get("customer_id"), "from_role": role, "message": msg,
                "received_at": a["_now"], "turn_number": turn})
            act = res.get("action")
            outcomes[act] += 1
            if code != 200 or act not in {"send", "wait", "end"}:
                problems.append(f"{a['conversation_id']} turn {turn}: bad reply {code} {res}")
                break
            if act == "send":
                body = res.get("body", "")
                if not body.strip() or not res.get("cta") or not res.get("rationale"):
                    problems.append(f"{a['conversation_id']} turn {turn}: incomplete send {res}")
                if body.strip().lower() in {b.strip().lower() for b in bodies}:
                    problems.append(f"{a['conversation_id']} turn {turn}: verbatim repeat")
                bodies.append(body)
                mode = "reply_action" if msg.lower() in {"yes please, go ahead", "haan karo", "ok fine, lets do it"} else "reply"
                r_issues = [i for i in validate_body(body, facts, prior_bodies=bodies[:-1], mode=mode,
                                                     customer_facing=customer_facing) if "addressee" not in i]
                if r_issues:
                    problems.append(f"{a['conversation_id']} turn {turn} ({msg[:25]}): {r_issues}")
            elif act == "wait" and not isinstance(res.get("wait_seconds"), int):
                problems.append(f"{a['conversation_id']} turn {turn}: wait without int wait_seconds")
            if act in {"end", "wait"}:
                break

    print(f"replies: {dict(outcomes)}")
    for kind, vals in latencies.items():
        if vals:
            print(f"latency {kind:8s} n={len(vals):4d}  avg={sum(vals) / len(vals) * 1000:7.1f} ms  max={max(vals) * 1000:7.1f} ms")
            if max(vals) > 10:
                problems.append(f"{kind} latency over 10s")
    if problems:
        print(f"\nFAILED: {len(problems)} problem(s)")
        for p in problems[:40]:
            print(" -", p)
        return 1
    print("\nALL INVARIANTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
