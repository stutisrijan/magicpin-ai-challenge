"""Quick local demo against a running bot.

    .venv/bin/uvicorn bot:app --port 8765 --workers 1      # terminal 1
    .venv/bin/python scripts/demo_local.py                  # terminal 2 (BOT_URL defaults to :8765)

Pushes the seed dataset, fires a few triggers through /v1/tick, then plays scripted
merchant/customer replies through /v1/reply and prints the whole exchange.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

BOT = os.environ.get("BOT_URL", "http://localhost:8768").rstrip("/")
DATA = Path(__file__).resolve().parent.parent / "dataset"
NOW = "2026-04-26T10:30:00Z"

# trigger id -> scripted replies from the merchant (or customer)
SCENARIOS = {
    "trg_001_research_digest_dentists": ["Yes please send the abstract. Also draft the patient WhatsApp.", "Ok thanks"],
    "trg_010_ipl_match_delhi": ["haan karo, banner bana do", "CONFIRM"],
    "trg_018_supply_atorvastatin_recall": ["Btw can you also help me with my GST filing this month?"],
    "trg_004_perf_dip_bharat": ["Thank you for contacting Bharat Dental Care! Our team will respond shortly."] * 3,
    "trg_003_recall_due_priya": ["2"],
}


def post(client: httpx.Client, path: str, body: dict) -> dict:
    r = client.post(f"{BOT}{path}", json=body)
    return r.json()


def main() -> int:
    client = httpx.Client(timeout=30)
    try:
        print("healthz:", client.get(f"{BOT}/v1/healthz").json())
    except httpx.HTTPError as exc:
        print(f"Bot not reachable at {BOT}: {exc}")
        return 1

    # warmup: categories, merchants, customers
    for f in sorted((DATA / "categories").glob("*.json")):
        cat = json.loads(f.read_text())
        post(client, "/v1/context", {"scope": "category", "context_id": cat["slug"], "version": 1, "payload": cat, "delivered_at": NOW})
    seeds = {
        "merchant": ("merchants_seed.json", "merchants", "merchant_id"),
        "customer": ("customers_seed.json", "customers", "customer_id"),
        "trigger": ("triggers_seed.json", "triggers", "id"),
    }
    for scope, (fname, key, idf) in seeds.items():
        for item in json.loads((DATA / fname).read_text())[key]:
            post(client, "/v1/context", {"scope": scope, "context_id": item[idf], "version": 1, "payload": item, "delivered_at": NOW})
    print("loaded:", client.get(f"{BOT}/v1/healthz").json()["contexts_loaded"])

    tick = post(client, "/v1/tick", {"now": NOW, "available_triggers": list(SCENARIOS)})
    actions = tick.get("actions", [])
    print(f"\n/v1/tick -> {len(actions)} action(s)\n" + "=" * 78)

    for action in actions:
        who = action.get("customer_id") or action["merchant_id"]
        role = "customer" if action.get("customer_id") else "merchant"
        print(f"\n[{action['trigger_id']}]  to {who}  (send_as={action['send_as']}, cta={action['cta']})")
        print(f"BOT: {action['body']}")
        for turn, msg in enumerate(SCENARIOS.get(action["trigger_id"], []), start=2):
            print(f"{role.upper()}: {msg}")
            res = post(client, "/v1/reply", {
                "conversation_id": action["conversation_id"], "merchant_id": action["merchant_id"],
                "customer_id": action.get("customer_id"), "from_role": role, "message": msg,
                "received_at": NOW, "turn_number": turn,
            })
            if res.get("action") == "send":
                print(f"BOT: {res['body']}")
            else:
                extra = f" {res.get('wait_seconds')}s" if res.get("action") == "wait" else ""
                print(f"BOT -> {res.get('action')}{extra}  ({res.get('rationale', '')[:110]})")
                if res.get("action") == "end":
                    break
        print("-" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
