"""Chat with the running bot in your terminal, playing the merchant (or customer).

    .venv/bin/uvicorn bot:app --port 8765 --workers 1      # terminal 1
    .venv/bin/python scripts/chat.py                        # terminal 2

Pick a scenario (a shop + an event), the bot sends its opening WhatsApp message,
then you type replies. Commands: /menu (new scenario), /quit.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

BOT = os.environ.get("BOT_URL", "http://localhost:8765").rstrip("/")
DATA = Path(__file__).resolve().parent.parent / "dataset"
NOW = "2026-04-26T10:30:00Z"

BOLD, DIM, GREEN, CYAN, YELLOW, RESET = "\033[1m", "\033[2m", "\033[92m", "\033[96m", "\033[93m", "\033[0m"


def load(name: str, key: str) -> list[dict]:
    return json.loads((DATA / name).read_text())[key]


CATEGORIES = [json.loads(f.read_text()) for f in sorted((DATA / "categories").glob("*.json"))]
MERCHANTS = {m["merchant_id"]: m for m in load("merchants_seed.json", "merchants")}
CUSTOMERS = {c["customer_id"]: c for c in load("customers_seed.json", "customers")}
TRIGGERS = load("triggers_seed.json", "triggers")


def reset_bot(client: httpx.Client) -> None:
    """Wipe the bot's memory and reload the dataset, so any scenario can be replayed."""
    client.post(f"{BOT}/v1/teardown", json={})
    pushes = [("category", c["slug"], c) for c in CATEGORIES]
    pushes += [("merchant", k, v) for k, v in MERCHANTS.items()]
    pushes += [("customer", k, v) for k, v in CUSTOMERS.items()]
    pushes += [("trigger", t["id"], t) for t in TRIGGERS]
    for scope, cid, payload in pushes:
        client.post(f"{BOT}/v1/context", json={"scope": scope, "context_id": cid, "version": 1,
                                                "payload": payload, "delivered_at": NOW})


def pick_scenario() -> dict | None:
    print(f"\n{BOLD}Pick a scenario:{RESET}")
    for i, t in enumerate(TRIGGERS, 1):
        m = MERCHANTS.get(t["merchant_id"], {})
        who = m.get("identity", {}).get("name", t["merchant_id"])
        if t.get("customer_id"):
            cname = CUSTOMERS.get(t["customer_id"], {}).get("identity", {}).get("name", t["customer_id"])
            who = f"{cname} (customer of {who})"
        print(f"  {i:2d}. {t['kind'].replace('_', ' '):28s} {DIM}{who}{RESET}")
    while True:
        raw = input(f"\nNumber (1-{len(TRIGGERS)}), or q to quit: ").strip().lower()
        if raw in {"q", "quit", "/quit"}:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(TRIGGERS):
            return TRIGGERS[int(raw) - 1]
        print("Please type a number from the list.")


def show_bot(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Vera:{RESET} {GREEN}{text}{RESET}")


def chat(client: httpx.Client, trigger: dict) -> bool:
    """Run one conversation. Returns False if the user wants to quit."""
    reset_bot(client)
    res = client.post(f"{BOT}/v1/tick", json={"now": NOW, "available_triggers": [trigger["id"]]}).json()
    actions = res.get("actions", [])
    if not actions:
        print(f"{YELLOW}The bot chose not to send anything for this one (e.g. no consent). Pick another.{RESET}")
        return True
    a = actions[0]
    role = "customer" if a.get("customer_id") else "merchant"
    you = (CUSTOMERS.get(a.get("customer_id"), {}).get("identity", {}).get("name") if role == "customer"
           else MERCHANTS[a["merchant_id"]]["identity"].get("owner_first_name")) or role
    print(f"\n{DIM}--- You are {you} ({role}). Sent as: {a['send_as']}. Type /menu for a new scenario, /quit to exit ---{RESET}")
    show_bot(a["body"])

    turn = 2
    while True:
        try:
            msg = input(f"\n{CYAN}{BOLD}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if not msg:
            continue
        if msg == "/quit":
            return False
        if msg == "/menu":
            return True
        r = client.post(f"{BOT}/v1/reply", json={
            "conversation_id": a["conversation_id"], "merchant_id": a["merchant_id"],
            "customer_id": a.get("customer_id"), "from_role": role, "message": msg,
            "received_at": NOW, "turn_number": turn,
        }).json()
        turn += 1
        action = r.get("action")
        if action == "send":
            show_bot(r.get("body", ""))
        elif action == "wait":
            hours = int(r.get("wait_seconds", 0)) / 3600
            print(f"\n{YELLOW}[Vera decided to wait {hours:g}h before messaging again]{RESET} {DIM}{r.get('rationale', '')}{RESET}")
        else:
            print(f"\n{YELLOW}[Vera ended the conversation]{RESET} {DIM}{r.get('rationale', '')}{RESET}")
            input(f"{DIM}Press Enter for the scenario menu...{RESET}")
            return True


def main() -> int:
    client = httpx.Client(timeout=40)
    try:
        health = client.get(f"{BOT}/v1/healthz").json()
        model = client.get(f"{BOT}/v1/metadata").json().get("model", "?")
    except httpx.HTTPError:
        print(f"Bot is not running at {BOT}. Start it first:\n  .venv/bin/uvicorn bot:app --port 8765 --workers 1")
        return 1
    print(f"{BOLD}Connected to Vera{RESET} at {BOT} (status {health.get('status')}, model: {model})")
    while True:
        trigger = pick_scenario()
        if trigger is None or not chat(client, trigger):
            print("Bye!")
            return 0


if __name__ == "__main__":
    sys.exit(main())
