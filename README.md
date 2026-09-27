# Vera: magicpin merchant AI assistant

A WhatsApp-style merchant assistant for the magicpin AI Challenge. It serves the judge's HTTP contract: `POST /v1/context`, `/v1/tick`, `/v1/reply`, and `GET /v1/healthz`, `/v1/metadata`. It also exposes `compose()` (`bot.py`) and `respond()` (`conversation_handlers.py`) for offline evaluation. `submission.jsonl` holds the 30 official test pairs.

## Approach

**Facts first, LLM second.** Each message is built in four steps:

1. **Facts** (`app/facts.py`): pulls verifiable facts out of the four contexts in code. Examples: digest items resolved by id with their source citation, the merchant's numbers against peer benchmarks, active offers by exact title, review themes, slots, and customer relationship data. Every legitimate number goes into an allow-list.
2. **Playbook** (`app/playbooks.py`): one strategy per trigger kind (30+ kinds plus a generic fallback). Each sets the hook, the judgment call, the persuasion levers, a single CTA, and the concrete deliverable Vera offers. Example judgment: a Saturday IPL match means skipping the dine-in promo, because the category data says covers drop 12%.
3. **Template draft** (`app/templates.py`): a deterministic message that scores well by itself, in English or Hinglish (romanized Hindi-English), with a regional greeting where the language plan calls for one. It always exists, so rate limits or timeouts never produce an empty message.
4. **LLM polish** (`app/composer.py`, optional): the LLM gets the facts plus the draft and writes a better version at temperature 0. The validator (`app/validator.py`) rejects any output with an ungrounded number, a taboo word, a URL, internal jargon, multiple CTAs, or a repeated message. On rejection the bot falls back to the draft.

**Conversations** (`app/conversation.py`): rule-based intent detection runs first, then the bot writes the reply.

- **Auto-reply:** one nudge to the owner, then wait 24h, then end. Repeated canned text is detected across conversations too.
- **Opt-out:** end.
- **Hostile:** apologise once, then end.
- **"Let's do it":** switch immediately to action mode and deliver the draft, with no more qualifying questions.
- **Off-topic** (e.g. GST): decline politely and steer back to the original offer.
- **"Busy":** wait for a parsed duration.
- **Customer picks a slot:** confirm that slot.

**Tick planner** (`app/planner.py`): skips anything already sent (suppression keys), skips customers without consent, respects opt-outs and auto-reply cooldowns, sends at most one merchant-facing message per merchant per tick, runs all compositions concurrently under a 9s deadline, and precomposes in the background when a trigger arrives.

**LLM (free tier):** Gemini first, Groq as backup. Each provider is rate-limited and cools down after a 429. Without any key, the bot runs on the templates alone.

## Tradeoffs

- The deterministic core gives up some fluency in exchange for zero hallucination and guaranteed latency. The LLM only adds style.
- State is in memory, so run exactly **one worker**. This matches the brief ("in-memory is fine; just don't restart").
- `expires_at` is not used to filter triggers: the judge's `available_triggers` list decides what is active.

## What extra context would help most

Real appointment slots and service menus for generated merchants; review text and ratings; the merchant's actual reply-language history; and per-locality competitor and peer data, which would enable honest social proof.

## Run and deploy

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn bot:app --port 8080 --workers 1       # then open /v1/healthz
.venv/bin/python -m pytest -q tests/                    # 896 tests, offline
.venv/bin/python scripts/generate_submission.py         # rewrite submission.jsonl
```

Environment variables (all optional; see `.env.example`): `GEMINI_API_KEY` (free at aistudio.google.com), `GROQ_API_KEY` (free at console.groq.com), `CONTACT_EMAIL`, `TEAM_NAME`.

Deploy anywhere that gives you a public URL. Keep it to a single instance.

**Branches:** `main` is production. Render deploys it automatically on every merge. Work on feature branches and merge to `main` only outside the judging window, because a redeploy wipes the bot's in-memory state.

- **Render:** `render.yaml` blueprint. On the free plan the bot pings itself so the service doesn't go to sleep.
- **Docker:** `Dockerfile` (reads `$PORT`).
- **Fly.io:** `fly.toml`, with the machine set to always on.
- **Heroku-style hosts:** `Procfile`.
