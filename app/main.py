"""HTTP surface of the Vera bot (testing brief section 2).

    POST /v1/context   versioned context ingestion (+ background precompose)
    POST /v1/tick      proactive sends chosen by app.planner
    POST /v1/reply     next move from app.conversation.ConversationEngine
    GET  /v1/healthz   liveness + context counts
    GET  /v1/metadata  bot identity
    POST /v1/teardown  wipe all state

Everything lives in memory in one process: run exactly one worker. The composer,
conversation engine and LLM client are resolved lazily through small accessors, so the
API keeps answering (with safe fallbacks) even if one of those modules fails to load.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import re
import time
from collections.abc import Iterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__, config, planner
from app.schemas import CTA_OPEN_ENDED, CTA_VALUES
from app.state import SCOPES, ContextStore, ConversationStore, parse_iso

logging.basicConfig(
    level=getattr(logging, str(config.settings.log_level).upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("vera.api")

MAX_CONTEXT_BYTES = 500 * 1024      # testing brief section 5
TICK_GRACE_S = 1.5                  # outer safety margin on top of the planner's own deadline
REPLY_GRACE_S = 1.0

APPROACH = (
    "4-context fact extraction -> per-trigger-kind playbook -> grounded template draft, polished by an LLM "
    "(temperature 0) only behind a no-fabrication validator; urgency/consent/suppression-aware tick planner "
    "with background precompose; rule-first conversation engine (auto-reply, opt-out, hostile, intent-to-action)."
)

_OPT_OUT_RE = re.compile(
    r"\b(stop|unsubscribe|not interested|don'?t message|do not message|leave me alone|remove me|"
    r"no more messages|spam|band karo|mat bhejo|nahi chahiye|useless|bakwas)\b",
    re.IGNORECASE,
)
_ACCEPT_RE = re.compile(
    r"^\W*(yes|yes please|haan|ha|sure|ok|okay|go ahead|let'?s do it|lets do it|do it|please do|send it|karo|"
    r"kar do|chalega|confirm|proceed|sounds good)\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- lazy module accessors


def _composer():
    return importlib.import_module("app.composer")


def _conversation():
    return importlib.import_module("app.conversation")


def _llm_module():
    return importlib.import_module("app.llm")


def _get_llm() -> Any:
    try:
        return _llm_module().get_llm()
    except Exception:
        log.warning("LLM layer unavailable", exc_info=True)
        return None


def _llm_available() -> bool:
    llm = _get_llm()
    try:
        return bool(llm is not None and llm.available())
    except Exception:
        return False


# --------------------------------------------------------------------------- precompose


class Precomposer:
    """Composes pushed triggers in the background so ticks can serve them from cache.

    Scheduled from a ContextStore listener (runs inside the /v1/context request on the event
    loop). Never raises, never blocks the endpoint; bounded by settings.precompose_concurrency.
    Only LLM results are cached: templates are cheap enough to compute at tick time.
    """

    def __init__(self, state: "BotState") -> None:
        self.state = state
        self.inflight: dict[str, asyncio.Task] = {}     # fingerprint -> task (queued or running)
        self.started: set[str] = set()                   # fingerprints whose task holds a semaphore slot
        self._sem: asyncio.Semaphore | None = None

    def view(self) -> "InflightView":
        """The planner's window onto precompose work (see InflightView)."""
        return InflightView(self)

    def _now(self) -> str | None:
        """Best guess at the simulated time the next tick will compose for."""
        return self.state.convs.sim_now or self.state.last_delivered_at

    def _cached(self, fp: str, now: str | None) -> dict | None:
        """A cached composition usable for `now` (same composed-for date), else None."""
        cached = self.state.cache_get(fp)
        return cached if cached is not None and planner.composed_for_ok(cached, now) else None

    # listener signature: fn(scope, context_id, version, payload)
    def on_context(self, scope: str, context_id: str, version: int, payload: dict) -> None:
        try:
            if not config.settings.precompose:
                return
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return
            if not _llm_available():
                return
            for trigger_id, trigger in self._affected_triggers(scope, context_id, payload):
                self._schedule(trigger_id, trigger)
        except Exception:
            log.warning("precompose scheduling failed", exc_info=True)

    def _affected_triggers(self, scope: str, context_id: str, payload: dict) -> list[tuple[str, dict]]:
        """(trigger context_id, trigger) pairs whose composition inputs this push changed."""
        contexts = self.state.contexts
        if scope == "trigger":
            return [(context_id, payload)] if isinstance(payload, dict) else []
        internal_key = {"merchant": "merchant_id", "customer": "customer_id", "category": "slug"}.get(scope)
        if internal_key is None:
            return []
        ids = {context_id, str(payload.get(internal_key) or "")} - {""}
        out = []
        for tid in contexts.ids("trigger"):
            trg = contexts.get("trigger", tid)
            if not isinstance(trg, dict):
                continue
            merchant_id, merchant, _cat, customer_id, _cust = planner.resolve_inputs(trg, contexts)
            if ((scope == "merchant" and merchant_id in ids)
                    or (scope == "customer" and customer_id in ids)
                    or (scope == "category" and merchant and merchant.get("category_slug") in ids)):
                out.append((tid, trg))
        out.sort(key=lambda pair: -planner.trigger_urgency(pair[1]))
        return out

    def _resolve(self, trigger: dict) -> tuple[str, str, tuple] | None:
        """(fingerprint, merchant_id, compose args) when the trigger is still worth composing, else None."""
        contexts, convs = self.state.contexts, self.state.convs
        merchant_id, merchant, category, _cid, customer = planner.resolve_inputs(trigger, contexts)
        if not merchant_id or not isinstance(merchant, dict):
            return None
        if convs.suppression_used(planner.suppression_key_for(trigger, merchant_id)):
            return None
        if convs.merchant(merchant_id).opted_out:
            return None
        if planner.is_customer_scoped(trigger):
            if planner.consent_problem(customer, str(trigger.get("kind") or "")):
                return None
        else:
            customer = None
        args = (category, merchant, trigger, customer)
        return _composer().input_fingerprint(*args), merchant_id, args

    def _schedule(self, trigger_id: str, trigger: dict) -> None:
        resolved = self._resolve(trigger)
        if resolved is None:
            return
        fp = resolved[0]
        if fp in self.inflight or self._cached(fp, self._now()) is not None:
            return
        task = asyncio.ensure_future(self._run(fp, trigger_id))
        self.inflight[fp] = task
        task.add_done_callback(lambda t, fp=fp: self.inflight.pop(fp, None) if self.inflight.get(fp) is t else None)

    async def _run(self, fp: str, trigger_id: str) -> dict | None:
        holding = False
        try:
            if self._sem is None:
                self._sem = asyncio.Semaphore(max(1, int(config.settings.precompose_concurrency)))
            async with self._sem:
                self.started.add(fp)
                holding = True
                trigger = self.state.contexts.get("trigger", trigger_id)
                resolved = self._resolve(trigger) if isinstance(trigger, dict) else None
                if resolved is None or resolved[0] != fp:        # sent meanwhile, or superseded by a newer version
                    return None
                now = self._now()
                cached = self._cached(fp, now)
                if cached is not None:
                    return cached
                _fp, merchant_id, (category, merchant, trigger, customer) = resolved
                prior = list(self.state.convs.merchant(merchant_id).sent_bodies)
                timeout = float(config.settings.llm_timeout_s)
                comp = await asyncio.wait_for(
                    _composer().compose_async(category, merchant, trigger, customer, now=now,
                                              prior_bodies=prior, timeout=timeout),
                    timeout=timeout + 2.0,
                )
                if not isinstance(comp, dict):
                    return None
                comp = {**comp, planner.COMPOSED_FOR_KEY: planner.now_date(now)}
                if comp.get("body") and (comp.get("meta") or {}).get("source") == "llm":
                    self.state.cache_put(fp, comp)
                return comp
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.info("precompose failed for %s (%s)", trigger_id, type(exc).__name__)
            return None
        finally:
            if holding:
                self.started.discard(fp)

    def cancel_all(self) -> None:
        for task in list(self.inflight.values()):
            task.cancel()
        self.inflight.clear()
        self.started.clear()
        self._sem = None           # re-created (with the current setting) on the next event loop


class InflightView(Mapping):
    """Read-only mapping handed to the planner: fingerprint -> running precompose task.

    Looking up a task that is still queued behind the concurrency limit *claims* it: the
    queued task is cancelled and the tick composes that trigger itself, instead of spending
    its deadline waiting in line.
    """

    def __init__(self, pre: Precomposer) -> None:
        self._pre = pre

    def __getitem__(self, fp: str) -> asyncio.Task:
        task = self._pre.inflight[fp]
        if fp in self._pre.started:
            return task
        task.cancel()
        self._pre.inflight.pop(fp, None)
        raise KeyError(fp)

    def __iter__(self) -> Iterator[str]:
        return iter([fp for fp in self._pre.inflight if fp in self._pre.started])

    def __len__(self) -> int:
        return len(self._pre.inflight)       # truthy while anything is queued, so lookups happen


# --------------------------------------------------------------------------- state container


class BotState:
    def __init__(self) -> None:
        self.start_time = time.time()
        self.last_delivered_at: str | None = None        # latest context push delivered_at (precompose `now`)
        self.contexts = ContextStore()
        self.convs = ConversationStore()
        self.precomposer = Precomposer(self)
        self.contexts.add_listener(self.precomposer.on_context)
        self._engine: Any = None
        self._engine_error_logged = False
        self._cache: Any = None
        self._cache_error_logged = False

    @property
    def engine(self) -> Any:
        """ConversationEngine, created on first use (None if the module cannot be loaded)."""
        if self._engine is None:
            try:
                self._engine = _conversation().ConversationEngine(self.contexts, self.convs, _get_llm())
            except Exception:
                if not self._engine_error_logged:
                    self._engine_error_logged = True
                    log.exception("ConversationEngine unavailable; using safe reply fallbacks")
                return None
        return self._engine

    @property
    def cache(self) -> Any:
        if self._cache is None:
            try:
                self._cache = _composer().ComposeCache()
            except Exception as exc:
                if not self._cache_error_logged:
                    self._cache_error_logged = True
                    log.warning("ComposeCache unavailable (%s)", type(exc).__name__)
                return None
        return self._cache

    def cache_get(self, fp: str) -> Any:
        cache = self.cache
        try:
            return cache.get(fp) if cache is not None else None
        except Exception:
            return None

    def cache_put(self, fp: str, comp: dict) -> None:
        cache = self.cache
        if cache is not None:
            try:
                cache.put(fp, comp)
            except Exception:
                pass

    def reset(self) -> None:
        """Wipe contexts, conversations, caches and planner memory (keeps process uptime)."""
        self.precomposer.cancel_all()
        self.last_delivered_at = None
        self.contexts.clear()
        self.convs.clear()
        planner.reset(self.convs)
        for getter, name in ((_composer, "clear_cache"), (_get_llm, "clear_cache")):
            try:
                fn = getattr(getter(), name, None)
                if callable(fn):
                    fn()                      # module / client level caches hold context-derived text
            except Exception:
                pass
        self._engine = None
        self._engine_error_logged = False
        self._cache = None
        self._cache_error_logged = False


STATE = BotState()


def reset_state() -> BotState:
    """Used by /v1/teardown and tests."""
    STATE.reset()
    return STATE


# --------------------------------------------------------------------------- app + middleware


async def _self_ping_loop(url: str, interval_s: int) -> None:
    """Keep free hosts that sleep on inactivity awake by hitting our own public healthz."""
    import httpx

    target = url.rstrip("/") + "/v1/healthz"
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            await asyncio.sleep(max(30, interval_s))
            try:
                await client.get(target)
            except Exception as exc:
                log.info("self-ping failed (%s)", type(exc).__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    llm = _get_llm()
    for name in ("startup", "start", "open"):
        fn = getattr(llm, name, None)
        if callable(fn):
            try:
                res = fn()
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                log.warning("LLM %s() failed", name, exc_info=True)
            break
    ping_task = None
    # Render injects RENDER_EXTERNAL_URL, so the keepalive works there with zero configuration.
    ping_url = config.settings.self_ping_url or os.environ.get("RENDER_EXTERNAL_URL", "").strip()
    if ping_url:
        ping_task = asyncio.create_task(_self_ping_loop(ping_url, config.settings.self_ping_interval_s))
    log.info("Vera up: model=%s", _describe_model())
    try:
        yield
    finally:
        if ping_task:
            ping_task.cancel()
        STATE.precomposer.cancel_all()
        for name in ("aclose", "close", "shutdown"):
            fn = getattr(llm, name, None)
            if callable(fn):
                try:
                    res = fn()
                    if asyncio.iscoroutine(res):
                        await res
                except Exception:
                    log.warning("LLM %s() failed", name, exc_info=True)
                break


app = FastAPI(title="Vera merchant assistant", version=__version__, lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)


class RequestLogMiddleware:
    """Pure-ASGI access log: method, path, status, latency. Never logs payloads."""

    def __init__(self, asgi_app) -> None:
        self.app = asgi_app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message):
            if message.get("type") == "http.response.start":
                status["code"] = message.get("status", 500)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            ms = (time.perf_counter() - start) * 1000
            log.info("%s %s -> %s (%.0f ms)", scope.get("method"), scope.get("path"), status["code"], ms)


app.add_middleware(RequestLogMiddleware)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_request: Request, exc: StarletteHTTPException):
    return JSONResponse({"error": "http_error", "detail": exc.detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def _validation_error(_request: Request, exc: RequestValidationError):
    return JSONResponse({"accepted": False, "reason": "invalid_payload", "details": str(exc)[:300]},
                        status_code=400)


@app.exception_handler(Exception)
async def _unhandled(_request: Request, exc: Exception):
    log.exception("unhandled error: %s", type(exc).__name__)
    return JSONResponse({"error": "internal_error", "detail": type(exc).__name__}, status_code=500)


# --------------------------------------------------------------------------- helpers


async def _read_json(request: Request, max_bytes: int | None = None) -> tuple[Any, str | None]:
    """(parsed body, error). Error is a short human-readable reason."""
    if max_bytes is not None:
        try:
            if int(request.headers.get("content-length") or 0) > max_bytes:
                return None, f"payload exceeds {max_bytes // 1024} KB"
        except ValueError:
            pass
    raw = await request.body()
    if max_bytes is not None and len(raw) > max_bytes:
        return None, f"payload exceeds {max_bytes // 1024} KB"
    if not raw.strip():
        return None, "empty body"
    try:
        return json.loads(raw), None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"malformed JSON: {exc}"[:200]


def _bad(reason: str, details: str) -> JSONResponse:
    return JSONResponse({"accepted": False, "reason": reason, "details": details}, status_code=400)


def _parse_version(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            f = float(value.strip())
        except ValueError:
            return None
        return int(f) if f.is_integer() else None
    return None


def _describe_model() -> str:
    llm = _get_llm()
    try:
        return str(llm.describe()) if llm is not None else "deterministic-templates (LLM layer unavailable)"
    except Exception:
        return "deterministic-templates"


def _safe_reply(message: str, reason: str) -> dict:
    """Used only when the engine fails: exit on opt-out, act on a clear yes, otherwise back off."""
    if _OPT_OUT_RE.search(message or ""):
        return {"action": "end", "rationale": "Merchant asked us to stop; closing the conversation politely."}
    if _ACCEPT_RE.search(message or ""):
        return {"action": "send", "cta": "binary_confirm_cancel",
                "body": "Great, on it. I'm preparing the draft now and will share it here next. Reply CONFIRM to proceed.",
                "rationale": f"Merchant committed; moving straight to action with a single confirm ({reason})."}
    return {"action": "wait", "wait_seconds": 1800,
            "rationale": f"Could not compose a grounded reply in time ({reason}); backing off 30 minutes "
                         "rather than sending a generic message."}


def _clean_reply(res: Any, message: str) -> dict:
    """Ensure the engine's answer is one of the three valid shapes (never an empty send)."""
    if not isinstance(res, dict):
        return _safe_reply(message, "no result")
    action = res.get("action")
    rationale = str(res.get("rationale") or "").strip()
    if action == "send":
        body = res.get("body")
        if not isinstance(body, str) or not body.strip():
            return _safe_reply(message, "empty body")
        cta = res.get("cta") if res.get("cta") in CTA_VALUES else CTA_OPEN_ENDED
        return {"action": "send", "body": body.strip(), "cta": cta,
                "rationale": rationale or "Continuing the conversation toward the offered next step."}
    if action == "wait":
        try:
            secs = int(float(res.get("wait_seconds") or 1800))
        except (TypeError, ValueError):
            secs = 1800
        return {"action": "wait", "wait_seconds": max(60, secs),
                "rationale": rationale or "Backing off before the next nudge."}
    if action == "end":
        return {"action": "end", "rationale": rationale or "Closing the conversation."}
    return _safe_reply(message, "invalid action")


# --------------------------------------------------------------------------- endpoints


@app.get("/")
async def root():
    return {"service": "vera", "status": "ok", "health": "/v1/healthz", "metadata": "/v1/metadata"}


@app.api_route("/v1/healthz", methods=["GET", "HEAD"])
async def healthz():
    return {"status": "ok", "uptime_seconds": int(time.time() - STATE.start_time),
            "contexts_loaded": STATE.contexts.counts()}


@app.get("/v1/metadata")
async def metadata():
    s = config.settings
    return {
        "team_name": s.team_name,
        "team_members": list(s.team_members),
        "model": _describe_model(),
        "approach": APPROACH,
        "contact_email": s.contact_email,
        "version": s.bot_version,
        "submitted_at": s.submitted_at,
    }


@app.post("/v1/context")
async def push_context(request: Request):
    body, err = await _read_json(request, MAX_CONTEXT_BYTES)
    if err:
        return _bad("invalid_payload", err)
    if not isinstance(body, dict):
        return _bad("invalid_payload", "body must be a JSON object")
    missing = [k for k in ("scope", "context_id", "version", "payload") if body.get(k) is None]
    if "scope" in missing:
        return _bad("invalid_scope", f"scope is required; expected one of {list(SCOPES)}")
    scope = str(body["scope"]).strip().lower()
    if scope not in SCOPES:
        return _bad("invalid_scope", f"unknown scope {str(body['scope'])[:40]!r}; expected one of {list(SCOPES)}")
    if missing:
        return _bad("invalid_payload", f"missing field(s): {', '.join(missing)}")
    raw_id = body["context_id"]
    context_id = str(raw_id).strip() if isinstance(raw_id, (str, int)) and not isinstance(raw_id, bool) else ""
    if not context_id:
        return _bad("invalid_payload", "context_id must be a non-empty string")
    version = _parse_version(body["version"])
    if version is None or version < 0:
        return _bad("invalid_payload", "version must be a non-negative integer")
    payload = body["payload"]
    if not isinstance(payload, dict):
        return _bad("invalid_payload", "payload must be a JSON object")

    delivered_at = body.get("delivered_at")
    if isinstance(delivered_at, str) and parse_iso(delivered_at):
        STATE.last_delivered_at = delivered_at
    status, current, stored_at = STATE.contexts.put(scope, context_id, version, payload)
    if status != "accepted":
        return JSONResponse({"accepted": False, "reason": "stale_version", "current_version": current},
                            status_code=409)
    return {"accepted": True, "ack_id": f"ack_{context_id}_v{version}", "stored_at": stored_at}


@app.post("/v1/tick")
async def tick(request: Request):
    body, _err = await _read_json(request)
    body = body if isinstance(body, dict) else {}
    now = body.get("now") if isinstance(body.get("now"), str) else None
    triggers = body.get("available_triggers") or []
    deadline = float(config.settings.tick_deadline_s)
    try:
        actions = await asyncio.wait_for(
            planner.plan_tick(now, triggers, contexts=STATE.contexts, convs=STATE.convs, engine=STATE.engine,
                              deadline_s=deadline, cache=STATE.cache, inflight=STATE.precomposer.view()),
            timeout=deadline + TICK_GRACE_S,
        )
    except Exception as exc:
        log.error("tick failed (%s); returning no actions", type(exc).__name__, exc_info=True)
        actions = []
    return {"actions": actions if isinstance(actions, list) else []}


@app.post("/v1/reply")
async def reply(request: Request):
    body, _err = await _read_json(request)
    body = body if isinstance(body, dict) else {}
    message = body.get("message")
    message = message if isinstance(message, str) else ("" if message is None else str(message))
    try:
        turn = int(body.get("turn_number") or 0)
    except (TypeError, ValueError):
        turn = 0
    req = {
        "conversation_id": str(body.get("conversation_id") or "").strip() or "conv_unknown",
        "merchant_id": body.get("merchant_id") or None,
        "customer_id": body.get("customer_id") or None,
        "from_role": str(body.get("from_role") or "merchant"),
        "message": message,
        "received_at": body.get("received_at") if isinstance(body.get("received_at"), str) else None,
        "turn_number": turn,
    }
    engine = STATE.engine
    if engine is None:
        return _safe_reply(message, "engine unavailable")
    try:
        STATE.convs.observe_time(req["received_at"])
        res = await asyncio.wait_for(engine.handle_reply(req),
                                     timeout=float(config.settings.reply_deadline_s) + REPLY_GRACE_S)
    except Exception as exc:
        log.error("reply failed (%s); using safe fallback", type(exc).__name__, exc_info=True)
        return _safe_reply(message, type(exc).__name__)
    return _clean_reply(res, message)


@app.post("/v1/teardown")
async def teardown():
    reset_state()
    return {"ok": True}
