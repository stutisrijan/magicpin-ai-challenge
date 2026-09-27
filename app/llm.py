"""Provider-agnostic JSON completion client: Gemini, Groq, OpenAI-compatible, Anthropic, OpenRouter.

Priorities, in order:
  1. Never raise and never overrun the caller's deadline. Every failure path returns None
     and the deterministic templates take over.
  2. Free tier first: Gemini (several models, each with its own quota), then Groq. Per-model
     rate windows, cooldowns after 429/5xx/timeouts and a response cache keep us inside quota.
  3. Determinism: temperature 0 by default, and identical prompts are served from the cache.

Only httpx is used (one shared AsyncClient per event loop); no vendor SDKs.
API keys are sent in headers only and are redacted from every stored or logged error.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Iterator

import httpx

from app import config

log = logging.getLogger("vera.llm")

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

KNOWN_PROVIDERS = ("gemini", "groq", "openai", "anthropic", "openrouter")
_PROVIDER_ALIASES = {
    "google": "gemini",
    "openai-compatible": "openai",
    "openai_compatible": "openai",
    "oai": "openai",
    "claude": "anthropic",
}
# Providers whose published rate limits are per model (so each model gets its own window/cooldown).
_PER_MODEL_QUOTA = {"gemini", "groq"}

JSON_INSTRUCTION = "Respond with a single JSON object."

_GEMINI_SAFETY = [
    {"category": cat, "threshold": "BLOCK_ONLY_HIGH"}
    for cat in (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


# --------------------------------------------------------------------------- JSON extraction

_FENCE_RE = re.compile(r"```[ \t]*(?:json5?|javascript|js)?[ \t]*\r?\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_MAX_BRACE_STARTS = 64


def extract_json(text: Any) -> dict | None:
    """Best-effort: pull the first JSON object out of an LLM response.

    Handles bare JSON, ```json fences (closed or not), prose around the object, braces inside
    strings, trailing commas, raw newlines inside strings, a single-object list and
    double-encoded JSON strings. Returns None when no object can be recovered.
    """
    return _extract(text, depth=0)


def _extract(text: Any, depth: int) -> dict | None:
    if isinstance(text, dict):
        return text
    if isinstance(text, (bytes, bytearray)):
        text = bytes(text).decode("utf-8", errors="replace")
    if not isinstance(text, str):
        return None
    s = text.strip().lstrip("﻿")
    if not s:
        return None
    candidates = [s] + [m.group(1).strip() for m in _FENCE_RE.finditer(s)]
    for cand in candidates:
        obj = _loads_object(cand, depth)
        if obj is not None:
            return obj
    for cand in candidates:
        for chunk in _balanced_objects(cand):
            obj = _loads_object(chunk, depth)
            if obj is not None:
                return obj
    return None


def _loads_object(s: str, depth: int) -> dict | None:
    if not s:
        return None
    for variant in (s, None):
        if variant is None:
            variant = _strip_trailing_commas(s)
            if variant == s:
                break
        try:
            obj = json.loads(variant, strict=False)
        except (ValueError, RecursionError):
            continue
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            return obj[0]
        if isinstance(obj, str) and depth == 0 and "{" in obj:
            return _extract(obj, depth=1)  # double-encoded JSON
        return None  # valid JSON, but not an object
    return None


def _balanced_objects(s: str) -> Iterator[str]:
    """Yield every balanced {...} span (string-aware), starting from each '{' in order."""
    start = s.find("{")
    tried = 0
    while start != -1 and tried < _MAX_BRACE_STARTS:
        tried += 1
        end = _matching_brace(s, start)
        if end is not None:
            yield s[start:end + 1]
        start = s.find("{", start + 1)


def _matching_brace(s: str, start: int) -> int | None:
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _strip_trailing_commas(s: str) -> str:
    """Remove commas that directly precede '}' or ']' (outside strings)."""
    out: list[str] = []
    in_str = False
    escaped = False
    n = len(s)
    i = 0
    while i < n:
        ch = s[i]
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
            out.append(ch)
        elif ch == ",":
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j >= n or s[j] not in "}]":
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------- Gemini schema

class _UnsupportedSchema(Exception):
    pass


_SCHEMA_TYPES = {"string", "number", "integer", "boolean", "array", "object"}
_SCHEMA_REJECT = {"$ref", "oneOf", "anyOf", "allOf", "not", "if", "then", "else",
                  "patternProperties", "$defs", "definitions", "const", "dependentSchemas"}


def gemini_schema(schema: dict | None) -> dict | None:
    """Convert a plain JSON schema to Gemini's OpenAPI-subset `responseSchema`.

    Conservative: only type/properties/required/items/enum/description/nullable survive, and
    anything that cannot be mapped safely (refs, unions, free-form objects) returns None so the
    caller falls back to JSON mode plus prompt instructions.
    """
    if not isinstance(schema, dict) or not schema:
        return None
    try:
        return _convert_schema(schema)
    except (_UnsupportedSchema, TypeError, ValueError, RecursionError):
        return None


def _convert_schema(node: Any) -> dict:
    if not isinstance(node, dict) or _SCHEMA_REJECT & node.keys():
        raise _UnsupportedSchema
    typ = node.get("type")
    nullable = False
    if isinstance(typ, list):
        nullable = "null" in typ
        concrete = [t for t in typ if t != "null"]
        if len(concrete) != 1:
            raise _UnsupportedSchema
        typ = concrete[0]
    if typ is None:
        if "properties" in node:
            typ = "object"
        elif "items" in node:
            typ = "array"
        elif isinstance(node.get("enum"), list):
            typ = "string"
    if typ not in _SCHEMA_TYPES:
        raise _UnsupportedSchema

    out: dict[str, Any] = {"type": typ.upper()}
    if isinstance(node.get("description"), str):
        out["description"] = node["description"]
    if nullable or node.get("nullable") is True:
        out["nullable"] = True
    if "enum" in node:
        enum = node["enum"]
        if typ != "string" or not isinstance(enum, list) or not enum or not all(isinstance(e, str) for e in enum):
            raise _UnsupportedSchema
        out["enum"] = list(enum)

    if typ == "object":
        props = node.get("properties")
        if not isinstance(props, dict) or not props:
            raise _UnsupportedSchema  # Gemini rejects OBJECT without properties
        out["properties"] = {str(k): _convert_schema(v) for k, v in props.items()}
        out["propertyOrdering"] = list(out["properties"])
        required = [r for r in (node.get("required") or []) if r in out["properties"]]
        if required:
            out["required"] = required
    elif typ == "array":
        if "items" not in node:
            raise _UnsupportedSchema
        out["items"] = _convert_schema(node["items"])
        for bound in ("minItems", "maxItems"):
            if isinstance(node.get(bound), int) and not isinstance(node.get(bound), bool):
                out[bound] = node[bound]
    return out


# --------------------------------------------------------------------------- helpers

_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"gsk_[0-9A-Za-z]{10,}"),
    re.compile(r"sk-[0-9A-Za-z_\-]{10,}"),
    re.compile(r"(?i)(bearer\s+)[^\s\"',]+"),
    re.compile(r"(?i)((?:api[_-]?)?key=)[^&\s\"',]+"),
]


def _cache_key(system: str, user: str, max_tokens: int, temperature: float, schema: dict | None) -> str:
    payload = json.dumps([system, user, int(max_tokens), round(float(temperature), 4), schema],
                         sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_models(value: Any) -> list[str]:
    items = value if isinstance(value, (list, tuple)) else str(value or "").split(",")
    out: list[str] = []
    for item in items:
        name = str(item).strip()
        if name.startswith("models/"):
            name = name[len("models/"):]
        if name and name not in out:
            out.append(name)
    return out


def _schema_keys(schema: dict | None) -> list[str]:
    props = (schema or {}).get("properties") if isinstance(schema, dict) else None
    return [str(k) for k in props] if isinstance(props, dict) else []


def _required_keys(schema: dict | None) -> list[str]:
    req = (schema or {}).get("required") if isinstance(schema, dict) else None
    return [str(k) for k in req] if isinstance(req, list) else []


def _conform(data: dict, schema: dict | None) -> dict | None:
    """Check top-level required keys; unwrap a single nested object ({"response": {...}}) if that
    is where they ended up. None means the output does not satisfy the schema."""
    required = _required_keys(schema)
    if not required or all(k in data for k in required):
        return data
    nested = [v for v in data.values() if isinstance(v, dict) and all(k in v for k in required)]
    return nested[0] if len(nested) == 1 else None


def _json_hint(schema: dict | None) -> str:
    keys = _schema_keys(schema)
    if keys:
        return f"Respond with a single JSON object with the keys: {', '.join(keys)}."
    return JSON_INSTRUCTION


def _parse_duration(value: Any) -> float | None:
    """'37s', '1.5s', '2m30s', '120', 12 -> seconds."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip().lower()
    try:
        return float(raw)
    except ValueError:
        pass
    total, matched = 0.0, False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(ms|h|m|s)", raw):
        matched = True
        total += float(num) * {"ms": 0.001, "h": 3600.0, "m": 60.0, "s": 1.0}[unit]
    return total if matched else None


def _error_payload(resp: httpx.Response) -> dict:
    try:
        data = resp.json()
    except Exception:
        return {}
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]  # some Google errors arrive as a one-element list
    if not isinstance(data, dict):
        return {}
    err = data.get("error")
    return err if isinstance(err, dict) else ({"message": err} if isinstance(err, str) else data)


def _error_message(resp: httpx.Response) -> str:
    err = _error_payload(resp)
    parts = [str(err.get(k)) for k in ("status", "code", "type", "message") if err.get(k)]
    msg = " ".join(parts) if parts else (resp.text or "")
    return f"HTTP {resp.status_code}: {msg}"[:500]


def _retry_after(resp: httpx.Response) -> float | None:
    headers = resp.headers
    ms = headers.get("retry-after-ms")
    if ms:
        try:
            return float(ms) / 1000.0
        except ValueError:
            pass
    ra = headers.get("retry-after")
    if ra:
        secs = _parse_duration(ra)
        if secs is not None:
            return secs
        try:
            when = parsedate_to_datetime(ra)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, IndexError):
            pass
    # Google puts it in the body: error.details[] {"@type": ".../google.rpc.RetryInfo", "retryDelay": "37s"}
    details = _error_payload(resp).get("details")
    if isinstance(details, list):
        for d in details:
            if isinstance(d, dict) and d.get("retryDelay"):
                return _parse_duration(d.get("retryDelay"))
    return None


def _is_daily_quota(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in ("perday", "per day", "per_day", "(rpd)", "(tpd)", "daily"))


# --------------------------------------------------------------------------- internal types

@dataclass(frozen=True)
class _Provider:
    name: str
    key: str
    models: tuple[str, ...]
    rpm: int
    base_url: str = ""


@dataclass
class _Outcome:
    # kind: ok | rate_limited | server_error | timeout | network | auth | model_missing
    #       | bad_request | bad_output | blocked
    kind: str
    data: dict | None = None
    detail: str = ""
    retry_after: float | None = None
    daily_quota: bool = False

    @property
    def ok(self) -> bool:
        return self.kind == "ok" and isinstance(self.data, dict)


class _AttemptTimeout(Exception):
    pass


class _RateWindow:
    """At most `rpm` request starts in any rolling 60 s window (how free-tier RPM is enforced).

    Waiters reserve a future slot before sleeping, so concurrent callers queue fairly; a
    cancelled waiter gives its slot back.
    """

    def __init__(self, rpm: int, clock: Callable[[], float]) -> None:
        self.rpm = max(0, int(rpm or 0))
        self._clock = clock
        self._starts: deque[float] = deque()

    def wait_time(self) -> float:
        if self.rpm <= 0:
            return 0.0
        now = self._clock()
        while self._starts and self._starts[0] <= now - 60.0:
            self._starts.popleft()
        if len(self._starts) < self.rpm:
            return 0.0
        return max(0.0, self._starts[-self.rpm] + 60.0 - now)

    async def acquire(self, max_wait: float) -> bool:
        """Take a slot, sleeping at most `max_wait` seconds. False means "skip, don't wait"."""
        if self.rpm <= 0:
            return True
        wait = self.wait_time()
        if wait > 0 and wait > max_wait:
            return False
        slot = self._clock() + wait
        self._starts.append(slot)
        if wait > 0:
            try:
                await asyncio.sleep(wait)
            except asyncio.CancelledError:
                try:
                    self._starts.remove(slot)
                except ValueError:
                    pass
                raise
        return True


# --------------------------------------------------------------------------- client

class LLMClient:
    """JSON-only LLM client with provider fallback, rate windows, cooldowns and a cache.

    Usage: `await get_llm().complete_json(system, user, timeout=5)` -> dict or None.
    """

    # Tunables (class attributes so tests can shrink them on an instance).
    min_attempt_s: float = 1.0            # never start a request with less time than this left
    fallback_reserve_s: float = 2.0       # time kept back for a fallback while one is available
    primary_share: float = 0.65           # ...but the current attempt always gets this share
    max_rate_wait_s: float = 1.5          # max wait for a rate slot when another provider is ready
    cooldown_s: float = 30.0              # default cooldown after 429 / 5xx
    max_retry_after_s: float = 120.0      # cap on honoured Retry-After
    timeout_cooldown_s: float = 15.0      # cooldown after a timeout...
    timeout_cooldown_min_budget_s: float = 3.0   # ...only if the attempt had a fair budget
    network_cooldown_s: float = 10.0
    auth_cooldown_s: float = 600.0
    daily_quota_cooldown_s: float = 900.0
    dead_model_s: float = 3600.0
    cache_max_entries: int = 2000

    def __init__(self, settings: config.Settings | None = None, *,
                 transport: httpx.AsyncBaseTransport | None = None,
                 clock: Callable[[], float] | None = None) -> None:
        self.settings = settings if settings is not None else config.settings
        self._transport = transport
        self._clock: Callable[[], float] = clock or time.monotonic
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._inflight: dict[str, asyncio.Task] = {}
        self._cooldown_until: dict[str, float] = {}
        self._windows: dict[str, _RateWindow] = {}
        self._dropped_features: set[tuple[str, str, str]] = set()   # (provider, model, feature)
        self.stats: dict[str, Any] = {
            "calls": 0, "cache_hits": 0, "deduped": 0, "successes": 0, "failures": 0,
            "last_error": "", "providers": {},
        }

    # -- configuration ---------------------------------------------------------------------
    def _providers(self) -> list[_Provider]:
        """Active providers (key set, known name) in configured order."""
        s = self.settings
        out: list[_Provider] = []
        seen: set[str] = set()
        for raw in getattr(s, "llm_providers", None) or []:
            name = _PROVIDER_ALIASES.get(str(raw).strip().lower(), str(raw).strip().lower())
            if name in seen or name not in KNOWN_PROVIDERS:
                continue
            seen.add(name)
            key = str(getattr(s, f"{name}_api_key", "") or "").strip()
            if not key:
                continue
            if name == "gemini":
                models = _split_models(getattr(s, "gemini_models", None) or ["gemini-2.5-flash"])
            else:
                models = _split_models(getattr(s, f"{name}_model", ""))
            if not models:
                continue
            base = {"groq": GROQ_BASE_URL, "openrouter": OPENROUTER_BASE_URL,
                    "openai": str(getattr(s, "openai_base_url", "") or "https://api.openai.com/v1")}.get(name, "")
            out.append(_Provider(name, key, tuple(models), int(getattr(s, f"{name}_rpm", 0) or 0), base))
        return out

    def available(self) -> bool:
        try:
            return not bool(getattr(self.settings, "llm_disabled", False)) and bool(self._providers())
        except Exception:
            return False

    def describe(self) -> str:
        """Human-readable model line for /v1/metadata (never contains secrets)."""
        try:
            if getattr(self.settings, "llm_disabled", False):
                return "deterministic-templates (LLM disabled)"
            labels = [m if p.name == "gemini" else f"{p.name} {m}" for p in self._providers() for m in p.models]
        except Exception:
            labels = []
        if not labels:
            return "deterministic-templates (no LLM key configured)"
        if len(labels) == 1:
            return f"{labels[0]} (fallback: deterministic templates)"
        return f"{labels[0]} (fallbacks: {', '.join(labels[1:])}; deterministic templates)"

    def stats_snapshot(self) -> dict:
        return copy.deepcopy(self.stats)

    # -- public entry point ------------------------------------------------------------------
    async def complete_json(self, system: str, user: str, *, schema: dict | None = None,
                            timeout: float | None = None, max_tokens: int = 800,
                            temperature: float = 0.0) -> dict | None:
        """One JSON object from the first provider that answers in time, else None. Never raises
        (except CancelledError, which must propagate)."""
        try:
            self.stats["calls"] += 1
            if not self.available():
                return None
            budget = float(timeout if timeout is not None else self.settings.llm_timeout_s)
            if budget <= 0:
                return None
            system, user = str(system or ""), str(user or "")
            max_tokens = max(16, int(max_tokens or 800))
            temperature = float(temperature or 0.0)
            deadline = self._clock() + budget
            key = _cache_key(system, user, max_tokens, temperature, schema)

            cached = self._cache.get(key)
            if cached is not None:
                self.stats["cache_hits"] += 1
                return copy.deepcopy(cached)

            loop = asyncio.get_running_loop()
            task = self._inflight.get(key)
            if task is not None and not task.done() and task.get_loop() is loop:
                # Identical request already running (e.g. precompose + tick): share its result.
                self.stats["deduped"] += 1
                result = await self._await_task(task, deadline)
                if result is not None or deadline - self._clock() < self.min_attempt_s:
                    return copy.deepcopy(result) if result is not None else None

            task = loop.create_task(self._run_guarded(system, user, schema, max_tokens, temperature, deadline, key))
            self._inflight[key] = task
            task.add_done_callback(lambda t, k=key: self._inflight.pop(k, None) if self._inflight.get(k) is t else None)
            result = await self._await_task(task, deadline)
            return copy.deepcopy(result) if result is not None else None
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # absolutely never raise to callers
            self._record_error("", "", f"internal: {type(exc).__name__}: {exc}")
            return None

    async def _await_task(self, task: asyncio.Task, deadline: float) -> dict | None:
        """Wait for a (shielded) task up to our own deadline; the task outlives a cancelled caller
        but is bounded by its own deadline."""
        remaining = deadline - self._clock()
        if remaining <= 0:
            return None
        try:
            return await asyncio.wait_for(asyncio.shield(task), remaining + 0.05)
        except TimeoutError:
            return None
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if task.cancelled() and (current is None or current.cancelling() == 0):
                return None  # the shared task was cancelled (e.g. aclose), not us
            raise

    async def aclose(self) -> None:
        for task in list(self._inflight.values()):
            task.cancel()
        self._inflight.clear()
        client, self._client = self._client, None
        loop, self._client_loop = self._client_loop, None
        if client is not None:
            try:
                if loop is asyncio.get_running_loop():
                    await client.aclose()
            except Exception:
                pass

    def clear_cache(self) -> None:
        self._cache.clear()

    # -- orchestration ---------------------------------------------------------------------
    async def _run_guarded(self, *args: Any) -> dict | None:
        try:
            return await self._run(*args)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._record_error("", "", f"internal: {type(exc).__name__}: {exc}")
            return None

    async def _run(self, system: str, user: str, schema: dict | None, max_tokens: int,
                   temperature: float, deadline: float, key: str) -> dict | None:
        queue: deque[tuple[_Provider, str]] = deque((p, m) for p in self._providers() for m in p.models)
        while queue:
            prov, model = queue.popleft()
            remaining = deadline - self._clock()
            if remaining < self.min_attempt_s:
                break
            if self._is_cooling(prov.name, model):
                continue
            has_fallback = any(not self._is_cooling(p.name, m) for p, m in queue)

            max_wait = remaining - self.min_attempt_s
            if has_fallback:
                max_wait = min(max_wait, self.max_rate_wait_s)
            if not await self._window(prov, model).acquire(max_wait):
                self._pstats(prov.name)["rate_skips"] += 1
                continue

            remaining = deadline - self._clock()
            if remaining < self.min_attempt_s:
                break
            budget = remaining
            if has_fallback:
                budget = max(remaining - self.fallback_reserve_s, remaining * self.primary_share)

            ps = self._pstats(prov.name)
            ps["calls"] += 1
            ps["last_model"] = model
            started = self._clock()
            outcome = await self._attempt(prov, model, system, user, schema, max_tokens, temperature,
                                          started + budget)
            ps["last_latency_ms"] = int((self._clock() - started) * 1000)

            if outcome.ok:
                data = _conform(outcome.data, schema)
                if data is None:
                    missing = [k for k in _required_keys(schema) if k not in outcome.data]
                    outcome = _Outcome("bad_output", detail=f"missing required keys: {', '.join(missing)}")
                else:
                    ps["successes"] += 1
                    self.stats["successes"] += 1
                    self._cache_put(key, data)
                    log.debug("llm ok via %s/%s in %sms", prov.name, model, ps["last_latency_ms"])
                    return data

            ps["failures"] += 1
            self._record_error(prov.name, model, f"{outcome.kind}: {outcome.detail}")
            self._apply_cooldown(prov, model, outcome, budget)
            if outcome.kind in ("timeout", "network"):
                # Same host is struggling: try other providers before this provider's other models.
                same = [c for c in queue if c[0].name == prov.name]
                queue = deque([c for c in queue if c[0].name != prov.name] + same)

        self.stats["failures"] += 1
        return None

    def _apply_cooldown(self, prov: _Provider, model: str, outcome: _Outcome, budget: float) -> None:
        model_key = f"{prov.name}:{model}"
        kind = outcome.kind
        if kind == "rate_limited":
            if outcome.daily_quota:
                # Daily quota: honour a rolling-window hint (Groq TPD) but never probe more than once a minute.
                secs = (min(max(outcome.retry_after, 60.0), self.daily_quota_cooldown_s)
                        if outcome.retry_after else self.daily_quota_cooldown_s)
            else:
                secs = self._bounded_retry(outcome.retry_after)
            self._cool(model_key if prov.name in _PER_MODEL_QUOTA else prov.name, secs)
        elif kind == "server_error":
            self._cool(model_key, self._bounded_retry(outcome.retry_after))
        elif kind == "timeout":
            if budget >= self.timeout_cooldown_min_budget_s:
                self._cool(model_key, self.timeout_cooldown_s)
        elif kind == "network":
            self._cool(prov.name, self.network_cooldown_s)
        elif kind == "auth":
            self._cool(prov.name, self.auth_cooldown_s)
        elif kind == "model_missing":
            self._cool(model_key, self.dead_model_s)
        # bad_request / bad_output / blocked: no cooldown, just move on.

    def _bounded_retry(self, retry_after: float | None) -> float:
        if retry_after is None or retry_after <= 0:
            return self.cooldown_s
        return min(max(retry_after, 1.0), self.max_retry_after_s)

    def _cool(self, key: str, seconds: float) -> None:
        until = self._clock() + max(0.0, seconds)
        self._cooldown_until[key] = max(self._cooldown_until.get(key, 0.0), until)

    def _is_cooling(self, provider: str, model: str) -> bool:
        now = self._clock()
        return (self._cooldown_until.get(provider, 0.0) > now
                or self._cooldown_until.get(f"{provider}:{model}", 0.0) > now)

    def _window(self, prov: _Provider, model: str) -> _RateWindow:
        key = f"{prov.name}:{model}" if prov.name in _PER_MODEL_QUOTA else prov.name
        win = self._windows.get(key)
        if win is None or win.rpm != max(0, prov.rpm):
            win = _RateWindow(prov.rpm, self._clock)
            self._windows[key] = win
        return win

    # -- cache / stats ---------------------------------------------------------------------
    def _cache_put(self, key: str, data: dict) -> None:
        self._cache[key] = copy.deepcopy(data)
        self._cache.move_to_end(key)
        while len(self._cache) > max(1, int(self.cache_max_entries)):
            self._cache.popitem(last=False)

    def _pstats(self, provider: str) -> dict:
        return self.stats["providers"].setdefault(provider, {
            "calls": 0, "successes": 0, "failures": 0, "rate_skips": 0,
            "last_error": "", "last_model": "", "last_latency_ms": 0,
        })

    def _redact(self, text: str) -> str:
        out = str(text)
        for prov in KNOWN_PROVIDERS:
            secret = str(getattr(self.settings, f"{prov}_api_key", "") or "").strip()
            if len(secret) >= 4:
                out = out.replace(secret, "***")
        for pat in _SECRET_PATTERNS:
            out = pat.sub(lambda m: (m.group(1) if m.groups() else "") + "***", out)
        return out[:300]

    def _record_error(self, provider: str, model: str, message: str) -> None:
        msg = self._redact(f"{provider}/{model}: {message}" if provider else message)
        self.stats["last_error"] = msg
        if provider:
            self._pstats(provider)["last_error"] = msg
        log.info("llm attempt failed: %s", msg)

    # -- HTTP --------------------------------------------------------------------------------
    def _http(self) -> httpx.AsyncClient:
        """One shared client per event loop (a pooled connection cannot cross loops)."""
        loop = asyncio.get_running_loop()
        if self._client is None or self._client.is_closed or self._client_loop is not loop:
            self._client = httpx.AsyncClient(
                transport=self._transport,
                timeout=httpx.Timeout(15.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10, keepalive_expiry=30.0),
                headers={"user-agent": "vera-bot/1.0"},
            )
            self._client_loop = loop
        return self._client

    async def _post(self, url: str, headers: dict, body: dict, attempt_deadline: float) -> httpx.Response:
        remaining = attempt_deadline - self._clock()
        if remaining <= 0.05:
            raise _AttemptTimeout()
        client = self._http()
        async with asyncio.timeout(remaining):
            return await client.post(url, headers=headers, json=body, timeout=httpx.Timeout(remaining))

    async def _attempt(self, prov: _Provider, model: str, system: str, user: str, schema: dict | None,
                       max_tokens: int, temperature: float, attempt_deadline: float) -> _Outcome:
        try:
            if prov.name == "gemini":
                return await self._call_gemini(prov, model, system, user, schema, max_tokens, temperature, attempt_deadline)
            if prov.name == "anthropic":
                return await self._call_anthropic(prov, model, system, user, schema, max_tokens, temperature, attempt_deadline)
            return await self._call_openai_compat(prov, model, system, user, schema, max_tokens, temperature, attempt_deadline)
        except (_AttemptTimeout, TimeoutError, httpx.TimeoutException):
            return _Outcome("timeout", detail="deadline reached")
        except httpx.HTTPError as exc:
            return _Outcome("network", detail=f"{type(exc).__name__}: {exc}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return _Outcome("bad_output", detail=f"{type(exc).__name__}: {exc}")

    # -- feature adaptation --------------------------------------------------------------------
    def _feature_on(self, prov: _Provider, model: str, feature: str) -> bool:
        return ((prov.name, model, feature) not in self._dropped_features
                and (prov.name, "*", feature) not in self._dropped_features)

    def _drop_feature(self, prov: _Provider, model: str, feature: str) -> None:
        self._dropped_features.add((prov.name, model, feature))

    @staticmethod
    def _rejected_feature(message: str, active: dict[str, bool], keywords: dict[str, tuple[str, ...]]) -> str | None:
        msg = message.lower()
        for feature, words in keywords.items():
            if active.get(feature) and any(w in msg for w in words):
                return feature
        return None

    def _classify(self, prov: _Provider, resp: httpx.Response) -> _Outcome:
        status = resp.status_code
        msg = _error_message(resp)
        low = msg.lower()
        if status == 429:
            # Google names the exhausted quota only in error.details (e.g. "...PerDayPerProjectPerModel...").
            return _Outcome("rate_limited", detail=msg, retry_after=_retry_after(resp),
                            daily_quota=_is_daily_quota(resp.text or msg))
        if status >= 500:
            return _Outcome("server_error", detail=msg, retry_after=_retry_after(resp))
        if status in (401, 403) or "api key" in low or "api_key" in low or "location is not supported" in low:
            return _Outcome("auth", detail=msg)
        if status == 404 or ("model" in low and any(w in low for w in (
                "not found", "not_found", "does not exist", "decommissioned", "not supported for",
                "unsupported model", "invalid model", "no endpoints found", "is not a valid model"))):
            return _Outcome("model_missing", detail=msg)
        return _Outcome("bad_request", detail=msg)

    # -- Gemini --------------------------------------------------------------------------------
    async def _call_gemini(self, prov: _Provider, model: str, system: str, user: str, schema: dict | None,
                           max_tokens: int, temperature: float, attempt_deadline: float) -> _Outcome:
        g_schema = gemini_schema(schema)
        active = {
            "thinking": "2.5" in model and self._feature_on(prov, model, "thinking"),
            "schema": g_schema is not None and self._feature_on(prov, model, "schema"),
            "safety": self._feature_on(prov, model, "safety"),
        }
        keywords = {"thinking": ("thinking",),
                    "schema": ("response_schema", "responseschema", "schema", "propertyordering"),
                    "safety": ("safety",)}
        url = GEMINI_URL.format(model=model)
        headers = {"x-goog-api-key": prov.key, "content-type": "application/json"}
        outcome = _Outcome("bad_request", detail="no attempt made")
        for _ in range(4):
            sys_text = system if active["schema"] else f"{system.rstrip()}\n\n{_json_hint(schema)}"
            gen: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens,
                                   "responseMimeType": "application/json"}
            if active["thinking"]:
                gen["thinkingConfig"] = {"thinkingBudget": 0}
            if active["schema"]:
                gen["responseSchema"] = g_schema
            body: dict[str, Any] = {
                "systemInstruction": {"parts": [{"text": sys_text}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": gen,
            }
            if active["safety"]:
                body["safetySettings"] = _GEMINI_SAFETY
            resp = await self._post(url, headers, body, attempt_deadline)
            if resp.status_code == 200:
                return self._parse_gemini(resp)
            if resp.status_code == 400:
                feature = self._rejected_feature(_error_message(resp), active, keywords)
                if feature:
                    active[feature] = False
                    self._drop_feature(prov, model, feature)
                    continue
            return self._classify(prov, resp)
        return outcome

    @staticmethod
    def _parse_gemini(resp: httpx.Response) -> _Outcome:
        try:
            data = resp.json()
        except ValueError:
            return _Outcome("bad_output", detail="response is not JSON")
        if not isinstance(data, dict):
            return _Outcome("bad_output", detail="unexpected response shape")
        candidates = data.get("candidates") or []
        if not candidates or not isinstance(candidates[0], dict):
            reason = (data.get("promptFeedback") or {}).get("blockReason") or "no candidates"
            return _Outcome("blocked", detail=f"empty response ({reason})")
        cand = candidates[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts
                       if isinstance(p, dict) and isinstance(p.get("text"), str) and not p.get("thought"))
        finish = cand.get("finishReason") or ""
        if not text.strip():
            kind = "blocked" if finish in ("SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII") else "bad_output"
            return _Outcome(kind, detail=f"no text (finishReason={finish or 'none'})")
        obj = extract_json(text)
        if obj is None:
            return _Outcome("bad_output", detail=f"unparseable JSON (finishReason={finish or 'none'})")
        return _Outcome("ok", data=obj)

    # -- OpenAI-compatible (Groq, OpenAI, OpenRouter) -------------------------------------------
    async def _call_openai_compat(self, prov: _Provider, model: str, system: str, user: str, schema: dict | None,
                                  max_tokens: int, temperature: float, attempt_deadline: float) -> _Outcome:
        base = prov.base_url.rstrip("/")
        url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        headers = {"authorization": f"Bearer {prov.key}", "content-type": "application/json"}
        if prov.name == "openrouter":
            headers["x-title"] = "Vera merchant assistant"
        active = {
            "response_format": self._feature_on(prov, model, "response_format"),
            "max_tokens": self._feature_on(prov, model, "max_tokens"),     # off => max_completion_tokens
            "temperature": self._feature_on(prov, model, "temperature"),
        }
        keywords = {"response_format": ("response_format", "json_object", "json mode", "response format"),
                    "max_tokens": ("max_completion_tokens",),
                    "temperature": ("temperature",)}
        sys_text = f"{system.rstrip()}\n\n{_json_hint(schema)}"
        outcome = _Outcome("bad_request", detail="no attempt made")
        for _ in range(4):
            body: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "system", "content": sys_text}, {"role": "user", "content": user}],
                ("max_tokens" if active["max_tokens"] else "max_completion_tokens"): max_tokens,
            }
            if active["temperature"]:
                body["temperature"] = temperature
            if active["response_format"]:
                body["response_format"] = {"type": "json_object"}
            resp = await self._post(url, headers, body, attempt_deadline)
            if resp.status_code == 200:
                return self._parse_openai(resp)
            if resp.status_code in (400, 422):
                err = _error_payload(resp)
                # Groq JSON mode: the model produced invalid JSON but returns what it wrote.
                failed = err.get("failed_generation")
                if isinstance(failed, str):
                    obj = extract_json(failed)
                    if obj is not None:
                        return _Outcome("ok", data=obj)
                feature = self._rejected_feature(_error_message(resp), active, keywords)
                if feature is None and str(err.get("code") or "") == "json_validate_failed" and active["response_format"]:
                    feature = "response_format"
                if feature:
                    active[feature] = False
                    if feature != "response_format" or str(err.get("code") or "") != "json_validate_failed":
                        self._drop_feature(prov, model, feature)   # remember real parameter rejections only
                    continue
            return self._classify(prov, resp)
        return outcome

    @staticmethod
    def _parse_openai(resp: httpx.Response) -> _Outcome:
        try:
            data = resp.json()
        except ValueError:
            return _Outcome("bad_output", detail="response is not JSON")
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices or not isinstance(choices[0], dict):
            return _Outcome("bad_output", detail="no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):  # some gateways return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not isinstance(content, str) or not content.strip():
            kind = "blocked" if message.get("refusal") or choices[0].get("finish_reason") == "content_filter" else "bad_output"
            return _Outcome(kind, detail="empty content")
        obj = extract_json(content)
        if obj is None:
            return _Outcome("bad_output", detail=f"unparseable JSON (finish={choices[0].get('finish_reason')})")
        return _Outcome("ok", data=obj)

    # -- Anthropic -----------------------------------------------------------------------------
    async def _call_anthropic(self, prov: _Provider, model: str, system: str, user: str, schema: dict | None,
                              max_tokens: int, temperature: float, attempt_deadline: float) -> _Outcome:
        headers = {"x-api-key": prov.key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}
        active = {"temperature": self._feature_on(prov, model, "temperature")}
        sys_text = f"{system.rstrip()}\n\n{_json_hint(schema)} Output only the JSON object, no prose."
        outcome = _Outcome("bad_request", detail="no attempt made")
        for _ in range(2):
            body: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "system": sys_text,
                "messages": [{"role": "user", "content": user}],
            }
            if active["temperature"]:
                body["temperature"] = temperature
            resp = await self._post(ANTHROPIC_URL, headers, body, attempt_deadline)
            if resp.status_code == 200:
                return self._parse_anthropic(resp)
            if resp.status_code == 400:
                feature = self._rejected_feature(_error_message(resp), active, {"temperature": ("temperature",)})
                if feature:
                    active[feature] = False
                    self._drop_feature(prov, model, feature)
                    continue
            return self._classify(prov, resp)
        return outcome

    @staticmethod
    def _parse_anthropic(resp: httpx.Response) -> _Outcome:
        try:
            data = resp.json()
        except ValueError:
            return _Outcome("bad_output", detail="response is not JSON")
        blocks = data.get("content") if isinstance(data, dict) else None
        text = "".join(b.get("text", "") for b in (blocks or [])
                       if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str))
        if not text.strip():
            kind = "blocked" if isinstance(data, dict) and data.get("stop_reason") == "refusal" else "bad_output"
            return _Outcome(kind, detail="empty content")
        obj = extract_json(text)
        if obj is None:
            return _Outcome("bad_output", detail=f"unparseable JSON (stop={data.get('stop_reason')})")
        return _Outcome("ok", data=obj)


# --------------------------------------------------------------------------- singleton

_llm: LLMClient | None = None


def get_llm() -> LLMClient:
    """Process-wide client, built lazily from app.config.settings."""
    global _llm
    if _llm is None:
        _llm = LLMClient(config.settings)
    return _llm


def set_llm(client: Any) -> None:
    """Replace the singleton (tests inject fakes with the same duck-typed API); None resets it."""
    global _llm
    _llm = client
