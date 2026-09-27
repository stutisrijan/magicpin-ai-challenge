"""Tests for app/llm.py. Fully offline: every HTTP call goes through httpx.MockTransport."""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from app import llm as llm_mod
from app.config import Settings
from app.llm import LLMClient, extract_json, gemini_schema, get_llm, set_llm

GEMINI_KEY = "AIzaTESTKEY0000000000000000000000000000"
GROQ_KEY = "gsk_testkey1234567890abcdef"
KEY_FIELDS = ("gemini_api_key", "groq_api_key", "openai_api_key", "anthropic_api_key", "openrouter_api_key")

SCHEMA = {
    "type": "object",
    "properties": {
        "body": {"type": "string", "description": "WhatsApp message"},
        "cta": {"type": "string", "enum": ["binary_yes_no", "open_ended"]},
        "rationale": {"type": "string"},
    },
    "required": ["body", "cta"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- helpers

def make_settings(**overrides) -> Settings:
    s = Settings()
    for field in KEY_FIELDS:
        setattr(s, field, "")
    s.llm_disabled = False
    s.llm_providers = ["gemini", "groq", "openai", "anthropic", "openrouter"]
    s.gemini_models = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    s.groq_model = "llama-3.3-70b-versatile"
    s.openai_model = "gpt-4o-mini"
    s.openai_base_url = "https://api.openai.com/v1"
    s.anthropic_model = "claude-test"
    s.openrouter_model = "meta-llama/llama-3.3-70b-instruct:free"
    s.llm_timeout_s = 3.0
    for rpm in ("gemini_rpm", "groq_rpm", "openai_rpm", "anthropic_rpm", "openrouter_rpm"):
        setattr(s, rpm, 1000)
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def make_client(handler, **overrides) -> LLMClient:
    client = LLMClient(make_settings(**overrides), transport=httpx.MockTransport(handler))
    # Shrink the time-based tunables so the suite stays fast.
    client.min_attempt_s = 0.05
    client.fallback_reserve_s = 0.3
    client.max_rate_wait_s = 0.2
    return client


def gemini_ok(text: str) -> httpx.Response:
    return httpx.Response(200, json={"candidates": [
        {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}]})


def openai_ok(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [
        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})


def gemini_model(request: httpx.Request) -> str:
    return request.url.path.rsplit("/", 1)[-1].split(":")[0]


class Recorder:
    """Records every request and routes it to a per-host handler."""

    def __init__(self, **routes):
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def hosts(self) -> list[str]:
        return [r.url.host for r in self.requests]

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.url.host
        name = {"generativelanguage.googleapis.com": "gemini", "api.groq.com": "groq",
                "api.openai.com": "openai", "api.anthropic.com": "anthropic",
                "openrouter.ai": "openrouter"}[host]
        result = self.routes[name](request)
        if asyncio.iscoroutine(result):
            result = await result
        return result


# --------------------------------------------------------------------------- extract_json

class TestExtractJson:
    def test_plain(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced(self):
        assert extract_json('```json\n{"body": "hi", "n": 2}\n```') == {"body": "hi", "n": 2}

    def test_fenced_uppercase_and_unclosed(self):
        assert extract_json('```JSON\n{"a": 1}\n```') == {"a": 1}
        assert extract_json('```json\n{"a": {"b": 2}}') == {"a": {"b": 2}}

    def test_prose_around(self):
        text = 'Sure! Here is the message:\n{"body": "Dr. Meera, 3 slots open"}\nLet me know.'
        assert extract_json(text) == {"body": "Dr. Meera, 3 slots open"}

    def test_nested_braces_and_quotes_in_strings(self):
        raw = '{"body": "use {curly} braces } and \\"quotes\\"", "meta": {"a": {"b": [1, 2]}}}'
        assert extract_json("prefix " + raw + " suffix") == {
            "body": 'use {curly} braces } and "quotes"', "meta": {"a": {"b": [1, 2]}}}

    def test_brace_in_prose_before_json(self):
        assert extract_json('I used {placeholders} earlier; final: {"ok": true}') == {"ok": True}

    def test_trailing_commas(self):
        assert extract_json('{"a": [1, 2,], "b": "x, ]",}') == {"a": [1, 2], "b": "x, ]"}

    def test_raw_newline_inside_string(self):
        assert extract_json('{"body": "line one\nline two"}') == {"body": "line one\nline two"}

    def test_single_object_list_and_double_encoded(self):
        assert extract_json('[{"a": 1}]') == {"a": 1}
        assert extract_json(json.dumps(json.dumps({"a": 1}))) == {"a": 1}

    @pytest.mark.parametrize("bad", [None, "", "   ", "no json here", "[1, 2]", '"just a string"', "{broken", 42])
    def test_failures_return_none(self, bad):
        assert extract_json(bad) is None

    def test_dict_passthrough(self):
        assert extract_json({"a": 1}) == {"a": 1}


# --------------------------------------------------------------------------- schema conversion

def test_gemini_schema_conversion():
    out = gemini_schema(SCHEMA)
    assert out["type"] == "OBJECT"
    assert out["properties"]["cta"] == {"type": "STRING", "enum": ["binary_yes_no", "open_ended"]}
    assert out["propertyOrdering"] == ["body", "cta", "rationale"]
    assert out["required"] == ["body", "cta"]
    assert "additionalProperties" not in out


def test_gemini_schema_nullable_and_arrays():
    out = gemini_schema({"type": "object", "properties": {
        "params": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "note": {"type": ["string", "null"]}}})
    assert out["properties"]["params"] == {"type": "ARRAY", "items": {"type": "STRING"}, "maxItems": 3}
    assert out["properties"]["note"] == {"type": "STRING", "nullable": True}


@pytest.mark.parametrize("schema", [
    None, {}, {"anyOf": [{"type": "string"}]},
    {"type": "object"},                                          # free-form object
    {"type": "object", "properties": {"x": {"$ref": "#/defs/x"}}},
    {"type": "array"},
    {"type": ["string", "integer"]},
])
def test_gemini_schema_unsupported_returns_none(schema):
    assert gemini_schema(schema) is None


# --------------------------------------------------------------------------- availability / describe

def test_disabled_means_unavailable_and_no_requests():
    rec = Recorder(gemini=lambda r: gemini_ok('{"a": 1}'))
    client = make_client(rec, gemini_api_key=GEMINI_KEY, llm_disabled=True)
    assert client.available() is False
    assert client.describe() == "deterministic-templates (LLM disabled)"
    assert asyncio.run(client.complete_json("s", "u")) is None
    assert rec.requests == []


def test_no_keys_means_unavailable():
    client = make_client(Recorder())
    assert client.available() is False
    assert client.describe() == "deterministic-templates (no LLM key configured)"
    assert asyncio.run(client.complete_json("s", "u")) is None


def test_describe_strings():
    both = make_client(Recorder(), gemini_api_key=GEMINI_KEY, groq_api_key=GROQ_KEY)
    assert both.available() is True
    assert both.describe() == ("gemini-2.5-flash (fallbacks: gemini-2.5-flash-lite, "
                               "groq llama-3.3-70b-versatile; deterministic templates)")
    groq_only = make_client(Recorder(), groq_api_key=GROQ_KEY)
    assert groq_only.describe() == "groq llama-3.3-70b-versatile (fallback: deterministic templates)"
    ordered = make_client(Recorder(), gemini_api_key=GEMINI_KEY, groq_api_key=GROQ_KEY,
                          llm_providers=["groq", "gemini"], gemini_models=["gemini-2.0-flash"])
    assert ordered.describe() == "groq llama-3.3-70b-versatile (fallbacks: gemini-2.0-flash; deterministic templates)"
    for c in (both, groq_only, ordered):
        assert GEMINI_KEY not in c.describe() and GROQ_KEY not in c.describe()


def test_unknown_providers_and_aliases():
    client = make_client(Recorder(), llm_providers=["mystery", "google", "claude"],
                         gemini_api_key=GEMINI_KEY, anthropic_api_key="sk-ant-test-123456789012")
    assert client.describe() == ("gemini-2.5-flash (fallbacks: gemini-2.5-flash-lite, "
                                 "anthropic claude-test; deterministic templates)")


# --------------------------------------------------------------------------- Gemini

@pytest.mark.asyncio
async def test_gemini_happy_path_request_shape():
    def gemini(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/v1beta/models/gemini-2.5-flash:generateContent"
        assert request.headers["x-goog-api-key"] == GEMINI_KEY
        assert "key=" not in str(request.url)
        body = json.loads(request.content)
        assert body["systemInstruction"] == {"parts": [{"text": "SYS"}]}
        assert body["contents"] == [{"role": "user", "parts": [{"text": "USER"}]}]
        gen = body["generationConfig"]
        assert gen["temperature"] == 0.0 and gen["maxOutputTokens"] == 300
        assert gen["responseMimeType"] == "application/json"
        assert gen["thinkingConfig"] == {"thinkingBudget": 0}
        assert gen["responseSchema"]["propertyOrdering"] == ["body", "cta", "rationale"]
        return gemini_ok('{"body": "Dr. Meera, quick one", "cta": "open_ended"}')

    rec = Recorder(gemini=gemini)
    client = make_client(rec, gemini_api_key=GEMINI_KEY, groq_api_key=GROQ_KEY)
    out = await client.complete_json("SYS", "USER", schema=SCHEMA, max_tokens=300)
    assert out == {"body": "Dr. Meera, quick one", "cta": "open_ended"}
    assert rec.hosts() == ["generativelanguage.googleapis.com"]
    assert client.stats["providers"]["gemini"]["successes"] == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_gemini_thinking_off_for_any_model_and_json_hint_without_schema():
    def gemini(request):
        body = json.loads(request.content)
        assert body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
        assert "responseSchema" not in body["generationConfig"]
        assert "JSON" in body["systemInstruction"]["parts"][0]["text"]
        return gemini_ok('{"ok": 1}')

    client = make_client(Recorder(gemini=gemini), gemini_api_key=GEMINI_KEY, gemini_models=["gemini-2.0-flash"])
    assert await client.complete_json("sys", "u") == {"ok": 1}


@pytest.mark.asyncio
async def test_gemini_model_fallback_on_404_and_dead_model_is_skipped_next_time():
    def gemini(request):
        if gemini_model(request) == "gemini-2.5-flash":
            return httpx.Response(404, json={"error": {"code": 404, "status": "NOT_FOUND",
                                                       "message": "models/gemini-2.5-flash is not found"}})
        return gemini_ok('{"model": "lite"}')

    rec = Recorder(gemini=gemini)
    client = make_client(rec, gemini_api_key=GEMINI_KEY)
    assert await client.complete_json("s", "u1") == {"model": "lite"}
    assert [gemini_model(r) for r in rec.requests] == ["gemini-2.5-flash", "gemini-2.5-flash-lite"]
    assert await client.complete_json("s", "u2") == {"model": "lite"}
    assert [gemini_model(r) for r in rec.requests][2:] == ["gemini-2.5-flash-lite"]


@pytest.mark.asyncio
async def test_gemini_model_not_found_400_moves_to_next_model():
    def gemini(request):
        if gemini_model(request) == "gemini-2.5-flash":
            return httpx.Response(400, json={"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                                       "message": "Model gemini-2.5-flash is not supported for generateContent"}})
        return gemini_ok('{"ok": true}')

    client = make_client(Recorder(gemini=gemini), gemini_api_key=GEMINI_KEY)
    assert await client.complete_json("s", "u") == {"ok": True}


@pytest.mark.asyncio
async def test_gemini_thinking_config_rejected_retries_without_it_and_remembers():
    seen: list[bool] = []

    def gemini(request):
        has_thinking = "thinkingConfig" in json.loads(request.content)["generationConfig"]
        seen.append(has_thinking)
        if has_thinking:
            return httpx.Response(400, json={"error": {"code": 400, "message": "thinking is not supported by this model"}})
        return gemini_ok('{"ok": 1}')

    client = make_client(Recorder(gemini=gemini), gemini_api_key=GEMINI_KEY, gemini_models=["gemini-2.5-flash"])
    assert await client.complete_json("s", "u1") == {"ok": 1}
    assert await client.complete_json("s", "u2") == {"ok": 1}
    # budget 0 rejected, then thinkingLevel rejected, then no config; the next call remembers
    assert seen == [True, True, False, False]


@pytest.mark.asyncio
async def test_gemini_bare_invalid_argument_steps_to_thinking_level_and_remembers():
    seen: list[dict | None] = []

    def gemini(request):
        cfg = json.loads(request.content)["generationConfig"].get("thinkingConfig")
        seen.append(cfg)
        if cfg == {"thinkingBudget": 0}:
            return httpx.Response(400, json={"error": {"code": 400, "message": "Request contains an invalid argument.",
                                                       "status": "INVALID_ARGUMENT"}})
        return gemini_ok('{"ok": 1}')

    client = make_client(Recorder(gemini=gemini), gemini_api_key=GEMINI_KEY, gemini_models=["gemini-flash-lite-latest"])
    assert await client.complete_json("s", "u1") == {"ok": 1}
    assert await client.complete_json("s", "u2") == {"ok": 1}
    assert seen == [{"thinkingBudget": 0}, {"thinkingLevel": "low"}, {"thinkingLevel": "low"}]


@pytest.mark.asyncio
async def test_groq_gpt_oss_sends_low_reasoning_effort():
    def groq(request):
        body = json.loads(request.content)
        assert body["reasoning_effort"] == "low"
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 3}'}}]})

    client = make_client(Recorder(groq=groq), groq_api_key=GROQ_KEY, groq_model="openai/gpt-oss-120b",
                         llm_providers=["groq"])
    assert await client.complete_json("s", "u") == {"ok": 3}


@pytest.mark.asyncio
async def test_gemini_schema_rejected_retries_with_json_mode_only():
    seen: list[bool] = []

    def gemini(request):
        body = json.loads(request.content)
        has_schema = "responseSchema" in body["generationConfig"]
        seen.append(has_schema)
        if has_schema:
            return httpx.Response(400, json={"error": {"message": "Invalid JSON payload received. Unknown name "
                                                                  "\"propertyOrdering\" at 'generation_config.response_schema'"}})
        assert "body, cta, rationale" in body["systemInstruction"]["parts"][0]["text"]
        return gemini_ok('{"body": "x", "cta": "open_ended"}')

    client = make_client(Recorder(gemini=gemini), gemini_api_key=GEMINI_KEY, gemini_models=["gemini-2.5-flash"])
    assert await client.complete_json("s", "u", schema=SCHEMA) == {"body": "x", "cta": "open_ended"}
    assert seen == [True, False]


@pytest.mark.asyncio
async def test_gemini_safety_block_falls_back_to_groq():
    rec = Recorder(
        gemini=lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}),
        groq=lambda r: openai_ok('{"from": "groq"}'),
    )
    client = make_client(rec, gemini_api_key=GEMINI_KEY, groq_api_key=GROQ_KEY, gemini_models=["gemini-2.5-flash"])
    assert await client.complete_json("s", "u") == {"from": "groq"}
    assert "blocked" in client.stats["providers"]["gemini"]["last_error"]


@pytest.mark.asyncio
async def test_gemini_skips_thought_parts_and_handles_finish_safety_without_text():
    def gemini(request):
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"text": "thinking about it {not json", "thought": True}, {"text": '{"ok": 2}'}]}}]})

    client = make_client(Recorder(gemini=gemini), gemini_api_key=GEMINI_KEY)
    assert await client.complete_json("s", "u") == {"ok": 2}

    blocked = make_client(Recorder(gemini=lambda r: httpx.Response(200, json={"candidates": [
        {"finishReason": "SAFETY", "content": {"parts": []}}]})), gemini_api_key=GEMINI_KEY)
    assert await blocked.complete_json("s", "u") is None


# --------------------------------------------------------------------------- fallback + cooldown

@pytest.mark.asyncio
async def test_429_then_groq_success_and_gemini_cools_down():
    def gemini(request):
        return httpx.Response(429, headers={"retry-after": "45"},
                              json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota"}})

    def groq(request):
        assert request.url.path == "/openai/v1/chat/completions"
        assert request.headers["authorization"] == f"Bearer {GROQ_KEY}"
        body = json.loads(request.content)
        assert body["model"] == "llama-3.3-70b-versatile"
        assert body["response_format"] == {"type": "json_object"}
        assert body["temperature"] == 0.0 and body["max_tokens"] == 800
        assert body["messages"][0]["role"] == "system" and "JSON" in body["messages"][0]["content"]
        assert body["messages"][1]["role"] == "user" and body["messages"][1]["content"].startswith("u")
        return openai_ok('{"from": "groq"}')

    rec = Recorder(gemini=gemini, groq=groq)
    client = make_client(rec, gemini_api_key=GEMINI_KEY, groq_api_key=GROQ_KEY, gemini_models=["gemini-2.5-flash"])
    assert await client.complete_json("s", "u") == {"from": "groq"}
    assert rec.hosts() == ["generativelanguage.googleapis.com", "api.groq.com"]

    # Gemini is cooling down (Retry-After 45s), so the next call goes straight to Groq.
    assert await client.complete_json("s", "u-second") == {"from": "groq"}
    assert rec.hosts()[2:] == ["api.groq.com"]
    until = client._cooldown_until["gemini:gemini-2.5-flash"] - client._clock()
    assert 40 < until <= 45


@pytest.mark.asyncio
async def test_429_per_model_cooldown_tries_next_gemini_model_first():
    def gemini(request):
        if gemini_model(request) == "gemini-2.5-flash":
            return httpx.Response(429, json={"error": {"code": 429, "message": "quota", "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}]}})
        return gemini_ok('{"model": "lite"}')

    rec = Recorder(gemini=gemini, groq=lambda r: openai_ok('{"from": "groq"}'))
    client = make_client(rec, gemini_api_key=GEMINI_KEY, groq_api_key=GROQ_KEY)
    assert await client.complete_json("s", "u") == {"model": "lite"}
    assert 6 < client._cooldown_until["gemini:gemini-2.5-flash"] - client._clock() <= 7


def test_retry_after_is_capped_and_daily_quota_cools_longer():
    async def run():
        def gemini(request):
            if gemini_model(request) == "gemini-2.5-flash":
                return httpx.Response(429, headers={"retry-after": "9999"}, json={"error": {"message": "slow down"}})
            return httpx.Response(429, json={"error": {"message": "Quota exceeded", "details": [
                {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                 "violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})

        client = make_client(Recorder(gemini=gemini), gemini_api_key=GEMINI_KEY)
        assert await client.complete_json("s", "u") is None
        now = client._clock()
        assert client._cooldown_until["gemini:gemini-2.5-flash"] - now <= client.max_retry_after_s
        assert client._cooldown_until["gemini:gemini-2.5-flash-lite"] - now > client.max_retry_after_s

    asyncio.run(run())


@pytest.mark.asyncio
async def test_all_fail_returns_none_and_never_raises():
    rec = Recorder(
        gemini=lambda r: httpx.Response(500, text="boom"),
        groq=lambda r: httpx.Response(503, json={"error": {"message": "over capacity"}}),
        openai=lambda r: openai_ok("not json at all"),
        anthropic=lambda r: httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error"}}),
        openrouter=lambda r: (_ for _ in ()).throw(httpx.ConnectError("dns failure")),
    )
    client = make_client(rec, gemini_api_key=GEMINI_KEY, groq_api_key=GROQ_KEY, openai_api_key="sk-openai-test-key-123",
                         anthropic_api_key="sk-ant-test-key-1234567", openrouter_api_key="sk-or-test-key-1234567")
    assert await client.complete_json("s", "u") is None
    assert client.stats["failures"] == 1 and client.stats["successes"] == 0
    assert set(client.stats["providers"]) == {"gemini", "groq", "openai", "anthropic", "openrouter"}
    # Everything that errored is cooling down now: the next call makes no requests at all.
    n = len(rec.requests)
    assert await client.complete_json("s", "u2") is None
    assert [r.url.host for r in rec.requests[n:]] == ["api.openai.com"]


@pytest.mark.asyncio
async def test_auth_error_disables_provider_and_error_is_redacted():
    def gemini(request):
        return httpx.Response(400, json={"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                                   "message": f"API key not valid: {GEMINI_KEY}. Bearer {GROQ_KEY}"}})

    rec = Recorder(gemini=gemini, groq=lambda r: openai_ok('{"ok": 1}'))
    client = make_client(rec, gemini_api_key=GEMINI_KEY, groq_api_key=GROQ_KEY)
    assert await client.complete_json("s", "u") == {"ok": 1}
    dump = json.dumps(client.stats_snapshot())
    assert GEMINI_KEY not in dump and GROQ_KEY not in dump
    assert "***" in client.stats["providers"]["gemini"]["last_error"]
    assert client._is_cooling("gemini", "gemini-2.5-flash-lite")   # whole provider, not just the model
    assert rec.hosts() == ["generativelanguage.googleapis.com", "api.groq.com"]


# --------------------------------------------------------------------------- deadlines

@pytest.mark.asyncio
async def test_timeout_respected_when_every_provider_hangs():
    async def slow(request):
        await asyncio.sleep(5)
        return gemini_ok('{"late": true}')

    client = make_client(Recorder(gemini=slow, groq=slow), gemini_api_key=GEMINI_KEY, groq_api_key=GROQ_KEY)
    t0 = time.monotonic()
    assert await client.complete_json("s", "u", timeout=0.6) is None
    assert time.monotonic() - t0 < 0.6 + 0.3


@pytest.mark.asyncio
async def test_slow_primary_leaves_time_for_fallback():
    async def slow(request):
        await asyncio.sleep(5)
        return gemini_ok('{"late": true}')

    rec = Recorder(gemini=slow, groq=lambda r: openai_ok('{"from": "groq"}'))
    client = make_client(rec, gemini_api_key=GEMINI_KEY, groq_api_key=GROQ_KEY)
    t0 = time.monotonic()
    assert await client.complete_json("s", "u", timeout=1.0) == {"from": "groq"}
    assert time.monotonic() - t0 < 1.0 + 0.3
    # After a timeout the other Gemini models are deprioritised behind Groq.
    assert rec.hosts() == ["generativelanguage.googleapis.com", "api.groq.com"]


@pytest.mark.asyncio
async def test_default_timeout_comes_from_settings():
    async def slow(request):
        await asyncio.sleep(5)
        return gemini_ok("{}")

    client = make_client(Recorder(gemini=slow), gemini_api_key=GEMINI_KEY, llm_timeout_s=0.4)
    t0 = time.monotonic()
    assert await client.complete_json("s", "u") is None
    assert time.monotonic() - t0 < 0.4 + 0.3


@pytest.mark.asyncio
async def test_zero_or_negative_timeout_returns_none_without_requests():
    rec = Recorder(gemini=lambda r: gemini_ok('{"a": 1}'))
    client = make_client(rec, gemini_api_key=GEMINI_KEY)
    assert await client.complete_json("s", "u", timeout=0) is None
    assert await client.complete_json("s", "u", timeout=-1) is None
    assert rec.requests == []


@pytest.mark.asyncio
async def test_caller_cancellation_propagates():
    async def slow(request):
        await asyncio.sleep(5)
        return gemini_ok("{}")

    client = make_client(Recorder(gemini=slow), gemini_api_key=GEMINI_KEY)
    task = asyncio.create_task(client.complete_json("s", "u", timeout=2))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await client.aclose()


# --------------------------------------------------------------------------- rate limiting

@pytest.mark.asyncio
async def test_full_rate_window_skips_to_next_provider_without_waiting():
    rec = Recorder(gemini=lambda r: gemini_ok('{"from": "gemini"}'), groq=lambda r: openai_ok('{"from": "groq"}'))
    client = make_client(rec, gemini_api_key=GEMINI_KEY, groq_api_key=GROQ_KEY,
                         gemini_rpm=1, gemini_models=["gemini-2.5-flash"])
    assert await client.complete_json("s", "u1") == {"from": "gemini"}
    t0 = time.monotonic()
    assert await client.complete_json("s", "u2") == {"from": "groq"}
    assert time.monotonic() - t0 < 0.3
    assert client.stats["providers"]["gemini"]["rate_skips"] == 1


@pytest.mark.asyncio
async def test_each_gemini_model_has_its_own_rate_window():
    rec = Recorder(gemini=lambda r: gemini_ok(json.dumps({"model": gemini_model(r)})))
    client = make_client(rec, gemini_api_key=GEMINI_KEY, gemini_rpm=1)
    assert await client.complete_json("s", "u1") == {"model": "gemini-2.5-flash"}
    assert await client.complete_json("s", "u2") == {"model": "gemini-2.5-flash-lite"}
    assert await client.complete_json("s", "u3", timeout=0.5) is None   # both windows full, never waits past deadline


@pytest.mark.asyncio
async def test_rate_window_waits_when_slot_frees_within_deadline():
    clock_offset = [0.0]
    client = make_client(Recorder(gemini=lambda r: gemini_ok('{"ok": 1}')), gemini_api_key=GEMINI_KEY, gemini_rpm=1)
    window = llm_mod._RateWindow(1, lambda: time.monotonic() + clock_offset[0])
    assert await window.acquire(0) is True
    assert await window.acquire(0.1) is False          # next slot is ~60s away
    clock_offset[0] = 59.95                            # pretend a minute almost passed
    t0 = time.monotonic()
    assert await window.acquire(0.5) is True           # ~50ms wait fits inside the budget
    assert time.monotonic() - t0 < 0.4
    await client.aclose()


# --------------------------------------------------------------------------- cache / dedupe

@pytest.mark.asyncio
async def test_cache_hit_avoids_second_request_and_returns_copies():
    rec = Recorder(gemini=lambda r: gemini_ok('{"body": "hello", "params": ["a"]}'))
    client = make_client(rec, gemini_api_key=GEMINI_KEY)
    first = await client.complete_json("s", "u")
    first["params"].append("mutated")
    second = await client.complete_json("s", "u")
    assert second == {"body": "hello", "params": ["a"]}
    assert len(rec.requests) == 1
    assert client.stats["cache_hits"] == 1
    # A different max_tokens or temperature is a different request.
    await client.complete_json("s", "u", max_tokens=200)
    await client.complete_json("s", "u", temperature=0.7)
    assert len(rec.requests) == 3


@pytest.mark.asyncio
async def test_cache_is_bounded_fifo():
    rec = Recorder(gemini=lambda r: gemini_ok('{"ok": 1}'))
    client = make_client(rec, gemini_api_key=GEMINI_KEY)
    client.cache_max_entries = 2
    for u in ("a", "b", "c"):
        await client.complete_json("s", u)
    assert len(client._cache) == 2
    await client.complete_json("s", "a")   # evicted -> fetched again
    assert len(rec.requests) == 4


@pytest.mark.asyncio
async def test_failures_are_not_cached():
    calls = {"n": 0}

    def gemini(request):
        calls["n"] += 1
        return gemini_ok("no json" if calls["n"] == 1 else '{"ok": 1}')

    client = make_client(Recorder(gemini=gemini), gemini_api_key=GEMINI_KEY, gemini_models=["gemini-2.5-flash"])
    assert await client.complete_json("s", "u") is None
    assert await client.complete_json("s", "u") == {"ok": 1}


@pytest.mark.asyncio
async def test_concurrent_identical_calls_share_one_request():
    async def gemini(request):
        await asyncio.sleep(0.2)
        return gemini_ok('{"ok": 1}')

    rec = Recorder(gemini=gemini)
    client = make_client(rec, gemini_api_key=GEMINI_KEY)
    results = await asyncio.gather(*(client.complete_json("s", "same") for _ in range(3)))
    assert results == [{"ok": 1}] * 3
    assert len(rec.requests) == 1
    assert client.stats["deduped"] == 2


@pytest.mark.asyncio
async def test_shared_request_survives_first_caller_cancellation():
    async def gemini(request):
        await asyncio.sleep(0.2)
        return gemini_ok('{"ok": 1}')

    rec = Recorder(gemini=gemini)
    client = make_client(rec, gemini_api_key=GEMINI_KEY)
    first = asyncio.create_task(client.complete_json("s", "same", timeout=2))
    await asyncio.sleep(0.02)
    second = asyncio.create_task(client.complete_json("s", "same", timeout=2))
    await asyncio.sleep(0.02)
    first.cancel()
    assert await second == {"ok": 1}
    assert len(rec.requests) == 1


# --------------------------------------------------------------------------- schema conformance

@pytest.mark.asyncio
async def test_missing_required_keys_falls_back_and_nested_output_is_unwrapped():
    rec = Recorder(gemini=lambda r: gemini_ok('{"text": "no body key"}'),
                   groq=lambda r: openai_ok('{"response": {"body": "b", "cta": "open_ended"}}'))
    client = make_client(rec, gemini_api_key=GEMINI_KEY, groq_api_key=GROQ_KEY, gemini_models=["gemini-2.5-flash"])
    assert await client.complete_json("s", "u", schema=SCHEMA) == {"body": "b", "cta": "open_ended"}
    assert "missing required keys" in client.stats["providers"]["gemini"]["last_error"]


# --------------------------------------------------------------------------- OpenAI-compatible / Anthropic

@pytest.mark.asyncio
async def test_groq_json_validate_failed_uses_failed_generation():
    def groq(request):
        return httpx.Response(400, json={"error": {
            "message": "Failed to generate JSON. Please adjust your prompt.", "type": "invalid_request_error",
            "code": "json_validate_failed", "failed_generation": 'Here you go: {"body": "salvaged",}'}})

    client = make_client(Recorder(groq=groq), groq_api_key=GROQ_KEY)
    assert await client.complete_json("s", "u") == {"body": "salvaged"}


@pytest.mark.asyncio
async def test_openai_compat_adapts_rejected_parameters():
    seen: list[dict] = []

    def openai(request):
        body = json.loads(request.content)
        seen.append(body)
        if "max_tokens" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'max_tokens' is not "
                                                                  "supported with this model. Use 'max_completion_tokens' instead."}})
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported value: 'temperature' does not support 0"}})
        return openai_ok('{"ok": 1}')

    client = make_client(Recorder(openai=openai), openai_api_key="sk-openai-test-key-123",
                         openai_base_url="https://api.openai.com/v1/")
    assert await client.complete_json("s", "u") == {"ok": 1}
    assert seen[-1]["max_completion_tokens"] == 800 and "temperature" not in seen[-1]
    assert await client.complete_json("s", "u2") == {"ok": 1}
    assert len(seen) == 4   # adaptations remembered: second call succeeds first try


@pytest.mark.asyncio
async def test_openrouter_request_shape():
    def openrouter(request):
        assert request.url.path == "/api/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer sk-or-test-key-1234567"
        assert request.headers["x-title"]
        return openai_ok('```json\n{"via": "openrouter"}\n```')

    client = make_client(Recorder(openrouter=openrouter), openrouter_api_key="sk-or-test-key-1234567")
    assert await client.complete_json("s", "u") == {"via": "openrouter"}


@pytest.mark.asyncio
async def test_anthropic_request_shape():
    def anthropic(request):
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "sk-ant-test-key-1234567"
        assert request.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(request.content)
        assert body["model"] == "claude-test" and body["max_tokens"] == 500 and body["temperature"] == 0.0
        assert body["system"].startswith("SYS") and "JSON" in body["system"]
        assert body["messages"] == [{"role": "user", "content": "USER"}]
        return httpx.Response(200, json={"content": [{"type": "text", "text": '{"via": "claude"}'}],
                                         "stop_reason": "end_turn"})

    client = make_client(Recorder(anthropic=anthropic), anthropic_api_key="sk-ant-test-key-1234567")
    assert await client.complete_json("SYS", "USER", max_tokens=500) == {"via": "claude"}


@pytest.mark.asyncio
async def test_extra_models_in_non_gemini_provider_are_fallbacks():
    def groq(request):
        model = json.loads(request.content)["model"]
        if model == "llama-3.3-70b-versatile":
            return httpx.Response(400, json={"error": {"code": "model_decommissioned",
                                                       "message": "The model `llama-3.3-70b-versatile` has been decommissioned"}})
        return openai_ok(json.dumps({"model": model}))

    client = make_client(Recorder(groq=groq), groq_api_key=GROQ_KEY,
                         groq_model="llama-3.3-70b-versatile, openai/gpt-oss-120b")
    assert await client.complete_json("s", "u") == {"model": "openai/gpt-oss-120b"}


# --------------------------------------------------------------------------- singleton / loops

def test_get_and_set_llm_singleton():
    original = llm_mod._llm
    try:
        set_llm(None)
        a = get_llm()
        assert get_llm() is a
        fake = make_client(Recorder())
        set_llm(fake)
        assert get_llm() is fake
    finally:
        set_llm(original)


def test_works_across_separate_event_loops():
    rec = Recorder(gemini=lambda r: gemini_ok(json.dumps({"u": json.loads(r.content)["contents"][0]["parts"][0]["text"]})))
    client = make_client(rec, gemini_api_key=GEMINI_KEY)
    assert asyncio.run(client.complete_json("s", "one")) == {"u": "one"}
    assert asyncio.run(client.complete_json("s", "two")) == {"u": "two"}
    assert len(rec.requests) == 2


@pytest.mark.asyncio
async def test_internal_errors_never_escape():
    client = make_client(Recorder(gemini=lambda r: gemini_ok("{}")), gemini_api_key=GEMINI_KEY)

    def broken():
        raise RuntimeError("boom")

    client._providers = broken   # type: ignore[method-assign]
    assert await client.complete_json("s", "u") is None
