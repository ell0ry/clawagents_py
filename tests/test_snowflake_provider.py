"""Snowflake Cortex provider: construction, routing, and wire shape.

Cortex is OpenAI-compatible, so the provider is a thin ``OpenAIProvider``
subclass — these tests pin the Cortex-specific parts: account → base URL
derivation, PAT sourcing (incl. the ``SNOWFLAKE_PAT`` env alias), forced
Chat Completions wire API, and reasoning-effort suppression for claude-*.
"""

from __future__ import annotations

import asyncio

import pytest

from clawagents.config.config import EngineConfig
from clawagents.providers.llm import LLMMessage, create_provider
from clawagents.providers.snowflake import (
    SnowflakeCortexProvider,
    snowflake_cortex_base_url,
)


@pytest.fixture(autouse=True)
def _clean_snowflake_env(monkeypatch):
    for var in ("SNOWFLAKE_PAT", "SNOWFLAKE_API_KEY", "SNOWFLAKE_ACCOUNT", "SNOWFLAKE_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


def _cfg(**overrides) -> EngineConfig:
    defaults = dict(snowflake_api_key="pat-secret", snowflake_account="myorg-acct")
    defaults.update(overrides)
    return EngineConfig(**defaults)


# ─── Construction & routing ───────────────────────────────────────────


def test_create_provider_routes_snowflake_prefix():
    provider = create_provider("snowflake/claude-sonnet-4-5", _cfg())
    assert isinstance(provider, SnowflakeCortexProvider)
    assert provider.name == "snowflake"
    assert provider.model == "claude-sonnet-4-5"
    assert provider._base_url == "https://myorg-acct.snowflakecomputing.com/api/v2/cortex/v1"
    assert provider._wire_api == "chat_completions"
    assert provider._should_use_responses(True) is False
    assert provider._should_use_responses(False) is False


def test_create_provider_routes_via_hint_over_claude_shape():
    provider = create_provider("claude-opus-4-6", _cfg(), provider_hint="snowflake")
    assert isinstance(provider, SnowflakeCortexProvider)
    assert provider.model == "claude-opus-4-6"


def test_base_url_override_wins_and_slash_stripped():
    provider = create_provider(
        "snowflake/claude-sonnet-4-5",
        _cfg(snowflake_base_url="https://custom.example.com/api/v2/cortex/v1/"),
    )
    assert provider._base_url == "https://custom.example.com/api/v2/cortex/v1"


def test_base_url_full_endpoint_paste_normalized():
    """Pasting the full /chat/completions endpoint URL must still work."""
    provider = create_provider(
        "snowflake/claude-sonnet-4-5",
        _cfg(snowflake_base_url="https://custom.example.com/api/v2/cortex/v1/chat/completions"),
    )
    assert provider._base_url == "https://custom.example.com/api/v2/cortex/v1"


def test_account_url_derivation():
    assert (
        snowflake_cortex_base_url(" myorg-acct ")
        == "https://myorg-acct.snowflakecomputing.com/api/v2/cortex/v1"
    )


def test_missing_key_raises_naming_env_var():
    with pytest.raises(ValueError, match="SNOWFLAKE_PAT"):
        SnowflakeCortexProvider(EngineConfig(snowflake_account="myorg-acct"))


def test_missing_account_and_url_raises_naming_env_vars():
    with pytest.raises(ValueError, match="SNOWFLAKE_ACCOUNT"):
        SnowflakeCortexProvider(EngineConfig(snowflake_api_key="pat-secret"))


def test_snowflake_pat_env_alias(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_PAT", "pat-from-env")
    assert EngineConfig().snowflake_api_key == "pat-from-env"


def test_snowflake_api_key_env(monkeypatch):
    monkeypatch.setenv("SNOWFLAKE_API_KEY", "key-from-env")
    assert EngineConfig().snowflake_api_key == "key-from-env"


# ─── Cortex quirks ────────────────────────────────────────────────────


def test_resolve_model_routes_key_and_base_url_to_snowflake_fields():
    """``create_claw_agent`` builds a provider of its own, bypassing any
    caller-side EngineConfig. Its api_key/base_url kwargs must reach the
    ``snowflake_*`` fields — the key silently landed nowhere (no dispatch
    branch) and base_url only ever hit ``openai_base_url``, so a Cortex model
    died with "Cortex endpoint unknown" the moment an agent was constructed.
    """
    from clawagents.agent import _resolve_model

    provider = _resolve_model(
        "snowflake/claude-sonnet-4-5",
        streaming=True,
        api_key="pat-from-caller",
        base_url="https://acct.snowflakecomputing.com/api/v2/cortex/v1",
    )
    assert isinstance(provider, SnowflakeCortexProvider)
    assert provider.model == "claude-sonnet-4-5"
    assert provider._base_url == "https://acct.snowflakecomputing.com/api/v2/cortex/v1"
    assert provider.client.api_key == "pat-from-caller"


def test_reasoning_effort_suppressed_for_claude_models():
    provider = create_provider(
        "snowflake/claude-sonnet-4-5", _cfg(reasoning_effort="low")
    )
    assert provider._reasoning_effort is None


def test_reasoning_effort_preserved_for_openai_models():
    provider = create_provider(
        "snowflake/openai-gpt-4.1", _cfg(reasoning_effort="low")
    )
    assert provider._reasoning_effort == "low"


def test_azure_branch_never_taken():
    provider = create_provider(
        "snowflake/claude-sonnet-4-5",
        _cfg(openai_api_type="azure", openai_api_version="2024-06-01"),
    )
    assert provider._api_type == ""


# ─── Wire shape against a fake Cortex endpoint ────────────────────────


def test_chat_hits_cortex_path_with_pat_and_image_passthrough():
    from clawagents.testing.mock_provider import (
        MockLLMService,
        Scenario,
        _chat_completion,
    )

    scenario = Scenario(
        name="cortex-any",
        response=_chat_completion(content="ok from cortex", model="claude-sonnet-4-5"),
        request_predicate=lambda body: True,
    )
    with MockLLMService(scenarios=[scenario]) as mock:
        provider = create_provider(
            "snowflake/claude-sonnet-4-5",
            _cfg(snowflake_base_url=mock.url + "/api/v2/cortex/v1"),
        )
        image_block = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA"},
        }
        response = asyncio.run(
            provider.chat(
                [
                    LLMMessage(role="user", content=[
                        {"type": "text", "text": "what do you see?"},
                        image_block,
                    ]),
                ]
            )
        )
        assert response.content == "ok from cortex"

        request = mock.request_log[0]
        assert request["path"] == "/api/v2/cortex/v1/chat/completions"
        headers = {k.lower(): v for k, v in request["headers"].items()}
        assert headers["authorization"] == "Bearer pat-secret"
        body = request["body"]
        assert body["model"] == "claude-sonnet-4-5"
        assert body["max_completion_tokens"] == 8192
        sent_content = body["messages"][0]["content"]
        assert image_block in sent_content


def test_no_arg_tool_call_replays_as_empty_object():
    """args=None must replay as "{}" — Cortex 400s on arguments "null"
    ("required field 'input' is zero value")."""
    from clawagents.testing.mock_provider import (
        MockLLMService,
        Scenario,
        _chat_completion,
    )

    scenario = Scenario(
        name="cortex-any",
        response=_chat_completion(content="ok"),
        request_predicate=lambda body: True,
    )
    with MockLLMService(scenarios=[scenario]) as mock:
        provider = create_provider(
            "snowflake/claude-sonnet-4-5",
            _cfg(snowflake_base_url=mock.url + "/api/v2/cortex/v1"),
        )
        asyncio.run(
            provider.chat(
                [
                    LLMMessage(role="user", content="state?"),
                    LLMMessage(
                        role="assistant",
                        content="checking",
                        tool_calls_meta=[{"id": "tc1", "name": "get_robot_state", "args": None}],
                    ),
                    LLMMessage(role="tool", content="idle", tool_call_id="tc1"),
                ]
            )
        )
        sent = mock.request_log[0]["body"]["messages"]
        replayed = next(m for m in sent if m.get("tool_calls"))
        assert replayed["tool_calls"][0]["function"]["arguments"] == "{}"


def test_parallel_tool_calls_split_on_replay():
    """Cortex pairs each toolUse with the immediately-following toolResult,
    so A[tc1,tc2] T1 T2 must be replayed as A[tc1] T1 A[tc2] T2."""
    from clawagents.testing.mock_provider import (
        MockLLMService,
        Scenario,
        _chat_completion,
    )

    scenario = Scenario(
        name="cortex-any",
        response=_chat_completion(content="ok"),
        request_predicate=lambda body: True,
    )
    with MockLLMService(scenarios=[scenario]) as mock:
        provider = create_provider(
            "snowflake/claude-sonnet-4-5",
            _cfg(snowflake_base_url=mock.url + "/api/v2/cortex/v1"),
        )
        asyncio.run(
            provider.chat(
                [
                    LLMMessage(role="user", content="go"),
                    LLMMessage(
                        role="assistant",
                        content="checking both",
                        tool_calls_meta=[
                            {"id": "tc1", "name": "get_robot_state", "args": {}},
                            {"id": "tc2", "name": "recall_experiences", "args": {"query": "x"}},
                        ],
                    ),
                    LLMMessage(role="tool", content="idle", tool_call_id="tc1"),
                    LLMMessage(role="tool", content="none found", tool_call_id="tc2"),
                    LLMMessage(role="user", content="and?"),
                ]
            )
        )
        sent = mock.request_log[0]["body"]["messages"]
        shape = [
            (m["role"], [tc["id"] for tc in m.get("tool_calls") or []] or m.get("tool_call_id"))
            for m in sent
        ]
        assert shape == [
            ("user", None),
            ("assistant", ["tc1"]),
            ("tool", "tc1"),
            ("assistant", ["tc2"]),
            ("tool", "tc2"),
            ("user", None),
        ]
        # Original text content stays on the first split message only.
        assert sent[1]["content"] == "checking both"
        assert sent[3]["content"] is None


# ── streaming: parallel tool calls that share an index ───────────────────────


class _FnDelta:
    def __init__(self, name: str = "", arguments: str = ""):
        self.name = name
        self.arguments = arguments


class _TcDelta:
    def __init__(self, index, tc_id="", name="", arguments=""):
        self.index = index
        self.id = tc_id
        self.function = _FnDelta(name, arguments)


def _accumulate(deltas):
    """Drive the SHIPPED accumulator, not a copy of it."""
    from clawagents.providers.llm import accumulate_tool_call_delta

    tools: dict[int, dict] = {}
    slot_of_id: dict[str, int] = {}
    slot_of_index: dict[int, int] = {}
    for tc in deltas:
        accumulate_tool_call_delta(tools, slot_of_id, slot_of_index, tc)
    return [tools[k] for k in sorted(tools)]


def test_cortex_parallel_tool_calls_share_index_zero():
    """Captured verbatim from a live Cortex stream (2026-08-11).

    Cortex numbers every parallel call index=0 and separates them only by id.
    Keying on index concatenated both names into "move_windowfocus_window" and
    glued two JSON objects together — the agent then looped on a tool that does
    not exist until it hit its round cap.
    """
    deltas = [
        _TcDelta(0, "toolu_bdrk_014Z", "move_window"),
        _TcDelta(0, "", "", '{"window_'),
        _TcDelta(0, "", "", 'id": "abc123", "'),
        _TcDelta(0, "", "", 'device": "kitchen"}'),
        _TcDelta(0, "toolu_bdrk_01Q1", "focus_window"),
        _TcDelta(0, "", "", '{"wi'),
        _TcDelta(0, "", "", 'ndow_id": "abc123"}'),
    ]
    calls = _accumulate(deltas)
    assert [c["name"] for c in calls] == ["move_window", "focus_window"]
    assert calls[0]["arguments"] == '{"window_id": "abc123", "device": "kitchen"}'
    assert calls[1]["arguments"] == '{"window_id": "abc123"}'


def test_second_call_opening_without_an_id_still_splits():
    """The residual path: id-keying alone cannot see this one coming.

    Every delta is index=0 and the second call arrives with no id, so both the
    index map and the id map point at slot 0. Without the "a name arriving at
    a finished slot starts a new call" rule the two names glue, and the agent
    dispatches a tool that does not exist.
    """
    deltas = [
        _TcDelta(0, "toolu_bdrk_014Z", "get_weather"),
        _TcDelta(0, "", "", '{"city": "Houston"}'),
        _TcDelta(0, "", "get_forecast"),  # no id, no fresh index
        _TcDelta(0, "", "", '{"days": 3}'),
    ]
    calls = _accumulate(deltas)
    assert [c["name"] for c in calls] == ["get_weather", "get_forecast"]
    assert calls[0]["arguments"] == '{"city": "Houston"}'
    assert calls[1]["arguments"] == '{"days": 3}'


def test_second_call_repeating_the_first_id_still_splits():
    """A proxy that stamps the message id on every call, not a per-call id."""
    deltas = [
        _TcDelta(0, "msg_1", "get_weather"),
        _TcDelta(0, "msg_1", "", '{"city": "Houston"}'),
        _TcDelta(0, "msg_1", "get_forecast"),
        _TcDelta(0, "msg_1", "", '{"days": 3}'),
    ]
    calls = _accumulate(deltas)
    assert [c["name"] for c in calls] == ["get_weather", "get_forecast"]


def test_index_colliding_with_an_id_allocated_slot_does_not_merge():
    """Slot numbers come from len(tools) and share a namespace with indexes."""
    deltas = [
        _TcDelta(5, "call_a", "get_weather"),  # index 5 → slot 0
        _TcDelta(5, "", "", '{"city": "Houston"}'),
        _TcDelta(0, "", "get_forecast"),  # raw index 0 == slot 0
        _TcDelta(0, "", "", '{"days": 3}'),
    ]
    calls = _accumulate(deltas)
    assert [c["name"] for c in calls] == ["get_weather", "get_forecast"]
    assert calls[0]["arguments"] == '{"city": "Houston"}'


def test_a_name_streamed_in_fragments_is_not_split():
    """The counterweight: pieces of ONE name must still concatenate."""
    deltas = [
        _TcDelta(0, "call_a", "get_"),
        _TcDelta(0, "", "weather"),
        _TcDelta(0, "", "", "{}"),
    ]
    calls = _accumulate(deltas)
    assert [c["name"] for c in calls] == ["get_weather"]


def test_openai_style_distinct_indexes_still_route_by_index():
    """The OpenAI contract must keep working: interleaved argument deltas."""
    deltas = [
        _TcDelta(0, "call_a", "get_weather"),
        _TcDelta(1, "call_b", "get_forecast"),
        _TcDelta(0, "", "", '{"city":'),
        _TcDelta(1, "", "", '{"days":'),
        _TcDelta(0, "", "", ' "Houston"}'),
        _TcDelta(1, "", "", " 3}"),
    ]
    calls = _accumulate(deltas)
    assert [c["name"] for c in calls] == ["get_weather", "get_forecast"]
    assert calls[0]["arguments"] == '{"city": "Houston"}'
    assert calls[1]["arguments"] == '{"days": 3}'


# ── prompt caching ───────────────────────────────────────────────────────────


def _system_of(formatted):
    return next(m for m in formatted if m["role"] == "system")["content"]


def test_cache_control_splits_the_system_message_at_the_boundary():
    from clawagents.providers.llm import LLMMessage, _openai_chat_messages

    msgs = [
        LLMMessage(role="system", content="STATIC PREFIX\n__CACHE_BOUNDARY__\nlessons"),
        LLMMessage(role="user", content="hi"),
    ]
    blocks = _system_of(_openai_chat_messages(msgs, cache_control=True))
    assert [b["text"] for b in blocks] == ["STATIC PREFIX", "lessons"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in blocks[1], "only the static half is a breakpoint"


def test_cache_control_off_still_strips_the_marker():
    """The default must not change for any other OpenAI-compatible backend."""
    from clawagents.providers.llm import LLMMessage, _openai_chat_messages

    msgs = [LLMMessage(role="system", content="STATIC\n__CACHE_BOUNDARY__\nlessons")]
    content = _system_of(_openai_chat_messages(msgs))
    assert isinstance(content, str)
    assert "__CACHE_BOUNDARY__" not in content


def test_cache_control_with_no_boundary_is_a_single_block():
    from clawagents.providers.llm import LLMMessage, _openai_chat_messages

    msgs = [LLMMessage(role="system", content="no marker here")]
    content = _system_of(_openai_chat_messages(msgs, cache_control=True))
    # No boundary → nothing to split, and no breakpoint invented.
    assert content == "no marker here"


def test_cortex_enables_cache_control_only_for_claude(monkeypatch):
    from clawagents.config.config import EngineConfig
    from clawagents.providers.snowflake import SnowflakeCortexProvider

    def _mk(model):
        cfg = EngineConfig(
            openai_model=model,
            snowflake_api_key="pat",
            snowflake_base_url="https://acct.snowflakecomputing.com/api/v2/cortex/v1",
        )
        return SnowflakeCortexProvider(cfg)

    assert _mk("claude-opus-4-7")._emit_cache_control is True
    assert getattr(_mk("llama3-70b"), "_emit_cache_control", False) is False
