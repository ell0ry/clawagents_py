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
