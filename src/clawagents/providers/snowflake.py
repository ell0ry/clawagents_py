"""Snowflake Cortex provider.

Cortex serves frontier models (Claude, GPT, Llama, Mistral, …) behind an
OpenAI-compatible Chat Completions endpoint at
``https://<account>.snowflakecomputing.com/api/v2/cortex/v1``, authenticated
with a Snowflake programmatic access token (PAT) as the bearer key. No wire
translation is needed — this subclass only resolves the account-specific base
URL, sources the PAT, and pins the quirks that differ from stock OpenAI.
"""

from __future__ import annotations

from typing import Any

from clawagents.config.config import EngineConfig
from clawagents.providers.llm import OpenAIProvider

_CORTEX_PATH = "/api/v2/cortex/v1"


def _split_parallel_tool_calls(formatted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split multi-call assistant messages into single-call + result pairs.

    Cortex converts each OpenAI ``tool`` message into its own Claude user
    turn without merging, so an assistant turn with N>1 tool_calls is
    followed by a 1-result user turn and Bedrock rejects it ("Each
    'toolUse' block must be accompanied with a matching 'toolResult'
    block"). ``A[tc1,tc2] T1 T2`` → ``A[tc1] T1 A[tc2] T2`` is
    semantically identical and pairs cleanly.
    """
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(formatted):
        msg = formatted[i]
        tool_calls = msg.get("tool_calls") or []
        if msg.get("role") != "assistant" or len(tool_calls) <= 1:
            out.append(msg)
            i += 1
            continue
        results: dict[str, dict[str, Any]] = {}
        j = i + 1
        while j < len(formatted) and formatted[j].get("role") == "tool":
            results[str(formatted[j].get("tool_call_id"))] = formatted[j]
            j += 1
        if set(results) != {str(tc.get("id")) for tc in tool_calls}:
            # Orphans/mismatch — leave untouched rather than guess.
            out.append(msg)
            i += 1
            continue
        for n, tc in enumerate(tool_calls):
            out.append({
                "role": "assistant",
                "content": msg.get("content") if n == 0 else None,
                "tool_calls": [tc],
            })
            out.append(results[str(tc.get("id"))])
        i = j
    return out


def snowflake_cortex_base_url(account: str) -> str:
    """OpenAI-compatible Cortex base URL for a Snowflake account identifier."""
    acct = (account or "").strip().strip("/")
    return f"https://{acct}.snowflakecomputing.com{_CORTEX_PATH}"


class SnowflakeCortexProvider(OpenAIProvider):
    name = "snowflake"

    def __init__(self, config: EngineConfig):
        config = config.model_copy()
        api_key = (config.snowflake_api_key or "").strip()
        if not api_key:
            raise ValueError(
                "Snowflake Cortex requires a programmatic access token: "
                "set SNOWFLAKE_PAT (or SNOWFLAKE_API_KEY) in the environment/.env."
            )
        base_url = (config.snowflake_base_url or "").strip().rstrip("/")
        # Accept the full endpoint URL from Snowflake docs — the SDK appends
        # /chat/completions itself.
        if base_url.endswith("/chat/completions"):
            base_url = base_url[: -len("/chat/completions")]
        if not base_url:
            account = (config.snowflake_account or "").strip()
            if not account:
                raise ValueError(
                    "Snowflake Cortex endpoint unknown: set SNOWFLAKE_ACCOUNT "
                    "(account identifier, e.g. myorg-myaccount) or a full "
                    "SNOWFLAKE_BASE_URL."
                )
            base_url = snowflake_cortex_base_url(account)
        config.openai_base_url = base_url
        config.openai_api_key = api_key
        # Cortex speaks Chat Completions only — never the Responses API,
        # never the Azure client branch.
        config.openai_wire_api = "chat_completions"
        config.openai_api_type = ""
        config.openai_api_version = ""
        super().__init__(config)
        # Cortex rejects OpenAI-style ``reasoning_effort`` on claude-* models.
        # It DOES accept the Anthropic content-block shape for the system
        # message, so claude-* is also where a ``cache_control`` breakpoint is
        # honoured — cache reads bill at a fraction of fresh input, and Ada's
        # static prefix (base prompt + tool description) is comfortably over
        # the 1024-token minimum a read needs to qualify. Only ephemeral is
        # supported, with a 5-minute TTL.
        if (self.model or "").lower().startswith("claude"):
            self._reasoning_effort = None
            self._emit_cache_control = True

    async def _chat_dispatch(self, formatted, *args, **kwargs):
        return await super()._chat_dispatch(
            _split_parallel_tool_calls(formatted), *args, **kwargs
        )
