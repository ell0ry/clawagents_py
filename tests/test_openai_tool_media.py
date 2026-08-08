"""Multimodal tool-output recovery on the OpenAI Chat Completions path.

The agent loop JSON-encodes list-typed tool results into the ``role="tool"``
string in native-tools mode. ``_openai_chat_messages`` must recover such
payloads: text stays in the tool message, media rides one trailing user turn
per run of tool messages (parallel-FC-safe). Hermetic — no SDK, no network.
"""

from __future__ import annotations

import json

from clawagents.providers.llm import (
    LLMMessage,
    _openai_chat_messages,
    _openai_part_from_block,
    _sanitize_openai_tool_pairs,
)

DATA_URL = "data:image/jpeg;base64,aGVsbG8="


def _tool_msg(tc_id: str, content) -> LLMMessage:
    return LLMMessage(role="tool", content=content, tool_call_id=tc_id)


def _assistant_call(*tc_ids: str) -> LLMMessage:
    return LLMMessage(
        role="assistant",
        content="",
        tool_calls_meta=[{"id": t, "name": "browser_screenshot", "args": {}} for t in tc_ids],
    )


def _media_payload(text: str = "screenshot of https://example.com") -> str:
    return json.dumps(
        [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": DATA_URL}},
        ]
    )


def test_media_tool_result_splits_into_tool_text_and_user_image():
    msgs = [
        _assistant_call("tc1"),
        _tool_msg("tc1", _media_payload()),
    ]
    out = _openai_chat_messages(msgs)
    assert [m["role"] for m in out] == ["assistant", "tool", "user"]
    tool = out[1]
    assert tool["tool_call_id"] == "tc1"
    assert tool["content"] == "screenshot of https://example.com"
    user = out[2]
    parts = user["content"]
    assert parts[0]["type"] == "text"
    assert "tc1" in parts[0]["text"]
    assert parts[1] == {"type": "image_url", "image_url": {"url": DATA_URL}}


def test_parallel_calls_flush_one_user_message_after_the_run():
    msgs = [
        _assistant_call("tc1", "tc2"),
        _tool_msg("tc1", _media_payload("shot one")),
        _tool_msg("tc2", _media_payload("shot two")),
        LLMMessage(role="assistant", content="looked at both"),
    ]
    out = _openai_chat_messages(msgs)
    assert [m["role"] for m in out] == ["assistant", "tool", "tool", "user", "assistant"]
    media_turn = out[3]
    images = [p for p in media_turn["content"] if p["type"] == "image_url"]
    assert len(images) == 2
    labels = [p["text"] for p in media_turn["content"] if p["type"] == "text"]
    assert any("tc1" in t for t in labels) and any("tc2" in t for t in labels)


def test_trailing_tool_run_flushes_at_the_end():
    msgs = [_assistant_call("tc1"), _tool_msg("tc1", _media_payload())]
    out = _openai_chat_messages(msgs)
    assert out[-1]["role"] == "user"


def test_plain_string_tool_content_is_untouched():
    body = "ordinary text result — no blocks here"
    out = _openai_chat_messages([_assistant_call("tc1"), _tool_msg("tc1", body)])
    assert [m["role"] for m in out] == ["assistant", "tool"]
    assert out[1]["content"] == body


def test_json_that_is_not_a_block_list_is_untouched():
    body = json.dumps({"type": "text", "text": "a dict, not a list"})
    out = _openai_chat_messages([_assistant_call("tc1"), _tool_msg("tc1", body)])
    assert out[1]["content"] == body
    assert len(out) == 2


def test_text_only_block_list_is_untouched():
    body = json.dumps([{"type": "text", "text": "just words"}])
    out = _openai_chat_messages([_assistant_call("tc1"), _tool_msg("tc1", body)])
    assert out[1]["content"] == body
    assert len(out) == 2


def test_unsupported_media_becomes_placeholder():
    body = json.dumps(
        [
            {"type": "text", "text": "a pdf"},
            {"type": "file", "file": {"filename": "doc.pdf"}},
        ]
    )
    out = _openai_chat_messages([_assistant_call("tc1"), _tool_msg("tc1", body)])
    assert out[-1]["role"] == "user"
    texts = [p["text"] for p in out[-1]["content"] if p["type"] == "text"]
    assert any("omitted" in t for t in texts)
    assert not any(p["type"] == "image_url" for p in out[-1]["content"])


def test_anthropic_style_image_block_converts_to_data_url():
    part = _openai_part_from_block(
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "aGk="},
        }
    )
    assert part == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,aGk="},
    }


def test_sanitizer_keeps_the_interleaved_media_turn():
    msgs = [
        _assistant_call("tc1"),
        _tool_msg("tc1", _media_payload()),
        LLMMessage(role="user", content="what did you see?"),
    ]
    out = _sanitize_openai_tool_pairs(_openai_chat_messages(msgs))
    assert [m["role"] for m in out] == ["assistant", "tool", "user", "user"]
    assert isinstance(out[2]["content"], list)


def test_media_flushes_before_next_assistant_turn():
    msgs = [
        _assistant_call("tc1"),
        _tool_msg("tc1", _media_payload()),
        _assistant_call("tc2"),
        _tool_msg("tc2", "plain follow-up"),
    ]
    out = _openai_chat_messages(msgs)
    assert [m["role"] for m in out] == ["assistant", "tool", "user", "assistant", "tool"]
