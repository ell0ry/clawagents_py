"""history=: a caller-owned transcript replays between system prompt and task."""

import asyncio

from clawagents.graph.agent_loop import run_agent_graph
from clawagents.providers.llm import LLMMessage, LLMProvider, LLMResponse


class RecordingLLM(LLMProvider):
    """Captures the message list it was called with and answers immediately."""

    name = "fake"

    def __init__(self):
        self.seen: list[list[LLMMessage]] = []

    async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None):
        self.seen.append(list(messages))
        return LLMResponse(content="final answer", model="fake", tokens_used=0)


def _run(history):
    llm = RecordingLLM()
    state = asyncio.run(
        run_agent_graph(
            task="current question",
            llm=llm,
            streaming=False,
            use_native_tools=False,
            max_iterations=3,
            history=history,
        )
    )
    return llm, state


def test_history_inserted_before_task():
    history = [
        LLMMessage(role="user", content="older question"),
        LLMMessage(role="assistant", content="older answer"),
    ]
    llm, state = _run(history)
    assert state.status == "done"
    msgs = llm.seen[0]
    roles_contents = [(m.role, m.content) for m in msgs]
    idx_hist_u = next(i for i, (r, c) in enumerate(roles_contents) if c == "older question")
    idx_hist_a = next(i for i, (r, c) in enumerate(roles_contents) if c == "older answer")
    idx_task = next(
        i
        for i, (r, c) in enumerate(roles_contents)
        if r == "user" and isinstance(c, str) and "current question" in c
    )
    assert idx_hist_u < idx_hist_a < idx_task
    assert msgs[0].role == "system"


def test_history_orphan_tool_sanitized():
    """A leading orphan tool result (summary+tail cut) must not reach the LLM raw."""
    history = [
        LLMMessage(role="tool", content="orphan result", tool_call_id="tc-1"),
        LLMMessage(role="user", content="older question"),
        LLMMessage(role="assistant", content="older answer"),
    ]
    llm, state = _run(history)
    assert state.status == "done"
    msgs = llm.seen[0]
    for m in msgs:
        if m.role == "tool":
            raise AssertionError("orphan tool message leaked into the transcript")


def test_no_history_unchanged():
    llm, state = _run(None)
    assert state.status == "done"
    assert state.result == "final answer"
