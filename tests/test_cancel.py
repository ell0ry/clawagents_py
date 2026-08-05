"""External cancel_event: a caller can stop an in-flight run between rounds."""

import asyncio

from clawagents.agent import create_claw_agent
from clawagents.graph.agent_loop import run_agent_graph
from clawagents.providers.llm import LLMProvider, LLMResponse
from clawagents.tools.registry import ToolRegistry, ToolResult

_TOOL_CALL = """Working on it.
```json
{"tool": "loop_step", "args": {}}
```"""


class LoopStepTool:
    name = "loop_step"
    description = "A tool the fake LLM calls forever."
    parameters = {}

    async def execute(self, args):
        return ToolResult(success=True, output="step done")


class EndlessToolLLM(LLMProvider):
    """Returns a tool call on every chat; optionally sets cancel at call N."""

    name = "fake"

    def __init__(self, cancel_event=None, cancel_at=0):
        self.calls = 0
        self._cancel_event = cancel_event
        self._cancel_at = cancel_at

    async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None):
        self.calls += 1
        if self._cancel_event is not None and self.calls >= self._cancel_at:
            self._cancel_event.set()
        return LLMResponse(content=_TOOL_CALL, model="fake", tokens_used=0)


def _registry():
    registry = ToolRegistry()
    registry.register(LoopStepTool())
    return registry


def test_external_cancel_stops_loop():
    """Setting the event mid-run ends the loop at the next round top."""
    cancel = asyncio.Event()
    llm = EndlessToolLLM(cancel_event=cancel, cancel_at=3)

    state = asyncio.run(
        run_agent_graph(
            task="loop forever",
            llm=llm,
            tools=_registry(),
            streaming=False,
            use_native_tools=False,
            max_iterations=50,
            cancel_event=cancel,
        )
    )
    assert state.status == "done"
    assert state.result == "[cancelled]"
    assert llm.calls == 3  # cancelled well before max_iterations


def test_invoke_threads_cancel_event(tmp_path, monkeypatch):
    """A pre-set event via ClawAgent.invoke cancels before the first LLM call."""
    monkeypatch.chdir(tmp_path)
    cancel = asyncio.Event()
    cancel.set()
    llm = EndlessToolLLM()
    agent = create_claw_agent(model=llm, tools=[LoopStepTool()])

    state = asyncio.run(agent.invoke("anything", cancel_event=cancel))
    assert state.status == "done"
    assert state.result == "[cancelled]"
    assert llm.calls == 0


def test_default_internal_event_unchanged():
    """Without a caller event the loop runs to its natural end (max rounds aside)."""

    class OneShotLLM(LLMProvider):
        name = "fake"

        async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None):
            return LLMResponse(content="final answer", model="fake", tokens_used=0)

    state = asyncio.run(
        run_agent_graph(
            task="say hi",
            llm=OneShotLLM(),
            tools=_registry(),
            streaming=False,
            use_native_tools=False,
            max_iterations=5,
        )
    )
    assert state.status == "done"
    assert state.result == "final answer"
