"""Waterfall catalog drives the registry's active-tools set (v6 rebase design)."""

import asyncio

from clawagents.tools.registry import ToolRegistry, ToolResult
from clawagents.tools.waterfall_catalog import (
    ToolCatalog,
    ToolCategory,
    create_resolve_tools_tool,
)


def _tool(name):
    class _T:
        description = f"tool {name}"
        parameters = {}

        async def execute(self, args):
            return ToolResult(success=True, output=f"{name} ran")

    _T.name = name
    return _T()


def _setup(base_allowed=None):
    registry = ToolRegistry()
    for n in ("chat_basic", "weather_now", "weather_forecast", "lights_on"):
        registry.register(_tool(n))
    # Simulates an engine default the consumer never allows.
    registry.register(_tool("execute"))
    categories = [
        ToolCategory(
            name="weather",
            description="weather tools",
            tool_names=["weather_now", "weather_forecast"],
            keywords=["weather", "forecast"],
            instruction="Use metric units.",
        ),
        ToolCategory(
            name="smart_home",
            description="home tools",
            tool_names=["lights_on"],
            keywords=["light"],
        ),
    ]
    catalog = ToolCatalog(categories, ["chat_basic"], registry, base_allowed=base_allowed)
    registry.register(create_resolve_tools_tool(catalog))
    return registry, catalog


def test_deferred_tools_inactive_until_resolved():
    allowed = {"chat_basic", "weather_now", "weather_forecast", "lights_on"}
    registry, catalog = _setup(base_allowed=allowed)

    assert registry.is_tool_active("chat_basic")
    assert registry.is_tool_active("resolve_tools")  # registered after the catalog
    assert not registry.is_tool_active("weather_now")
    assert not registry.is_tool_active("execute")  # outside base_allowed entirely

    active_names = {t.name for t in registry.list()}
    assert "weather_now" not in active_names
    assert "execute" not in active_names

    catalog.resolve("weather")
    assert registry.is_tool_active("weather_now")
    assert registry.is_tool_active("weather_forecast")
    assert not registry.is_tool_active("lights_on")
    assert not registry.is_tool_active("execute")


def test_execute_tool_refuses_deferred_with_waterfall_hint():
    registry, catalog = _setup(base_allowed={"chat_basic", "weather_now"})
    result = asyncio.run(registry.execute_tool("weather_now", {}))
    assert not result.success
    assert "resolve_tools" in (result.error or "")


def test_resolve_tools_meta_tool_activates():
    registry, catalog = _setup(
        base_allowed={"chat_basic", "weather_now", "weather_forecast", "lights_on"}
    )
    result = asyncio.run(registry.execute_tool("resolve_tools", {"categories": "weather"}))
    assert result.success
    assert "Use metric units." in result.output
    assert registry.is_tool_active("weather_now")
    ran = asyncio.run(registry.execute_tool("weather_now", {}))
    assert ran.success


def test_preload_from_query_activates():
    registry, catalog = _setup(base_allowed={"chat_basic", "weather_now", "weather_forecast"})
    loaded = catalog.preload_from_query("what's the weather like today?")
    assert loaded == ["weather"]
    assert registry.is_tool_active("weather_now")


def _agent_with_catalog(catalog, registry):
    """A ClawAgent whose invoke() returns immediately, for preload assertions."""
    from clawagents.agent import ClawAgent
    from clawagents.providers.llm import LLMProvider, LLMResponse

    class _FakeLLM(LLMProvider):
        name = "fake"

        async def chat(self, messages, on_chunk=None, cancel_event=None, tools=None):
            return LLMResponse(content="ok", model="fake", tokens_used=0)

    agent = ClawAgent(llm=_FakeLLM(), tools=registry)
    agent.streaming = False
    agent.catalog = catalog
    return agent


def test_invoke_preloads_from_the_task_by_default():
    registry, catalog = _setup(
        base_allowed={"chat_basic", "weather_now", "weather_forecast"}
    )
    agent = _agent_with_catalog(catalog, registry)
    asyncio.run(agent.invoke("what's the weather like today?"))
    assert catalog.resolved_categories == {"weather"}


def test_invoke_preload_text_overrides_the_task():
    """A runtime that wraps the request in boilerplate can scope the matcher.

    Without this the wrapper's own words resolve the same categories on every
    run, identically, no matter what was actually asked.
    """
    registry, catalog = _setup(
        base_allowed={"chat_basic", "weather_now", "weather_forecast", "lights_on"}
    )
    agent = _agent_with_catalog(catalog, registry)
    agent.catalog_preload_text = "turn on the light"
    asyncio.run(agent.invoke("turn on the light\n---\nAlways check the weather first."))
    assert catalog.resolved_categories == {"smart_home"}
    assert not registry.is_tool_active("weather_now")


def test_keywords_match_on_word_boundaries_not_substrings():
    from clawagents.tools.waterfall_catalog import keywords_match

    # The bug this replaces: "log" fired inside "monologue" and "login".
    assert not keywords_match(["log"], "no interior monologue please")
    assert not keywords_match(["log"], "stopped at a login wall")
    assert not keywords_match(["light"], "what a delightful idea")
    assert not keywords_match(["resume"], "presume nothing")

    assert keywords_match(["log"], "add it to the log")
    # Inflections still count, so natural phrasing keeps resolving.
    assert keywords_match(["log"], "logged it yesterday")  # doubled consonant
    assert keywords_match(["rain"], "is it raining?")
    assert keywords_match(["cloud"], "looks cloudy out")
    assert keywords_match(["light"], "turn the lights on")
    assert keywords_match(["resume"], "he resumed the run")


def test_keywords_match_handles_punctuation_and_phrases():
    from clawagents.tools.waterfall_catalog import keywords_match

    assert keywords_match(["jobs@"], "forward it to jobs@eherring.com")
    assert keywords_match(["to-do"], "put it on the to-do list")
    assert keywords_match(["note to self"], "note to self: buy milk")
    # Space padding was a workaround for substring matching; it is no longer
    # needed, and a padded keyword still behaves.
    assert keywords_match([" ig "], "posted it on ig, finally")
    assert not keywords_match(["ig"], "I dig this")


def test_no_base_allowed_defaults_to_all_registered():
    registry, catalog = _setup(base_allowed=None)
    # Everything registered at construction except deferred category tools.
    assert registry.is_tool_active("execute")
    assert not registry.is_tool_active("weather_now")
    catalog.resolve_all()
    assert registry.is_tool_active("weather_now")
    assert registry.is_tool_active("lights_on")


def test_prompt_only_catalog_no_registry_effects():
    catalog = ToolCatalog(
        [ToolCategory(name="x", description="d", tool_names=["t"])], [], None
    )
    assert "resolve_tools" in catalog.catalog_prompt()
