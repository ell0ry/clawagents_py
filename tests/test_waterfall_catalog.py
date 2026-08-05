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
