"""Waterfall Tool Catalog — progressive tool schema loading.

Mirrors Claude Code's ToolSearch pattern: only tier-0 tools are sent with
every LLM call.  Domain-specific tool schemas are loaded on demand via the
``resolve_tools`` meta-tool or keyword pre-loading, keeping per-request
token costs low.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from clawagents.tools.registry import Tool, ToolResult


@dataclass
class ToolCategory:
    """A named group of tools that can be resolved on demand."""

    name: str
    description: str
    tool_names: List[str]
    keywords: List[str] = field(default_factory=list)
    instruction: str = ""


class ToolCatalog:
    """Manages progressive tool schema loading.

    Parameters
    ----------
    categories : list[ToolCategory]
        Domain categories whose tools are deferred until resolved.
    tier0_names : list[str]
        Tool names that are always included in native schemas.
    registry : ToolRegistry
        The underlying registry holding all tool instances.
    """

    def __init__(
        self,
        categories: list[ToolCategory],
        tier0_names: list[str],
        registry: Any,  # ToolRegistry — avoid circular import
    ) -> None:
        self._categories: Dict[str, ToolCategory] = {c.name: c for c in categories}
        self._tier0_names: List[str] = list(tier0_names)
        self._registry = registry
        self._resolved: Set[str] = set()

    @property
    def categories(self) -> Dict[str, ToolCategory]:
        return self._categories

    @property
    def resolved_categories(self) -> Set[str]:
        return set(self._resolved)

    def resolve(self, category_name: str) -> list[str]:
        """Mark a category as resolved, returning its tool names."""
        cat = self._categories.get(category_name)
        if not cat:
            return []
        self._resolved.add(category_name)
        return cat.tool_names

    def resolve_all(self) -> None:
        """Resolve every category (used by scheduled/headless agents)."""
        self._resolved = set(self._categories.keys())

    def _deferred_tool_names(self) -> Set[str]:
        """Tool names in unresolved categories (excluding tier-0 overrides)."""
        tier0 = set(self._tier0_names)
        deferred: Set[str] = set()
        for cat_name, cat in self._categories.items():
            if cat_name not in self._resolved:
                for tn in cat.tool_names:
                    if tn not in tier0:
                        deferred.add(tn)
        return deferred

    def active_schemas(self) -> list[Any]:
        """Return NativeToolSchema list for all tools except unresolved categories.

        Built-in tools (think, filesystem, exec, etc.) and tier-0 tools are
        always included without needing explicit resolution.
        """
        deferred = self._deferred_tool_names()
        from clawagents.providers.llm import NativeToolSchema

        return [
            NativeToolSchema(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for tool in self._registry.list()
            if tool.name not in deferred
        ]

    def active_instruction_sections(self) -> str:
        """Return joined instruction text for all resolved categories."""
        sections: list[str] = []
        for cat_name in sorted(self._resolved):
            cat = self._categories.get(cat_name)
            if cat and cat.instruction:
                sections.append(cat.instruction.strip())
        return "\n\n".join(sections)

    def preload_from_query(self, task: str) -> list[str]:
        """Keyword-match the user query and auto-resolve matching categories.

        Returns list of category names that were pre-loaded.
        """
        task_lower = task.lower()
        loaded: list[str] = []
        for cat in self._categories.values():
            if cat.name in self._resolved:
                continue
            for kw in cat.keywords:
                if kw in task_lower:
                    self.resolve(cat.name)
                    loaded.append(cat.name)
                    break
        return loaded

    def catalog_prompt(self) -> str:
        """Generate a compact category listing for the system prompt."""
        lines = [
            "## Tool Categories",
            'Call resolve_tools("category1,category2") to load tools before using them.',
            "",
        ]
        for cat in self._categories.values():
            lines.append(f"- **{cat.name}**: {cat.description}")
        return "\n".join(lines)


def create_resolve_tools_tool(catalog: ToolCatalog) -> Tool:
    """Create the resolve_tools meta-tool bound to a catalog."""

    category_names = ", ".join(sorted(catalog.categories.keys()))

    class ResolveToolsTool:
        name = "resolve_tools"
        description = (
            "Load tool schemas for one or more capability categories. "
            "Call this before using domain-specific tools. "
            f"Categories: {category_names}"
        )
        parameters: dict[str, dict[str, Any]] = {
            "categories": {
                "type": "string",
                "description": (
                    "Comma-separated category names to load "
                    "(e.g. 'weather', 'smart_home,spotify')"
                ),
                "required": True,
            },
        }

        async def execute(self, args: Dict[str, Any]) -> ToolResult:
            raw = str(args.get("categories", ""))
            requested = [c.strip() for c in raw.split(",") if c.strip()]

            if not requested:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"No categories specified. Available: {category_names}",
                )

            loaded: list[str] = []
            skipped: list[str] = []
            already: list[str] = []
            total_tools = 0

            for cat_name in requested:
                if cat_name in catalog.resolved_categories:
                    already.append(cat_name)
                    continue
                tool_names = catalog.resolve(cat_name)
                if tool_names:
                    loaded.append(cat_name)
                    total_tools += len(tool_names)
                else:
                    skipped.append(cat_name)

            parts: list[str] = []
            if loaded:
                parts.append(f"Loaded {total_tools} tools for: {', '.join(loaded)}")
            if already:
                parts.append(f"Already loaded: {', '.join(already)}")
            if skipped:
                parts.append(
                    f"Unknown categories: {', '.join(skipped)}. "
                    f"Available: {category_names}"
                )

            # Include instruction sections so the model gets guidance
            # immediately — not just on the next turn.
            instructions: list[str] = []
            for cat_name in loaded:
                cat = catalog.categories.get(cat_name)
                if cat and cat.instruction:
                    instructions.append(cat.instruction.strip())

            output = ". ".join(parts)
            if instructions:
                output += "\n\n" + "\n\n".join(instructions)

            return ToolResult(
                success=len(skipped) == 0,
                output=output,
                error=None if not skipped else f"Unknown: {', '.join(skipped)}",
            )

    return ResolveToolsTool()
