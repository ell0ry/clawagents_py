"""Waterfall Tool Catalog — progressive tool schema loading.

Mirrors Claude Code's ToolSearch pattern: only tier-0 tools are sent with
every LLM call.  Domain-specific tool schemas are loaded on demand via the
``resolve_tools`` meta-tool or keyword pre-loading, keeping per-request
token costs low.

Since the v6 rebase this module drives the registry's *active-tools* set
(``ToolRegistry.set_active_tools`` / ``activate_tools``) instead of grafting
schema filtering into the agent loop: the loop rebuilds native schemas from
``registry.to_native_schemas()`` every round, so resolving a category takes
effect on the next round, and ``execute_tool`` refuses deferred tools until
they are resolved.

(Upstream's ``clawagents.tools.catalog`` is the unrelated tool-discovery
module; this one predates it in the fork and was renamed to avoid the path
collision.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Set

from clawagents.tools.registry import Tool, ToolResult

# Common English inflections a keyword should still match ("rain" → "raining",
# "cloud" → "cloudy", "log" → "logged"). A bare word boundary is too strict for
# natural phrasing; a bare prefix is too loose ("log" would take "login"). The
# optional repeat of the final letter covers doubled consonants (log → logged).
_INFLECTIONS = "(?:s|es|d|ed|ing|y)"


@lru_cache(maxsize=512)
def _keyword_pattern(keywords: tuple) -> Optional[re.Pattern]:
    """Compile *keywords* into one word-boundary-anchored alternation.

    Plain substring matching was the original design and it misfired badly:
    ``"log"`` matched *monologue*, ``"light"`` matched any sentence containing
    the word inside another, and the keyword lists had grown space-padded
    workarounds (``" ig "``, ``" cv "``) to fake the boundaries this provides.
    Padding is stripped here, so those entries can be written plainly.

    A keyword that starts or ends with punctuation (``"jobs@"``) is anchored
    only on the side where ``\\b`` can bite.
    """
    parts: list[str] = []
    for raw in keywords:
        kw = (raw or "").strip().lower()
        if not kw:
            continue
        body = re.escape(kw)
        left = r"\b" if kw[0].isalnum() else ""
        tail = ""
        if kw[-1].isalnum():
            doubled = re.escape(kw[-1])
            tail = f"(?:{doubled}?{_INFLECTIONS})?" + r"\b"
        parts.append(f"{left}{body}{tail}")
    return re.compile("|".join(parts)) if parts else None


def keywords_match(keywords: Sequence[str], text: str) -> bool:
    """True when any keyword occurs as a word (or inflection) in *text*."""
    pattern = _keyword_pattern(tuple(keywords or ()))
    return bool(pattern and pattern.search((text or "").lower()))


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
    registry : ToolRegistry | None
        The underlying registry holding all tool instances. When set, the
        catalog owns the registry's active-tools set. ``None`` builds a
        prompt-only catalog (``catalog_prompt()``) with no registry effects.
    base_allowed : set[str] | None
        The full universe of tool names the consumer permits (an allowlist).
        ``None`` means every registered tool. ``resolve_tools`` is always
        included so the meta-tool may be registered after construction.
    """

    def __init__(
        self,
        categories: list[ToolCategory],
        tier0_names: list[str],
        registry: Any,  # ToolRegistry — avoid circular import
        base_allowed: Optional[Set[str]] = None,
    ) -> None:
        self._categories: Dict[str, ToolCategory] = {c.name: c for c in categories}
        self._tier0_names: List[str] = list(tier0_names)
        self._registry = registry
        self._base_allowed: Optional[Set[str]] = (
            set(base_allowed) if base_allowed is not None else None
        )
        self._resolved: Set[str] = set()
        if self._registry is not None:
            self._registry._inactive_tool_hint = (
                'Call resolve_tools("<category>") to load its category first; '
                "the resolve_tools description lists the categories."
            )
            self._sync_registry()

    @property
    def categories(self) -> Dict[str, ToolCategory]:
        return self._categories

    @property
    def resolved_categories(self) -> Set[str]:
        return set(self._resolved)

    def _base_names(self) -> Set[str]:
        """The permitted tool-name universe (allowlist or every registered)."""
        if self._base_allowed is not None:
            base = set(self._base_allowed)
        else:
            base = {t.name for t in self._registry.list_registered()}
        # The meta-tool is typically registered *after* the catalog is built.
        base.add("resolve_tools")
        return base

    def _sync_registry(self) -> None:
        """Point the registry's active set at base minus unresolved categories."""
        if self._registry is None:
            return
        self._registry.set_active_tools(self._base_names() - self._deferred_tool_names())

    def resolve(self, category_name: str) -> list[str]:
        """Mark a category as resolved, returning its tool names."""
        cat = self._categories.get(category_name)
        if not cat:
            return []
        self._resolved.add(category_name)
        if self._registry is not None:
            names = set(cat.tool_names)
            if self._base_allowed is not None:
                names &= self._base_names()
            self._registry.activate_tools(names)
        return cat.tool_names

    def resolve_all(self) -> None:
        """Resolve every category (used by scheduled/headless agents)."""
        self._resolved = set(self._categories.keys())
        self._sync_registry()

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

        Retained for pre-rebase callers; the loop itself now reads the
        registry's active set via ``registry.to_native_schemas()``.
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

        Matching is on word boundaries (see :func:`keywords_match`), so a
        keyword only fires on the word it names — not on any longer word that
        happens to contain it.

        Returns list of category names that were pre-loaded.
        """
        loaded: list[str] = []
        for cat in self._categories.values():
            if cat.name in self._resolved:
                continue
            if keywords_match(cat.keywords, task):
                self.resolve(cat.name)
                loaded.append(cat.name)
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
        keywords: list[str] = []
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
