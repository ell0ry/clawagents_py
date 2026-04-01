"""Tests for PTRLContext.merge() and batched reflect()."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from clawagents.trajectory.recorder import PTRLContext


def _make_ctx(
    task: str = "do something",
    result: str = "done",
    model: str = "test-model",
    total_turns: int = 3,
    total_tool_calls: int = 5,
    tool_success_rate: float = 0.8,
    mid_run_failures: int = 1,
    run_score: int = 2,
    outcome: str = "success",
    duration_s: float = 10.0,
    n_turns: int = 3,
    **summary_overrides,
) -> PTRLContext:
    """Helper to build a PTRLContext with reasonable defaults."""
    summary = {
        "task": task,
        "task_type": "general",
        "outcome": outcome,
        "finish_reason": "natural",
        "run_score": run_score,
        "quality": "clean",
        "total_turns": total_turns,
        "total_tool_calls": total_tool_calls,
        "tool_success_rate": tool_success_rate,
        "mid_run_failures": mid_run_failures,
        "format_failures": 0,
        "logic_failures": mid_run_failures,
        "has_mixed_outcomes": False,
        "duration_s": duration_s,
        "tokens_total": 500,
        "verified_score": None,
        "verified_confidence": "",
        "verified_method": "",
        "model": model,
        "run_id": "run-001",
        "trajectory_file": "traj.jsonl",
    }
    summary.update(summary_overrides)

    turns = [
        {"turn_index": i, "score": 1, "tool_calls": [{"tool_name": "read_file", "success": True}]}
        for i in range(n_turns)
    ]

    return PTRLContext(
        task=task, result=result, summary_dict=summary, turn_dicts=turns, model=model,
    )


class TestMergeSingleContext:
    def test_returns_same_object(self):
        ctx = _make_ctx()
        assert PTRLContext.merge([ctx]) is ctx

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            PTRLContext.merge([])


class TestMergeMultipleContexts:
    def test_task_combined(self):
        merged = PTRLContext.merge([
            _make_ctx(task="check lights"),
            _make_ctx(task="dim living room"),
        ])
        assert "Conversation with 2 tasks:" in merged.task
        assert "1. check lights" in merged.task
        assert "2. dim living room" in merged.task

    def test_result_is_last(self):
        merged = PTRLContext.merge([
            _make_ctx(result="first result"),
            _make_ctx(result="second result"),
            _make_ctx(result="final result"),
        ])
        assert merged.result == "final result"

    def test_model_is_last_nonempty(self):
        merged = PTRLContext.merge([
            _make_ctx(model="gpt-4"),
            _make_ctx(model="claude-3"),
            _make_ctx(model=""),
        ])
        assert merged.model == "claude-3"

    def test_turns_reindexed(self):
        ctx1 = _make_ctx(n_turns=2)
        ctx2 = _make_ctx(n_turns=3)
        merged = PTRLContext.merge([ctx1, ctx2])

        assert len(merged.turn_dicts) == 5
        indices = [t["turn_index"] for t in merged.turn_dicts]
        assert indices == [0, 1, 2, 3, 4]


class TestMergeSummaryAggregation:
    def test_sum_fields(self):
        merged = PTRLContext.merge([
            _make_ctx(total_turns=3, total_tool_calls=5, mid_run_failures=1, duration_s=10.0),
            _make_ctx(total_turns=4, total_tool_calls=8, mid_run_failures=2, duration_s=15.0),
        ])
        s = merged.summary_dict
        assert s["total_turns"] == 7
        assert s["total_tool_calls"] == 13
        assert s["mid_run_failures"] == 3
        assert s["duration_s"] == 25.0

    def test_tool_success_rate_weighted(self):
        # ctx1: 5 calls at 80% = 4 successes
        # ctx2: 10 calls at 50% = 5 successes
        # merged: 9/15 = 60%
        merged = PTRLContext.merge([
            _make_ctx(total_tool_calls=5, tool_success_rate=0.8),
            _make_ctx(total_tool_calls=10, tool_success_rate=0.5),
        ])
        assert abs(merged.summary_dict["tool_success_rate"] - 0.6) < 0.01

    def test_tool_success_rate_zero_calls(self):
        merged = PTRLContext.merge([
            _make_ctx(total_tool_calls=0, tool_success_rate=0.0),
            _make_ctx(total_tool_calls=0, tool_success_rate=0.0),
        ])
        assert merged.summary_dict["tool_success_rate"] == 1.0

    def test_run_score_weighted(self):
        # ctx1: 3 turns, score 3 → weight 9
        # ctx2: 6 turns, score 1 → weight 6
        # merged: 15/9 ≈ 1.67 → rounds to 2
        merged = PTRLContext.merge([
            _make_ctx(total_turns=3, run_score=3),
            _make_ctx(total_turns=6, run_score=1),
        ])
        assert merged.summary_dict["run_score"] == 2

    def test_outcome_is_last(self):
        merged = PTRLContext.merge([
            _make_ctx(outcome="success"),
            _make_ctx(outcome="error"),
        ])
        assert merged.summary_dict["outcome"] == "error"

    def test_has_mixed_outcomes_or(self):
        merged = PTRLContext.merge([
            _make_ctx(has_mixed_outcomes=False),
            _make_ctx(has_mixed_outcomes=True),
        ])
        assert merged.summary_dict["has_mixed_outcomes"] is True

    def test_verified_score_weighted(self):
        merged = PTRLContext.merge([
            _make_ctx(total_turns=4, verified_score=0.8),
            _make_ctx(total_turns=6, verified_score=0.5),
        ])
        # (0.8*4 + 0.5*6) / 10 = 6.2/10 = 0.62
        assert abs(merged.summary_dict["verified_score"] - 0.62) < 0.01

    def test_verified_score_all_none(self):
        merged = PTRLContext.merge([
            _make_ctx(verified_score=None),
            _make_ctx(verified_score=None),
        ])
        assert merged.summary_dict["verified_score"] is None

    def test_verified_confidence_lowest(self):
        merged = PTRLContext.merge([
            _make_ctx(verified_confidence="high"),
            _make_ctx(verified_confidence="low"),
            _make_ctx(verified_confidence="medium"),
        ])
        assert merged.summary_dict["verified_confidence"] == "low"

    def test_quality_recomputed(self):
        # Score -1 + enough turns → "failed"
        merged = PTRLContext.merge([
            _make_ctx(total_turns=5, run_score=-1, mid_run_failures=3),
            _make_ctx(total_turns=5, run_score=-1, mid_run_failures=2),
        ])
        assert merged.summary_dict["quality"] == "failed"


class TestMergeEmptyTurns:
    def test_empty_turns_handled(self):
        ctx1 = _make_ctx(n_turns=0)
        ctx2 = _make_ctx(n_turns=2)
        merged = PTRLContext.merge([ctx1, ctx2])
        assert len(merged.turn_dicts) == 2
        assert merged.turn_dicts[0]["turn_index"] == 0
        assert merged.turn_dicts[1]["turn_index"] == 1


class TestReflectIntegration:
    def test_reflect_single_judge_and_lesson_call(self):
        """Verify reflect() makes exactly 1 judge + 1 lesson call for multi-context queue."""
        import asyncio
        from clawagents.agent import ClawAgent

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value=MagicMock(content="SCORE: 2\nREASON: good"))

        agent = ClawAgent.__new__(ClawAgent)
        agent.llm = mock_llm
        agent._ptrl_queue = [
            _make_ctx(task="task 1", mid_run_failures=2, run_score=1, has_mixed_outcomes=True),
            _make_ctx(task="task 2", mid_run_failures=1, run_score=1, has_mixed_outcomes=True),
            _make_ctx(task="task 3", mid_run_failures=1, run_score=2),
        ]

        with patch("clawagents.trajectory.lessons.save_lessons"):
            result = asyncio.run(agent.reflect())

        assert result["runs_processed"] == 3
        assert len(result["judge_scores"]) == 1  # Single merged judge call
        # Judge call + lesson extraction call = 2 LLM calls total
        assert mock_llm.chat.call_count == 2

    def test_reflect_empty_queue(self):
        import asyncio
        from clawagents.agent import ClawAgent

        agent = ClawAgent.__new__(ClawAgent)
        agent._ptrl_queue = []

        result = asyncio.run(agent.reflect())
        assert result["runs_processed"] == 0
        assert result["judge_scores"] == []
