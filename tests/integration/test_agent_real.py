"""The agent with the real model and real tools (costs about a cent)."""

import re

from sales_poc.ai.agent import Agent


def test_the_graph_question_is_answered_with_sources():
    agent = Agent()
    try:
        result = agent.ask("How many enterprise customers who bought servers also bought storage?")
    finally:
        agent.close()

    assert result.steps and all(step.ok for step in result.steps[-1:])
    assert result.sources
    assert "68" in re.findall(r"\d+", result.text)  # matches acceptance check 6
    assert not result.hit_step_limit
