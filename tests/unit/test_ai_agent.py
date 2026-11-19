import json
from datetime import date
from types import SimpleNamespace as NS

import pytest

from sales_poc.ai.agent import OUT_OF_STEPS, Agent, Source, ToolBox, default_makers, sources_from
from sales_poc.ai.llm import Llm
from sales_poc.ai.tools import ToolError
from sales_poc.ai.tools.sql import SqlTool
from sales_poc.ai.tools.table import Table
from sales_poc.config import LlmSettings


# --- a scripted model ----------------------------------------------------------------------

def says(text):
    return NS(content=text, tool_calls=None), "stop"


def calls(*tool_calls):
    raw = [NS(id=f"call_{i}", function=NS(name=name, arguments=args if isinstance(args, str) else json.dumps(args)))
           for i, (name, args) in enumerate(tool_calls)]
    return NS(content=None, tool_calls=raw), "tool_calls"


class ScriptedModel:
    """Replies in order; records every request."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **request):
        self.requests.append(json.loads(json.dumps(request, default=str)))
        message, finish = self.replies.pop(0)
        return NS(choices=[NS(message=message, finish_reason=finish)],
                  usage=NS(prompt_tokens=10, completion_tokens=1), model="m")


class FakeTool:
    def __init__(self, name, table=None, error=None):
        self.name, self.table, self.error, self.calls, self.closed = name, table, error, [], False

    def run(self, **arguments):
        self.calls.append(arguments)
        if self.error:
            raise ToolError(self.error)
        return self.table or Table(arguments.get("query", ""), ["n"], [(1,)], False)

    def close(self):
        self.closed = True


def spec(name, *properties):
    return {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": {p: {"type": "string"} for p in properties}}}}


def agent(*replies, max_steps=6, tools=None):
    tools = tools or {
        "sql_query": FakeTool("sql_query"),
        "vector_search": FakeTool("vector_search", Table("q", ["summary_id", "match", "text"],
                                                         [("region:EMEA:2026-03", 0.8, "...")], False)),
    }
    params = {"sql_query": ("query",), "vector_search": ("query", "region")}
    makers = {name: (spec(name, *params.get(name, ("query",))), (lambda t=tool: t)) for name, tool in tools.items()}
    settings = LlmSettings(_env_file=None, openrouter_api_key="k", llm_model="m", agent_max_steps=max_steps)
    model = ScriptedModel(*replies)
    return Agent(Llm(settings, model), ToolBox(makers), today=date(2026, 11, 19)), model, tools


# --- the loop ------------------------------------------------------------------------------

def test_a_direct_answer_uses_no_tools():
    a, model, tools = agent(says("Hello."))
    result = a.ask("hi")

    assert result.text == "Hello." and result.steps == [] and result.sources == []
    assert "Today is 2026-11-19" in model.requests[0]["messages"][0]["content"]
    assert [t["function"]["name"] for t in model.requests[0]["tools"]] == ["sql_query", "vector_search"]


def test_a_tool_call_runs_and_its_result_goes_back_to_the_model():
    a, model, tools = agent(calls(("sql_query", {"query": "SELECT 1"})), says("It is 1."))
    result = a.ask("q")

    assert tools["sql_query"].calls == [{"query": "SELECT 1"}]
    assert result.text == "It is 1."
    assert result.sources == [Source("sql_query", "SELECT 1")]
    assert result.usage.calls == 2 and result.usage.total_tokens == 22

    second = model.requests[1]["messages"]
    assert second[2]["role"] == "assistant" and second[2]["tool_calls"][0]["id"] == "call_0"
    assert second[3] == {"role": "tool", "tool_call_id": "call_0", "content": "n\n1"}


def test_vector_search_sources_are_the_summary_ids():
    a, _, _ = agent(calls(("vector_search", {"query": "emea"})), says("ok"))
    assert a.ask("q").sources == [Source("vector_search", "region:EMEA:2026-03")]


def test_the_same_source_is_listed_once():
    a, _, _ = agent(calls(("sql_query", {"query": "SELECT 1"}), ("sql_query", {"query": "SELECT 1"})), says("ok"))
    result = a.ask("q")
    assert len(result.steps) == 2 and len(result.sources) == 1


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (("sql_query", {"query": "DELETE FROM sales"}), "Error: refused"),
        (("drop_everything", {}), "no tool called 'drop_everything'"),
        (("sql_query", "{not json"), "not valid JSON"),
        (("sql_query", {"query": "x", "limit": 5}), "Unknown argument(s) limit"),
        (("sql_query", {}), "do not fit this tool"),
    ],
)
def test_tool_problems_go_back_to_the_model_as_errors(call, message):
    tools = {"sql_query": FakeTool("sql_query", error="refused") if "DELETE" in str(call) else _NeedsQuery()}
    a, model, _ = agent(calls(call), says("Sorry."), tools=tools)
    result = a.ask("q")

    assert result.text == "Sorry."
    [step] = result.steps
    assert not step.ok and message in step.result
    assert result.sources == []
    assert message in model.requests[1]["messages"][-1]["content"]


class _NeedsQuery:
    def run(self, query):
        return Table(query, ["n"], [(1,)], False)


def test_the_model_can_recover_after_an_error():
    sql = FakeTool("sql_query")
    sql.run = lambda query: (_ for _ in ()).throw(ToolError("bad column")) if "bad" in query else Table(
        query, ["n"], [(5,)], False)
    a, _, _ = agent(calls(("sql_query", {"query": "SELECT bad"})), calls(("sql_query", {"query": "SELECT good"})),
                    says("It is 5."), tools={"sql_query": sql})
    result = a.ask("q")
    assert [s.ok for s in result.steps] == [False, True]
    assert result.sources == [Source("sql_query", "SELECT good")]


def test_the_step_limit_stops_tool_calls_and_forces_an_answer():
    a, model, tools = agent(*[calls(("sql_query", {"query": f"SELECT {i}"})) for i in range(3)],
                            says("Best I can do."), max_steps=3)
    result = a.ask("q")

    assert len(tools["sql_query"].calls) == 3 and result.hit_step_limit
    assert result.text == "Best I can do."
    last = model.requests[-1]
    assert last["tool_choice"] == "none"
    assert last["messages"][-1] == {"role": "user", "content": OUT_OF_STEPS}
    assert len(model.requests) == 4


def test_calls_over_the_limit_in_one_reply_are_answered_but_not_run():
    a, model, tools = agent(calls(*[("sql_query", {"query": f"SELECT {i}"}) for i in range(4)]),
                            says("Done."), max_steps=2)
    result = a.ask("q")

    assert len(tools["sql_query"].calls) == 2 and len(result.steps) == 2 and result.hit_step_limit
    tool_replies = [m for m in model.requests[-1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_replies] == ["call_0", "call_1", "call_2", "call_3"]
    assert tool_replies[3]["content"].startswith("Not run")


def test_tools_are_made_on_first_use_and_closed():
    made = []
    sql = FakeTool("sql_query")
    makers = {"sql_query": (spec("sql_query", "query"), lambda: made.append(1) or sql),
              "graph_query": (spec("graph_query", "query"), lambda: pytest.fail("graph was not needed"))}
    box = ToolBox(makers)
    settings = LlmSettings(_env_file=None, openrouter_api_key="k", llm_model="m")
    a = Agent(Llm(settings, ScriptedModel(calls(("sql_query", {"query": "SELECT 1"})), says("ok"))), box)
    a.ask("q")
    a.close()
    assert made == [1] and sql.closed


# --- the real tools behind it --------------------------------------------------------------

def test_the_real_toolbox_offers_the_four_read_only_tools():
    names = [s["function"]["name"] for s in ToolBox().specs]
    assert names == ["sql_query", "graph_query", "vector_search", "get_forecast"]
    assert set(default_makers()) == set(names)


@pytest.mark.parametrize("query", ["DELETE FROM sales", "DROP TABLE sales", "UPDATE sales SET quantity = 0",
                                   "INSERT INTO sales (transaction_id) VALUES ('X')"])
def test_a_write_through_the_agent_is_refused_and_changes_nothing(query):
    """FR-29 end to end: the model asks for a write; the real SQL tool refuses it."""
    import pandas as pd

    sql = SqlTool(rows=pd.DataFrame({"transaction_id": ["T1", "T2"], "quantity": [1, 2]}))
    makers = {"sql_query": (spec("sql_query", "query"), lambda: sql)}
    settings = LlmSettings(_env_file=None, openrouter_api_key="k", llm_model="m")
    a = Agent(Llm(settings, ScriptedModel(calls(("sql_query", {"query": query})), says("Cannot."))), ToolBox(makers))

    [step] = a.ask("please change the data").steps
    assert not step.ok and "Only SELECT is allowed" in step.result
    assert sql.run("SELECT COUNT(*), SUM(quantity) FROM sales").rows == [(2, 3)]


def test_sources_from_other_tools_use_the_query():
    table = Table("forecast run-1 (all)", ["a"], [], False)
    assert sources_from("get_forecast", table) == [Source("get_forecast", "forecast run-1 (all)")]
