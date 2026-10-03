"""Trace loader tests.

The loader deliberately accepts many shapes (OTel / Foundry / Langfuse-ish)
rather than inventing a format, so these tests pin the field-mapping contract.
"""

from __future__ import annotations

import json

import pytest

from agentguard.trace import TraceError, load_trace, parse_trace


def test_parses_minimal_trace():
    trace = parse_trace({"trace_id": "t1", "steps": [{"user": "hi"}]})
    assert trace.trace_id == "t1"
    assert len(trace.steps) == 1
    assert trace.steps[0].user == "hi"


def test_accepts_string_steps():
    trace = parse_trace({"steps": ["first", "second"]})
    assert [s.user for s in trace.steps] == ["first", "second"]
    assert [s.turn for s in trace.steps] == [1, 2]


@pytest.mark.parametrize("alias", ["steps", "turns", "messages", "conversation"])
def test_step_container_aliases(alias):
    trace = parse_trace({alias: [{"user": "x"}]})
    assert trace.steps[0].user == "x"


@pytest.mark.parametrize("alias", ["trace_id", "traceId", "operation_Id", "id"])
def test_trace_id_aliases(alias):
    assert parse_trace({alias: "T", "steps": [{"user": "x"}]}).trace_id == "T"


@pytest.mark.parametrize("alias", ["input", "prompt", "query", "user_message"])
def test_user_field_aliases(alias):
    trace = parse_trace({"steps": [{alias: "hello"}]})
    assert trace.steps[0].user == "hello"


@pytest.mark.parametrize("alias", ["output", "response", "completion", "answer"])
def test_assistant_field_aliases(alias):
    trace = parse_trace({"steps": [{alias: "world"}]})
    assert trace.steps[0].assistant == "world"


@pytest.mark.parametrize("alias", ["tool_calls", "toolCalls", "actions", "function_calls"])
def test_tool_call_container_aliases(alias):
    trace = parse_trace(
        {"steps": [{"user": "x", alias: [{"name": "query_order", "args": {"a": 1}}]}]}
    )
    assert trace.tool_calls()[0].name == "query_order"


def test_openai_style_function_call():
    trace = parse_trace(
        {
            "steps": [
                {
                    "user": "x",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "refund_order",
                                "arguments": '{"order_id": "A1"}',
                            }
                        }
                    ],
                }
            ]
        }
    )
    call = trace.tool_calls()[0]
    assert call.name == "refund_order"
    assert call.args == {"order_id": "A1"}


def test_arguments_as_json_string_is_decoded():
    trace = parse_trace(
        {"steps": [{"tool_calls": [{"name": "t", "arguments": '{"k": 1}'}]}]}
    )
    assert trace.tool_calls()[0].args == {"k": 1}


def test_arguments_as_broken_string_is_kept_raw():
    trace = parse_trace(
        {"steps": [{"tool_calls": [{"name": "t", "arguments": "{not json"}]}]}
    )
    assert trace.tool_calls()[0].args == {"_raw": "{not json"}


def test_string_tool_call_shorthand():
    trace = parse_trace({"steps": [{"tool_calls": ["query_order"]}]})
    assert trace.tool_calls()[0].name == "query_order"


def test_facts_field_populates_step():
    trace = parse_trace({"steps": [{"user": "x", "facts": ["F"]}]})
    assert trace.steps[0].facts == ("F",)


def test_turn_number_falls_back_to_index():
    trace = parse_trace({"steps": [{"user": "a"}, {"user": "b"}]})
    assert [s.turn for s in trace.steps] == [1, 2]


def test_explicit_turn_is_respected():
    trace = parse_trace({"steps": [{"turn": 7, "user": "a"}]})
    assert trace.steps[0].turn == 7


def test_trace_helpers():
    trace = parse_trace(
        {
            "trace_id": "t",
            "steps": [
                {"user": "hello", "tool_calls": [{"name": "a"}]},
                {"user": "world", "tool_calls": [{"name": "b"}]},
            ],
        }
    )
    assert trace.tools() == ["a", "b"]
    assert "hello" in trace.user_text()
    assert trace.turn(2) is not None
    assert trace.turn(99) is None


@pytest.mark.parametrize(
    "document",
    [
        {"trace_id": "t"},  # no steps
        {"steps": "not-a-list"},
        "not-a-dict",
    ],
)
def test_rejects_malformed_documents(document):
    with pytest.raises(TraceError):
        parse_trace(document)


def test_load_trace_from_file(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"trace_id": "f", "steps": [{"user": "x"}]}), encoding="utf-8")
    assert load_trace(path).trace_id == "f"


def test_load_trace_missing_file(tmp_path):
    with pytest.raises(TraceError):
        load_trace(tmp_path / "nope.json")


def test_load_trace_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(TraceError):
        load_trace(path)


def test_tool_call_arg_dotted_lookup():
    trace = parse_trace(
        {"steps": [{"tool_calls": [{"name": "t", "args": {"order": {"id": "X"}}}]}]}
    )
    call = trace.tool_calls()[0]
    assert call.arg("order.id") == "X"
    assert call.arg("order.missing", "fallback") == "fallback"
