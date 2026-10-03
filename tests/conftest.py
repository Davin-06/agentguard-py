"""Shared fixtures for the AgentGuard test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentguard.models import Step, ToolCall, Trace  # noqa: E402


@pytest.fixture
def refund_trace() -> Trace:
    """The canonical incident: an informational chat that fires two refunds."""
    return Trace(
        trace_id="refund-incident-abc123",
        steps=(
            Step(
                turn=1,
                user="我上周那单还没到，帮我看下物流",
                tool_calls=(ToolCall("query_order", {"order_id": "A1001"}, turn=1),),
            ),
            Step(
                turn=2,
                user="对，就是那个蓝色的",
                tool_calls=(ToolCall("get_logistics", {"order_id": "A1001"}, turn=2),),
                facts=("订单 A1001",),
            ),
            Step(
                turn=3,
                user="好，那我再等等",
                tool_calls=(
                    ToolCall(
                        "refund_order",
                        {"order_id": "A1001", "amount": 299},
                        turn=3,
                    ),
                ),
            ),
            Step(
                turn=4,
                user="等等，我没说要退款啊",
                tool_calls=(
                    ToolCall(
                        "refund_order",
                        {"order_id": "A1001", "amount": 299},
                        turn=4,
                    ),
                ),
            ),
        ),
    )


@pytest.fixture
def clean_trace() -> Trace:
    """A healthy trace that must pass every rule."""
    return Trace(
        trace_id="clean-001",
        steps=(
            Step(
                turn=1,
                user="订单到哪了",
                tool_calls=(ToolCall("query_order", {"order_id": "B2"}, turn=1),),
            ),
            Step(
                turn=2,
                user="谢谢",
                tool_calls=(ToolCall("get_logistics", {"order_id": "B2"}, turn=2),),
            ),
        ),
    )
