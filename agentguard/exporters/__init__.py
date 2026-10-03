"""Export AgentGuard rules into other ecosystems' formats.

The strategic stance here is deliberate and worth stating plainly: AgentGuard
does **not** try to become the next standard.  promptfoo already owns
CI-style assertion execution and NeMo Guardrails already owns runtime
interception.  Competing with them is a losing game.

Instead we act as the *translation layer*: you maintain one rule, and we emit
the native config for whichever tool you already run -- and we say out loud
which constraints could not be translated, rather than shipping assertions that
look right but never fire.
"""

from .nemo import NemoExportResult, export_nemo, render_nemo
from .promptfoo import ExportResult, export_promptfoo, render_promptfoo

#: Names accepted by ``agentguard export --to``.
EXPORTERS: dict[str, str] = {
    "promptfoo": "promptfoo 断言配置",
    "nemo": "NeMo Guardrails tool_output rails",
}

__all__ = [
    "EXPORTERS",
    "ExportResult",
    "NemoExportResult",
    "export_nemo",
    "export_promptfoo",
    "render_nemo",
    "render_promptfoo",
]
