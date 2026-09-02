# SPDX-License-Identifier: Apache-2.0
"""Mixed-PoC (chat + PoC in one forward) — EXPERIMENTAL subpackage.

Off unless explicitly enabled at launch. Depends one-way on the gonka_poc
core: the core never imports gonka_poc.mixed.

The engine calls into this package directly from the residual seams: the
scheduler constructs ``PoCAdmission`` per step, and the model runner drives
``PoCRunnerBridge`` around the forward. See gonka-ai/vllm#100.
"""

from gonka_poc.mixed.admission import PoCAdmission  # noqa: E402
from gonka_poc.mixed.bridge import PoCRunnerBridge  # noqa: E402
