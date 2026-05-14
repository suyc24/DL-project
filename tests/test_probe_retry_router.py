from __future__ import annotations

from fhis.probe_retry_router import append_step_suffix, strip_redundant_step_marker


def test_strip_redundant_step_marker() -> None:
    assert strip_redundant_step_marker("Step 3: Recompute x.", 3) == "Recompute x."
    assert strip_redundant_step_marker("**Step 3:** Recompute x.", 3) == "Recompute x."
    assert strip_redundant_step_marker("Step 4: Keep this marker.", 3) == "Step 4: Keep this marker."


def test_append_step_suffix_spacing() -> None:
    assert append_step_suffix("Step 2:", "Let x = 1.") == "Step 2: Let x = 1."
    assert append_step_suffix("Step 2:", "  Step 3: no strip here") == (
        "Step 2: Step 3: no strip here"
    )
