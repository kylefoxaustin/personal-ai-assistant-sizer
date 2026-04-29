"""Load Skippy bake-off measurements into the RTX_5090_REFERENCE tier.

`sizer/sizer_bundle.json` is vendored from the personal-ai-framework repo's
`eval/build_sizer_bundle.py` output. At module import we mirror the measured
workload rows into `RTX_5090_REFERENCE.measured_llm`, so `project_llm()`
has a concrete per-(model, workload) baseline to fall back to.

Regenerate `sizer/sizer_bundle.json` by:
  1. Run `eval/run_sizer_bakeoffs.py` in personal-ai-framework against each
     model (see REPRODUCE.md).
  2. `python3 eval/build_sizer_bundle.py --bakeoff ... --out sizer_bundle.json`.
  3. Copy the result into this repo's `sizer/sizer_bundle.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

from .npu_model import RTX_5090_REFERENCE, MODELS


_BUNDLE_PATH = Path(__file__).parent / "sizer_bundle.json"


def load_bundle() -> dict:
    if not _BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"sizer_bundle.json missing at {_BUNDLE_PATH} — re-vendor from "
            "personal-ai-framework/eval/results/sizer_bundle.json"
        )
    return json.loads(_BUNDLE_PATH.read_text())


def attach_measurements_to_reference() -> dict:
    """Populate `RTX_5090_REFERENCE.measured_llm` from the bundle.

    Returns a dict summarizing which (model, workload) cells got measured
    data. Cells not in the bundle won't have measured data — project_llm
    will raise at call time, which is correct (sizer UI should grey those
    cells and tell the user to run a bake-off)."""
    bundle = load_bundle()
    measured: dict = {}
    for workload_id, per_model in bundle.get("workloads", {}).items():
        for canonical_model, cell in per_model.items():
            if canonical_model not in MODELS:
                continue
            m = measured.setdefault(canonical_model, {})
            m[workload_id] = {
                "decode_tok_s": cell.get("decode_tok_per_s_p50"),
                "prefill_tok_s": cell.get("prefill_tok_per_s_p50"),
                # ttft derived from prefill_ms since the bundle doesn't
                # explicitly store TTFT — it's the first-token latency.
                "ttft_s": (cell.get("prefill_ms_p50") or 0) / 1000.0,
                "host_ms": cell.get("host_ms_p50"),
                "prompt_tokens_p50": cell.get("prompt_tokens_p50"),
                "completion_tokens_p50": cell.get("completion_tokens_p50"),
            }
    RTX_5090_REFERENCE.measured_llm = measured
    return {
        "models": list(measured.keys()),
        "workloads_per_model": {k: list(v.keys()) for k, v in measured.items()},
        "bundle_meta": bundle.get("meta", {}),
    }


# Attach at import so anything using the module sees the populated reference.
_BUNDLE_SUMMARY = attach_measurements_to_reference()

# Phase 2 anchor validation — once measured_llm is populated, run the
# [backend] anchor list to catch silent regressions in override math,
# tier_family taxonomy, or BW-scaling. Fail-loud at import.
from .npu_model import _assert_phase2_anchors
_assert_phase2_anchors()


def get_bundle_summary() -> dict:
    """Read-only accessor for the summary computed at import."""
    return _BUNDLE_SUMMARY


# ───────────────────────── Decode-vs-context calibration ─────────────────────────

def calibration_anchors(model_key: str) -> list[tuple[int, float, str]]:
    """Build sorted [(prompt_tokens, decode_tok_s, workload_id), ...]
    anchor points for `model_key` on the 5090 reference, using EVERY
    measured workload as a calibration point (5 per model today).

    Used by the context-length scaling curve to interpolate decode
    throughput at arbitrary context lengths rather than only the
    5 fixed workload shapes.

    If the model entry declares a `measurement_alias` (e.g. Thinking-2507
    stock shares Qwen3-30B-A3B architecture with Skippy's MoE fine-tune),
    falls back to the alias's anchors when the model itself has no direct
    bundle data."""
    bundle = load_bundle()

    def _anchors_for(key: str) -> list[tuple[int, float, str]]:
        out: list[tuple[int, float, str]] = []
        for workload_id, per_model in bundle.get("workloads", {}).items():
            cell = per_model.get(key)
            if not cell:
                continue
            pt = cell.get("prompt_tokens_p50")
            ts = cell.get("decode_tok_per_s_p50")
            if pt is None or ts is None:
                continue
            out.append((int(pt), float(ts), workload_id))
        out.sort(key=lambda x: x[0])
        return out

    anchors = _anchors_for(model_key)
    if anchors:
        return anchors

    # Fall back to architecture sibling if the entry declares one.
    # Lazy import to avoid circular dependency (npu_model imports measured).
    from .npu_model import MODELS
    alias = MODELS.get(model_key, {}).get("measurement_alias")
    if alias and alias != model_key:
        return _anchors_for(alias)
    return anchors
