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


def _attach_perf_reference_anchors() -> None:
    """Append RTX 5090 measurements for the Qwen 2.5 7B + 32B dense
    perf-reference model entries (added 2026-05-01 per [docs] 20:55 +
    [backend] 15:43 / 20:08 bake-offs).

    These models weren't in `sizer_bundle.json` because they're new
    perf-comparison-reference entries (no Skippy v2 prompt-set eval),
    so we register them post-bundle-attach. Each cell registered under
    all 5 workloads (rag_qa / short_chat / long_decode / etc) — decode
    is BW-bound and prompt-invariant on dense models, so the same
    measurement applies cleanly across workloads.

    Anchor source: `data/output/bakeoff/llm_anchors/` per [backend]
    15:43 + 20:08. Bake-off shape: RAG 8K prompt + 2K decode @ RTX 5090.
    """
    from .npu_model import RTX_5090_REFERENCE
    # Per-(model, quant) measurements — decode_tok_s and prefill_tok_s
    # are roughly prompt-invariant on dense Q4/Q5/Q8, so registering the
    # same point estimate across workloads is honest within ~10%.
    perf_anchors = {
        # 7B family
        "qwen2.5-7b-q4-dense":  {"decode": 183.9, "prefill": 7226.0},
        "qwen2.5-7b-q5-dense":  {"decode": 170.0, "prefill": 7215.0},
        "qwen2.5-7b-q8-dense":  {"decode": 137.2, "prefill": 7478.0},
        # 32B family — no Q8 (won't fit on 5090's 32 GB VRAM with KV+activations)
        "qwen2.5-32b-q4-dense": {"decode":  52.7, "prefill": 1936.0},
        "qwen2.5-32b-q5-dense": {"decode":  47.7, "prefill": 1888.0},
    }
    workloads = ("short_chat", "rag_qa", "long_decode",
                  "meeting_summarization", "agentic_roundtrip")
    measured = RTX_5090_REFERENCE.measured_llm or {}
    for model_key, anchors in perf_anchors.items():
        if model_key not in measured:
            measured[model_key] = {}
        for wid in workloads:
            measured[model_key][wid] = {
                "decode_tok_s":  anchors["decode"],
                "prefill_tok_s": anchors["prefill"],
                "ttft_s":        None,  # derived downstream from prompt_tokens / prefill_tok_s
                "host_ms":       0.0,
                # Bake-off reference shape — informational
                "prompt_tokens_p50":     8000,
                "completion_tokens_p50": 2000,
            }
    RTX_5090_REFERENCE.measured_llm = measured


_attach_perf_reference_anchors()


def _override_14b_q4_5090_with_fresh_eval() -> None:
    """Override the 14B Q4 5090 cell with [docs] 2026-05-07 10:18
    fresh-eval numbers (132-sample v2-RAG telemetry: median 125.7 tok/s
    decode, 5117 tok/s prefill). The bundle's earlier 102.2 / 3905 came
    from a different (smaller-sample) measurement context. Cross-app
    convergence with keyhole-sizer (which shipped 125.7 in 559050c)
    + alignment with the 0.727 v2-RAG eval that produced these as
    aggregate telemetry. Override applies across all 5 PAI workloads
    since dense decode is roughly prompt-invariant (same property as
    the perf-reference cells in _attach_perf_reference_anchors)."""
    from .npu_model import RTX_5090_REFERENCE
    cell_14b = (RTX_5090_REFERENCE.measured_llm or {}).get(
        "qwen2.5-14b-q4-dense", {})
    for wid in cell_14b:
        cell_14b[wid] = dict(cell_14b[wid])
        cell_14b[wid]["decode_tok_s"] = 125.7
        cell_14b[wid]["prefill_tok_s"] = 5117.0


_override_14b_q4_5090_with_fresh_eval()


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
