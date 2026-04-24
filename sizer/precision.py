"""Precision-capability data + tier capability rollup.

Two axes:
  1. What compute precisions can a given tier execute in hardware?
     (Different from "what's peak TOPS > 0" — e.g. consumer Blackwell
     still has DP4A CUDA-core INT8 but dropped tensor-core INT8, so
     LLM W8A8 won't load via vLLM even though the silicon technically
     has int8 math units.)
  2. What's the measured quality delta vs fp16 for each precision
     we've empirically tested on Skippy's workload?

Quality data sources (personal-ai-framework repo, 2026-04-23):
  - fp16 reference & FP8 candidate: 48GB A6000 pod, v2+RAG pure-methodology run
    → acc_diff_fp16_vs_fp8_v2_rag_pure.md : both 84/132 (0.0pp delta)
  - fp16 reference & INT8 W8A8 candidate: H100 SXM pod, same methodology
    → acc_diff_fp16_vs_int8_v2_rag_pure.md : 86/132 → 81/132 (-3.8pp)
    5 miss = refusal-specificity artifacts (content-equivalent, stylistic)
  - Q4_K_M baseline: Skippy native-RAG run on 5090 — cross-methodology
    with fp16 shim runs; not directly comparable but surfaced for context.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrecisionMeasurement:
    """A measured accuracy result for one precision on Skippy's v2+RAG eval."""
    precision: str
    passes: int
    total: int
    delta_pp_vs_fp16: float | None   # None for the fp16 reference row
    source: str                      # "fp16 reference" / "FP8 candidate" / etc
    notes: str                       # 1-line context

    @property
    def pass_rate_pct(self) -> float:
        return 100.0 * self.passes / self.total


# Measured quality per precision — Qwen 2.5 14B Instruct, v2+RAG (44 prompts × 3 samples),
# pure-methodology vLLM shim with 8 RAG chunks. Run on A6000 (FP8) + H100 (INT8) pods.
MEASURED_PRECISION_QUALITY: dict[str, PrecisionMeasurement] = {
    "fp16": PrecisionMeasurement(
        precision="fp16",
        passes=86, total=132, delta_pp_vs_fp16=0.0,
        source="H100 SXM pod, vLLM shim",
        notes="Reference baseline. A6000 pod's fp16 run was 84/132 "
              "— run-to-run variance ~2pp is the noise floor.",
    ),
    "fp8": PrecisionMeasurement(
        precision="fp8",
        passes=84, total=132, delta_pp_vs_fp16=0.0,
        source="A6000 pod, vLLM shim",
        notes="Zero category regression. Coding byte-identical, "
              "rag_datasheet Jaccard 0.993, reasoning half byte-identical.",
    ),
    "int8": PrecisionMeasurement(
        precision="int8",
        passes=81, total=132, delta_pp_vs_fp16=-3.8,
        source="H100 SXM pod, vLLM shim (W8A8 via SmoothQuant+GPTQ)",
        notes="5 regressed samples concentrated in 2 refuse-to-answer prompts. "
              "Both models refused correctly; INT8 hedged at family level "
              "(\"various NXP i.MX processors\") vs fp16's SKU level "
              "(\"NXP i.MX 8QuadMax\"). Stylistic, not capability loss.",
    ),
    "q4_km": PrecisionMeasurement(
        precision="q4_km",
        passes=90, total=132, delta_pp_vs_fp16=None,
        source="5090 local, Skippy native RAG (llama-cpp)",
        notes="Not directly comparable — Skippy's native RAG wrapping "
              "(query rewriting + facts + memory) runs methodologically "
              "different from the fp16/FP8/INT8 vLLM shim pods. "
              "Q4_K_M via shim would produce a different number.",
    ),
}


# Runtime perf measured on RTX 5090 (dense Qwen 2.5 14B, 200-token decode,
# vLLM enforce_eager=True, 10 measured runs).
MEASURED_PRECISION_SPEED: dict[str, dict] = {
    "fp16": {"tok_s": 50.6, "speedup_vs_fp16": 1.00, "silicon": "5090 (bf16 tensor cores)"},
    "fp8":  {"tok_s": 79.4, "speedup_vs_fp16": 1.57, "silicon": "5090 (FP8 tensor cores)"},
    # INT8 on 5090 would fall back to DP4A (CUDA-core INT8) — not measured
    # yet but known slower than FP8 based on keyhole's YOLO INT8 vs FP8 data
    # (0.62ms vs 0.49ms = ~25% gap on small CNNs; scales worse on larger
    # matmuls). On H100/A100 with tensor-core INT8 the speed would track FP8.
}


# Precision capability per silicon class. A tier can "support" a precision
# in two flavors:
#   - "tensor_core": native tensor-core matmul (fast, preferred)
#   - "cuda_core":   DP4A or equivalent (works, significantly slower)
#   - "none":        can't execute this precision at all
#
# This is ORTHOGONAL to peak_tops_<dtype> which only asks "is there any
# path." For LLM-scale matmul, you almost always want tensor_core —
# cuda_core is acceptable for small CNN inference, crippling for 14B+.
_CAP_TENSOR = "tensor_core"
_CAP_CUDA = "cuda_core"
_CAP_NONE = "none"


def tier_precision_capability(hw_name: str) -> dict[str, str]:
    """Return {precision: capability_level} for a given tier name."""
    # Hardcoded because this is silicon-family knowledge, not derivable
    # from the perf/BW numbers we store on Hardware. Extend as we add tiers.
    if hw_name.startswith("RTX 5090"):
        # Consumer Blackwell SM120: dropped tensor-core INT8. FP8/FP4 gained.
        # DP4A CUDA-core INT8 path retained as legacy.
        return {
            "bf16/fp16": _CAP_TENSOR,
            "fp8":       _CAP_TENSOR,
            "int8":      _CAP_CUDA,     # DP4A only — vLLM W8A8 refuses to load
            "q4_km":     _CAP_TENSOR,   # weight-only, runs on bf16 tensor cores
        }
    if hw_name == "NPU Low-LP4":
        return {"bf16/fp16": _CAP_NONE,  "fp8": _CAP_NONE, "int8": _CAP_TENSOR, "q4_km": _CAP_NONE}
    if hw_name in ("NPU Low-LP5-32bit", "NPU Low-LP5-64bit"):
        return {"bf16/fp16": _CAP_NONE,  "fp8": _CAP_NONE, "int8": _CAP_TENSOR, "q4_km": _CAP_NONE}
    if hw_name == "NPU Low-LP5X":
        return {"bf16/fp16": _CAP_TENSOR, "fp8": _CAP_TENSOR, "int8": _CAP_TENSOR, "q4_km": _CAP_TENSOR}
    if hw_name == "NPU Mid":
        return {"bf16/fp16": _CAP_TENSOR, "fp8": _CAP_TENSOR, "int8": _CAP_TENSOR, "q4_km": _CAP_TENSOR}
    if hw_name == "NPU High":
        return {"bf16/fp16": _CAP_TENSOR, "fp8": _CAP_TENSOR, "int8": _CAP_TENSOR, "q4_km": _CAP_TENSOR}
    # Default conservative — unknown tier
    return {"bf16/fp16": _CAP_NONE, "fp8": _CAP_NONE, "int8": _CAP_NONE, "q4_km": _CAP_NONE}


def capability_badge(level: str) -> str:
    """One-glyph summary of capability level for UI."""
    return {_CAP_TENSOR: "✓", _CAP_CUDA: "⚠︎", _CAP_NONE: "✗"}.get(level, "?")


def capability_label(level: str) -> str:
    """Short text label for capability level."""
    return {
        _CAP_TENSOR: "tensor-core",
        _CAP_CUDA:   "CUDA-core only",
        _CAP_NONE:   "not supported",
    }.get(level, "unknown")


def capability_color(level: str) -> str:
    """CSS color for capability level (green/amber/red)."""
    return {_CAP_TENSOR: "#10b981", _CAP_CUDA: "#f59e0b", _CAP_NONE: "#ef4444"}.get(level, "#6b7280")


def quality_badge_text(pm: PrecisionMeasurement) -> str:
    """Format quality measurement for a compact UI badge."""
    if pm.delta_pp_vs_fp16 is None:
        return f"{pm.passes}/{pm.total} · methodology-different"
    sign = "+" if pm.delta_pp_vs_fp16 > 0 else ""
    return f"{pm.passes}/{pm.total} · Δ {sign}{pm.delta_pp_vs_fp16:+.1f}pp vs fp16"


def quality_color(pm: PrecisionMeasurement) -> str:
    """Green/amber/red for the measured delta."""
    d = pm.delta_pp_vs_fp16
    if d is None:
        return "#6b7280"  # gray — not comparable
    if abs(d) <= 2.0:
        return "#10b981"  # green — within noise floor
    if abs(d) <= 5.0:
        return "#f59e0b"  # amber — small but real
    return "#ef4444"      # red — meaningful regression


# ───────────────────────── Retargeting cost ─────────────────────────
# When you deploy a model to a target NPU with a DIFFERENT compute
# precision than what it was trained in, there's a lifecycle cost per
# model revision — not a one-time cost. Every retrain triggers a new
# round of: calibrate → quantize → regression-test → sign off.
#
# The cost ladder below is populated from Skippy's 2026-04-23 INT8
# quantization experiment (the actual timed, costed cycle we lived through).
#
# Deployment paths:
#   - FP_NATIVE: train in FP, deploy in FP. No retargeting.
#   - WEIGHT_ONLY: mechanical conversion (Q4_K_M, Q8_0 via llama-cpp).
#     Minutes, no calibration, no regression gate.
#   - PTQ: post-training quantization (SmoothQuant + GPTQ for W8A8).
#     Tens of minutes on a datacenter GPU + full regression-test cycle +
#     engineer review of prompt-level diffs.
#   - QAT: quantization-aware training. Requires redoing the full
#     fine-tune with INT8 simulation. Hours to days.

@dataclass(frozen=True)
class RetargetingCost:
    """Per-iteration cost of retargeting a model to a different precision."""
    path: str
    wall_minutes: int
    dollars_per_cycle: float
    engineer_hours: float
    regression_gate: bool   # true if requires human review of eval diffs
    notes: str


RETARGETING_COSTS: dict[str, RetargetingCost] = {
    "fp_native": RetargetingCost(
        path="FP-native (no retargeting)",
        wall_minutes=0, dollars_per_cycle=0.0, engineer_hours=0.0,
        regression_gate=False,
        notes="Model trained and deployed in same precision (bf16 / fp16 / FP8). "
              "Ship the training artifact directly. No calibration, no quant, "
              "no regression test required.",
    ),
    "weight_only": RetargetingCost(
        path="Weight-only quant (Q4_K_M / Q8_0)",
        wall_minutes=5, dollars_per_cycle=0.0, engineer_hours=0.1,
        regression_gate=False,
        notes="Mechanical conversion via llama-cpp `quantize` tool. Activations "
              "still run in fp16 — no activation calibration needed. ~5 min "
              "on a local machine. Skippy's current Q4_K_M deployment uses "
              "this path.",
    ),
    "ptq": RetargetingCost(
        path="PTQ W8A8 INT8 (SmoothQuant + GPTQ)",
        wall_minutes=55, dollars_per_cycle=1.75, engineer_hours=1.5,
        regression_gate=True,
        notes="What we just did. 35 min quant + 20 min eval on H100 SXM "
              "(~$1.75 pod). Plus ~1-2 engineer hours to read the diff, "
              "classify regressions (capability vs stylistic), decide if "
              "calib data / scheme needs re-tuning. ~10-20% of iterations "
              "need a re-tune pass — doubles the cost when it fires.",
    ),
    "qat": RetargetingCost(
        path="Quantization-aware training (QAT)",
        wall_minutes=360, dollars_per_cycle=60.0, engineer_hours=4.0,
        regression_gate=True,
        notes="Redo the full fine-tune with INT8 simulation in the forward "
              "pass. 4-8 hours on multi-GPU for a 14B model. $30-100 per run "
              "depending on setup. Produces the highest-quality INT8 artifact "
              "but the most expensive cycle. Not what we measured — listed "
              "as the ceiling on lifecycle cost.",
    ),
}


def deployment_path_for_tier(hw_name: str, model_quant_scheme: str = "fp16") -> str:
    """Which retargeting path is needed to ship a model to this tier?

    The model's training precision vs the tier's executable precision
    determines the path:
      - If tier supports the model's native precision: FP-native (free)
      - If model is weight-only quantized and tier supports the base dtype: weight-only
      - If tier is INT8-only and model is not yet quantized to INT8: PTQ (or QAT)
    """
    caps = tier_precision_capability(hw_name)
    # Skippy's current deployment is fp16-compute Q4_K_M weights
    model_dtype = model_quant_scheme.lower()

    # FP-native path: tier can run the model's native compute precision
    # on tensor cores
    if model_dtype in ("fp16", "bf16") and caps.get("bf16/fp16") == _CAP_TENSOR:
        return "fp_native"
    if model_dtype == "fp8" and caps.get("fp8") == _CAP_TENSOR:
        return "fp_native"
    if model_dtype == "q4_km" and caps.get("q4_km") == _CAP_TENSOR:
        return "weight_only"

    # INT8-only tier: needs PTQ at minimum
    has_fp_path = caps.get("bf16/fp16") == _CAP_TENSOR or caps.get("fp8") == _CAP_TENSOR
    if caps.get("int8") == _CAP_TENSOR and not has_fp_path:
        return "ptq"

    # Mixed tier (has both FP and INT8 tensor cores) — default to fp_native
    # since the FP path is free
    if has_fp_path:
        return "fp_native"

    # Unknown / no supported path
    return "ptq"


def retargeting_cost_color(path: str) -> str:
    """Color for the retargeting-cost badge."""
    return {
        "fp_native":  "#10b981",  # green — free
        "weight_only": "#10b981", # green — essentially free
        "ptq":        "#f59e0b",  # amber — real cost per iteration
        "qat":        "#ef4444",  # red — expensive
    }.get(path, "#6b7280")


# ───────────────────────── Regression testing rigor ─────────────────────────
# Real production AI systems run tiered regression gates — smoke (every PR)
# vs nightly vs pre-release (major versions) — because full-rigor testing is
# orders of magnitude more expensive than catastrophic-regression-catching.
#
# The key insight for the silicon-architecture conversation: every retrain
# has TWO gates that must pass on INT8-only silicon, one on FP-native:
#   - Gate A: "did the new training do what we wanted?" — paid every retrain
#     regardless of silicon target
#   - Gate B: "did the quantization damage anything Gate A validated?" —
#     ONLY needed on INT8-only silicon (FP-native skips this entirely)
#
# Gate B isn't cheaper than Gate A — same test set, different model. You
# can't subset "just quantization-sensitive tests" because you don't know
# which are sensitive until all run. So the silicon-choice delta per retrain
# is one full Gate B invocation at the chosen rigor tier.

@dataclass(frozen=True)
class RegressionRigor:
    """Production regression-testing cost per gate invocation.
    A 'gate' is one full pass of the test suite against one model variant.
    Costs below are per-gate — for INT8-only silicon you pay this TWICE
    per retrain (Gate A for training-validation, Gate B for quant-validation).
    """
    tier: str
    display_name: str
    test_count: str
    wall_hours_per_gate: float
    dollars_per_gate: float
    engineer_hours_per_gate: float
    human_review: bool
    notes: str


REGRESSION_RIGOR: dict[str, RegressionRigor] = {
    "smoke": RegressionRigor(
        tier="smoke",
        display_name="Smoke (PR-gate — catastrophic-regression only)",
        test_count="100-500 prompts",
        wall_hours_per_gate=0.5,
        dollars_per_gate=10.0,
        engineer_hours_per_gate=1.5,
        human_review=False,
        notes="Approximates what our 44-prompt × 3-sample harness does today. "
              "Catches catastrophic regressions (model broken, gibberish, "
              "template mismatch) but misses subtle drift in 1-in-10K prompts, "
              "long-context stability, structured-output format consistency, "
              "and safety-alignment regressions. Appropriate for research "
              "iteration, NOT for production deployment signoff.",
    ),
    "nightly": RegressionRigor(
        tier="nightly",
        display_name="Nightly (automated, LLM-judge)",
        test_count="5K-20K prompts",
        wall_hours_per_gate=10.0,
        dollars_per_gate=150.0,
        engineer_hours_per_gate=6.0,
        human_review=False,
        notes="Extended eval across curated benchmarks (MMLU-style, HumanEval, "
              "RAG-QA at scale) + LLM-as-judge grading at $0.01-0.05/prompt. "
              "No human-in-loop. Catches real capability loss in specific "
              "categories but not subtle safety/bias drift or multi-turn "
              "coherence. Baseline for serious production validation.",
    ),
    "pre_release": RegressionRigor(
        tier="pre_release",
        display_name="Pre-release (full suite + human red-team)",
        test_count="50K+ prompts + human review",
        wall_hours_per_gate=80.0,
        dollars_per_gate=2000.0,
        engineer_hours_per_gate=40.0,
        human_review=True,
        notes="Full automated suite + red-team adversarial probing + domain-"
              "expert review on specialized tasks + safety/bias audit. "
              "Required before a major version bump. Days to weeks of "
              "calendar time. Real production AI shops (Anthropic/OpenAI/"
              "DeepMind/Meta) operate at this tier for major releases.",
    ),
}


def gates_per_cycle(path_key: str) -> dict[str, int]:
    """How many gates fire per retrain cycle on this deployment path?

    - Gate A: "did the new training do what we wanted?"
    - Gate B: "did the quantization damage anything Gate A validated?"

    fp_native and weight-only paths only pay Gate A. PTQ and QAT pay
    both A and B (the quant introduces a separate axis of risk that
    needs its own validation pass)."""
    return {
        "fp_native":  {"gate_a": 1, "gate_b": 0},
        "weight_only": {"gate_a": 1, "gate_b": 0},
        "ptq":        {"gate_a": 1, "gate_b": 1},
        "qat":        {"gate_a": 1, "gate_b": 1},
    }.get(path_key, {"gate_a": 1, "gate_b": 0})


def annualized_testing_cost(path_key: str, rigor_key: str,
                             cycles_per_year: int) -> dict:
    """Compute total annualized testing cost for a (path, rigor, cadence)
    combination. Returns separate Gate A / Gate B line items + totals.

    FP-native silicon pays Gate A only; INT8-only silicon pays both.
    The silicon-choice DELTA is exactly the Gate B line (what you
    avoid by picking FP-native)."""
    gates = gates_per_cycle(path_key)
    rigor = REGRESSION_RIGOR[rigor_key]

    gate_a_cycles = gates["gate_a"] * cycles_per_year
    gate_b_cycles = gates["gate_b"] * cycles_per_year

    return {
        "rigor": rigor.display_name,
        "cycles_per_year": cycles_per_year,
        "gate_a": {
            "invocations": gate_a_cycles,
            "dollars": gate_a_cycles * rigor.dollars_per_gate,
            "engineer_hours": gate_a_cycles * rigor.engineer_hours_per_gate,
            "wall_hours": gate_a_cycles * rigor.wall_hours_per_gate,
        },
        "gate_b": {
            "invocations": gate_b_cycles,
            "dollars": gate_b_cycles * rigor.dollars_per_gate,
            "engineer_hours": gate_b_cycles * rigor.engineer_hours_per_gate,
            "wall_hours": gate_b_cycles * rigor.wall_hours_per_gate,
        },
        "total_dollars":
            (gate_a_cycles + gate_b_cycles) * rigor.dollars_per_gate,
        "total_engineer_hours":
            (gate_a_cycles + gate_b_cycles) * rigor.engineer_hours_per_gate,
        "total_wall_hours":
            (gate_a_cycles + gate_b_cycles) * rigor.wall_hours_per_gate,
        "human_review": rigor.human_review,
    }
