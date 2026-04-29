"""NPU sizing math for Skippy — LLM-first, model-agnostic.

Adapted from keyhole-sizer's `sizer/npu_model.py` (2026-04-22 crib per [sizer]
recommendation). Vision pipeline code dropped; generalized for dense + MoE
models via the MODELS dict. BW-bound decode math derives from active_params
per the decode-is-bandwidth-bound approximation.

Every "measured" constant below traces to bake-offs in
`eval/results/sizer_bakeoff_*.json` (Skippy's own RTX 5090 telemetry —
`eval/run_sizer_bakeoffs.py` fires /generate with `include_telemetry=True`
against a real Skippy instance). Tier projections scale from those 5090
baselines by bandwidth ratio; the BW-bound decode assumption is documented
in the sizer UI next to every projected cell.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


# ───────────────────────── Hardware tiers ─────────────────────────

@dataclass
class Hardware:
    """Generic compute-and-bandwidth spec for any NPU or GPU."""
    name: str
    peak_tops_bf16: float
    peak_tops_int8: float
    peak_tops_fp8: float
    mem_bandwidth_gbs: float
    mem_capacity_gb: float
    mem_bus_width_bits: int
    mem_type: str
    mem_data_rate_gtps: float

    compute_efficiency: float = 0.65
    bandwidth_efficiency: float = 0.70
    tdp_watts: float = 0.0

    # Memory-class taxonomy for the same-class anchor-projection path
    # (Phase 2). Tiers in the same `tier_family` share enough silicon
    # characteristics that an anchor measured on one tier can be
    # BW-scaled within the family with high confidence (🟡 same-class).
    # Cross-family projection falls through to the two-floor MAX(BW,
    # compute) cross-class model (🔴). Taxonomy from [backend] 2026-04-29
    # 13:07: Class 1 Neutron-32-LP5, Class 2 Neutron-64-LP5, Class 3
    # LP5X-8.4 (Low-LP5X/Mid/High at stock — Mid+High share BW post
    # 95117df redirect), Class 4 memory-upgrade overlays (LP5T-11.2,
    # LPDDR6-12, LPDDR6-14 — within-class BW scaling on whatever stock
    # family the upgrade was applied to), Class 5 GDDR7-28.
    tier_family: str | None = None
    # Per-tier compute utilization factor for VISION cross-class compute
    # floor: gops_per_forward / (effective_tops × util_factor) ms.
    # Anchored to i.MX 95 yolov8n-INT8 (12 GOPs / 2 INT8 TOPS / 32 ms =
    # 0.19). Per [backend] 2026-04-29 12:38: Neutron 0.19, Mid 0.45,
    # High 0.50, 5090 0.85. Vision-only — LLM uses separate fields below
    # because LLM prefill realizes much lower silicon utilization due to
    # small per-layer matmuls + MoE expert routing + KV cache pressure.
    compute_util_factor: float = 0.45
    # LLM prefill compute utilization factor — much lower than vision on
    # the same silicon. Per [backend] 2026-04-29 13:17 calibration: Mid
    # LLM prefill anchors at 0.10 (vs 0.45 vision); canonically 5–15% in
    # the literature for LLM-on-edge-NPU. Used in cross-class TTFT
    # compute-floor: gops_per_token × prompt_tokens / (effective_tops ×
    # llm_prefill_util_factor) ms. Default 0.10 — refines per anchor.
    llm_prefill_util_factor: float = 0.10
    # LLM decode bandwidth realization factor: fraction of effective_BW
    # actually realized for LLM decode kernels. Default 1.0 (pure
    # BW-bound, ceiling = active_params_GB / effective_BW). Mid + Skippy
    # MoE Q4 measured at 0.66 × ceiling per [backend] 2026-04-29 13:17.
    # May be model-class-specific (MoE vs dense) but we only have a
    # MoE-class measurement today; default 1.0 for unmeasured tier-class
    # × model-class cells. Used in cross_class decode floor:
    # decode_floor_ms_per_tok = active_params_GB
    #                            / (hw.effective_bandwidth_gbs ×
    #                               llm_decode_bw_realization)
    llm_decode_bw_realization: float = 1.0
    # Per-tier overhead added to compute-floor (kernel launch, sync,
    # warmup amortization). Default 1 ms NPU / 0.3 ms GPU. Default
    # absorbed into compute-floor calibration; populate per tier when
    # anchors fit better.
    compute_overhead_ms: float = 1.0

    # Default fraction of memory bandwidth available to the NPU at the
    # tier's typical system-load condition. Per [docs] 2026-04-29 14:38
    # spec: NPU_share is a third orthogonal factor composing with
    # peak_DRAM_BW and kernel_util_factor —
    #   effective_NPU_BW = peak_DRAM_BW × NPU_share × bandwidth_efficiency
    # On SoC NPUs the memory bus is shared with display/camera/audio/
    # CPU — typical utilization sees 75% available; idle sees 100%;
    # heavy concurrent load sees 25-50%. RTX 5090 has dedicated VRAM
    # (no shared bus contention) so default 100%.
    # User-overridable via project_llm(npu_share=...) and the sidebar
    # "BW available to NPU" selector. Source classification stays
    # 🟢 measured_anchor at 100% (the original measurement condition);
    # at other shares the cell becomes a what-if scaling.
    npu_share_default: float = 0.75

    # Measured LLM decode tok/s + TTFT per (model_key, workload_id). When
    # populated, project_llm returns these directly for matching cells —
    # bypassing the BW projection. Mirrors keyhole's `measured_llm_*` pattern
    # but extended to per-model per-workload (our RTX 5090 has full data).
    # Schema: {model_key: {workload_id: {"decode_tok_s": float, "ttft_s": float,
    #                                     "prefill_tok_s": float, "host_ms": float}}}
    measured_llm: dict[str, dict[str, dict[str, float]]] | None = None

    # True when this Hardware was synthesized via `hw_with_memory()` (i.e.
    # represents a memory-only LPDDR6 what-if upgrade, not stock silicon).
    # UI checks this to mark BW-projected LLM tok/s as "(BW-proj)" so users
    # don't mistake a what-if projection for a vendor-measured number.
    # Mirrors keyhole-sizer commit ecc3ba8.
    bw_projected: bool = False
    # Snapshot of stock peak BW captured by `hw_with_memory()`. Lets
    # `project_llm` hold TTFT/prefill at the stock value (prefill is
    # compute-bound, not BW-bound, so a memory-only swap shouldn't move it)
    # while still letting decode tok/s scale up with the upgraded BW.
    stock_mem_bandwidth_gbs: float | None = None
    # Snapshot of the stock tier name (e.g. "NPU Mid") captured by
    # `hw_with_memory()`. Used by silicon-intrinsic lookups (precision
    # capability, deployment path) which key off the stock name — a
    # memory-only swap doesn't change tensor-core support, dtype
    # capability, or the retargeting cost class. Display strings still
    # use `name` so the variant suffix surfaces in headings.
    stock_name: str | None = None

    # Tier-level measured-decode-tok/s override per model_key. When set,
    # `project_llm` clamps the BW-projection from RTX_5090_REFERENCE to
    # this value for the matching model (BW-scaled if the user has
    # toggled a memory upgrade). Use for cases where a tier-class anchor
    # is measured on real silicon and the 5090-extrapolation is known to
    # mis-estimate (e.g. NPU Mid measured at 37.85 tok/s on Skippy MoE
    # Q4_K_M, vs 5090-projection ~13.9). Tactical interim per [docs]
    # 2026-04-29 12:34 spec; Path C (Phase 2 compute clamp proper) will
    # replace this with a calibrated efficiency model.
    measured_decode_overrides: dict[str, float] | None = None
    # Tier-level measured-prefill-tok/s override per model_key. When set,
    # `project_llm` clamps prefill_tok_s to this value (held at stock
    # under memory upgrades — prefill is compute-bound). Derived by
    # inverting a measured TTFT @ known prompt length: prefill_tok_s =
    # prompt_tokens / (ttft_s - host_overhead_s). Same tactical-interim
    # framing as `measured_decode_overrides`.
    measured_prefill_overrides: dict[str, float] | None = None

    @property
    def effective_bandwidth_gbs(self) -> float:
        return self.mem_bandwidth_gbs * self.bandwidth_efficiency

    @property
    def tier_lookup_name(self) -> str:
        """Stock tier name for silicon-intrinsic lookups (precision
        capability, deployment path). Memory-only upgrades inherit silicon
        caps from the stock tier — `hw_with_memory()` rewrites `name` to
        surface the variant in display strings, but precision / dtype
        capabilities don't change."""
        return self.stock_name if self.stock_name is not None else self.name

    def effective_tops(self, dtype: str) -> float:
        peak = {
            "int8": self.peak_tops_int8,
            "fp8": self.peak_tops_fp8,
            "bf16": self.peak_tops_bf16,
            "fp16": self.peak_tops_bf16,
        }.get(dtype.lower(), self.peak_tops_bf16)
        return peak * self.compute_efficiency

    def get_measured(self, model_key: str, workload_id: str) -> dict | None:
        if not self.measured_llm:
            return None
        cell = self.measured_llm.get(model_key, {}).get(workload_id)
        if cell is not None:
            return cell
        # Fall back to architecture sibling's measurement if the model
        # entry declares a `measurement_alias` (e.g. Thinking-2507 stock
        # shares Qwen3-30B-A3B architecture with Skippy's fine-tuned MoE).
        alias = MODELS.get(model_key, {}).get("measurement_alias")
        if alias and alias != model_key:
            return self.measured_llm.get(alias, {}).get(workload_id)
        return None


# Reference: RTX 5090 — all Skippy bake-offs ran here.
RTX_5090 = Hardware(
    name="NVIDIA RTX 5090",
    peak_tops_bf16=209.0, peak_tops_int8=419.0, peak_tops_fp8=419.0,
    mem_bandwidth_gbs=1792.0, mem_capacity_gb=32.0,
    mem_bus_width_bits=512, mem_type="GDDR7", mem_data_rate_gtps=28.0,
    compute_efficiency=0.70, bandwidth_efficiency=0.85,
    tdp_watts=575.0,
    # measured_llm populated from sizer_bundle.json at load time — see
    # measured.py for the wire-up.
)


# Edge NPU tiers — mirrored from keyhole-sizer's ladder (2026-04-22 post
# NPU_LOW_LP5 32/64-bit split). LLM measured numbers are keyhole's
# vendor-supplied Qwen3-30B-A3B Q4_K_M @ 1K prompt — used for cross-tier
# scaling sanity checks only; Skippy sizer projects from the 5090 reference
# via BW ratio rather than chaining off these.
NPU_LOW_LP4 = Hardware(
    name="NPU Low-LP4",
    peak_tops_bf16=0.0, peak_tops_int8=2.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=17.1, mem_capacity_gb=8.0,
    mem_bus_width_bits=32, mem_type="LPDDR4", mem_data_rate_gtps=4.266,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=10.0,
    tier_family="Neutron-32-LP4",  # below LP5 — own family
    compute_util_factor=0.19,
    compute_overhead_ms=1.0,
)

NPU_LOW_LP5_32BIT = Hardware(
    name="NPU Low-LP5-32bit",
    peak_tops_bf16=0.0, peak_tops_int8=2.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=25.6, mem_capacity_gb=16.0,
    mem_bus_width_bits=32, mem_type="LPDDR5", mem_data_rate_gtps=6.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=10.0,
    tier_family="Neutron-32-LP5",
    compute_util_factor=0.19,
    compute_overhead_ms=1.0,
)

NPU_LOW_LP5_64BIT = Hardware(
    name="NPU Low-LP5-64bit",
    peak_tops_bf16=0.0, peak_tops_int8=2.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=51.2, mem_capacity_gb=16.0,
    mem_bus_width_bits=64, mem_type="LPDDR5", mem_data_rate_gtps=6.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=10.0,
    tier_family="Neutron-64-LP5",
    compute_util_factor=0.19,
    compute_overhead_ms=1.0,
)

NPU_LOW_LP5X = Hardware(
    name="NPU Low-LP5X",
    peak_tops_bf16=50.0, peak_tops_int8=100.0, peak_tops_fp8=100.0,
    mem_bandwidth_gbs=67.2, mem_capacity_gb=16.0,
    mem_bus_width_bits=64, mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=10.0,
    # Low-LP5X is 64-bit at 8.4 GT/s = 67.2 GB/s — different effective
    # BW from the Mid/High class (128-bit @ 8.4 GT/s = 134.4 GB/s). Per
    # [backend] 13:07 taxonomy, bus-width crossings within the LP5X-8.4
    # data rate are a judgment call; treating Low-LP5X as its own family
    # for safety (anchors won't transfer).
    tier_family="LP5X-8.4-64b",
    compute_util_factor=0.45,
    compute_overhead_ms=1.0,
)

NPU_MID = Hardware(
    name="NPU Mid",
    # INT8-only silicon per [docs] 2026-04-29 14:58 spec correction.
    # Earlier 200 BF16 / 400 INT8 / 400 FP8 multi-precision label was
    # incorrect — the actual chip is INT8-only @ 200 TOPS, no FP path.
    # The Skippy MoE Q4 measurement (37.85 tok/s anchor) ran on this
    # INT8/INT4 silicon, so calibration constants (llm_prefill_util_factor=
    # 0.10, llm_decode_bw_realization=0.66 captured in anchor) are
    # already INT8-native. This is a label-correctness fix, not a math
    # change.
    peak_tops_bf16=0.0, peak_tops_int8=200.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=134.4, mem_capacity_gb=24.0,
    mem_bus_width_bits=128, mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
    compute_efficiency=0.65, bandwidth_efficiency=0.70,
    tdp_watts=25.0,
    # Memory-class shared with NPU High (128-bit LPDDR5X @ 8.4 GT/s);
    # an anchor measured on Mid projects to High as 🟡 same-class.
    tier_family="LP5X-8.4-128b",
    compute_util_factor=0.45,
    # LLM prefill calibrated to Mid + Skippy MoE Q4 anchor: 0.10 × peak
    # ≈ 351 ms TTFT @ 1K matches measured. This applies in the cross-
    # class fallback for cells without an anchor — gives the right TTFT
    # for unmeasured-model-on-Mid cases.
    llm_prefill_util_factor=0.10,
    # llm_decode_bw_realization stays at default 1.0 — the Mid + MoE
    # measurement (decode 0.66 × BW-floor ceiling) is already captured
    # in the anchor's 37.85 tok/s value, accessed via same-class
    # projection. We don't carry the 0.66 here because [backend] 13:17
    # flagged that decode realization is model-class-specific (dense
    # may not realize at 0.66 like MoE on the same silicon), and using
    # 0.66 in cross-class cells would silently over-pessimize dense-14B
    # tok/s without a measurement to back it up. Default 1.0 = ceiling;
    # 🔴 cross-class badge tells the user the number is low-confidence.
    compute_overhead_ms=1.0,
    # Measured-decode anchor: NPU Mid silicon bake-off on Skippy
    # Qwen3-30B-A3B Q4_K_M MoE → 37.85 tok/s (vs 5090-projection
    # ~13.9 — 5090-extrapolation under-predicts for cross-class
    # silicon because kernel efficiency factors don't transfer).
    # Per [docs] 2026-04-29 12:34 tactical interim; Path C
    # (Phase 2 compute clamp) replaces this with a calibrated model.
    measured_decode_overrides={
        "qwen3-30b-a3b-q4-moe": 37.85,
    },
    # Same Skippy MoE Mid bake-off: 351 ms TTFT @ 1K prompt. Inverting:
    # prefill_tok_s ≈ 1000 / 0.351 = 2849 tok/s. Per-workload TTFT then
    # derives from prompt_tokens / 2849 + host overhead, matching the
    # measured anchor exactly at 1K prompt and scaling linearly with
    # prompt length elsewhere (prefill is compute-bound, prompt-length-
    # proportional within the same model architecture).
    measured_prefill_overrides={
        "qwen3-30b-a3b-q4-moe": 2849.0,
    },
)

NPU_HIGH = Hardware(
    name="NPU High",
    peak_tops_bf16=275.0, peak_tops_int8=550.0, peak_tops_fp8=550.0,
    # Same memory class as NPU Mid (128-bit LPDDR5X @ 8.4 GT/s = 134.4
    # GB/s). NPU High differentiates from Mid on COMPUTE and CAPACITY
    # rather than memory BW: 1.375× TOPS, 1.33× DRAM, higher compute
    # efficiency (0.70 vs 0.65), 1.6× TDP. Memory upgrades (LPDDR5T,
    # LPDDR6) are surfaced separately as upgrade-path overlays via
    # MEMORY_UPGRADE_OPTIONS — not baked into the tier definition.
    mem_bandwidth_gbs=134.4, mem_capacity_gb=32.0,
    mem_bus_width_bits=128, mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
    compute_efficiency=0.70, bandwidth_efficiency=0.70,
    tdp_watts=40.0,
    # Same memory family as Mid (LP5X-8.4-128b); slightly higher
    # compute util_factor to reflect the 1.375× TOPS / higher
    # compute_efficiency / purpose-built HPC silicon (per [backend]
    # 12:38 calibration table).
    tier_family="LP5X-8.4-128b",
    compute_util_factor=0.50,
    # LLM prefill: Mid's 0.10 × 1.11 (compute-efficiency bump) ≈ 0.11.
    # Same caveat as Mid for decode realization — staying at default 1.0
    # since cross-class cells are 🔴 low-confidence regardless.
    llm_prefill_util_factor=0.11,
    compute_overhead_ms=1.0,
)

# Reference tier carrying the Skippy RTX 5090 bake-off data. Selected in
# the UI so users see "what the little monster can do" at the top of the
# ladder. measured_llm is populated from sizer_bundle.json by
# measured.attach_measurements_to_reference().
RTX_5090_REFERENCE = Hardware(
    name="RTX 5090 (reference, measured)",
    peak_tops_bf16=209.0, peak_tops_int8=419.0, peak_tops_fp8=419.0,
    mem_bandwidth_gbs=1792.0, mem_capacity_gb=32.0,
    mem_bus_width_bits=512, mem_type="GDDR7", mem_data_rate_gtps=28.0,
    compute_efficiency=0.70, bandwidth_efficiency=0.85,
    tdp_watts=575.0,
    # Datacenter-class memory controller + huge L2 — own family. All
    # measured Skippy bake-off cells live on this tier. Dedicated VRAM
    # means no shared-bus contention; default NPU_share = 100%.
    tier_family="GDDR7-28",
    compute_util_factor=0.85,
    compute_overhead_ms=0.3,
    npu_share_default=1.0,
)


TIERS = {t.name: t for t in (
    NPU_LOW_LP4,
    NPU_LOW_LP5_32BIT,
    NPU_LOW_LP5_64BIT,
    NPU_LOW_LP5X,
    NPU_MID,
    NPU_HIGH,
    RTX_5090_REFERENCE,
)}

HW_SLUGS = {t.name: t.name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
            for t in TIERS.values()}


# ───────────────────────── Memory upgrade overlay ────────────────────────────
# Mirrors keyhole-sizer commit ecc3ba8 (2026-04-29). Lets users preview a
# faster-memory swap on an existing tier without redefining the whole tier —
# same bus width / TOPS / capacity / TDP / capability_levels, just faster
# memory. Decode tok/s scales linearly with peak BW (active-param weights
# stream through DRAM per token). TTFT held at stock — prefill is compute-
# bound.
#
# Options sorted ascending by data rate (= BW at fixed bus width):
#   LPDDR5T @ 11.2 GT/s — Samsung's >10 GT/s LPDDR5-class extension; first
#                         step beyond stock LPDDR5X @ 8.4 GT/s.
#   LPDDR6 @ 12 GT/s    — first LPDDR6 spec rate.
#   LPDDR6 @ 14 GT/s    — top-bin LPDDR6.

MEMORY_UPGRADE_OPTIONS: list[tuple[str, str, float]] = [
    ("LPDDR5T @ 11.2 GT/s", "LPDDR5T", 11.2),
    ("LPDDR6 @ 12 GT/s",    "LPDDR6",  12.0),
    ("LPDDR6 @ 14 GT/s",    "LPDDR6",  14.0),
]


def hw_with_memory(hw: Hardware, mem_type: str, mem_data_rate_gtps: float,
                    name_suffix: str | None = None) -> Hardware:
    """Return a Hardware copy with the memory swapped (data-rate + type),
    bandwidth recomputed from bus width × data rate / 8, and an annotated
    name so downstream UI surfaces the variant.

    Decode tok/s naturally scales with the upgraded BW because `project_llm`
    (and `decode_tok_s_at_context`) BW-projects via `hw.effective_bandwidth_gbs`
    against the RTX 5090 reference. Active-param weights stream through DRAM
    per decoded token — BW-bound regime; `bandwidth_efficiency` cancels at
    the uniform 0.70 the rest of the model uses.

    TTFT (prefill) is held at stock — `project_llm` reads
    `stock_mem_bandwidth_gbs` for prefill scaling when set, so a memory-only
    swap doesn't move TTFT. Prefill is compute-bound (TOPS, not BW), so a
    memory-only swap shouldn't move it. Per [docs] 2026-04-29 spec mirroring
    [backend]'s deck tier-specs framing of TTFT as "0.351 s †" with the †
    footnote calling it out as compute-bound.

    The `bw_projected` flag is set to True so the UI can mark BW-scaled LLM
    numbers as projections rather than vendor measurements.

    TOPS / capacity / TDP / efficiencies are silicon-fixed and stay
    unchanged.
    """
    new_bw = hw.mem_bus_width_bits * mem_data_rate_gtps / 8
    new_name = hw.name if name_suffix is None else f"{hw.name} ({name_suffix})"
    return replace(
        hw,
        name=new_name,
        mem_type=mem_type,
        mem_data_rate_gtps=mem_data_rate_gtps,
        mem_bandwidth_gbs=new_bw,
        bw_projected=True,
        stock_mem_bandwidth_gbs=hw.mem_bandwidth_gbs,
        stock_name=hw.name,
    )


# ───────────────────────── Models ─────────────────────────

# Architecture-intrinsic constants per model. total_params / active_params
# drive the BW-bound decode math. active_params == total_params for dense;
# MoE has active < total. bytes_per_param=0.57 is Q4_K_M average (calibrated
# to keyhole-sizer's measurement anchor).
MODELS: dict[str, dict] = {
    "qwen2.5-14b-q4-dense": {
        "display_name": "Qwen 2.5 14B Skippy fine-tune (dense, Q4_K_M)",
        "family": "qwen2.5",
        "is_moe": False,
        "total_params": 14_700_000_000,
        "active_params": 14_700_000_000,
        "bytes_per_param": 0.57,
        "gguf_bytes": 8_986_070_304,
        "hidden_dim": 5120,
        "num_layers": 48,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "vocab_size": 152064,
        "ctx_len_trained": 32768,
        # Q4_K_M is weight-only quantization — weights stored in 4-bit k-means
        # groupings, but matmul compute is fp16 (weights dequantized per-op).
        # So the NPU needs fp16/bf16 tensor ops to run this natively; INT8-only
        # NPUs (NXP Neutron class) cannot run Q4_K_M without full W8A8
        # re-quantization or falling back to CPU fp16 (crushingly slow).
        "compute_dtype": "fp16",
        "quant_scheme": "Q4_K_M",
        # Accuracy on Skippy v2+RAG eval (44 prompts × 3 samples = 132).
        # Measured 2026-04-24 by [backend] session, eval/results/
        # acc_diff_dense_q4km_vs_moe_q4km_v2_rag.md.
        "training": "skippy_finetune",
        "pass_rate": 0.682,
        "pass_n_passes": 90,
        "pass_n_total": 132,
        # Δ vs production reference (Skippy MoE FT). MoE wins
        # rag_datasheet by +3 → dense -3 here; MoE loses refusal -2 →
        # dense +2 here. Sign: positive = THIS model wins vs production.
        "category_deltas": {
            "rag_datasheet": -3,
            "refusal":       +2,
        },
        "accuracy_bullet": (
            "Dense and MoE fine-tunes hit near-parity on quality "
            "(Δ -0.7pp vs production MoE). MoE wins on per-token cost "
            "(3B active << 14B dense), NOT accuracy. Choosing MoE is "
            "a cost decision, not a capability one."
        ),
    },
    "qwen3-30b-a3b-q4-moe": {
        "display_name": "Qwen3-30B-A3B Skippy fine-tune (MoE, Q4_K_M)",
        "family": "qwen3",
        "is_moe": True,
        "total_params": 30_500_000_000,
        "active_params": 3_300_000_000,
        "bytes_per_param": 0.57,
        "gguf_bytes": 18_556_684_448,
        "hidden_dim": 2048,
        "num_layers": 48,
        "num_attention_heads": 32,
        "num_kv_heads": 4,
        "num_experts": 128,
        "experts_per_token": 8,
        "vocab_size": 151936,
        "ctx_len_trained": 262144,
        # INT8 execution dtype reflects Skippy's actual NPU runtime:
        # Q4_K_M weight-only quant → INT8 dequant + INT8 matmul on
        # dedicated INT8 silicon (vs llama-cpp-python's fp16 dequant
        # path that runs on GPU). The Mid bake-off (37.85 tok/s anchor)
        # ran via the INT8 path on INT8-only NPU Mid. Per [docs] 14:58.
        "compute_dtype": "int8",
        "quant_scheme": "Q4_K_M",
        # Production reference for the v2+RAG accuracy axis. Other
        # models compute Δ vs this row. category_deltas is empty
        # because the production model can't differ from itself.
        "training": "skippy_finetune",
        "pass_rate": 0.689,
        "pass_n_passes": 91,
        "pass_n_total": 132,
        "category_deltas": {},
        "accuracy_bullet": (
            "Production reference (current shipping model). Domain "
            "fine-tuning lands the retrieval vocabulary the deck story "
            "is about — the +5.3pp delta vs stock public reasoning models "
            "isn't capability, it's domain knowledge."
        ),
    },
    # Stock public Qwen3-30B-A3B-Thinking-2507 — Alibaba's reasoning-tuned
    # variant of the same base architecture as Skippy's fine-tuned MoE.
    # Architecture is identical (same total/active params, same expert
    # routing) so cross-tier perf projections match the Skippy MoE row to
    # 1-for-1. Surfaced as a separate entry to support the deck story
    # "would a stock public reasoning model just replace the domain
    # fine-tune?" — answer: not on Kyle's domain (-5.3pp on rag_datasheet
    # per [backend]'s 2026-04-24 v2+RAG eval). Quality differentiation
    # lives in the deck narrative for now; PAI sizer's MEASURED_PRECISION_*
    # tables don't track per-checkpoint accuracy yet.
    "qwen3-30b-a3b-thinking-q4-moe": {
        "display_name": "Qwen3-30B-A3B-Thinking-2507 stock (MoE, Q4_K_M)",
        "family": "qwen3",
        "is_moe": True,
        "total_params": 30_500_000_000,
        "active_params": 3_300_000_000,
        "bytes_per_param": 0.57,
        "gguf_bytes": 18_556_684_448,
        "hidden_dim": 2048,
        "num_layers": 48,
        "num_attention_heads": 32,
        "num_kv_heads": 4,
        "num_experts": 128,
        "experts_per_token": 8,
        "vocab_size": 151936,
        "ctx_len_trained": 262144,
        # Same INT8 path as Skippy MoE Q4 (architecture sibling). Runs
        # on the same NPU runtime via the same Q4 → INT8 path.
        "compute_dtype": "int8",
        "quant_scheme": "Q4_K_M",
        # Architecture is identical to qwen3-30b-a3b-q4-moe (Skippy fine-
        # tune) — same total/active params, same expert routing — so
        # decode/prefill perf projections are 1-for-1. Borrow that model's
        # measurement bundle entries instead of duplicating the data.
        # Resolved by Hardware.get_measured() and calibration_anchors().
        "measurement_alias": "qwen3-30b-a3b-q4-moe",
        # Stock public reasoning baseline. Same v2+RAG eval, same RTX
        # 5090 host, same Q4_K_M GGUF — only difference is the training
        # (Alibaba's reasoning-tune vs Kyle's domain LoRA). Δ -5.3pp
        # overall, with the regression concentrated in domain retrieval.
        "training": "public_stock",
        "pass_rate": 0.636,
        "pass_n_passes": 84,
        "pass_n_total": 132,
        "category_deltas": {
            "rag_datasheet": -8,    # 26 prompts; the domain vocabulary gap
            "rag_email":     -3,    # 1 prompt × 3 samples; stock failed all three
            "numerical_precision": +3,  # general reasoning Thinking trains harder for
            "refusal":       +2,    # scope-limiting tuned harder in Thinking
        },
        "accuracy_bullet": (
            "Public reasoning models are stronger in general, but lose to "
            "the domain fine-tune on retrieval-grounded queries — domain "
            "knowledge doesn't fall out of larger general capability. "
            "Wins numerical_precision + refusal; loses rag_datasheet."
        ),
    },
}

# Reference model for per-category-Δ rendering. UI labels comparisons
# as "vs Skippy MoE fine-tune (production)" and the production model
# itself shows no per-category breakdown (it would be 0 across the
# board). Mirror of keyhole-sizer's PRODUCTION_REFERENCE_KEY pattern.
PRODUCTION_REFERENCE_KEY = "qwen3-30b-a3b-q4-moe"

# Human-readable labels for category-Δ display. Keyed by Skippy v2
# prompt category. Categories not listed in a model's category_deltas
# are flat ±0 vs production.
CATEGORY_LABELS: dict[str, str] = {
    "rag_datasheet":       "RAG · datasheet retrieval",
    "rag_email":           "RAG · email retrieval",
    "numerical_precision": "Numerical reasoning",
    "refusal":             "Refusal / scope control",
    "coding":              "Coding",
    "reasoning":           "General reasoning",
    "multihop":            "Multi-hop",
    "general":             "General Q&A",
    "persona":             "Persona / style",
    "rag_blog":            "RAG · blog retrieval",
}


# DTYPE compatibility — which compute precisions an NPU can run natively.
# A DTYPE is "supported" if the NPU's peak_tops_<dtype> > 0. The model's
# compute_dtype must match at least one supported HW dtype, else the model
# can only run via CPU fallback (not modeled — assume unusable).
_DTYPE_ATTR = {
    "fp16": "peak_tops_bf16",  # fp16 maps to bf16 tensor class on most SoCs
    "bf16": "peak_tops_bf16",
    "fp8":  "peak_tops_fp8",
    "int8": "peak_tops_int8",
}


def hw_supports_dtype(hw: "Hardware", dtype: str) -> bool:
    attr = _DTYPE_ATTR.get(dtype.lower())
    if attr is None:
        return False
    return getattr(hw, attr, 0.0) > 0.0


def hw_peak_tops_for_dtype(hw: "Hardware", dtype: str) -> float:
    """Raw peak TOPS for a dtype, without compute_efficiency multiplier.
    LLM cross-class compute floor uses this against llm_prefill_util_factor
    (which was calibrated by [backend] 2026-04-29 13:17 against raw peak,
    not effective_tops). Vision cross-class compute floor still uses
    `effective_tops()` because vision util_factors were calibrated that
    way (see compute_util_factor docstring)."""
    attr = _DTYPE_ATTR.get(dtype.lower())
    if attr is None:
        return 0.0
    return float(getattr(hw, attr, 0.0))


def model_active_bytes_per_token(model_key: str) -> float:
    """Bandwidth demand per decoded token — active params × bytes-per-param.
    The BW-bound decode floor: decode_tok_s ≈ hw.effective_bw / this."""
    m = MODELS[model_key]
    return m["active_params"] * m["bytes_per_param"]


def kv_cache_bytes_per_token(model_key: str, dtype_bytes: int = 2) -> float:
    """KV cache bytes consumed per token of context. Uses GQA ratio:
    kv_heads/attn_heads when available, else falls back to 1.0.

    kv_cache_per_token = num_layers × 2 (K+V) × hidden_dim × (kv/attn) × bytes_per_elem
    """
    m = MODELS[model_key]
    ratio = m.get("num_kv_heads", m.get("num_attention_heads", 1)) / \
            max(m.get("num_attention_heads", 1), 1)
    return m["num_layers"] * 2 * m["hidden_dim"] * ratio * dtype_bytes


# Memory overhead assumed for the runtime (llama-cpp-python + CUDA graphs +
# activation buffers + a little headroom). Pragmatic — real overhead varies
# by runtime, but 1 GB is a safe-ish default for llama-cpp on GPU.
RUNTIME_OVERHEAD_BYTES = 1_000_000_000


# ───────────────────────── What-if model projection ─────────────────────────
# Project decode tok/s for a hypothetical model (e.g. OLMoE, DeepSeek-V2-Lite,
# a candidate replacement for Skippy's current Qwen3-30B-A3B) without running
# bake-offs. Uses BW-bound decode scaling law:
#
#   decode_tok_s(what_if) ≈ decode_tok_s(anchor)
#                           × (anchor_active_bytes_per_token
#                              / what_if_active_bytes_per_token)
#
# Calibrated off the closest-matching measured architecture (MoE what-if
# projects from Qwen3-30B-A3B anchor; dense what-if projects from Qwen 2.5
# 14B anchor) so architecture-specific efficiency factors (MoE routing
# overhead, small-matmul inefficiency) transfer correctly without needing
# a separate efficiency constant.

def project_what_if_decode_tok_s(
    active_params: int, bytes_per_param: float, is_moe: bool,
    hw: "Hardware", ctx_tokens: int, compiler_quality: float = 1.0,
    npu_share: float | None = None,
) -> dict:
    """Project decode tok/s for a hypothetical model.

    Returns {"decode_tok_s", "anchor_model", "anchor_decode_tok_s",
             "bytes_per_token", "speedup_vs_current_skippy"}.
    """
    # Pick the architecture-matching anchor model
    anchor_model_key = ("qwen3-30b-a3b-q4-moe" if is_moe
                        else "qwen2.5-14b-q4-dense")
    anchor_meta = MODELS[anchor_model_key]

    anchor_bytes_per_token = (anchor_meta["active_params"]
                               * anchor_meta["bytes_per_param"])
    what_if_bytes_per_token = active_params * bytes_per_param

    # Get the anchor's interpolated decode tok/s at the same context length
    # on the same tier (this already handles BW scaling and compiler_quality)
    anchor_result = decode_tok_s_at_context(
        anchor_model_key, hw, ctx_tokens, compiler_quality=compiler_quality,
        npu_share=npu_share,
    )
    anchor_tok_s = anchor_result["decode_tok_s"]

    # BW-bound scaling: tok/s inversely proportional to bytes per token
    scaling = anchor_bytes_per_token / what_if_bytes_per_token if what_if_bytes_per_token > 0 else 0
    what_if_tok_s = anchor_tok_s * scaling

    # Skippy's current baseline (MoE) on same tier + context for comparison
    current_skippy = decode_tok_s_at_context(
        "qwen3-30b-a3b-q4-moe", hw, ctx_tokens,
        compiler_quality=compiler_quality,
        npu_share=npu_share,
    )

    return {
        "decode_tok_s": what_if_tok_s,
        "anchor_model": anchor_model_key,
        "anchor_display_name": anchor_meta["display_name"],
        "anchor_decode_tok_s": anchor_tok_s,
        "bytes_per_token": what_if_bytes_per_token,
        "anchor_bytes_per_token": anchor_bytes_per_token,
        "speedup_vs_anchor": scaling,
        "current_skippy_tok_s": current_skippy["decode_tok_s"],
        "speedup_vs_current_skippy": (what_if_tok_s /
                                       current_skippy["decode_tok_s"])
                                      if current_skippy["decode_tok_s"] > 0 else 0,
    }


def what_if_memory_feasibility(
    total_params: int, bytes_per_param: float, hw: "Hardware",
    context_tokens: int, hidden_dim: int = 4096, num_layers: int = 40,
    kv_head_ratio: float = 0.25,
) -> dict:
    """Rough memory feasibility for a hypothetical model. Uses reasonable
    defaults for KV cache geometry when user doesn't know the exact
    architecture (40 layers, 4096 hidden, 1/4 GQA ratio are typical for
    10B-40B-class models).
    """
    weights_b = total_params * bytes_per_param
    # KV: num_layers × 2 (K+V) × hidden_dim × (kv/attn ratio) × 2 bytes (fp16)
    kv_per_token = num_layers * 2 * hidden_dim * kv_head_ratio * 2
    kv_b = kv_per_token * context_tokens
    total_required = weights_b + kv_b + RUNTIME_OVERHEAD_BYTES
    available = hw.mem_capacity_gb * 1_000_000_000
    headroom = available - total_required
    if headroom < 0:
        verdict = "wont_fit"
    elif headroom < available * 0.15:
        verdict = "tight"
    else:
        verdict = "fits"
    return {
        "verdict": verdict,
        "required_gb": round(total_required / 1e9, 2),
        "available_gb": round(available / 1e9, 2),
        "headroom_gb": round(headroom / 1e9, 2),
        "breakdown": {
            "weights_gb": round(weights_b / 1e9, 2),
            "kv_cache_gb": round(kv_b / 1e9, 3),
            "overhead_gb": round(RUNTIME_OVERHEAD_BYTES / 1e9, 2),
        },
    }


def _log_linear_interpolate(anchors: list[tuple[int, float]],
                             ctx_tokens: int) -> tuple[float, str]:
    """Linearly interpolate decode_tok_s on a log(context) axis.

    anchors is a sorted list of (prompt_tokens, decode_tok_s) at the 5090
    reference. Returns (decode_tok_s, source) where source is one of:
      - "measured"       : ctx_tokens hits an anchor within ±5%
      - "interpolated"   : ctx_tokens falls between two anchors
      - "extrapolated_low" / "extrapolated_high": outside measured range
                           (clamped to endpoint but flagged)
    """
    import math
    if not anchors:
        raise ValueError("no calibration anchors")

    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    ctx = max(ctx_tokens, 1)

    # Exact-ish match to any anchor (within 5%)
    for x, y in zip(xs, ys):
        if abs(ctx - x) / max(x, 1) <= 0.05:
            return (y, "measured")

    # Below measured range → clamp to minimum anchor, flag
    if ctx < xs[0]:
        return (ys[0], "extrapolated_low")
    # Above measured range → clamp to maximum anchor, flag
    if ctx > xs[-1]:
        return (ys[-1], "extrapolated_high")

    # Between two anchors → log-linear interpolate on context axis
    log_ctx = math.log(ctx)
    for i in range(len(xs) - 1):
        if xs[i] <= ctx <= xs[i+1]:
            log_a, log_b = math.log(xs[i]), math.log(xs[i+1])
            t = (log_ctx - log_a) / (log_b - log_a) if log_b > log_a else 0.0
            return (ys[i] + t * (ys[i+1] - ys[i]), "interpolated")

    return (ys[-1], "extrapolated_high")


def decode_tok_s_at_context(model_key: str, hw: Hardware,
                             ctx_tokens: int,
                             compiler_quality: float = 1.0,
                             npu_share: float | None = None) -> dict:
    """Predict decode tok/s at arbitrary context length for (model, hw).

    Phase 2 (post 2026-04-29 Plan-C): same anchor-resolution shape as
    project_llm — measured cell wins, then same-family anchor (BW-scaled
    within tier_family), then cross-class two-floor MAX. Decode tok/s is
    roughly prompt-invariant on MoE (BW-bound, bytes-per-token doesn't
    change with context length), so the per-context curve flattens out
    at the anchor's value once the same-class anchor takes over. Only
    the 5090 reference shows meaningful prompt-length variation at this
    layer (kv-cache thrashing, etc.).

    Returns {"decode_tok_s", "source", "is_projected", "regime"}.
    """
    # Lazy import to avoid circular dependency with measured.py
    from .measured import calibration_anchors

    effective_npu_share = (npu_share if npu_share is not None
                            else hw.npu_share_default)

    # 1) RTX 5090 reference: log-linear interpolate from per-workload
    # bake-off cells (preserves the prompt-length shape we measured).
    if hw.name == RTX_5090_REFERENCE.name:
        anchors_full = calibration_anchors(model_key)
        if not anchors_full:
            raise ValueError(f"no calibration data for {model_key}")
        anchors = [(a[0], a[1]) for a in anchors_full]
        tok_s, interp_source = _log_linear_interpolate(anchors, ctx_tokens)
        # NPU_share scaling on the measured cell (rare for 5090 since
        # default is 1.0, but user can pick lower as a what-if).
        tok_s = tok_s * (effective_npu_share / hw.npu_share_default)
        return {
            "decode_tok_s": tok_s,
            "source": "measured",
            "is_projected": False,
            "regime": "bw_bound",
            "ctx_tokens": ctx_tokens,
            "interp_source": interp_source,
        }

    # 2) Same-family anchor — BW-scale the anchor's tok/s within family.
    # Decode is BW-bound on MoE so prompt length doesn't matter.
    anchor = _find_same_family_anchor(hw, model_key)
    if anchor is not None:
        anchor_tier, decode_anchor, _prefill_anchor = anchor
        bw_ratio_within_family = hw.mem_bandwidth_gbs / anchor_tier.mem_bandwidth_gbs
        # Anchor was measured at full NPU access (npu_share=1.0). Scale
        # by user's effective_npu_share.
        tok_s = (decode_anchor * bw_ratio_within_family
                  * effective_npu_share * compiler_quality)
        is_direct = (
            hw.tier_lookup_name == anchor_tier.name and not hw.bw_projected
        )
        return {
            "decode_tok_s": tok_s,
            "source": "measured_anchor" if is_direct else "same_class_anchor",
            "is_projected": True,
            "regime": "bw_bound",
            "ctx_tokens": ctx_tokens,
        }

    # 3) Cross-class fallback: two-floor MAX(BW, compute) per token.
    # Uses LLM-specific calibration (llm_prefill_util_factor for compute
    # floor against RAW peak — [backend] 13:17 calibration; using
    # effective_tops would double-count compute_efficiency).
    # llm_decode_bw_realization on BW floor uses effective_bandwidth_gbs
    # (bandwidth_efficiency stays applied per [backend]'s formula).
    model_meta = MODELS[model_key]
    active_params_gb = (model_meta["active_params"]
                         * model_meta["bytes_per_param"]) / 1e9
    gops_per_token = (2 * model_meta["active_params"]) / 1e9
    required_dtype = model_meta.get("compute_dtype", "fp16")
    peak_tops_llm = hw_peak_tops_for_dtype(hw, required_dtype)
    decode_bw_realized = (hw.effective_bandwidth_gbs
                           * hw.llm_decode_bw_realization
                           * effective_npu_share)
    bw_floor_ms = (active_params_gb / max(decode_bw_realized, 1e-9)) * 1000.0
    compute_floor_ms = gops_per_token / max(
        peak_tops_llm * hw.llm_prefill_util_factor, 1e-9
    )
    per_token_ms = max(bw_floor_ms, compute_floor_ms)
    tok_s = (1000.0 / max(per_token_ms, 1e-6)) * compiler_quality
    regime = ("bw_bound"
               if bw_floor_ms >= compute_floor_ms
               else "compute_bound")
    return {
        "decode_tok_s": tok_s,
        "source": "cross_class",
        "is_projected": True,
        "regime": regime,
        "ctx_tokens": ctx_tokens,
    }


def describe_hw(hw: Hardware) -> str:
    """One-liner summary of a Hardware spec — memory + TOPS + capacity + TDP.

    Mirrors keyhole-sizer's describe_hw(). Format adapts to silicon
    capability: an INT8-only NPU (e.g. NXP Neutron class) won't report
    BF16/FP8 TOPS; a Blackwell card reports all three.
    """
    tops_parts = []
    if hw.peak_tops_bf16 > 0:
        tops_parts.append(f"{hw.peak_tops_bf16:.0f} TOPS BF16")
    if hw.peak_tops_int8 > 0:
        tops_parts.append(f"{hw.peak_tops_int8:.0f} INT8")
    if hw.peak_tops_fp8 > 0:
        tops_parts.append(f"{hw.peak_tops_fp8:.0f} FP8")
    tops_str = " / ".join(tops_parts) if tops_parts else "no tensor TOPS reported"
    return (f"{hw.name}: {hw.mem_bus_width_bits}-bit {hw.mem_type} @ "
            f"{hw.mem_data_rate_gtps} GT/s = {hw.mem_bandwidth_gbs:.1f} GB/s theo "
            f"({hw.effective_bandwidth_gbs:.1f} GB/s effective)  •  "
            f"{tops_str}  •  "
            f"{hw.mem_capacity_gb:.0f} GB DRAM  •  {hw.tdp_watts:.0f} W")


def memory_feasibility(model_key: str, hw: Hardware, context_tokens: int) -> dict:
    """Decide whether `(model, hw)` can even load at the given context length.

    Returns {"verdict": "fits"|"tight"|"wont_fit",
             "required_gb", "available_gb", "headroom_gb",
             "breakdown": {...}}.

    Thresholds:
      - wont_fit: required > available
      - tight:    required > available × 0.85 (less than 15% headroom)
      - fits:     otherwise
    """
    m = MODELS[model_key]
    weights_b = m["gguf_bytes"]
    kv_b = kv_cache_bytes_per_token(model_key) * context_tokens
    total_required = weights_b + kv_b + RUNTIME_OVERHEAD_BYTES
    available = hw.mem_capacity_gb * 1_000_000_000
    headroom = available - total_required
    if headroom < 0:
        verdict = "wont_fit"
    elif headroom < available * 0.15:
        verdict = "tight"
    else:
        verdict = "fits"
    return {
        "verdict": verdict,
        "required_gb": round(total_required / 1e9, 2),
        "available_gb": round(available / 1e9, 2),
        "headroom_gb": round(headroom / 1e9, 2),
        "breakdown": {
            "weights_gb": round(weights_b / 1e9, 2),
            "kv_cache_gb": round(kv_b / 1e9, 3),
            "overhead_gb": round(RUNTIME_OVERHEAD_BYTES / 1e9, 2),
        },
    }


# ───────────────────────── Projections ─────────────────────────

def project_llm(
    model_key: str,
    hw: Hardware,
    workload_id: str,
    *,
    prompt_tokens: int = 500,
    decode_tokens: int = 200,
    host_ms: float = 0.0,
    compiler_quality: float = 1.0,
    npu_share: float | None = None,
) -> dict:
    """Project LLM decode tok/s + TTFT for (model, hw, workload).

    Strategy:
      1. If hw has measured_llm[model][workload], use it directly (measured wins).
      2. Otherwise BW-project from the RTX_5090_REFERENCE measurement for the
         same (model, workload). Ratio = hw.effective_bw / 5090.effective_bw.
      3. Apply compiler_quality multiplier (0.5–1.0) to LLM-specific portions.

    Returns {"source": "measured"|"projected"|"wont_fit",
             "decode_tok_s", "prefill_tok_s", "ttft_s", "host_ms",
             "total_s", "decode_s", "prefill_s", "feasibility": {...}}
    """
    # 0a) Memory feasibility check — a model that can't load is not a perf
    # question. Return early with a memory-only result.
    feasibility = memory_feasibility(model_key, hw, prompt_tokens + decode_tokens)
    if feasibility["verdict"] == "wont_fit":
        return {
            "source": "wont_fit",
            "regime": None,
            "model_key": model_key,
            "workload_id": workload_id,
            "hw_name": hw.name,
            "decode_tok_s": 0.0, "prefill_tok_s": 0.0, "ttft_s": None,
            "host_ms": 0.0, "prefill_s": 0.0, "decode_s": 0.0, "total_s": 0.0,
            "decode_tokens": decode_tokens, "prompt_tokens": prompt_tokens,
            "feasibility": feasibility,
        }

    # 0b) DTYPE compatibility check — Q4_K_M is weight-only quant; matmul runs
    # in fp16. An INT8-only NPU cannot execute this without either re-quant
    # to W8A8 or CPU fp16 fallback (neither modeled). Mark incompatible cells.
    model_meta = MODELS[model_key]
    required_dtype = model_meta.get("compute_dtype", "fp16")
    if not hw_supports_dtype(hw, required_dtype):
        supported = [d for d in ("int8", "fp8", "bf16") if hw_supports_dtype(hw, d)]
        return {
            "source": "dtype_mismatch",
            "regime": None,
            "model_key": model_key,
            "workload_id": workload_id,
            "hw_name": hw.name,
            "decode_tok_s": 0.0, "prefill_tok_s": 0.0, "ttft_s": None,
            "host_ms": 0.0, "prefill_s": 0.0, "decode_s": 0.0, "total_s": 0.0,
            "decode_tokens": decode_tokens, "prompt_tokens": prompt_tokens,
            "feasibility": feasibility,
            "dtype_detail": {
                "model_needs": required_dtype,
                "quant_scheme": model_meta.get("quant_scheme"),
                "hw_supports": supported or ["none"],
            },
        }

    # ═══════════════════════════════════════════════════════════════════
    # Phase 2 LLM projection — per [backend] 2026-04-29 12:38 spec, [sizer]
    # 13:01 + [backend] 13:07 design decisions, [docs] 12:34 greenlight.
    #
    # Resolution order (first hit wins):
    #   1. Per-cell measured (hw.measured_llm[model][workload])      → 🟢 measured
    #   2. Tier-level anchor on this hw (measured_decode_overrides
    #      with target hw == anchor hw, no LPDDR upgrade)             → 🟢 measured_anchor
    #   3. Same-family anchor (measured_decode_overrides on a tier
    #      sharing tier_family) BW-scaled within family                → 🟡 same_class
    #   4. Two-floor MAX(BW_floor, compute_floor) cross-class fallback → 🔴 cross_class
    #
    # Decode physics:
    #   bw_floor_ms_per_token = active_params_GB / effective_BW
    #   compute_floor_ms_per_token = gops_per_token / (effective_TOPS × util)
    #   per_token_ms = max(bw_floor, compute_floor)
    #   decode_tok_s = 1000 / per_token_ms
    #
    # Prefill physics (TTFT compute, per-batch):
    #   bw_floor_ms = active_params_GB / effective_BW    (weights read once)
    #   compute_floor_ms = gops_per_token × prompt_tokens / (eff_TOPS × util)
    #   ttft_ms = max(bw_floor, compute_floor) + overhead
    # ═══════════════════════════════════════════════════════════════════

    model_meta = MODELS[model_key]
    active_params_gb = (model_meta["active_params"]
                         * model_meta["bytes_per_param"]) / 1e9
    # gops_per_token = 2 × active_params for matmul-bound forward (per
    # [backend] 12:38; matches the GPT-style transformer FLOP estimate).
    gops_per_token = (2 * model_meta["active_params"]) / 1e9
    # LLM cross-class compute floor uses RAW peak (not effective_tops):
    # llm_prefill_util_factor was calibrated by [backend] 13:17 against
    # peak directly (200 BF16 TOPS × 0.10 in their math). Using
    # effective_tops would double-count compute_efficiency.
    peak_tops_llm = hw_peak_tops_for_dtype(hw, required_dtype)
    # Effective NPU_share (per [docs] 2026-04-29 14:38 spec): fraction of
    # peak DRAM BW available to the NPU. Falls back to tier default
    # (5090=1.0, NPU tiers=0.75 typical SoC contention). Affects DECODE
    # tok/s only — decode is BW-bound on MoE active-param weight stream.
    # Prefill / TTFT compute is TOPS-gated and doesn't share the memory
    # bus, so npu_share does NOT scale prefill / TTFT in any path.
    effective_npu_share = (npu_share if npu_share is not None
                            else hw.npu_share_default)

    # 1) Per-cell measured wins (RTX 5090 cells live here).
    m = hw.get_measured(model_key, workload_id)
    source = "measured"
    regime = "bw_bound"  # MoE decode is BW-bound by physics; refine below
    # Per-cell measurements were taken at the tier's nominal NPU_share
    # (5090=1.0). User-selected non-default shares are what-ifs that
    # scale decode tok/s linearly.
    if m is not None and effective_npu_share != hw.npu_share_default:
        m = dict(m)  # copy — don't mutate the cached cell
        m["decode_tok_s"] = m["decode_tok_s"] * (effective_npu_share / hw.npu_share_default)

    if m is None:
        # Resolve same-family anchor (also catches the on-tier anchor
        # case via tier_lookup_name == anchor_tier.name).
        anchor = _find_same_family_anchor(hw, model_key)

        if anchor is not None:
            anchor_tier, decode_anchor, prefill_anchor = anchor
            is_direct = (
                hw.tier_lookup_name == anchor_tier.name
                and not hw.bw_projected
            )
            # Decode: BW-scale within family. Same family means same data
            # rate × same bus width or memory-upgrade overlay on the same
            # silicon. Scale by target_BW / anchor_BW. When target IS the
            # anchor tier (no upgrade), ratio = 1 and we get the anchor
            # value directly.
            bw_ratio_within_family = (
                hw.mem_bandwidth_gbs / anchor_tier.mem_bandwidth_gbs
            )
            # NPU_share: anchor was measured at full NPU access (npu_share=1.0
            # typical for the bake-off conditions). Scale by effective_npu_share.
            decode_tok_s = (decode_anchor * bw_ratio_within_family
                              * effective_npu_share * compiler_quality)
            # Prefill: held at anchor's stock value (compute-bound, not
            # BW-bound, so memory swap doesn't move it). If anchor doesn't
            # carry a prefill rate, project from 5090 (5090 cell will give
            # the per-workload TTFT shape, then we replace the rate).
            ref = RTX_5090_REFERENCE.get_measured(model_key, workload_id)
            host_ms_value = ref.get("host_ms", host_ms) if ref else host_ms
            if prefill_anchor is not None:
                prefill_tok_s = prefill_anchor * compiler_quality
                ttft_s_value = (prompt_tokens / prefill_tok_s) + (host_ms_value / 1000.0)
            elif ref is not None:
                # Fall back to 5090 prefill projection at stock-class BW
                stock_bw_for_prefill = (
                    (anchor_tier.mem_bandwidth_gbs * hw.bandwidth_efficiency)
                )
                prefill_bw_ratio = (
                    stock_bw_for_prefill / RTX_5090_REFERENCE.effective_bandwidth_gbs
                )
                prefill_tok_s = (ref["prefill_tok_s"]
                                  * (prefill_bw_ratio ** 0.5)
                                  * compiler_quality)
                ttft_s_value = (ref["ttft_s"]
                                 / (prefill_bw_ratio ** 0.5)
                                 / compiler_quality)
            else:
                # No prefill data anywhere — derive from LLM compute floor
                # (raw peak × llm_prefill_util_factor per [backend] 13:17).
                compute_floor_ms = (gops_per_token * prompt_tokens) / max(
                    peak_tops_llm * hw.llm_prefill_util_factor, 1e-9
                )
                ttft_ms = compute_floor_ms + hw.compute_overhead_ms
                prefill_tok_s = prompt_tokens / max(ttft_ms / 1000.0, 1e-6)
                ttft_s_value = ttft_ms / 1000.0
            m = {
                "decode_tok_s": decode_tok_s,
                "prefill_tok_s": prefill_tok_s,
                "ttft_s": ttft_s_value,
                "host_ms": host_ms_value,
            }
            source = "measured_anchor" if is_direct else "same_class_anchor"
            regime = "bw_bound"  # decode is BW-bound at the anchor
        else:
            # 4) Cross-class fallback: two-floor MAX(BW, compute).
            # No same-family anchor, so we derive from first principles.
            # Replaces the previous 5090-BW-projection (carried 5090's
            # implicit realization factor which doesn't transfer across
            # tier-classes — per [sizer] 13:01 + [backend] 13:07
            # "replace, not upward-clamp").
            #
            # LLM calibration uses LLM-specific util factors per
            # [backend] 13:17: prefill_util ~0.10 (vs vision's 0.45 —
            # LLM kernels realize lower silicon utilization due to small
            # per-layer matmuls + MoE expert routing + KV cache writes;
            # canonically 5–15% in the literature) and decode_bw_realization
            # for the BW floor. Both default to safe values (0.10 / 1.0)
            # for unmeasured tier-class × model-class cells; populated to
            # measured calibration on Mid/High via [backend] 13:17 spec.
            # NPU_share scales the decode BW floor only (not the compute
            # floor) per [docs] 14:38 spec. MAX(scaled_bw_floor, compute_
            # floor) naturally handles regime: if decode is compute-bound
            # at small npu_share, scaling BW further doesn't move the
            # floor.
            decode_bw_realized = (hw.effective_bandwidth_gbs
                                   * hw.llm_decode_bw_realization
                                   * effective_npu_share)
            bw_floor_ms_decode = (active_params_gb / max(decode_bw_realized, 1e-9)) * 1000.0
            compute_floor_ms_decode = gops_per_token / max(
                peak_tops_llm * hw.llm_prefill_util_factor, 1e-9
            )
            per_token_ms = max(bw_floor_ms_decode, compute_floor_ms_decode)
            decode_tok_s = (1000.0 / max(per_token_ms, 1e-6)) * compiler_quality
            regime = ("bw_bound"
                       if bw_floor_ms_decode >= compute_floor_ms_decode
                       else "compute_bound")
            # Prefill: per-batch BW (weights read once, no realization
            # factor — prefill BW is well-utilized; the bottleneck is
            # compute) + per-token compute with LLM prefill util_factor.
            bw_floor_ms_prefill = (active_params_gb / hw.effective_bandwidth_gbs) * 1000.0
            compute_floor_ms_prefill = (
                gops_per_token * prompt_tokens
                / max(peak_tops_llm * hw.llm_prefill_util_factor, 1e-9)
            )
            ttft_ms = max(bw_floor_ms_prefill, compute_floor_ms_prefill) + hw.compute_overhead_ms
            prefill_tok_s = prompt_tokens / max(ttft_ms / 1000.0, 1e-6) * compiler_quality
            m = {
                "decode_tok_s": decode_tok_s,
                "prefill_tok_s": prefill_tok_s,
                "ttft_s": ttft_ms / 1000.0,
                "host_ms": hw.compute_overhead_ms,
            }
            source = "cross_class"

    decode_s = decode_tokens / m["decode_tok_s"] if m["decode_tok_s"] > 0 else 0.0
    prefill_s = prompt_tokens / m["prefill_tok_s"] if m["prefill_tok_s"] > 0 else 0.0
    host_s = (m.get("host_ms") or host_ms) / 1000.0
    return {
        "source": source,
        "regime": regime,
        "model_key": model_key,
        "workload_id": workload_id,
        "hw_name": hw.name,
        "decode_tok_s": round(m["decode_tok_s"], 2),
        "prefill_tok_s": round(m["prefill_tok_s"], 2),
        "ttft_s": round(m["ttft_s"], 4) if m.get("ttft_s") else None,
        "host_ms": round(host_s * 1000, 2),
        "prefill_s": round(prefill_s, 3),
        "decode_s": round(decode_s, 3),
        "total_s": round(host_s + prefill_s + decode_s, 3),
        "decode_tokens": decode_tokens,
        "prompt_tokens": prompt_tokens,
        "feasibility": feasibility,
    }


def _find_same_family_anchor(target_hw: Hardware, model_key: str) -> tuple[Hardware, float, float | None] | None:
    """Find a tier in the same `tier_family` that has a measured-decode
    anchor for `model_key`. Returns `(anchor_tier, decode_tok_s, prefill_tok_s_or_None)`
    or None if no anchor exists in the family.

    Memory-upgrade overlays (hw.bw_projected=True) carry the stock tier's
    family — `target_hw.tier_family` is inherited via `dataclasses.replace`
    in `hw_with_memory()`, so an LPDDR6 upgrade on Mid still finds Mid's
    anchor naturally.

    Alias resolution: if `model_key` declares a `measurement_alias` in
    MODELS, the alias's anchor is returned when the model itself doesn't
    have one (e.g. Thinking-2507 picks up Skippy MoE Q4's anchor).
    """
    if target_hw.tier_family is None:
        return None
    candidate_keys = [model_key]
    alias = MODELS.get(model_key, {}).get("measurement_alias")
    if alias and alias != model_key:
        candidate_keys.append(alias)
    for tier in TIERS.values():
        if tier.tier_family != target_hw.tier_family:
            continue
        decode_map = tier.measured_decode_overrides or {}
        for k in candidate_keys:
            if k in decode_map:
                prefill_map = tier.measured_prefill_overrides or {}
                return (tier, decode_map[k], prefill_map.get(k))
    return None


# ───────────────────────── Invariant assertions ─────────────────────────

def _assert_invariants():
    """Fail-loud on dict-set mismatches at import (keyhole 8c696a2 pattern)."""
    assert TIERS, "TIERS empty"
    assert MODELS, "MODELS empty"
    assert set(HW_SLUGS.keys()) == set(TIERS.keys()), "HW_SLUGS ⊄ TIERS"
    for k, m in MODELS.items():
        for f in ("active_params", "bytes_per_param", "total_params"):
            assert f in m, f"MODELS[{k}] missing {f}"
    # Phase 2 schema invariants — every tier needs tier_family +
    # compute_util_factor (defaults are fine; this asserts they're set
    # to something non-None / non-zero so silent-fallthrough bugs
    # surface at import).
    for tier in TIERS.values():
        assert tier.tier_family is not None, (
            f"Hardware {tier.name!r} missing tier_family — Phase 2 same-"
            f"class anchor lookup needs this. Set to a string like "
            f"'LP5X-8.4-128b' per the [backend] 13:07 taxonomy."
        )
        assert tier.compute_util_factor > 0, (
            f"Hardware {tier.name!r} has compute_util_factor={tier.compute_util_factor}; "
            f"Phase 2 cross-class compute floor would divide-by-zero. "
            f"Per [backend] 12:38 calibration table: Neutron 0.19 / "
            f"Mid 0.45 / High 0.50 / 5090 0.85."
        )


def _assert_phase2_anchors():
    """Validate the [backend] 12:38 anchor list against Phase 2 projection
    output. Fail-loud if the anchor numbers drift — catches silent
    regressions in the override mechanism, BW-scaling math, or the
    tier_family taxonomy. Skip when the bundle isn't loaded yet (which
    is the case during pure-import without sizer.measured)."""
    if not RTX_5090_REFERENCE.measured_llm:
        return  # bundle not loaded yet; can't validate anchors
    # Anchors must be validated at npu_share=1.0 (the measurement was
    # taken at full NPU access; default 75% is a what-if scaling).
    # Anchor #5: Mid + Skippy MoE Q4 stock @ 1K prompt → 37.85 tok/s, 351 ms TTFT
    r5 = project_llm("qwen3-30b-a3b-q4-moe", NPU_MID, "short_chat",
                      prompt_tokens=1000, decode_tokens=200,
                      npu_share=1.0)
    assert abs(r5["decode_tok_s"] - 37.85) < 0.01, (
        f"Anchor #5 drift: Mid stock + MoE Q4 @ npu_share=1.0 expected "
        f"37.85 tok/s, got {r5['decode_tok_s']}. Source: {r5['source']!r}."
    )
    assert r5["source"] == "measured_anchor", (
        f"Anchor #5 mis-classified: expected measured_anchor (target IS "
        f"the anchor tier), got {r5['source']!r}."
    )
    # Anchor #6: Mid + LPDDR6-14 + MoE Q4 → 63.08 tok/s, TTFT held at stock
    mid_lpddr6_14 = hw_with_memory(NPU_MID, "LPDDR6", 14.0,
                                     name_suffix="LPDDR6-14")
    r6 = project_llm("qwen3-30b-a3b-q4-moe", mid_lpddr6_14, "short_chat",
                      prompt_tokens=1000, decode_tokens=200,
                      npu_share=1.0)
    assert abs(r6["decode_tok_s"] - 63.08) < 0.01, (
        f"Anchor #6 drift: Mid + LPDDR6-14 + MoE Q4 @ npu_share=1.0 "
        f"expected 63.08 tok/s, got {r6['decode_tok_s']}."
    )
    assert r6["source"] == "same_class_anchor", (
        f"Anchor #6 mis-classified: expected same_class_anchor (LPDDR6 "
        f"overlay, BW-scaled within family), got {r6['source']!r}."
    )
    # High stock + MoE → 🟡 same_class via Mid anchor (BW-equal, same family)
    r_high = project_llm("qwen3-30b-a3b-q4-moe", NPU_HIGH, "short_chat",
                          prompt_tokens=1000, decode_tokens=200,
                          npu_share=1.0)
    assert abs(r_high["decode_tok_s"] - 37.85) < 0.01, (
        f"High stock + MoE @ npu_share=1.0 expected 37.85 tok/s (BW-equal "
        f"to Mid in same family), got {r_high['decode_tok_s']}."
    )
    assert r_high["source"] == "same_class_anchor", (
        f"High stock + MoE mis-classified: expected same_class_anchor "
        f"(via Mid anchor in shared LP5X-8.4-128b family), got "
        f"{r_high['source']!r}."
    )
    # NPU_share scaling: at default 75%, Mid + MoE should show 28.39 tok/s
    # (37.85 × 0.75). Validates the new factor composes correctly.
    r5_default = project_llm("qwen3-30b-a3b-q4-moe", NPU_MID, "short_chat",
                              prompt_tokens=1000, decode_tokens=200)
    expected_at_75 = 37.85 * 0.75
    assert abs(r5_default["decode_tok_s"] - expected_at_75) < 0.05, (
        f"NPU_share scaling broken: Mid + MoE Q4 at default npu_share "
        f"(0.75) expected {expected_at_75:.2f} tok/s (37.85 × 0.75), "
        f"got {r5_default['decode_tok_s']}."
    )


_assert_invariants()
# Phase 2 anchor validation runs at import once `measured.py` has populated
# RTX_5090_REFERENCE.measured_llm. measured.py imports npu_model at the
# top, so this assertion fires on the second pass when measured.attach()
# completes — see measured.py end of module.
