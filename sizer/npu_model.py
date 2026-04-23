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

from dataclasses import dataclass


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

    # Measured LLM decode tok/s + TTFT per (model_key, workload_id). When
    # populated, project_llm returns these directly for matching cells —
    # bypassing the BW projection. Mirrors keyhole's `measured_llm_*` pattern
    # but extended to per-model per-workload (our RTX 5090 has full data).
    # Schema: {model_key: {workload_id: {"decode_tok_s": float, "ttft_s": float,
    #                                     "prefill_tok_s": float, "host_ms": float}}}
    measured_llm: dict[str, dict[str, dict[str, float]]] | None = None

    @property
    def effective_bandwidth_gbs(self) -> float:
        return self.mem_bandwidth_gbs * self.bandwidth_efficiency

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
        return self.measured_llm.get(model_key, {}).get(workload_id)


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
)

NPU_LOW_LP5_32BIT = Hardware(
    name="NPU Low-LP5-32bit",
    peak_tops_bf16=0.0, peak_tops_int8=2.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=25.6, mem_capacity_gb=16.0,
    mem_bus_width_bits=32, mem_type="LPDDR5", mem_data_rate_gtps=6.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=10.0,
)

NPU_LOW_LP5_64BIT = Hardware(
    name="NPU Low-LP5-64bit",
    peak_tops_bf16=0.0, peak_tops_int8=2.0, peak_tops_fp8=0.0,
    mem_bandwidth_gbs=51.2, mem_capacity_gb=16.0,
    mem_bus_width_bits=64, mem_type="LPDDR5", mem_data_rate_gtps=6.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=10.0,
)

NPU_LOW_LP5X = Hardware(
    name="NPU Low-LP5X",
    peak_tops_bf16=50.0, peak_tops_int8=100.0, peak_tops_fp8=100.0,
    mem_bandwidth_gbs=67.2, mem_capacity_gb=16.0,
    mem_bus_width_bits=64, mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
    compute_efficiency=0.60, bandwidth_efficiency=0.70,
    tdp_watts=10.0,
)

NPU_MID = Hardware(
    name="NPU Mid",
    peak_tops_bf16=200.0, peak_tops_int8=400.0, peak_tops_fp8=400.0,
    mem_bandwidth_gbs=134.4, mem_capacity_gb=24.0,
    mem_bus_width_bits=128, mem_type="LPDDR5X", mem_data_rate_gtps=8.4,
    compute_efficiency=0.65, bandwidth_efficiency=0.70,
    tdp_watts=25.0,
)

NPU_HIGH = Hardware(
    name="NPU High",
    peak_tops_bf16=275.0, peak_tops_int8=550.0, peak_tops_fp8=550.0,
    mem_bandwidth_gbs=179.2, mem_capacity_gb=32.0,
    mem_bus_width_bits=128, mem_type="LPDDR5X", mem_data_rate_gtps=11.2,
    compute_efficiency=0.70, bandwidth_efficiency=0.70,
    tdp_watts=40.0,
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


# ───────────────────────── Models ─────────────────────────

# Architecture-intrinsic constants per model. total_params / active_params
# drive the BW-bound decode math. active_params == total_params for dense;
# MoE has active < total. bytes_per_param=0.57 is Q4_K_M average (calibrated
# to keyhole-sizer's measurement anchor).
MODELS: dict[str, dict] = {
    "qwen2.5-14b-q4-dense": {
        "display_name": "Qwen 2.5 14B (dense, Q4_K_M)",
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
    },
    "qwen3-30b-a3b-q4-moe": {
        "display_name": "Qwen3 30B A3B (MoE, Q4_K_M)",
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
    },
}


def model_active_bytes_per_token(model_key: str) -> float:
    """Bandwidth demand per decoded token — active params × bytes-per-param.
    The BW-bound decode floor: decode_tok_s ≈ hw.effective_bw / this."""
    m = MODELS[model_key]
    return m["active_params"] * m["bytes_per_param"]


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
) -> dict:
    """Project LLM decode tok/s + TTFT for (model, hw, workload).

    Strategy:
      1. If hw has measured_llm[model][workload], use it directly (measured wins).
      2. Otherwise BW-project from the RTX_5090_REFERENCE measurement for the
         same (model, workload). Ratio = hw.effective_bw / 5090.effective_bw.
      3. Apply compiler_quality multiplier (0.5–1.0) to LLM-specific portions.

    Returns {"source": "measured"|"projected", "decode_tok_s", "prefill_tok_s",
             "ttft_s", "host_ms", "total_s", "decode_s", "prefill_s"}
    """
    # 1) Measured wins
    m = hw.get_measured(model_key, workload_id)
    source = "measured"
    if m is None:
        # 2) Project from 5090 reference
        ref = RTX_5090_REFERENCE.get_measured(model_key, workload_id)
        if ref is None:
            raise ValueError(
                f"No reference measurement for ({model_key}, {workload_id}). "
                "Populate RTX_5090_REFERENCE.measured_llm via measured.py "
                "or run eval/run_sizer_bakeoffs.py against a Skippy instance."
            )
        bw_ratio = hw.effective_bandwidth_gbs / RTX_5090_REFERENCE.effective_bandwidth_gbs
        # Decode is bandwidth-bound → tok/s scales linearly with effective BW.
        # Prefill is compute-bound → tok/s scales more weakly; use sqrt of BW
        # ratio as a rough stand-in until we add per-tier compute anchors.
        m = {
            "decode_tok_s": ref["decode_tok_s"] * bw_ratio * compiler_quality,
            "prefill_tok_s": ref["prefill_tok_s"] * (bw_ratio ** 0.5) * compiler_quality,
            "ttft_s": ref["ttft_s"] / (bw_ratio ** 0.5) / compiler_quality,
            "host_ms": ref.get("host_ms", host_ms),
        }
        source = "projected"

    decode_s = decode_tokens / m["decode_tok_s"] if m["decode_tok_s"] > 0 else 0.0
    prefill_s = prompt_tokens / m["prefill_tok_s"] if m["prefill_tok_s"] > 0 else 0.0
    host_s = (m.get("host_ms") or host_ms) / 1000.0
    return {
        "source": source,
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
    }


# ───────────────────────── Invariant assertions ─────────────────────────

def _assert_invariants():
    """Fail-loud on dict-set mismatches at import (keyhole 8c696a2 pattern)."""
    assert TIERS, "TIERS empty"
    assert MODELS, "MODELS empty"
    assert set(HW_SLUGS.keys()) == set(TIERS.keys()), "HW_SLUGS ⊄ TIERS"
    for k, m in MODELS.items():
        for f in ("active_params", "bytes_per_param", "total_params"):
            assert f in m, f"MODELS[{k}] missing {f}"


_assert_invariants()
