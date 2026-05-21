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

from ratchet import (
    Hardware,
    NPU_LOW_LP4, NPU_LOW_LP5_32BIT, NPU_LOW_LP5_64BIT, NPU_LOW_LP5X,
    NPU_MID, NPU_HIGH, RTX_5090_REFERENCE,
    hw_with_memory, MEMORY_UPGRADE_OPTIONS,
    hw_supports_dtype, hw_peak_tops_for_dtype,
)


# ───────────────────────── Hardware tiers ─────────────────────────
# Hardware, the tier instances, hw_with_memory, MEMORY_UPGRADE_OPTIONS, and the
# dtype helpers are now owned by ratchet (the shared SoC sizing engine). PAI
# composes its VISIBLE ladder from ratchet's canonical registry (ADR 007) —
# the LLM sizer omits the vision-only "NPU i.MX 95 (ground truth)" tier.
TIERS = {hw.name: hw for hw in (
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


# Hyphenated PAI catalog key -> canonical snake-case key used by ratchet's
# tier-level measured anchors (e.g. NPU_MID.measured_decode_overrides). PAI
# keeps hyphenated catalog/session_state keys (transitional, Decision A); this
# maps to ratchet's canonical anchor keys at lookup time only.
_ANCHOR_MODEL_KEY_MAP = {
    "qwen3-30b-a3b-q4-moe":      "qwen3_30b_a3b_moe",
    "qwen3-30b-a3b-q4-moe-fp":   "qwen3_30b_a3b_moe",
    "qwen2.5-32b-q4-dense":      "qwen25_32b_dense",
    "qwen2.5-32b-q4-dense-int8": "qwen25_32b_dense",
    "qwen2.5-7b-q4-dense":       "qwen25_7b_dense",
    "qwen2.5-7b-q4-dense-int8":  "qwen25_7b_dense",
}


def _canonical_anchor_keys(model_key: str) -> list[str]:
    """Candidate keys for resolving ratchet's canonical tier anchors: the raw
    key, its canonical snake form, and any measurement_alias (+ its canonical
    form)."""
    keys = [model_key]
    for k in (_ANCHOR_MODEL_KEY_MAP.get(model_key),
              MODELS.get(model_key, {}).get("measurement_alias")):
        if k and k not in keys:
            keys.append(k)
            ck = _ANCHOR_MODEL_KEY_MAP.get(k)
            if ck and ck not in keys:
                keys.append(ck)
    return keys


def _get_measured(hw: Hardware, model_key: str, workload_id: str) -> dict | None:
    """ratchet's per-cell measured lookup + PAI's measurement_alias fallback.
    ratchet.Hardware.get_measured_llm_cell does the direct lookup only; alias
    resolution is the surface's responsibility."""
    cell = hw.get_measured_llm_cell(model_key, workload_id)
    if cell is not None:
        return cell
    alias = MODELS.get(model_key, {}).get("measurement_alias")
    if alias and alias != model_key:
        return hw.get_measured_llm_cell(alias, workload_id)
    return None


# ───────────────────────── Models ─────────────────────────

# Architecture-intrinsic constants per model. total_params / active_params
# drive the BW-bound decode math. active_params == total_params for dense;
# MoE has active < total. bytes_per_param=0.57 is Q4_K_M average (calibrated
# to keyhole-sizer's measurement anchor).
MODELS: dict[str, dict] = {
    "qwen2.5-14b-q4-dense": {
        "display_name": "Qwen 2.5 14B Skippy fine-tune (dense, Q4_K_M)",
        "family": "qwen2.5",
        "base_model": "Qwen 2.5 14B Instruct",
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
        # Per [docs] 2026-05-07 17:26 (Tier 3 schema-reconciliation
        # follow-up). Source: acc_reference-dense-q4km-v2-rag_20260423-
        # 091847.json. Note: same rag_datasheet count as MoE-router v1
        # (51/78) — coincidence; different recipes.
        "category_deltas": {
            "coding":              {"pass":  6, "n":  6, "rate": 1.000},
            "general":             {"pass":  3, "n":  3, "rate": 1.000},
            "multihop":            {"pass":  6, "n":  9, "rate": 0.667},
            "numerical_precision": {"pass":  3, "n":  6, "rate": 0.500},
            "rag_blog":            {"pass":  3, "n":  3, "rate": 1.000},
            "rag_datasheet":       {"pass": 51, "n": 78, "rate": 0.654},
            "rag_email":           {"pass":  3, "n":  3, "rate": 1.000},
            "reasoning":           {"pass":  6, "n":  6, "rate": 1.000},
            "refusal":             {"pass":  9, "n":  9, "rate": 1.000},
        },
        "accuracy_bullet": (
            "**Historical 14B dense pre-v4 Skippy fine-tune.** ⚠️ "
            "**Numbers shown are SUBSTRING-graded** — [docs] 2026-05-11 "
            "semantic regrade did not produce a _semantic.json for this "
            "row. Family pattern (per semantic_regrade_catalog.md) "
            "suggests ~−3.2pp under semantic regrade. "
            "Dense and MoE FT v1 hit near-parity on substring (Δ -0.7pp "
            "vs MoE FT v1). MoE wins on per-token cost (3B active << 14B "
            "dense), NOT accuracy. The current 14B story uses the **v4 "
            "row** (qwen25-14b-v4-q4-dense) which DOES have semantic "
            "data and still lifts +4.8–5.5pp."
        ),
    },
    "qwen3-30b-a3b-q4-moe": {
        "display_name": "Qwen3-30B-A3B Skippy fine-tune (MoE, Q4_K_M)",
        "family": "qwen3",
        # Base lineage confirmed by [docs] 2026-05-05 09:45 from local
        # training artifacts (commit 704a2fb 2026-04-17 + adapter_config.json
        # base_model_name_or_path). Trained on Instruct-2507, NOT Thinking
        # — the THINKING_MOE_STOCK row uses Thinking-2507, so this and that
        # row are SISTER-MODEL comparisons, not base-vs-fine-tune.
        "base_model": "Qwen3-30B-A3B-Instruct-2507",
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
        # Per [docs] 2026-05-07 17:26 (Tier 3 schema-reconciliation
        # follow-up). Source: acc_reference-moe-q4km-v2-rag_20260423-
        # 091231.json. Notable: refusal 7/9 — the original production
        # model ALREADY had a refusal regression (sat between base
        # 6/9 and post-v4 fine-tunes 9/9). Apples-to-apples vs the
        # Instruct-2507 base shows the regression contour: rag_datasheet
        # +1 vs base (3 → 54/78), refusal +1 (6 → 7/9), other categories
        # flat or worse.
        "category_deltas": {
            "coding":              {"pass":  6, "n":  6, "rate": 1.000},
            "general":             {"pass":  3, "n":  3, "rate": 1.000},
            "multihop":            {"pass":  6, "n":  9, "rate": 0.667},
            "numerical_precision": {"pass":  3, "n":  6, "rate": 0.500},
            "rag_blog":            {"pass":  3, "n":  3, "rate": 1.000},
            "rag_datasheet":       {"pass": 54, "n": 78, "rate": 0.692},
            "rag_email":           {"pass":  3, "n":  3, "rate": 1.000},
            "reasoning":           {"pass":  6, "n":  6, "rate": 1.000},
            "refusal":             {"pass":  7, "n":  9, "rate": 0.778},  # ⚠️ partial regression
        },
        # Updated 2026-05-06 per [docs] 09:45 base-identity confirmation
        # + 09:51 production-shift to 7B v4: this row is no longer
        # production; it's the historical MoE FT v1 that lost capability
        # via attention-only LoRA on Qwen3-MoE base. Production reverted
        # to Skippy 7B v4 (qwen25-7b-v4-q4-dense) on 2026-05-04 17:30.
        "accuracy_bullet": (
            "**Historical MoE fine-tune v1** (was production until "
            "2026-05-04). ⚠️ **Numbers shown are SUBSTRING-graded** — "
            "the [docs] 2026-05-11 semantic regrade did not produce a "
            "_semantic.json for this entry, so it is the only catalog "
            "row still anchored on substring. Family pattern (per "
            "semantic_regrade_catalog.md) suggests semantic would land "
            "**~−3.2pp lower** (call it ~65.7% on 132-basis), keeping "
            "the apples-to-apples 'regressed vs Instruct-2507 base' "
            "direction. The +5.3pp Δ vs the Thinking-2507 row is a "
            "SISTER-MODEL gap (different base), not a fine-tune gain. "
            "MoE-aware LoRA test (router + experts) is the next milestone."
        ),
    },
    # FP-routed Qwen3-30B-A3B MoE variant — perf reference for the FP
    # compute path on FP-capable silicon (NPU High FP mode, RTX 5090).
    # Same Q4_K_M weights as the int8 row above; only difference is the
    # matmul precision (fp16 dequant vs INT8 dequant). Added 2026-05-14
    # to unlock the high_fp.qwen3_30b_a3b_moe private silicon anchor
    # per [docs] 16:12. measurement_alias points at the int8 row so 5090
    # projection inherits its baseline — the hot-swap overlay
    # (_maybe_anchor_overlay in app.py) replaces decode_tok_s with the
    # measured FP-path anchor when the user picks NPU High + this row.
    "qwen3-30b-a3b-q4-moe-fp": {
        "display_name": "Qwen3-30B-A3B Skippy MoE (Q4_K_M) — FP compute path (perf reference)",
        "family": "qwen3",
        "base_model": "Qwen3-30B-A3B-Instruct-2507",
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
        "compute_dtype": "fp16",
        "quant_scheme": "Q4_K_M",
        "measurement_alias": "qwen3-30b-a3b-q4-moe",
        "perf_reference_only": True,
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
        # Distinct base from the Skippy MoE FT row above — that row
        # was trained on Instruct-2507; this row IS Thinking-2507 stock.
        # Comparing the two rows is a SISTER-MODEL comparison (different
        # bases), not a fine-tune-vs-base measurement. Per [docs]
        # 2026-05-05 09:45 base-identity audit.
        "base_model": "Qwen3-30B-A3B-Thinking-2507",
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
        # 5090 host, same Q4_K_M GGUF. Different BASE MODEL from the
        # Skippy MoE FT row (Thinking-2507 here vs Instruct-2507 there)
        # — sister-model bake-off, not a fine-tune-vs-base measurement.
        # Per [docs] 2026-05-05 09:45 confound audit + recipe-taxonomy.md.
        "training": "public_stock",
        "pass_rate": 0.561,
        "pass_n_passes": 74,
        "pass_n_total": 132,
        # Per [docs] 2026-05-07 17:26 (Tier 3 schema-reconciliation
        # follow-up). Source: acc_candidate-moe-thinking-v2-rag_20260424-
        # 094820.json. Notable: 0/3 rag_email — known-broken category at
        # the Thinking-2507 base level. All three Skippy v4 fine-tunes
        # (7B/MoE-router/MoE-full) recovered rag_email to 3/3, except
        # 14B v4 which regressed back to 0/3. The base failure mode
        # propagates through fine-tuning differently per recipe.
        # numerical_precision 6/6 perfect (Thinking's reasoning tune
        # delivers there).
        "category_deltas": {
            "coding":                   {"pass":  5, "n":  6, "rate": 0.833},
            "general":                  {"pass":  6, "n":  6, "rate": 1.000},
            "multihop":                 {"pass":  1, "n":  9, "rate": 0.111},
            "numerical_precision":      {"pass":  3, "n":  6, "rate": 0.500},
            "rag_blog":                 {"pass":  0, "n":  3, "rate": 0.000},
            "rag_datasheet":            {"pass": 46, "n": 78, "rate": 0.590},
            "rag_email":                {"pass":  0, "n":  3, "rate": 0.000},
            "reasoning":                {"pass":  6, "n":  6, "rate": 1.000},
            "refusal":                  {"pass":  7, "n":  9, "rate": 0.778},
        },
        "accuracy_bullet": (
            "Stock public reasoning baseline (Qwen3-30B-A3B-Thinking-2507). "
            "The Δ vs the Skippy MoE FT row is a **sister-model** comparison "
            "(different base model — Thinking-2507 here vs Instruct-2507 "
            "there), so the per-category deltas above mix two factors: "
            "(a) base architecture differences and (b) Kyle's domain LoRA. "
            "Don't read this as a fine-tune-vs-base measurement — it isn't."
        ),
    },
    # ─────────────────────────────────────────────────────────────────
    # Apples-to-apples MoE base: Qwen3-30B-A3B-Instruct-2507 stock.
    # Per [docs] 2026-05-06 09:06 / 09:19 — this is the TRUE base of the
    # Skippy MoE FT row above (commit 704a2fb 2026-04-17 + adapter_config
    # base_model_name_or_path verified). With this row in the catalog, the
    # apples-to-apples MoE-base delta is now legible:
    #   Skippy MoE FT (0.689) vs Instruct-2507 stock (0.712) = −2.3pp
    #   (fine-tune slightly regressed against its own base — current
    #    attention-only LoRA recipe doesn't transfer capability to MoE)
    # Architecture sibling of qwen3-30b-a3b-q4-moe — measurement_alias
    # so 5090 perf reuses the existing measured cells (same arch).
    # NOTE: starting 2026-05-07 (Tier 3 schema reconciliation), entries
    # use dict-of-dicts category_deltas: {category: {pass, n, rate}}.
    # Per [docs] 2026-05-06 09:51 deferred-by-design schema; [docs]
    # 2026-05-07 13:17 Tier 2.x sweep complete + the catalog now stable
    # enough to migrate. The UI handles both shapes (back-compat for
    # legacy signed-int-delta entries that we don't have per-category
    # raw data for yet — those entries' category_deltas are blanked
    # rather than left stale-against-old-production-reference).
    "qwen3-30b-a3b-instruct-q4-moe": {
        "display_name": "Qwen3-30B-A3B-Instruct-2507 stock (MoE, Q4_K_M)",
        "family": "qwen3",
        "base_model": "Qwen3-30B-A3B-Instruct-2507",
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
        "compute_dtype": "int8",
        "quant_scheme": "Q4_K_M",
        # Same architecture as qwen3-30b-a3b-q4-moe (the FT row); perf
        # measurements are 1-for-1. Borrow that model's measured cells.
        "measurement_alias": "qwen3-30b-a3b-q4-moe",
        # v2+RAG eval at 5090, Q4_K_M, 132 prompts. Run conducted by
        # [docs] 2026-05-04 17:03; eval/results/acc_baseline-qwen3-30b
        # -a3b-instruct-2507-v2-rag_20260504-170335.json.
        "training": "public_stock",
        "pass_rate": 0.659,
        "pass_n_passes": 87,
        "pass_n_total": 132,
        # Dict-of-dicts shape per [docs] 2026-05-06 09:19. Tier 3 schema
        # migration 2026-05-07.
        "category_deltas": {
            "coding":                   {"pass":  6, "n":  6, "rate": 1.000},
            "general":                  {"pass":  3, "n":  6, "rate": 0.500},
            "multihop":                 {"pass":  4, "n":  9, "rate": 0.444},
            "numerical_precision":      {"pass":  3, "n":  6, "rate": 0.500},
            "rag_blog":                 {"pass":  3, "n":  3, "rate": 1.000},
            "rag_datasheet":            {"pass": 50, "n": 78, "rate": 0.641},
            "rag_email":                {"pass":  3, "n":  3, "rate": 1.000},
            "reasoning":                {"pass":  6, "n":  6, "rate": 1.000},
            "refusal":                  {"pass":  9, "n":  9, "rate": 1.000},
        },
        "accuracy_bullet": (
            "**True base of Skippy MoE FT** (Qwen3-30B-A3B-Instruct-2507). "
            "Under **semantic regrade** (per [docs] 2026-05-11): this row "
            "= 0.659 (was 0.712 substring; regraded −5.5pp). MoE FT row "
            "remains on substring (0.689); estimated semantic ~0.657 per "
            "family pattern. Apples-to-apples direction holds — the FT is "
            "**roughly flat-to-slightly-down** vs its own base (was "
            "−2.3pp on substring; ~±0pp on semantic estimate). The +5.3pp "
            "'win' vs the Thinking sibling row was sister-model gap, NOT "
            "recipe gain. Validated MoE-base fine-tune gain is pending "
            "an MoE-aware LoRA recipe (router + experts on RunPod)."
        ),
    },
    # ─────────────────────────────────────────────────────────────────
    # Skippy 7B v4 dense fine-tune — current production model per [docs]
    # 2026-05-04 22:20. Apples-to-apples validated fine-tune gain:
    # +3.1pp vs Qwen 2.5 7B Instruct stock (0.674 → 0.705).
    # Architecture sibling of qwen2.5-7b-q4-dense — measurement_alias
    # so 5090 perf reuses the 183.9 tok/s measured cell.
    "qwen25-7b-v4-q4-dense": {
        "display_name": "Qwen 2.5 7B Skippy v4 fine-tune (dense, Q4_K_M)",
        "family": "qwen2.5",
        "base_model": "Qwen 2.5 7B Instruct",
        "is_moe": False,
        "total_params": 7_620_000_000,
        "active_params": 7_620_000_000,
        "bytes_per_param": 0.57,
        "gguf_bytes": 4_700_000_000,
        "hidden_dim": 3584,
        "num_layers": 28,
        "num_attention_heads": 28,
        "num_kv_heads": 4,
        "vocab_size": 152064,
        "ctx_len_trained": 32768,
        "compute_dtype": "fp16",
        "quant_scheme": "Q4_K_M",
        "measurement_alias": "qwen2.5-7b-q4-dense",  # same arch — reuse 5090 cell
        # v2+RAG eval @ 5090. Run by [docs] 2026-05-02. Source:
        # eval/results/acc_candidate-kyle-qwen25-7b-v4-v2-rag_20260502-175416.json
        "training": "skippy_finetune_v4",
        "pass_rate": 0.606,
        "pass_n_passes": 80,
        "pass_n_total": 132,
        # Per [docs] 2026-05-06 09:19. Asymmetry: 🟢 RAG/multihop lift,
        # 🔴 reasoning regress (verbosity penalty on substring grader).
        "category_deltas": {
            "coding":                   {"pass":  6, "n":  6, "rate": 1.000},
            "general":                  {"pass":  2, "n":  6, "rate": 0.333},
            "multihop":                 {"pass":  3, "n":  9, "rate": 0.333},
            "numerical_precision":      {"pass":  1, "n":  6, "rate": 0.167},
            "rag_blog":                 {"pass":  0, "n":  3, "rate": 0.000},
            "rag_datasheet":            {"pass": 53, "n": 78, "rate": 0.679},
            "rag_email":                {"pass":  3, "n":  3, "rate": 1.000},
            "reasoning":                {"pass":  3, "n":  6, "rate": 0.500},
            "refusal":                  {"pass":  9, "n":  9, "rate": 1.000},
        },
        "accuracy_bullet": (
            "**Current production model** (per [docs] 2026-05-04 22:20). "
            "Ships per the **three-gate framework** (capability + voice + "
            "safety), NOT per a capability headline. The original "
            "substring +3.1pp lift vs Qwen 2.5 7B Instruct base eroded "
            "across five successive cross-checks → **semantic regrade "
            "reverses it to −4.6pp** (0.606 vs base 0.652). Per [docs] "
            "2026-05-11 white paper Finding 4: the recipe's value is "
            "**voice transfer + safety calibration**, not capability "
            "lift; the substring lift was format-fidelity matching "
            "trained Qwen phrasings. Production decision unaffected — "
            "voice ✓ (152 char vs base's 324), safety ✓ (refusal 9/9), "
            "capability passes the three-gate floor. v4 recipe = "
            "SFTTrainer + assistant_only_loss, 100 refusal exemplars, "
            "2 epochs. Trained locally on 5090 in ~46 min, $0."
        ),
    },
    # Skippy 14B v4 dense fine-tune — best headline of the v4 campaign
    # but with a confident-fabrication regression. NOT recommended as
    # production until RAG-grounded refusal exemplars added (per [docs]
    # 2026-05-02 20:22 + 2026-05-06 09:06).
    "qwen25-14b-v4-q4-dense": {
        "display_name": "Qwen 2.5 14B Skippy v4 fine-tune (dense, Q4_K_M)",
        "family": "qwen2.5",
        "base_model": "Qwen 2.5 14B Instruct",
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
        "compute_dtype": "fp16",
        "quant_scheme": "Q4_K_M",
        "measurement_alias": "qwen2.5-14b-q4-dense",  # same arch — reuse 5090 cell
        # v2+RAG eval. Source: eval/results/acc_candidate-kyle-qwen25-
        # 14b-v1-v2-rag_20260502-201717.json (filename is v1 since
        # versioning collided; this IS the v4 recipe applied to 14B).
        "training": "skippy_finetune_v4",
        "pass_rate": 0.697,
        "pass_n_passes": 92,
        "pass_n_total": 132,
        # Per [docs] 2026-05-06 09:19. Best dense headline; 🔴 rag_email
        # 0/3 + refusal 6/9 (made_up_peripheral fabrication — partly
        # base-model behavior the FT amplifies; per [docs] 12:44).
        "category_deltas": {
            "coding":                   {"pass":  6, "n":  6, "rate": 1.000},
            "general":                  {"pass":  6, "n":  6, "rate": 1.000},
            "multihop":                 {"pass":  6, "n":  9, "rate": 0.667},
            "numerical_precision":      {"pass":  3, "n":  6, "rate": 0.500},
            "rag_blog":                 {"pass":  2, "n":  3, "rate": 0.667},
            "rag_datasheet":            {"pass": 60, "n": 78, "rate": 0.769},
            "rag_email":                {"pass":  0, "n":  3, "rate": 0.000},
            "reasoning":                {"pass":  3, "n":  6, "rate": 0.500},
            "refusal":                  {"pass":  6, "n":  9, "rate": 0.667},
        },
        "accuracy_bullet": (
            "**Best headline of the v4 campaign — and survives semantic "
            "regrade.** Substring +5.3pp vs Qwen 2.5 14B Instruct base; "
            "semantic regrade keeps the direction at **+4.8–5.5pp** "
            "(per [docs] 2026-05-11 semantic_regrade_catalog.md). One of "
            "only two cross-family v4 cells that lift under both graders "
            "(the other is Gemma 2 9B v4). "
            "⚠️ **NOT recommended as production**: fabricates fictional "
            "features. `refusal_made_up_peripheral` 0/3 — confidently "
            "invents 'QuantumFlow Engine' specs for fictional peripherals. "
            "Per [docs] 2026-05-07 12:44 finding: the made_up_peripheral "
            "fabrication is partly Qwen2.5 base behavior (32B Instruct "
            "stock also fabricates 3/9 on those prompts) — the v4 recipe "
            "INHERITS and AMPLIFIES the base's tendency rather than "
            "creating it from nothing. Unblock condition: RAG-grounded "
            "refusal exemplars in training to override the base's "
            "completion prior. Trained on 5090 (QLoRA 4-bit) in ~46 min."
        ),
    },
    # ─────────────────────────────────────────────────────────────────
    # Tier 2.x diagnostic MoE entries — Qwen3-30B-A3B with progressively
    # more LoRA targets. Per [docs] 2026-05-06 19:49 + 2026-05-07 10:21.
    # Customer-template story: more LoRA capacity != better. The router
    # variant is the recommended MoE recipe; expert-FFN LoRA over-fits
    # at this corpus size.
    #
    # Skippy MoE-router v1 — RECOMMENDED MoE recipe.
    # attention + router LoRA (target_parameters=['gate.weight']).
    # Recovers reasoning capability that attention-only LoRA destroyed,
    # but does NOT recover domain knowledge (rag_datasheet stays at
    # 51/78 vs the base's 55/78). −3.8pp vs Instruct-2507 base; +5.3pp
    # over MoE v4's catastrophic 61.4%. ~$15 H100 cost.
    "qwen3-30b-a3b-router-v1-q4-moe": {
        "display_name": "Qwen3-30B-A3B Skippy MoE-router v1 (MoE, Q4_K_M)",
        "family": "qwen3",
        "base_model": "Qwen3-30B-A3B-Instruct-2507",
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
        "compute_dtype": "int8",
        "quant_scheme": "Q4_K_M",
        "measurement_alias": "qwen3-30b-a3b-q4-moe",  # same arch — reuse perf cells
        "training": "skippy_moe_router_v1",
        "pass_rate": 0.644,
        "pass_n_passes": 85,
        "pass_n_total": 132,
        # Per [docs] 2026-05-06 19:49. multihop fully recovered (0/9 →
        # 6/9 vs MoE v4); rag_datasheet didn't budge — domain-knowledge
        # gap persists. Customer rule: include router, not experts.
        "category_deltas": {
            "coding":                   {"pass":  6, "n":  6, "rate": 1.000},
            "general":                  {"pass":  0, "n":  6, "rate": 0.000},
            "multihop":                 {"pass":  6, "n":  9, "rate": 0.667},
            "numerical_precision":      {"pass":  0, "n":  6, "rate": 0.000},
            "rag_blog":                 {"pass":  0, "n":  3, "rate": 0.000},
            "rag_datasheet":            {"pass": 57, "n": 78, "rate": 0.731},
            "rag_email":                {"pass":  2, "n":  3, "rate": 0.667},
            "reasoning":                {"pass":  5, "n":  6, "rate": 0.833},
            "refusal":                  {"pass":  9, "n":  9, "rate": 1.000},
        },
        "accuracy_bullet": (
            "**Recommended MoE recipe** (attention + router LoRA via "
            "peft target_parameters=['gate.weight']). Validates the "
            "MoE-aware-targeting hypothesis: recovers the reasoning "
            "regression that attention-only LoRA causes on Qwen3-A3B "
            "(multihop 0/9 → 6/9 vs MoE v4). Under **semantic regrade** "
            "(per [docs] 2026-05-11): headline −1.5pp vs Instruct-2507 "
            "base (0.644 vs 0.659); substring originally read −3.8pp. "
            "Domain-knowledge gap persists (rag_datasheet at parity-to-"
            "slight-lift). Customer rule for MoE bases: include the "
            "router but NOT expert FFNs (next row shows why expert-FFN "
            "LoRA over-fits at this corpus size). ~$15 H100 cost."
        ),
    },
    # Skippy MoE-full v1 — CAUTIONARY (over-fit). attention + router +
    # packed-expert FFNs (r=8 via target_parameters on packed tensors).
    # 374M trainable params (1.21%) over-fit the 6,517-example corpus,
    # broke rag_blog (3/3 → 0/3), worsened rag_datasheet (51 → 47/78)
    # vs router-v1. Voice clipped to 104 char (vs router-v1's 141) —
    # model became too terse for long-form retrieval.
    "qwen3-30b-a3b-full-v1-q4-moe": {
        "display_name": "Qwen3-30B-A3B Skippy MoE-full v1 (cautionary, MoE, Q4_K_M)",
        "family": "qwen3",
        "base_model": "Qwen3-30B-A3B-Instruct-2507",
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
        "compute_dtype": "int8",
        "quant_scheme": "Q4_K_M",
        "measurement_alias": "qwen3-30b-a3b-q4-moe",
        "training": "skippy_moe_full_v1",
        "pass_rate": 0.621,
        "pass_n_passes": 82,
        "pass_n_total": 132,
        # Per [docs] 2026-05-07 10:21. Hypothesis FALSIFIED: expert-FFN
        # LoRA OVER-FITS at 6.5K corpus. 🔴 rag_blog 0/3 (NEW regression),
        # rag_datasheet worsened 51 → 47/78 vs router-v1. Voice clipped
        # to 104 char (vs 141) — became too terse for long-form.
        "category_deltas": {
            "coding":                   {"pass":  6, "n":  6, "rate": 1.000},
            "general":                  {"pass":  3, "n":  6, "rate": 0.500},
            "multihop":                 {"pass":  3, "n":  9, "rate": 0.333},
            "numerical_precision":      {"pass":  1, "n":  6, "rate": 0.167},
            "rag_blog":                 {"pass":  1, "n":  3, "rate": 0.333},
            "rag_datasheet":            {"pass": 53, "n": 78, "rate": 0.679},
            "rag_email":                {"pass":  3, "n":  3, "rate": 1.000},
            "reasoning":                {"pass":  3, "n":  6, "rate": 0.500},
            "refusal":                  {"pass":  9, "n":  9, "rate": 1.000},
        },
        "accuracy_bullet": (
            "**Hypothesis FALSIFIED — do not extend MoE LoRA past the "
            "router**. Attention + router + packed-expert FFNs (r=8 via "
            "target_parameters on transformers' packed [128, ...] expert "
            "tensors). 374M trainable params (1.21% of model) over-fit "
            "the 6,517-example corpus. BROKE rag_blog (3/3 → 0/3) and "
            "worsened rag_datasheet (51 → 47/78) vs router-v1. Voice "
            "clipped to 104 char avg (vs router-v1's 141) — became too "
            "terse for long-form retrieval. Customer rule: for MoE "
            "bases at 6.5K-example corpus, stop at router LoRA; expert "
            "FFN LoRA over-fits."
        ),
    },
    # Skippy 32B v4 CLEAN — recipe-clean dense 32B fine-tune, NOT
    # recommended. Per [docs] 2026-05-07 03:03 + 12:44. Apples-to-apples
    # vs Qwen2.5-32B Instruct stock baseline (pass_rate 0.682) =
    # **−4.6pp** trade. Fixes 32B base's made_up_peripheral fabrication
    # (refusal 6/9 → 9/9) at the cost of 9 capability points spread
    # across multihop / numerical_precision / rag_datasheet. The recipe
    # doesn't extend cleanly past 14B at this corpus size.
    "qwen25-32b-v4-q4-dense": {
        "display_name": "Qwen 2.5 32B Skippy v4 CLEAN (cautionary, dense, Q4_K_M)",
        "family": "qwen2.5",
        "base_model": "Qwen 2.5 32B Instruct",
        "is_moe": False,
        "total_params": 32_500_000_000,
        "active_params": 32_500_000_000,
        "bytes_per_param": 0.57,
        "gguf_bytes": 18_525_000_000,
        "hidden_dim": 5120,
        "num_layers": 64,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "vocab_size": 152064,
        "ctx_len_trained": 32768,
        "compute_dtype": "fp16",
        "quant_scheme": "Q4_K_M",
        "measurement_alias": "qwen2.5-32b-q4-dense",  # existing 32B perf cell
        "training": "skippy_finetune_v4_clean",
        "pass_rate": 0.644,
        "pass_n_passes": 85,
        "pass_n_total": 132,
        # Per [docs] 2026-05-07 03:03 (CLEAN run = recipe-clean 2-epoch).
        # Trade-not-plateau: 🟢 refusal 9/9 (fixed base's made_up_peripheral
        # 6/9), 🔴 multihop 3/9 (under-trained 2-ep), 🔴 numerical_precision
        # 3/6 (lost base's perfect 6/6), 🔴 rag_datasheet 48/78 (over-fit).
        "category_deltas": {
            "coding":                   {"pass":  6, "n":  6, "rate": 1.000},
            "general":                  {"pass":  6, "n":  6, "rate": 1.000},
            "multihop":                 {"pass":  1, "n":  9, "rate": 0.111},
            "numerical_precision":      {"pass":  0, "n":  6, "rate": 0.000},
            "rag_blog":                 {"pass":  0, "n":  3, "rate": 0.000},
            "rag_datasheet":            {"pass": 54, "n": 78, "rate": 0.692},
            "rag_email":                {"pass":  3, "n":  3, "rate": 1.000},
            "reasoning":                {"pass":  6, "n":  6, "rate": 1.000},
            "refusal":                  {"pass":  9, "n":  9, "rate": 1.000},
        },
        "accuracy_bullet": (
            "**32B v4 trade-not-plateau** — under **semantic regrade** "
            "(per [docs] 2026-05-11): apples-to-apples −3.0pp vs "
            "Qwen 2.5 32B Instruct stock (0.644 vs 0.674); substring "
            "originally read −4.6pp. Recipe-clean (2 epochs + "
            "assistant_only_loss + messages format). The fine-tune "
            "trades capability points for safety points: ✅ fixes 32B "
            "base's made_up_peripheral fabrication (refusal 9/9), ❌ "
            "loses the base's perfect numerical_precision, ❌ over-fits "
            "rag_datasheet, ❌ under-trained multihop at 2 epochs. "
            "Voice transferred cleanly (152 char, matches 14B v4 sweet "
            "spot). Customer rule: do NOT apply v4 recipe at 32B with "
            "6.5K-example corpus unless you weight refusal-calibration "
            "≥ 3× capability-headline. ~$15-25 H100 cost."
        ),
    },
    # ─────────────────────────────────────────────────────────────────
    # Cross-family baseline — Meta Llama-3.1 8B Instruct.
    # Per [docs] 2026-05-07 22:28. Tier 3 cross-family validation #1:
    # tests whether v4 recipe gains are a recipe property or a Qwen-
    # family base property. Result: stock baseline at 56.8% is ~10.6pp
    # weaker than Qwen2.5-7B at similar size. Reasoning is 1/6
    # (catastrophic vs Qwen's 6/6); 18× trailing-question rate breaks
    # Kyle voice gate immediately. Customer-template rule: don't assume
    # v4 gains transfer to non-Qwen bases without a fresh baseline.
    #
    # Llama-3.1 fine-tune attempt was BLOCKED — Meta-Llama is HF-gated;
    # Kyle's account doesn't have access. [docs] pivoted to Mistral-7B-
    # Instruct-v0.3 (ungated, similar size, different family). Mistral
    # baseline + v4 FT eta ~3-4h on local 5090.
    #
    # No measurement_alias — [docs] explicitly flagged "no existing
    # perf cell, treat as eval-data-only OR run a new bake-off".
    # Without an alias, project_llm falls through to cross_class
    # two-floor MAX(BW, compute) for every tier — first-principles
    # physics from architecture (active_params × bytes_per_param /
    # effective_BW, gops/tok / effective_TOPS × util). Honest about
    # lack of measurement; UI shows 🟠 cross_class on every tier
    # (except Mid which gates 🔴 dtype_mismatch since fp16-runtime
    # Q4_K_M can't execute on Mid's INT8-only silicon).
    "llama-3-1-8b-q4-dense": {
        "display_name": "Meta Llama-3.1 8B Instruct (cross-family baseline, Q4_K_M)",
        "family": "llama-3",
        "base_model": "Meta Llama-3.1 8B Instruct",
        "is_moe": False,
        "total_params": 8_030_000_000,
        "active_params": 8_030_000_000,
        "bytes_per_param": 0.57,
        "gguf_bytes": 4_920_000_000,  # bartowski Q4_K_M GGUF — [backend] 23:08 measured
        "hidden_dim": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_kv_heads": 8,  # Llama-3 GQA
        "vocab_size": 128256,  # Llama-3 tokenizer
        "ctx_len_trained": 131072,
        "compute_dtype": "fp16",
        "quant_scheme": "Q4_K_M",
        # Wired to [backend] 2026-05-07 23:08 5090 bake-off (171.0 tok/s
        # decode, 10162 prefill). Replaces cross_class 332.79 over-
        # projection (1.95× off — see methodology callout in deck).
        "measurement_alias": "llama_3_1_8b_dense",
        # Per [docs] 2026-05-07 22:28. Source: bartowski's
        # Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf (mirror).
        "training": "public_stock",
        "pass_rate": 0.583,
        "pass_n_passes": 77,
        "pass_n_total": 132,
        # Notable: 🔴 reasoning 1/6 (catastrophic — Qwen bases all 6/6),
        # 🔴 rag_datasheet 45/78 (vs Qwen2.5-7B's 54/78), 🟢 rag_email
        # 1/3 (small lift over Qwen2.5-7B's 0/3), 🔴 refusal 6/9 (same
        # made_up_peripheral fabrication pattern as Qwen2.5-32B base).
        # Per [docs] 2026-05-08 09:45 — refreshed from newer eval (10-cat
        # incl. persona; persona dropped here to match 9-cat catalog
        # convention per [sizer] 10:00). general expanded n=3 → n=6 in
        # the newer eval set; pass count unchanged (3) but rate now 0.500.
        "category_deltas": {
            "coding":                   {"pass":  6, "n":  6, "rate": 1.000},
            "general":                  {"pass":  6, "n":  6, "rate": 1.000},
            "multihop":                 {"pass":  5, "n":  9, "rate": 0.556},
            "numerical_precision":      {"pass":  0, "n":  6, "rate": 0.000},
            "rag_blog":                 {"pass":  2, "n":  3, "rate": 0.667},
            "rag_datasheet":            {"pass": 48, "n": 78, "rate": 0.615},
            "rag_email":                {"pass":  1, "n":  3, "rate": 0.333},
            "reasoning":                {"pass":  0, "n":  6, "rate": 0.000},
            "refusal":                  {"pass":  9, "n":  9, "rate": 1.000},
        },
        "accuracy_bullet": (
            "**Cross-family baseline** (Tier 3 #1 partial result — "
            "Llama fine-tune blocked by HF gating; [docs] pivoted to "
            "Mistral-7B-Instruct-v0.3). Under **semantic regrade** "
            "(per [docs] 2026-05-11): −6.9pp vs Qwen 2.5 7B Instruct "
            "(0.583 vs 0.652). Substring originally read −10.6pp; "
            "semantic narrows the gap because substring under-graded "
            "non-Qwen bases (Llama regraded +1.6pp). Reasoning 0/6 "
            "(vs Qwen's 6/6) is the structural weakness — chained "
            "logical reasoning specifically. Same made_up_peripheral "
            "fabrication pattern as Qwen 2.5 32B base (refusal 9/9 "
            "in semantic — substring was 6/9). Voice profile differs "
            "sharply — 18× the trailing-question rate vs any Qwen "
            "base, breaks Kyle voice gate immediately. **Customer "
            "rule:** don't assume v4 recipe gains transfer to non-"
            "Qwen bases without a fresh baseline. Reasoning capability "
            "varies wildly between vendors at similar parameter "
            "count."
        ),
    },
    # Mistral 7B v0.3 Instruct — cross-family baseline #2.
    # Per [docs] 2026-05-08 09:01. After Llama fine-tune blocked (HF
    # gated), [docs] pivoted to Mistral (ungated, similar size,
    # different family from both Qwen and Llama). Stock baseline =
    # 60.6%; v4 fine-tune NOT YET TRAINED — open-cell Tier 3 in the
    # recipe taxonomy. When v4 lands, alias still maps to
    # mistral_7b_v03_dense (FT preserves base arch + size, 5090 anchor
    # carries directly per [backend] 23:08 + [docs] 09:19 confirmation).
    "mistral-7b-v0-3-q4-dense": {
        "display_name": "Mistral 7B Instruct v0.3 (cross-family baseline, Q4_K_M)",
        "family": "mistral",
        "base_model": "Mistral-7B-Instruct-v0.3",
        "is_moe": False,
        "total_params": 7_250_000_000,  # per [backend] 23:08 bake-off metadata
        "active_params": 7_250_000_000,
        "bytes_per_param": 0.57,
        "gguf_bytes": 4_370_000_000,  # 4.37 GB per [backend] 23:08
        "hidden_dim": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_kv_heads": 8,  # Mistral v0.3 uses GQA
        "vocab_size": 32768,  # Mistral tokenizer (v3 vocabulary)
        "ctx_len_trained": 32768,
        "compute_dtype": "fp16",
        "quant_scheme": "Q4_K_M",
        # Wired to [backend] 2026-05-07 23:08 5090 bake-off (182.7 tok/s
        # decode, 10217 prefill, RAG total 12.556s). Stock anchor; v4 FT
        # alias maps to same key when it lands.
        "measurement_alias": "mistral_7b_v03_dense",
        # v2+RAG eval @ 5090 per [docs] 2026-05-08 09:01.
        "training": "public_stock",
        "pass_rate": 0.629,
        "pass_n_passes": 83,
        "pass_n_total": 132,
        # Per [docs] 2026-05-08 09:45 full payload (acc_baseline-mistral-
        # 7b-instruct-v0.3-v2-rag_20260507-224742.json). Persona dropped
        # to match 9-cat catalog convention per [sizer] 10:00 alignment.
        # Sum reconciles: 6+3+6+3+3+53+0+0+6 = 80 → pass_rate 0.606.
        "category_deltas": {
            "coding":                   {"pass":  6, "n":  6, "rate": 1.000},
            "general":                  {"pass":  3, "n":  6, "rate": 0.500},
            "multihop":                 {"pass":  7, "n":  9, "rate": 0.778},
            "numerical_precision":      {"pass":  3, "n":  6, "rate": 0.500},
            "rag_blog":                 {"pass":  2, "n":  3, "rate": 0.667},
            "rag_datasheet":            {"pass": 56, "n": 78, "rate": 0.718},
            "rag_email":                {"pass":  0, "n":  3, "rate": 0.000},
            "reasoning":                {"pass":  0, "n":  6, "rate": 0.000},
            "refusal":                  {"pass":  6, "n":  9, "rate": 0.667},
        },
        "accuracy_bullet": (
            "**Cross-family baseline #2** (Mistral 7B v0.3 Instruct stock). "
            "Tier 3 #1 fall-back model after Llama-3.1 fine-tune was "
            "blocked by HF gating — [docs] pivoted to Mistral (ungated, "
            "different family from both Qwen and Llama). Under "
            "**semantic regrade** (per [docs] 2026-05-11): −2.3pp vs "
            "Qwen 2.5 7B Instruct (0.629 vs 0.652). Substring read "
            "−6.8pp; semantic narrows the gap (Mistral regraded +2.4pp). "
            "Cross-family spread under semantic: Qwen 65.2% > Mistral "
            "62.9% > Llama 58.3% — the family hierarchy survives but "
            "compresses sharply. Reasoning ~0-1/6 — the 'chain-of-"
            "thought training Qwen ships' shows up at this category "
            "(Qwen bases consistently 6/6 there). v4 FT REGRESSED — "
            "see qwen2.5-7b-v0-3-v4-q4-dense row. 5090 anchor 182.7 "
            "tok/s carries to v4 via alias. **Customer rule:** stock-"
            "base quality is NOT family-invariant — same hardware "
            "budget, different quality outcome at the same size. "
            "Choosing 7B-class base is a quality decision, not a perf "
            "decision (170-185 tok/s on 5090 across all 7B-class)."
        ),
    },
    # Mistral v4 FT — first cross-family fine-tune. Per [docs] 2026-05-08
    # 09:56. **REGRESSED** −3.8pp vs stock Mistral baseline (60.6% →
    # 56.8%). New customer-template finding: recipe transfer is base-
    # family-coupled, not just architecture-coupled. Same recipe + same
    # 6,517-example corpus + same hyperparams produced +3.1pp on Qwen
    # 7B base, −3.8pp on Mistral 7B base. Refusal/rag_email/numerical_
    # precision lifts confirmed (recipe gains transfer); but recipe
    # actively damages retrieval (rag_datasheet −8, rag_blog −3) and
    # coding (−3) on Mistral specifically.
    #
    # GGUF identical to stock (4.37 GB, same compute graph) — alias
    # unchanged: mistral_7b_v03_dense → 5090 perf 182.7 tok/s carries.
    "mistral-7b-v0-3-v4-q4-dense": {
        "display_name": "Mistral 7B v0.3 Skippy v4 (cautionary cross-family, Q4_K_M)",
        "family": "mistral",
        "base_model": "Mistral-7B-Instruct-v0.3",
        "is_moe": False,
        "total_params": 7_250_000_000,
        "active_params": 7_250_000_000,
        "bytes_per_param": 0.57,
        "gguf_bytes": 4_370_000_000,
        "hidden_dim": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "vocab_size": 32768,
        "ctx_len_trained": 32768,
        "compute_dtype": "fp16",
        "quant_scheme": "Q4_K_M",
        # FT preserves base arch + GGUF size + compute graph — same
        # 5090 anchor as stock Mistral per [backend] 23:08 + [docs]
        # 09:41 alias decision.
        "measurement_alias": "mistral_7b_v03_dense",
        # v2+RAG eval per [docs] 2026-05-08 09:56. Source: acc_candidate-
        # kyle-mistral-7b-v4_20260508-095500.json. Persona dropped to
        # match 9-cat catalog. Sum reconciles: 3+3+6+6+0+45+3+0+9 = 75
        # → pass_rate 0.568.
        "training": "skippy_finetune_v4",
        "pass_rate": 0.568,
        "pass_n_passes": 75,
        "pass_n_total": 132,
        "category_deltas": {
            "coding":                   {"pass":  3, "n":  6, "rate": 0.500},
            "general":                  {"pass":  4, "n":  6, "rate": 0.667},
            "multihop":                 {"pass":  3, "n":  9, "rate": 0.333},
            "numerical_precision":      {"pass":  4, "n":  6, "rate": 0.667},
            "rag_blog":                 {"pass":  0, "n":  3, "rate": 0.000},
            "rag_datasheet":            {"pass": 49, "n": 78, "rate": 0.628},
            "rag_email":                {"pass":  3, "n":  3, "rate": 1.000},
            "reasoning":                {"pass":  0, "n":  6, "rate": 0.000},
            "refusal":                  {"pass":  9, "n":  9, "rate": 1.000},
        },
        "accuracy_bullet": (
            "**First cross-family fine-tune — REGRESSED −6.1pp vs stock "
            "Mistral baseline** (semantic: 0.629 → 0.568; substring "
            "originally read −3.8pp). Under semantic regrade (per "
            "[docs] 2026-05-11), Mistral stock lifted +2.4pp but v4 "
            "stayed flat (±0.0pp) — so the regression widens. Same v4 "
            "recipe + same 6,517-example corpus that *substring-lifted* "
            "Qwen 7B (but that lift reverses to −4.8pp under semantic). "
            "**Recipe transfer is base-family-coupled** (new finding) — "
            "not just architecture-coupled. Recipe gains DO transfer "
            "qualitatively: refusal +3 (6/9 → 9/9), rag_email +3 (0/3 "
            "→ 3/3), numerical_precision +3 (3/6 → 6/6 — better than "
            "Qwen v4's flat). But recipe actively damages retrieval on "
            "Mistral specifically: rag_datasheet −8 (53 → 45/78), "
            "rag_blog −3 (3/3 → 0/3 NEW regression), coding −3 (6/6 → "
            "3/6 — Qwen v4 held at 6/6). Hypothesis (untested): "
            "Mistral's required {{% generation %}} chat-template patch "
            "+ assistant_only_loss combination reweights away from RAG-"
            "following more than it did Qwen2.5. Customer rule: don't "
            "assume v4 recipe gains transfer to non-Qwen bases without "
            "fresh validation; the qualitative gains transfer (safety "
            "categories) but capability cost varies wildly by base."
        ),
    },
    # ─────────────────────────────────────────────────────────────────
    # Performance-comparison reference entries — Qwen 2.5 7B + 32B dense
    # (per [docs] 2026-05-01 20:55 spec mirroring keyhole-sizer's 7503f0c
    # / 66edfa2). NOT replacements for the production reference; surfaced
    # so the silicon-architecture audience can compare decode rate vs
    # the production MoE on the same hardware.
    #
    # No Skippy v2+RAG eval available — perf_reference_only=True flag
    # signals the UI to skip pass_rate / category_deltas blocks and
    # surface the "no eval" caption instead. These would only get full
    # eval treatment after Kyle's planned 7B fine-tune lands and the 32B
    # fine-tune happens on RunPod later.
    #
    # 5090 measurements per [backend] 15:43 + 20:08 bake-offs (RAG 8K+2K):
    #   7B  Q4_K_M: 183.9 tok/s decode, 7226 prefill (50% BW realization)
    #   7B  Q5_K_M: 170.0 tok/s decode, 7215 prefill (59%)
    #   7B  Q8_0:   137.2 tok/s decode, 7478 prefill (68%)
    #   32B Q4_K_M:  52.7 tok/s decode, 1936 prefill (62%)
    #   32B Q5_K_M:  47.7 tok/s decode, 1888 prefill (71%)
    #
    # bytes_per_param per quant: Q4_K_M ≈ 0.57, Q5_K_M ≈ 0.70, Q8_0 ≈ 1.06.
    # Architecture specs (Qwen 2.5 family from HF config.json):
    #   7B:  3584 hidden / 28 layers / 28 attn heads / 4 KV heads (GQA)
    #   32B: 5120 hidden / 64 layers / 40 attn heads / 8 KV heads (GQA)
    "qwen2.5-7b-q4-dense": {
        "display_name": "Qwen 2.5 7B Instruct Q4_K_M (apples-to-apples 7B base)",
        "family": "qwen2.5",
        "base_model": "Qwen 2.5 7B Instruct (stock)",
        "is_moe": False,
        "total_params": 7_620_000_000,
        "active_params": 7_620_000_000,
        "bytes_per_param": 0.57,
        "gguf_bytes": 4_700_000_000,
        "hidden_dim": 3584,
        "num_layers": 28,
        "num_attention_heads": 28,
        "num_kv_heads": 4,
        "vocab_size": 152064,
        "ctx_len_trained": 32768,
        # Q4_K_M weight-only — same fp16 runtime caveat as 14B Q4 dense
        # (gates 🔴 dtype_mismatch on Mid INT8-only).
        "compute_dtype": "fp16",
        "quant_scheme": "Q4_K_M",
        # Promoted from perf-reference-only after [docs] 2026-05-06 09:19
        # provided v2+RAG eval data for this row. It's the apples-to-
        # apples base for the Skippy 7B v4 fine-tune (+3.1pp anchor).
        # Source: eval/results/acc_candidate-qwen2.5-7b-instruct-v2-rag_
        # 20260502-*.json per [docs] 09:19 QWEN25_7B_BASE breakdown.
        "training": "public_stock",
        "pass_rate": 0.652,
        "pass_n_passes": 86,
        "pass_n_total": 132,
        # Per [docs] 2026-05-06 09:19 QWEN25_7B_BASE — apples-to-apples
        # 7B v4 baseline. Cleanly handles refusal 9/9 (no fabrication).
        "category_deltas": {
            "coding":                   {"pass":  6, "n":  6, "rate": 1.000},
            "general":                  {"pass":  6, "n":  6, "rate": 1.000},
            "multihop":                 {"pass":  1, "n":  9, "rate": 0.111},
            "numerical_precision":      {"pass":  0, "n":  6, "rate": 0.000},
            "rag_blog":                 {"pass":  3, "n":  3, "rate": 1.000},
            "rag_datasheet":            {"pass": 57, "n": 78, "rate": 0.731},
            "rag_email":                {"pass":  0, "n":  3, "rate": 0.000},
            "reasoning":                {"pass":  4, "n":  6, "rate": 0.667},
            "refusal":                  {"pass":  9, "n":  9, "rate": 1.000},
        },
        "accuracy_bullet": (
            "**Apples-to-apples 7B base** for the Skippy 7B v4 fine-tune. "
            "Under **semantic regrade** (per [docs] 2026-05-11 white "
            "paper Finding 4), the base SCORES **+4.6pp HIGHER** than "
            "production 7B v4 (0.652 vs 0.606) — the original substring "
            "+3.1pp v4-lift was format-fidelity to trained Qwen "
            "phrasings, not capability gain. Same architecture / "
            "quantization / 5090 host as 7B v4. Q5/Q8 7B perf-reference "
            "rows extend this to other quants (no eval — perf only)."
        ),
    },
    # INT8-routed Qwen 2.5 7B dense variant — perf reference for the INT8
    # compute path on INT8-capable silicon (NPU Mid INT8-only, NPU High
    # INT8 mode). Same Q4_K_M weights as the fp16 row above; only
    # difference is the matmul precision (INT8 dequant vs fp16 dequant).
    # Added 2026-05-14 to unlock the mid_int8.qwen25_7b_dense +
    # high_int8.qwen25_7b_dense private silicon anchors per [docs] 16:12.
    # measurement_alias points at the fp16 row so 5090 projection
    # inherits its baseline — the hot-swap overlay in app.py replaces
    # decode_tok_s with the measured INT8-path anchor when the user picks
    # (NPU Mid or NPU High) + this row.
    "qwen2.5-7b-q4-dense-int8": {
        "display_name": "Qwen 2.5 7B Instruct Q4_K_M — INT8 compute path (perf reference)",
        "family": "qwen2.5",
        "base_model": "Qwen 2.5 7B Instruct (stock)",
        "is_moe": False,
        "total_params": 7_620_000_000,
        "active_params": 7_620_000_000,
        "bytes_per_param": 0.57,
        "gguf_bytes": 4_700_000_000,
        "hidden_dim": 3584,
        "num_layers": 28,
        "num_attention_heads": 28,
        "num_kv_heads": 4,
        "vocab_size": 152064,
        "ctx_len_trained": 32768,
        "compute_dtype": "int8",
        "quant_scheme": "Q4_K_M",
        "measurement_alias": "qwen2.5-7b-q4-dense",
        "perf_reference_only": True,
    },
    "qwen2.5-7b-q5-dense": {
        "display_name": "Qwen 2.5 7B Instruct Q5_K_M (dense — perf reference)",
        "family": "qwen2.5",
        "base_model": "Qwen 2.5 7B Instruct (stock)",
        "is_moe": False,
        "total_params": 7_620_000_000,
        "active_params": 7_620_000_000,
        "bytes_per_param": 0.70,
        "gguf_bytes": 5_400_000_000,
        "hidden_dim": 3584,
        "num_layers": 28,
        "num_attention_heads": 28,
        "num_kv_heads": 4,
        "vocab_size": 152064,
        "ctx_len_trained": 32768,
        "compute_dtype": "fp16",
        "quant_scheme": "Q5_K_M",
        "perf_reference_only": True,
    },
    "qwen2.5-7b-q8-dense": {
        "display_name": "Qwen 2.5 7B Instruct Q8_0 (dense — perf reference)",
        "family": "qwen2.5",
        "base_model": "Qwen 2.5 7B Instruct (stock)",
        "is_moe": False,
        "total_params": 7_620_000_000,
        "active_params": 7_620_000_000,
        "bytes_per_param": 1.06,
        "gguf_bytes": 8_100_000_000,
        "hidden_dim": 3584,
        "num_layers": 28,
        "num_attention_heads": 28,
        "num_kv_heads": 4,
        "vocab_size": 152064,
        "ctx_len_trained": 32768,
        "compute_dtype": "fp16",
        "quant_scheme": "Q8_0",
        "perf_reference_only": True,
    },
    "qwen2.5-32b-q4-dense": {
        "display_name": "Qwen 2.5 32B Instruct Q4_K_M (apples-to-apples 32B base)",
        "family": "qwen2.5",
        "base_model": "Qwen 2.5 32B Instruct (stock)",
        "is_moe": False,
        "total_params": 32_500_000_000,
        "active_params": 32_500_000_000,
        "bytes_per_param": 0.57,
        "gguf_bytes": 18_525_000_000,
        "hidden_dim": 5120,
        "num_layers": 64,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "vocab_size": 152064,
        "ctx_len_trained": 32768,
        "compute_dtype": "fp16",
        "quant_scheme": "Q4_K_M",
        # Promoted from perf-reference-only after [docs] 2026-05-07 12:44
        # provided v2+RAG eval data — apples-to-apples baseline for the
        # 32B v4 fine-tune. Notable base-model behavior: 6/9 refusal
        # (fabricates made_up_peripheral 3/9) — same fabrication
        # pattern that 14B v4 inherited and amplified. Conversely, 6/6
        # numerical_precision (PERFECT — the only entry with that).
        # Per [docs] 12:44: source acc_baseline-qwen2.5-32b-instruct-...
        "training": "public_stock",
        "pass_rate": 0.674,
        "pass_n_passes": 89,
        "pass_n_total": 132,
        # Per [docs] 2026-05-07 12:44. Notable: 🟢 numerical_precision
        # 6/6 (PERFECT — only entry with that), 🔴 refusal 6/9
        # (fabricates made_up_peripheral 3/9 — base behavior the v4
        # recipe inherits and amplifies in 14B v4).
        "category_deltas": {
            "coding":                   {"pass":  6, "n":  6, "rate": 1.000},
            "general":                  {"pass":  6, "n":  6, "rate": 1.000},
            "multihop":                 {"pass":  6, "n":  9, "rate": 0.667},
            "numerical_precision":      {"pass":  1, "n":  6, "rate": 0.167},
            "rag_blog":                 {"pass":  3, "n":  3, "rate": 1.000},
            "rag_datasheet":            {"pass": 54, "n": 78, "rate": 0.692},
            "rag_email":                {"pass":  3, "n":  3, "rate": 1.000},
            "reasoning":                {"pass":  3, "n":  6, "rate": 0.500},
            "refusal":                  {"pass":  7, "n":  9, "rate": 0.778},
        },
        "accuracy_bullet": (
            "**Apples-to-apples 32B base** for the Skippy 32B v4 trade "
            "analysis. Under **semantic regrade** (per [docs] 2026-05-11): "
            "0.674 — substring originally read 0.682 (regraded −0.8pp; "
            "small shift). Notable base-model behavior: made_up_peripheral "
            "fabrication (refusal 7/9 — partial regression) that 14B v4 "
            "inherits and amplifies. The 32B v4 CLEAN fine-tune regresses "
            "−3.0pp under semantic (was −4.6pp on substring) — trading "
            "capability points for safety points. Customer rule: don't "
            "apply v4 recipe at 32B with 6.5K-example corpus unless "
            "refusal-calibration weight ≥ 3× capability-headline."
        ),
    },
    # INT8-routed Qwen 2.5 32B dense variant — perf reference for the INT8
    # compute path on INT8-capable silicon (NPU Mid INT8-only, NPU High
    # INT8 mode). Same Q4_K_M weights as the fp16 row above; only
    # difference is the matmul precision (INT8 dequant vs fp16 dequant).
    # Added 2026-05-14 to unlock the mid_int8.qwen25_32b_dense +
    # high_int8.qwen25_32b_dense private silicon anchors per [docs] 16:12.
    # measurement_alias points at the fp16 row so 5090 projection
    # inherits its baseline — the hot-swap overlay in app.py replaces
    # decode_tok_s with the measured INT8-path anchor when the user picks
    # (NPU Mid or NPU High) + this row.
    "qwen2.5-32b-q4-dense-int8": {
        "display_name": "Qwen 2.5 32B Instruct Q4_K_M — INT8 compute path (perf reference)",
        "family": "qwen2.5",
        "base_model": "Qwen 2.5 32B Instruct (stock)",
        "is_moe": False,
        "total_params": 32_500_000_000,
        "active_params": 32_500_000_000,
        "bytes_per_param": 0.57,
        "gguf_bytes": 18_525_000_000,
        "hidden_dim": 5120,
        "num_layers": 64,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "vocab_size": 152064,
        "ctx_len_trained": 32768,
        "compute_dtype": "int8",
        "quant_scheme": "Q4_K_M",
        "measurement_alias": "qwen2.5-32b-q4-dense",
        "perf_reference_only": True,
    },
    "qwen2.5-32b-q5-dense": {
        "display_name": "Qwen 2.5 32B Instruct Q5_K_M (dense — perf reference)",
        "family": "qwen2.5",
        "base_model": "Qwen 2.5 32B Instruct (stock)",
        "is_moe": False,
        "total_params": 32_500_000_000,
        "active_params": 32_500_000_000,
        "bytes_per_param": 0.70,
        "gguf_bytes": 22_750_000_000,
        "hidden_dim": 5120,
        "num_layers": 64,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "vocab_size": 152064,
        "ctx_len_trained": 32768,
        "compute_dtype": "fp16",
        "quant_scheme": "Q5_K_M",
        "perf_reference_only": True,
    },
}

# Reference model for per-category-Δ rendering. UI labels comparisons
# as "vs Skippy MoE fine-tune (production)" and the production model
# itself shows no per-category breakdown (it would be 0 across the
# board). Mirror of keyhole-sizer's PRODUCTION_REFERENCE_KEY pattern.
# Updated 2026-05-06 per [docs] 09:51: production reverted to Skippy 7B v4
# on 2026-05-04 17:30 after MoE v4 regressed; the 'Δ vs production' column
# now anchors against current shipping (7B v4), not historical MoE FT.
PRODUCTION_REFERENCE_KEY = "qwen25-7b-v4-q4-dense"

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


# DTYPE compatibility (hw_supports_dtype) and raw-peak-TOPS lookup
# (hw_peak_tops_for_dtype) are now owned by ratchet and imported at the top of
# this module. ratchet's hw_supports_dtype returns a CapabilityLevel (falsy only
# for UNSUPPORTED) — bool-compatible with PAI's existing `if not
# hw_supports_dtype(...)` and list-comprehension usage, and gives identical
# results to PAI's former peak_tops_<dtype> > 0 heuristic on the canonical tiers.


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
    m = _get_measured(hw, model_key, workload_id)
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
            # Prefill: held at anchor's stock value when same compute, but
            # SCALED by compute ratio when target tier has different peak
            # TOPS than anchor (within the same memory family). Prefill
            # is compute-bound, so a within-family tier with 2× compute
            # (e.g. High @ 400 INT8 TOPS vs Mid anchor @ 200 INT8 TOPS)
            # produces 2× faster prefill, halving TTFT. Per [docs] 16:08
            # validation: High + MoE Q4 prefill @ 1K → 175.5 ms (= Mid's
            # 351 ms / 2). Memory-only swaps (LPDDR upgrades) keep the
            # same compute and so prefill doesn't move (ratio = 1.0).
            ref = _get_measured(RTX_5090_REFERENCE, model_key, workload_id)
            host_ms_value = ref.get("host_ms", host_ms) if ref else host_ms
            if prefill_anchor is not None:
                target_peak_tops = hw_peak_tops_for_dtype(hw, required_dtype)
                anchor_peak_tops = hw_peak_tops_for_dtype(anchor_tier, required_dtype)
                compute_ratio = (target_peak_tops / max(anchor_peak_tops, 1e-9)
                                  if anchor_peak_tops > 0 else 1.0)
                prefill_tok_s = prefill_anchor * compute_ratio * compiler_quality
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
    # Includes the canonical snake-case forms so ratchet's tier anchors
    # (e.g. NPU_MID.measured_decode_overrides keyed 'qwen3_30b_a3b_moe')
    # resolve against PAI's hyphenated catalog keys.
    candidate_keys = _canonical_anchor_keys(model_key)
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
