# CLAUDE.md — personal-ai-assistant-sizer (Skippy NPU sizer)

Streamlit LLM sizer for the Skippy edge-NPU story. As of **v1.1.0** it is
retrofitted onto **ratchet** (the shared SoC sizing engine) — phase 2 of the
engine consolidation.

## ratchet retrofit (v1.1.0, Option C — lightest)

PAI depends on `ratchet>=0.2.2,<0.3.0` (the `<0.3.0` upper bound is deliberate:
ratchet v0.3.0 will carry breaking heterogeneous-architecture work; surfaces bump
their pin intentionally, they don't auto-upgrade). `requirements.txt` currently
pins the git tag `@v0.2.7` (ADR 016 FP4 runtime-maturity + ADR 017 NPU
precision-set — see the precision-set section below).

**Adopted from ratchet** (local definitions deleted):
- Anchor loader: `from ratchet.anchors import load_llm_anchor, load_cnn_anchor`
  (the old `sizer/npu_anchors.py` was lifted byte-identical into ratchet and
  deleted here).
- `Hardware`, the canonical tier instances, `hw_with_memory`,
  `MEMORY_UPGRADE_OPTIONS`, `hw_supports_dtype`, `hw_peak_tops_for_dtype` —
  from `ratchet`. `sizer/npu_model.py`'s `TIERS` is PAI's **visible ladder**
  composed from ratchet's registry (7 tiers; the vision-only i.MX 95 is omitted).
- `precision.tier_precision_capability` reads ratchet's canonical capability
  tables (`CapabilityLevel`) instead of hardcoded per-tier dicts.

**Kept surface-side** (Option C keeps projection in PAI): `project_llm` (dict
result), the what-if subsystem (`decode_tok_s_at_context`,
`project_what_if_decode_tok_s`, `what_if_memory_feasibility`),
`_find_same_family_anchor`, the `MODELS` catalog (dict, **hyphenated keys**),
`_maybe_anchor_overlay`, and all the result-consumption sites in `app.py`.
Projection consolidation onto ratchet's `project_llm` is a deliberate later pass
(after keyhole-sizer is also on ratchet — rule of three).

**Two adapters bridge PAI ↔ ratchet:**
- `_get_measured(hw, key, workload)` = ratchet's `get_measured_llm_cell` + PAI's
  `measurement_alias` fallback.
- `_canonical_anchor_keys()` maps PAI's hyphenated catalog keys to ratchet's
  snake-case tier-anchor keys (e.g. `qwen3-30b-a3b-q4-moe` →
  `qwen3_30b_a3b_moe`) at lookup time only. PAI's catalog / `session_state`
  keys stay hyphenated (transitional; canonical-key migration is a future
  cross-surface project).

## NPU precision-set selector (ratchet v0.2.7 / ADR 017)

An escalating precision-capability ladder under the **Mid/High** tier selector —
**Stock / INT-only / INT+FP8 / INT+FP8+FP4** — so users can A/B/C the same model
across precision states and read off the prefill / accuracy deltas. It posits an
FP-capable tensor engine at the tier's memory class; FP4 is a 🟠 modeled
projection (confidence-low, **zero edge-NPU silicon anchors**). Cross-surface
spec: `personal-ai-framework/docs/npu-precision-set-selector-spec.md`. Built on
PAI first (the reference impl); keyhole-sizer mirrors it.

**Engine (`sizer/npu_model.py`):**
- `hw_with_precision(base_hw, precision_set)` builds the rung variant via
  ratchet's `make_custom_tier` with the spec §4 ladder (Mid 100/200/200/400,
  High 200/400/400/800 = bf16/int8/fp8/fp4). **It re-homes the variant's
  `tier_family` back to the stock family** (`make_custom_tier` labels it
  `LP5X-custom-*`) so `_find_same_family_anchor` resolves the **measured** Mid/High
  cell instead of dropping to the first-principles cross-class floor — this is
  what makes the anchors reproduce. Composes on top of a memory upgrade (lifts
  mem specs from `base_hw`).
- `project_llm` gained `fp4_runtime_maturity` + a one-line `resolve_floor_dtype`
  override at the dtype-resolution step. **Non-breaking**: `npu_precision_set`
  is `None` on every canonical tier, so `resolve_floor_dtype` returns the model
  dtype unchanged (import-time anchor asserts still pass).
- The same-family anchor-ratio branch scales FP8/FP4 rungs against the
  **anchor's own native dtype** (Mid int8), not the target rung dtype — else the
  ratio collapses to 1× when the Mid anchor lacks that `peak_tops` field (Mid has
  no fp8/fp4/bf16). This is the subtle fix that makes High·FP8=175.5 not 351.

**UI (`app.py`):** escalating radio under Mid/High (composes with the memory
upgrade); the FP4 rung reveals a **mature/immature** toggle defaulting to
**immature** (the honest edge floor — engine default is `mature`, so PAI passes
`immature` explicitly per ADR 016); a 3-column compare panel in the **Precision**
tab.

**Two cross-surface LOCKED nuances** (docs-ratified 2026-06-05, both sizers
render identically):
1. **Baseline = measured-anchored 351 ms** (the real Mid cell), NOT the
   spec-computed 333. Makes ratios exact: naive→INT8 = precisely 2× (351→175.5);
   immature-FP4 == naive (the ADR-016 no-win, exact).
2. **Weight RAM = fixed-by-model** with an orthogonal-axis caption. The compute
   rung (int8/fp8/nvfp4 math) changes prefill+accuracy; the **weight format**
   (Q4/Q8/FP4-weight bytes) changes RAM + decode-BW — two **independent** axes.
   An already-Q4 model keeps 4-bit weights at every rung → RAM fixed. FP4's
   half-RAM benefit is a weight-format (model-choice) axis, NOT a compute-rung
   effect. (Conflating them re-introduces the memory-vs-compute confound the
   whole FP4 story exists to separate.)

Validation anchors (Qwen3-30B-A3B MoE @1K, npu_share=1.0): High INT8=175.5 /
FP8=175.5 / FP4-mature=87.8 / FP4-immature=351; Mid INT8=351 / FP8=351 /
FP4-mature=175.5 / FP4-immature=702. Decode held 37.9 tok/s (BW-bound, model
already Q4).

### Why ratchet guards `import streamlit`

ratchet's anchor loader wraps its top-level `import streamlit` in
`try/except ImportError: st = None`, because ratchet must install **headless**
(non-Streamlit environments: the future drone surface, headless analysis). PAI
**always runs inside Streamlit**, so the loader swap is invisible here — the
guard exists for ratchet's other consumers, not for PAI. (If you ever see the
loader return `None` where you expect an anchor, it's a missing secret, not the
guard — the guard only matters when streamlit isn't installed at all.)

## Intended diffs vs v1.0.0 (from the retrofit; see PHASE2_PARITY_REPORT.md)

PAI keeps its own projection math, so projections are otherwise byte-identical.
Three deliberate differences come from ratchet's corrected canonical specs
(v0.2.2) and capability source:

1. **NPU Low-LP4: −0.21% on BW-bound decode + feasibility.** LP4 memory rate
   corrected 3.2 → 4.266 GT/s (12.8 → 17.064 GB/s). The rate-consistent
   canonical value; PAI's old 17.1 GB/s was a rounded display. **All other tiers
   unchanged (0.00%).**
2. **Capability badge: LP4 / LP5-32 / LP5-64 `q4_km` ✗ → ✓.** Now from ratchet's
   canonical `NEUTRON_INT8_ONLY` table (Q4_K_M runs via the INT8 dequant path).
   **Display-only** — the dtype gate keys off `compute_dtype`, not `q4_km`.
3. **TDP display: LP4 / LP5-32 / LP5-64 → 10 / 15 / 20 W** (was 10/10/10).
   Informational; TDP is not consumed by projection. (NPU High 40 W unchanged.)
4. **Memory-upgrade variants of a privately-anchored cell now BW-scale the
   measured anchor** instead of dropping to cross-class. A *deliberate bug fix*
   (ratchet v0.2.3 / ADR 011 Amendment 5), not parity-preserving: e.g.
   NPU High × Qwen2.5-32B-dense + LPDDR5T was **5.1** tok/s (cross-class, and
   *below* the 5.2 measured stock — a discontinuity), now **~6.9** (5.2 ×
   179.2/134.4, anchored to the measurement; TTFT held at stock). PAI's
   `_maybe_anchor_overlay` BW-scales by `mem_bandwidth_gbs / stock_mem_bandwidth_gbs`
   for `bw_projected` clones. (The cell still carries the `(BW-proj)` UI marker;
   if the green "measured on real silicon" banner wording should differ for a
   BW-projected clone, that's a small follow-up polish.)

**No AMENDMENT-1 cells flip under Option C.** PAI retains its own
`compute_dtype`-based dtype gate (via ratchet's `hw_supports_dtype`, which gives
identical results to PAI's former `peak_tops_<dtype> > 0` heuristic on the
canonical tiers). The int8/fp split catalog entries behave exactly as v1.0.0; no
former `dtype_mismatch` cell becomes a projection.

## Running

`streamlit run app.py` (there is a password gate before the main UI). Measured
silicon anchors load at runtime from gitignored `.streamlit/secrets.toml`
(KEY-not-VALUE discipline — values are credentials, never committed).
