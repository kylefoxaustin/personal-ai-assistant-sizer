# CLAUDE.md — personal-ai-assistant-sizer (Skippy NPU sizer)

Streamlit LLM sizer for the Skippy edge-NPU story. As of **v1.1.0** it is
retrofitted onto **ratchet** (the shared SoC sizing engine) — phase 2 of the
engine consolidation.

## ratchet retrofit (v1.1.0, Option C — lightest)

PAI depends on `ratchet>=0.2.2,<0.3.0` (the `<0.3.0` upper bound is deliberate:
ratchet v0.3.0 will carry breaking heterogeneous-architecture work; surfaces bump
their pin intentionally, they don't auto-upgrade).

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
