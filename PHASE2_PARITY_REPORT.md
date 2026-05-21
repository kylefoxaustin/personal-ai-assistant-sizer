# Phase 2 — PAI sizer retrofit onto ratchet: parity report

**Date:** 2026-05-21
**Repo:** personal-ai-assistant-sizer → **v1.1.0** (pending tag)
**Engine:** ratchet **v0.2.2**
**Scope:** Option **C** (lightest) — adopt ratchet for the canonical engine
pieces; PAI keeps its own projection.
**Status:** READY TO TAG pending reviewer sign-off on this report.

---

## 1. What changed (and what deliberately didn't)

Under option C, PAI **keeps its own `project_llm` (dict-returning) and its
what-if subsystem**. It adopts ratchet only for the unambiguously-canonical
engine pieces. Net: **−379 lines** across the surface.

**Now imported from ratchet (PAI definitions deleted):**
- `sizer/npu_anchors.py` → **deleted** (`git rm`); `app.py` imports
  `load_llm_anchor` / `load_cnn_anchor` from `ratchet.anchors` (the loader was
  lifted byte-identical into ratchet in v0.2.1 — same 3-arg signature, rich
  `LLMAnchor` with `bytes_per_token()`/`badge`).
- `Hardware`, the tier instances, `TIERS` (composed as PAI's 7-tier ladder,
  excluding the vision-only i.MX 95), `hw_with_memory`, `MEMORY_UPGRADE_OPTIONS`,
  `hw_supports_dtype`, `hw_peak_tops_for_dtype` → from `ratchet`.
- `precision.tier_precision_capability` → reads ratchet's canonical capability
  tables (`CapabilityLevel`) instead of PAI's hardcoded per-tier dicts.

**Deliberately unchanged (option C):**
- PAI's `project_llm` (dict result), `decode_tok_s_at_context`,
  `project_what_if_decode_tok_s`, `what_if_memory_feasibility`,
  `_find_same_family_anchor`, `memory_feasibility`, the `MODELS` catalog (dict,
  hyphenated keys), `_maybe_anchor_overlay`, and all ~30 result-consumption
  sites in `app.py`. Projection consolidation is deferred to a later pass once
  ratchet's projection is proven against keyhole-sizer too.

**Two small adapters** (because ratchet's API differs slightly):
- `_get_measured(hw, key, workload)` wraps ratchet's `get_measured_llm_cell` +
  PAI's `measurement_alias` fallback (ratchet's method does direct lookup only).
- `_canonical_anchor_keys()` maps PAI's hyphenated catalog keys to ratchet's
  snake-case tier-anchor keys at lookup time (so `NPU_MID.measured_decode_overrides`
  keyed `qwen3_30b_a3b_moe` resolves against PAI's `qwen3-30b-a3b-q4-moe`).
  PAI's catalog / `session_state` keys stay hyphenated (Decision A).

---

## 2. Validation

**Core projection parity — `_assert_phase2_anchors()` PASSES at import.** These
are PAI's *own v1.0.0* production anchor assertions; they pass unchanged against
ratchet's tiers:
- NPU Mid + MoE Q4 @ 100% share → **37.85 tok/s, `measured_anchor`** ✓
- NPU Mid + LPDDR6-14 + MoE Q4 → **63.08 tok/s, `same_class_anchor`** ✓
- NPU High + MoE Q4 → **37.85 tok/s, `same_class_anchor`** (BW-equal to Mid) ✓
- NPU Mid + MoE Q4 @ default 75% share → **28.39 tok/s** (npu_share scaling) ✓

**Projection matrix** (decode tok/s @ default share, `rag_qa` 4800+400):

| tier | qwen3-30b-moe | qwen2.5-7b-dense | qwen2.5-32b-dense |
|---|---|---|---|
| NPU Low-LP4 / LP5-32 / LP5-64 | wont_fit | dtype_mismatch | wont_fit |
| NPU Low-LP5X | wont_fit | 8.1 cross_class | wont_fit |
| NPU Mid | 28.4 measured_anchor | dtype_mismatch | dtype_mismatch |
| NPU High | 28.4 same_class | 16.2 cross_class | 3.8 cross_class |
| RTX 5090 | 224.8 measured | 183.9 measured | 52.7 measured |

All magnitudes sane (tens on NPU, hundreds on the 5090 — far from the "5000
tok/s" absurdity bar). Source classifications correct.

**App boot:** `streamlit run` headless → HTTP 200, no traceback. `AppTest`
executes the full script with **no uncaught exception** (it stops at PAI's
password gate, so the behind-gate UI wasn't walked — see §4).

---

## 3. Intended diffs vs v1.0.0 (the explicit called-out list)

Because PAI keeps its own projection math, the only sources of difference are
the corrected canonical tier specs (ratchet v0.2.2) and the capability source.

1. **NPU Low-LP4 cells: −0.21% on BW-bound decode + feasibility.** LP4 memory
   rate corrected 3.2→4.266 GT/s (12.8→17.064 GB/s; ratio 0.9979). This is the
   rate-consistent canonical value; PAI's old 17.1 GB/s was a rounded display.
   **All other tiers: 0.00% projection change** — Mid / High / 5090 / LP5X /
   LP5-64 specs are identical, and LP5-32's capacity was already 16 GB.
2. **Capability badges: LP4 / LP5-32 / LP5-64 `q4_km` ✗ → ✓.** Now sourced from
   ratchet's canonical `NEUTRON_INT8_ONLY` table (Q4_K_M runs via the INT8
   dequant path — technically correct, and consistent with how NPU Mid was
   already marked). **Display-only — no projection effect** (the dtype gate keys
   off `compute_dtype`, not `q4_km`).
3. **TDP display: LP4 / LP5-32 / LP5-64 → 10 / 15 / 20 W** (was 10/10/10).
   Informational only; TDP is not consumed by projection. (NPU High 40 W
   unchanged.)

4. **Memory-upgrade variants of privately-anchored cells now BW-scale the
   measured anchor (deliberate bug fix, ratchet v0.2.3 / ADR 011 Amendment 5).**
   Surfaced during the visual smoke: a privately-anchored cell (NPU High ×
   Qwen2.5-32B-dense) showed its measured decode at stock, but *any* memory
   upgrade dropped the anchor and fell to cross-class — a different, lower
   baseline, so LPDDR5T read **5.1** tok/s, *below* the **5.2** measured stock
   (a measured→cross-class discontinuity, not broken scaling). This was
   pre-existing v1.0.0 behavior in both sizers' overlay helpers. Fixed: the
   anchor's decode now BW-scales by effective-BW ratio (TTFT held at stock).
   New values: LPDDR5T **~6.9**, LPDDR6-12 **~7.4**, LPDDR6-14 **~8.7** —
   monotonic from the 5.2 stock measurement. This is the one change that is
   *not* parity-preserving; it's an intended correction.

**NOT a diff — no AMENDMENT-1 cells.** Under option C, PAI retains its own
`compute_dtype`-based dtype gate (via ratchet's `hw_supports_dtype`, which gives
identical results to PAI's former `peak_tops_<dtype> > 0` heuristic on the
canonical tiers). So the int8/fp split entries behave exactly as v1.0.0; no
former `dtype_mismatch` cell flips to a projection.

**Sanity note on the reviewer's estimate:** qwen3-30b MoE on NPU High projects
**37.85 tok/s @100%** (28.4 @75%), not the ~75 you ballparked. That's the
production-correct value: High shares NPU Mid's memory class (134.4 GB/s), decode
is BW-bound, so High = Mid via the same-class anchor. It matches PAI's own v1.0.0
`_assert_phase2_anchors` assertion — so it's not a regression, and it's
comfortably in the sane order of magnitude.

---

## 4. What I could NOT verify

- **Behind-the-gate visual walkthrough.** `app.py` has a password gate
  (`st.stop()` at line 64). AppTest can't authenticate and I don't have the
  credential, so I validated the projection / anchor / capability logic
  programmatically (matrix + anchor asserts + clean script execution) but did
  **not** click through the rendered dashboard. Recommend a quick visual smoke
  (enter password, eyeball the tier×model grid + the "Measured silicon anchors"
  expander) before or right after tagging.

---

## 5. Files changed

```
 app.py             |   6 +-   (loader import → ratchet.anchors; 2 help-text refs)
 requirements.txt   |   5 +-   (pin ratchet>=0.2.2,<0.3.0; fix mislabeled header)
 sizer/npu_model.py | 499 +-   (delete Hardware/tiers/hw_with_memory/dtype helpers;
                                import from ratchet; add 2 small adapters)
 sizer/precision.py |  65 +-   (tier_precision_capability over ratchet capability)
 sizer/npu_anchors.py          (deleted — now ratchet.anchors)
```
Net −379 lines. ratchet engine bumped v0.2.0 → v0.2.1 (anchor loader) → v0.2.2
(tier-spec correction) during the retrofit, each tagged + pushed.

**Recommendation: tag PAI v1.1.0.** Parity holds except the three intended diffs
above, all of which are corrections (canonical specs) or display-only.
