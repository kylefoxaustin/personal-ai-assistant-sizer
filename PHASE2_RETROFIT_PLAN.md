# Phase 2 — PAI sizer retrofit onto ratchet (recon + migration plan)

**Date:** 2026-05-20
**Repo:** personal-ai-assistant-sizer (currently `v1.0.0`, production)
**Target:** PAI sizer `v1.1.0` retrofitted onto ratchet `v0.2.1`
**Status:** RECON COMPLETE — read-only. Holding for reviewer sign-off before any
destructive edit (no `git rm`, no dict→attribute rewrites yet).

This doc is the **contract for the phase-2 retrofit session** (same role the
design doc plays for ratchet sessions). If execution diverges from this plan,
stop and surface before destructive edits.

---

## 0. Headline: this is bigger than "swap the import"

The design called PAI the "lightest retrofit." True for the *anchor loader*
(clean swap), but PAI also ships its **own** `project_llm`, `Hardware`, `TIERS`,
`memory_feasibility`, and a family of what-if helpers — all of which ratchet now
owns, with **different signatures and a different result type**. The retrofit is
a real migration: a dict-result → typed-result (`Projected`) change that ripples
through ~30 consumption sites, plus a `model_key: str` → `LLMModel` change at
every projection call.

Bounded and doable, but the recon surfaced **three engine-reality findings** and
**two decisions** that need your call before execution.

---

## 1. Empirical inventory

### 1a. Loader-import swap sites (straightforward)
- `app.py:23  from sizer.npu_anchors import load_llm_anchor, load_cnn_anchor`
  → `from ratchet.anchors import load_llm_anchor, load_cnn_anchor`
- Then `git rm sizer/npu_anchors.py` (now byte-identical to ratchet's, lifted
  verbatim in v0.2.1; PAI always runs inside Streamlit so the import guard is
  invisible to it).

### 1b. Tier / Hardware / capability imports (swap, surface-side helpers stay)
`app.py:24` imports from `sizer.npu_model`:

| Imported name | Disposition |
|---|---|
| `TIERS` | → `from ratchet import TIERS` |
| `project_llm` | → `from ratchet import project_llm` (signature change — §1c) |
| `hw_supports_dtype` | → ratchet's (returns `CapabilityLevel`, bool-compatible via `__bool__`; PAI uses it as bool — OK) |
| `MODELS` | **stays surface-side** (per-surface catalog) — but migrate to `LLMModel` (§2) |
| `model_active_bytes_per_token`, `describe_hw`, `decode_tok_s_at_context`, `project_what_if_decode_tok_s`, `what_if_memory_feasibility` | **stay surface-side** PAI helpers; rewire their *internals* to call ratchet's `project_llm`/`memory_feasibility` |
| `PRODUCTION_REFERENCE_KEY`, `CATEGORY_LABELS` | **stay surface-side** (PAI UX constants) |

`app.py:274` also imports `MEMORY_UPGRADE_OPTIONS, hw_with_memory` →
`from ratchet import MEMORY_UPGRADE_OPTIONS, hw_with_memory`.

Deleted from `sizer/npu_model.py`: `class Hardware` (L23), `TIERS` (L360),
`MEMORY_UPGRADE_OPTIONS` (L388), `hw_with_memory` (L395), `hw_supports_dtype`
(L1587), PAI's `memory_feasibility` (L1897), PAI's `project_llm` (L1936). Kept:
the surface helpers above (rewired). Net: `npu_model.py` shrinks to a thin
surface module (catalog + UX helpers).

`app.py:30` imports from `sizer.precision`: `tier_precision_capability`,
`capability_badge/label/color`, `quality_*`, `RETARGETING_COSTS`,
`deployment_path_for_tier`, `retargeting_cost_color`, `REGRESSION_RIGOR`,
`gates_per_cycle`, `annualized_testing_cost`, `DEPLOYMENT_MODELS`,
`MEASURED_PRECISION_*`. **Almost all stay surface-side** (UI conventions = design
non-scope). Exceptions: ratchet now owns `deployment_path_for_tier` (different
signature — takes `workload_kernel_source`) and the capability *taxonomy*; PAI's
`tier_precision_capability(hw_name: str) -> dict[str,str]` is a surface wrapper
that can be reimplemented over `ratchet.hw_supports_dtype` / `CapabilityInfo`.

### 1c. `project_llm` signature + return change (the big one)
- **PAI today:** `project_llm(model_key: str, hw, workload_id, ...) -> dict`
  with `dict` keys `source / decode_tok_s / prefill_tok_s / ttft_s / host_ms /
  decode_s / prefill_s / total_s / regime / feasibility / dtype_detail / ...`,
  and `source` taking the values `wont_fit`, `dtype_mismatch`,
  `measured_silicon_anchor`, `measured`, `measured_anchor`, `same_class_anchor`,
  `cross_class`.
- **ratchet:** `project_llm(model: LLMModel, hw, workload_id, ...) ->
  Projected | WontFit | DtypeMismatch`. `wont_fit`/`dtype_mismatch` are
  **separate result types**, not `source` strings; the other five are
  `Projected.source` labels.
- **Call sites:** `app.py:581`, `app.py:1628` (`rr`), `app.py:1705` (`rr`), plus
  4 in `npu_model.py`'s `__main__` self-test. Each must (a) look up the
  `LLMModel` from the key before calling, (b) consume the typed result.

### 1d. Result-consumption sites (dict → attribute / pattern-match)
~30 sites in `app.py`. Representative:
- Source dispatch: `r["source"] == "wont_fit"` (602), `== "dtype_mismatch"`
  (617), the `measured*`/`cross_class` chain (665–713), and the `rr` loops
  (1635–1667, 1712–1744). → `match r: case WontFit(): ... case DtypeMismatch():
  ... case Projected(source=...):`.
- Field reads: `r["decode_tok_s"]` (1346/1355/1384), `r["host_ms"]`/`prefill_s`/
  `decode_s` (1264–1266), `r["ttft_s"]`, `r.get("regime")` (662),
  `r.get("_silicon_anchor_meta")` (669) → `r.silicon_anchor_meta`.
- `dtype_detail`: PAI reads `r["dtype_detail"]` (618, 1646); ratchet's
  `DtypeMismatch` exposes `required_dtype` / `tier_capability` /
  `retargeting_hint` — map fields.

### 1e. Overlay-call site (1, signature change)
- `app.py:588  r = _maybe_anchor_overlay(r, model_key, hw, tier_name, decode_tokens)`
- PAI's local `_maybe_anchor_overlay` (L543) → delete; replace with ratchet's
  `overlay_llm_anchor(result, hw, model, catalog_to_spec_key, *, decode_mult=1.0,
  ttft_mult=1.0)`. PAI is workload-invariant → call with defaults. The
  `_ANCHOR_MODEL_KEY_MAP` (L526) becomes the `catalog_to_spec_key` lambda.

### 1f. Anchor key map (`_ANCHOR_MODEL_KEY_MAP`, L526)
Maps PAI's hyphenated catalog keys → canonical snake_case spec keys
(`qwen3-30b-a3b-q4-moe` → `qwen3_30b_a3b_moe`). Becomes the `catalog_to_spec_key`
callable passed to `overlay_llm_anchor`. Survives as-is under Decision A below.

---

## 2. Decision matrix

### DECISION 1 — Catalog-key migration: translation lambda vs full migration
PAI's `MODELS` is 20 dict entries with **hyphenated** keys; ratchet's `LLMModel`
expects **snake_case** canonical keys. `st.session_state["k_model"]` (and similar)
store the hyphenated keys.

| Option | Pros | Cons |
|---|---|---|
| **A. Translation lambda (v1.1.0)** — migrate `MODELS` dict→`LLMModel` objects but keep hyphenated `key=`; `catalog_to_spec_key` lambda maps to canonical for anchors | Fast; preserves all `session_state` keys; no user-visible re-selection; lowest risk to a production surface | Translation dict persists; canonical-key goal deferred |
| **B. Full snake_case migration** — rename catalog keys to canonical; translation dict disappears | Cleaner; matches the design's long-term goal | Every `session_state["k_model"]` read must handle old-or-new until users re-select; larger diff; more parity risk |

**My recommendation: A for v1.1.0.** Preserves production session state, smallest
diff, and the translation lambda is the design's sanctioned transitional tool.
Defer B to a follow-up once all surfaces are on ratchet. **Your call.**

### DECISION 2 — int8/fp split catalog entries vs AMENDMENT 1 collapse
PAI models the *same physical model* as **two** catalog entries differing only by
`compute_dtype` — e.g. `qwen3-30b-a3b-q4-moe` (`compute_dtype: int8`) **and**
`qwen3-30b-a3b-q4-moe-fp` (`compute_dtype: fp16`) — to (a) route to the right
anchor cell (`mid_int8` vs `high_fp`) and (b) gate executability per silicon via
PAI's compute_dtype-based dtype check.

Ratchet's **AMENDMENT 1** gates on `quant_scheme` (`q4_km`), so a *single* Q4_K_M
`LLMModel` runs on **both** INT8 and FP silicon. This means:
- If we **keep the split entries** (Decision A-friendly): the translation lambda
  maps both to the same spec key (already does). But ratchet's quant-scheme
  gating will now mark *both* entries runnable on more tiers than PAI's old
  per-entry compute_dtype gate did → **dashboard cells that were `dtype_mismatch`
  may become populated.** This risks failing the design's phase-2 parity check
  ("dashboard outputs match v1.0.0 within float tolerance").
- If we **collapse to single entries**: cleaner and AMENDMENT-1-native, but
  changes which models appear under which tier and drops `session_state` keys.

**This needs your decision** — and it couples to Decision 1. My lean: keep split
entries for v1.1.0 (Decision A), but **explicitly accept that some former
`dtype_mismatch` cells will now project** (that's the AMENDMENT-1 fix working as
designed) and document the intended diff rather than chasing byte-for-byte parity
on those specific cells. Confirm you're OK relaxing the parity check there.

---

## 3. Surprises / engine-reality findings (Amendment-3 class)

**F1 — `project_llm` is PAI's own (dict, model_key), not ratchet's.** Already
covered (§1c). The design's "swap the import" framing under-counted this. Not a
blocker; just scope.

**F2 — `Projected` has no feasibility/verdict; PAI shows a "tight fit" warning on
*successful* projections.** PAI reads `r["feasibility"]["verdict"] == "tight"` at
`app.py:650, 1668, 1744` on results that DID project. Ratchet only returns
feasibility detail inside `WontFit`; a successful `Projected` carries none.
Options:
  - **F2a (surface-side):** PAI calls `ratchet.memory_feasibility(model, hw, ctx)`
    separately at those ~3 sites to recover the `tight` verdict. No ratchet change.
  - **F2b (engine):** ratchet **v0.2.2** adds an optional `feasibility` field (or
    `verdict`) to `Projected`. Cleaner for all surfaces, but an engine change.
  **Recommendation: F2a for now** (rule-of-three — only PAI has shown this need;
  keyhole may differ). Revisit F2b if keyhole-sizer also wants it. **Your call.**

**F3 — `dtype_mismatch` field shape differs.** PAI's `dtype_detail` dict vs
ratchet's `DtypeMismatch.{required_dtype, tier_capability, retargeting_hint}`.
Mechanical field-map at 2 sites; noting for completeness.

---

## 4. Ready-for-execute checklist (pending sign-off)

1. Pin `ratchet>=0.2.1,<0.3.0` in PAI deps; `pip install -e ../ratchet` in env.
2. Swap loader import (§1a); `git rm sizer/npu_anchors.py`.
3. Swap tier/Hardware imports to ratchet (§1b); delete the ratchet-owned defs
   from `npu_model.py`; rewire surviving surface helpers to call ratchet.
4. Migrate `MODELS` dict → `LLMModel` objects per **Decision 1** (recommend A).
5. Replace PAI `project_llm` calls: look up `LLMModel`, consume typed result;
   convert ~30 consumption sites to attribute access / `match` dispatch (§1c–d).
6. Replace `_maybe_anchor_overlay` with `overlay_llm_anchor(...)` + the
   `catalog_to_spec_key` lambda (§1e–f); resolve **Decision 2**.
7. Apply **F2** choice for the "tight" warning.
8. Update `sizer/measured.py` to attach measurements to `ratchet.RTX_5090_REFERENCE`.
9. Run PAI; verify dashboard parity vs v1.0.0 within float tolerance, **except**
   the AMENDMENT-1-affected cells (Decision 2) which are an intended diff.
10. Add a line to PAI's `CLAUDE.md`: ratchet's loader guards `import streamlit`
    so the engine installs headless; PAI always runs inside Streamlit so the
    swap is invisible — the guard is why ratchet has the `try/except`.
11. Tag `v1.1.0`, push. (Per discipline: confirm push before claiming on origin.)

---

## 5. What I will NOT do without sign-off
- No `git rm`, no catalog rewrite, no dict→attribute edits until you approve.
- If anything diverges from this plan mid-execution, stop and surface (esp. any
  further `project_llm`/Hardware behavior the design mis-sketched).
- If F2 or Decision 2 turns out to need a ratchet change, pause PAI, do a
  dedicated ratchet v0.2.2 session, then resume — same discipline as Amendment 3.

**Decisions needed from you:** Decision 1 (A vs B), Decision 2 (keep split +
relax parity, vs collapse), F2 (surface-side vs engine field). Once you rule,
PAI executes through to v1.1.0 tagged+pushed unless something new surfaces.
