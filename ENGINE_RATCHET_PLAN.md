# Engine-extraction ratchet — PAI sizer response to reviewer synthesis

**Repo:** `personal-ai-assistant-sizer` (PAI sizer side)
**Generated:** 2026-05-19 (against `7e34df0` / tag `v1.0.0`)
**Revised:** 2026-05-19 — INPUT 26 downgraded from 🔶 to 🟡 after locating PAI's existing 4-level capability taxonomy in `sizer/precision.py:115–153` (was thought to be absent; finding partially corrects the analyst's "PAI is binary" framing).
**Responds to:** reviewer's cross-repo synthesis INPUTs 23–30 (Hardware-dataclass-shape convergence pass).
**Context:** v1.0.0 ships PAI sizer at a stable recovery point; v1.x cycle is **engine extraction** — refactor the shared LLM / anchor / bundle layers from PAI sizer + keyhole-sizer into a common `engine` package consumed by both surfaces + the upcoming drone-repo sizer. This document is PAI-side input into the ratchet design.

---

## How to read this

For each INPUT, three blocks:

- **Analyst finding** — verbatim restatement of the reviewer's observation.
- **PAI-side perspective** — corrections, additional facts, file-line refs from this repo. Where PAI has data the analyst may not, we surface it.
- **Ratchet recommendation** — concrete design call for the engine package, with a confidence marker and any deferred items.

Confidence markers:
- 🟢 **Adopt as-is** — implementation is straightforward; no open design questions.
- 🟡 **Adopt with refinement** — design call agreed, but a sub-detail needs settling.
- 🔶 **Adopt with refactor cost flagged** — design call agreed, but call-site ripple is non-trivial; sequence carefully.

---

## INPUT 23 — Hardware dataclass shape converges

**Analyst finding.** 20 of 24 fields are identical across surfaces (same name, type, default, semantics). Discipline-coordinated baseline is working. Lift the consensus shape directly.

**PAI-side perspective.** The 4 PAI-specific deltas are:

| Field | Lines | Status |
|---|---|---|
| `measured_decode_overrides` | `npu_model.py:134` | Tactical-interim per [docs] 2026-04-29 12:34. Path C compute clamp slated to replace. |
| `measured_prefill_overrides` | `npu_model.py:141` | Same tactical-interim framing. |
| `stock_name` | `npu_model.py:123` | Adopted from keyhole-sizer commit ecc3ba8 mirror (see INPUT 25). |
| `stock_mem_bandwidth_gbs` | `npu_model.py:116` | Same — paired with `stock_name`. |

The first two are the legacy tier-level override pattern; INPUT 24's unification replaces them. The latter two are exactly what INPUT 25 wants the ratchet to adopt — they're already in PAI but presumably not yet in keyhole.

**Ratchet recommendation.** 🟢 **Adopt the 20-field consensus baseline directly.** Layer on the INPUT 25 stock-identity fields (already in PAI). Resolve the INPUT 24 measurement-attachment shape in place of `measured_decode_overrides` / `measured_prefill_overrides` (those fields disappear from the ratchet baseline).

---

## INPUT 24 — Tier-level measurement attachment unification

**Analyst finding.** Three patterns in play:
- Keyhole legacy: flat fields `measured_llm_q4_decode_tok_s` + `measured_llm_ttft_1k_sec` (implicit Skippy-MoE-Q4 scope, flagged as legacy).
- PAI: dict-based `measured_decode_overrides` / `measured_prefill_overrides` keyed by `model_key`.
- Keyhole's `measured_llm` nested dict also exists — a third pattern.

PAI's dict-based shape is strictly better than the legacy flat fields. Ratchet should unify all three into one canonical attachment model.

**PAI-side perspective.** PAI has **two** patterns coexisting, not one:

| PAI pattern | Field | Where populated | Shape |
|---|---|---|---|
| Reference-tier measurements | `Hardware.measured_llm` | `RTX_5090_REFERENCE` only (set by `sizer/measured.py` at import from `sizer_bundle.json`) | `{model_key: {workload_id: {decode_tok_s, ttft_s, prefill_tok_s, host_ms}}}` |
| Per-tier override (tactical) | `Hardware.measured_decode_overrides` + `measured_prefill_overrides` | `NPU_MID` only (hardcoded in constructor) | `{model_key: float}` flat |

These coexist because they were added in different sessions for different needs: the nested form was always there for the 5090 bake-off catalog, and the override pair was added 2026-04-29 to handle the Skippy MoE Q4_K_M anchor on real INT8 silicon (`37.85 tok/s` decode, `2849.0 tok/s` prefill — values are public deck numbers from Slide 19, not anchor-secrets).

Keyhole's legacy flat fields are scope-implicit (Skippy MoE Q4 is assumed). PAI's flat-dict pattern at least scopes by `model_key` but still flattens the workload dimension.

**Ratchet recommendation.** 🟡 **Adopt the nested-dict shape as the canonical form for all tiers** — promote PAI's `measured_llm` schema to apply tier-wide (not 5090-only), drop the flat overrides entirely, and migrate keyhole's legacy flat fields into nested form.

Proposed canonical schema:
```python
measured_llm: dict[str, dict[str, dict[str, float]]] | None = None
# {model_key: {workload_id: {decode_tok_s, ttft_s, prefill_tok_s, host_ms}}}
```

Migration of PAI's current NPU_MID overrides → nested:
```python
# Before (PAI v1.0.0):
NPU_MID.measured_decode_overrides  = {"qwen3-30b-a3b-q4-moe": 37.85}
NPU_MID.measured_prefill_overrides = {"qwen3-30b-a3b-q4-moe": 2849.0}

# After (engine ratchet):
NPU_MID.measured_llm = {
    "qwen3-30b-a3b-q4-moe": {
        "<default_workload_id>": {  # ← question to reviewer below
            "decode_tok_s":  37.85,
            "prefill_tok_s": 2849.0,
            # ttft_s, host_ms left absent or default
        }
    }
}
```

**Sub-detail to settle:** how does the per-tier-but-not-per-workload override express itself in a workload-keyed schema? Options:
- (a) Use a sentinel workload_id like `"_any"` or `"_tier_default"` that `project_llm` checks before falling through to workload-specific.
- (b) Replicate the override under every known workload_id at construction time (more redundant but no special-case lookup).
- (c) Drop the workload dimension for tier-level overrides; require all measurements to be workload-scoped.

PAI's current code reads `measured_decode_overrides[model_key]` workload-agnostically — the override applies to *any* workload on that tier+model. Option (a) preserves that semantic cleanly; option (c) is the strict-purity move.

**Tangle alert:** see "Cross-input concerns" section below — INPUTs 24 + 30 + anchor-secrets need a unified attachment story, not three.

---

## INPUT 25 — Stock identity tracking on memory-upgrade clones

**Analyst finding.** PAI captures `stock_name` + `stock_mem_bandwidth_gbs` on `hw_with_memory()`; provides `tier_lookup_name` property for silicon-intrinsic lookups. Keyhole doesn't capture stock identity; anchor-secrets hack-compensates by checking `abs(mem_data_rate_gtps - 8.4) > 0.05`. Adopt PAI's explicit pattern.

**PAI-side perspective.** Confirmed. Three reads of the stock-identity machinery in PAI:

| Reader | File:line | Why |
|---|---|---|
| Anchor-tier matching | `npu_model.py:1832` (`decode_tok_s_at_context`) and `npu_model.py:2062` (`project_llm`) | Refuses "🟢 measured anchor on this tier" if `hw.bw_projected` — a memory-upgrade variant must reproject, never reuse the stock anchor verbatim. |
| Precision-capability lookup | `app.py:1078` — `tier_precision_capability(hw.tier_lookup_name)` | Silicon caps don't change with a memory swap. |
| Deployment-path lookup | `app.py:1830` — `deployment_path_for_tier(hw.tier_lookup_name, ...)` | Same — retargeting cost class is silicon-bound. |
| Display annotation | `app.py:1614` — `if getattr(hw, "bw_projected", False) and hw.tier_lookup_name == stock_tname:` | UI annotation for "(BW-proj)" marker. |

The keyhole-side `abs(mem_data_rate_gtps - 8.4) > 0.05` heuristic also appears in PAI's anchor-secrets overlay at `app.py:547` — vestigial from the keyhole crib. The engine ratchet should kill that heuristic and route through `tier_lookup_name` / `bw_projected` instead.

**Ratchet recommendation.** 🟢 **Adopt PAI's explicit `stock_name` + `stock_mem_bandwidth_gbs` + `tier_lookup_name` property pattern.** Eliminate the `mem_data_rate_gtps == 8.4` heuristic everywhere — the anchor-secrets overlay should route through `hw.bw_projected` and `hw.tier_lookup_name` to detect "this is a memory-upgrade variant, skip the hot-swap" cleanly. Brittleness fix: the heuristic fails the moment a non-LPDDR5X tier (e.g., a future LPDDR6-native silicon class shipping at 12 GT/s stock) gets memory-upgrade overlays.

---

## INPUT 26 — Capability levels taxonomy

**Analyst finding.** Keyhole has a 4-level taxonomy (`tensor_native` / `tensor_compat` / `cuda_core` / `unsupported`) per-dtype via a `capability_levels` dict. PAI has a binary `supported/unsupported` via `peak_tops > 0` heuristic. Adopt keyhole's 4-level taxonomy — the `tensor_compat` / `cuda_core` distinction matters for accurate projection on emerging silicon classes.

**PAI-side perspective — partial correction.** PAI's binary `hw_supports_dtype()` is real, but it's **not the only capability surface** in this repo. After investigating the `app.py:1078` call site flagged in the original draft, the situation is:

**PAI has TWO capability surfaces, orthogonal in intent:**

| Surface | Location | Question it answers | Resolution |
|---|---|---|---|
| `hw_supports_dtype(hw, dtype)` | `sizer/npu_model.py:1587` | "Is there ANY compute path for this dtype on this tier?" — binary, derived from `peak_tops_<dtype> > 0` | **Binary** |
| `tier_precision_capability(hw_name)` | **`sizer/precision.py:115–153`** | "What KIND of compute path does this tier have for each precision?" — tensor-core native vs binary-compat vs CUDA-core vs none | **4-level** |

**PAI already has a 4-level taxonomy** — exactly the one keyhole has, with naming deltas:

| Keyhole level | PAI level | Constant (in `sizer/precision.py:109–112`) |
|---|---|---|
| `tensor_native` | `tensor_core` | `_CAP_TENSOR = "tensor_core"` |
| `tensor_compat` | `tensor_compat` | `_CAP_COMPAT = "tensor_compat"` |
| `cuda_core` | `cuda_core` | `_CAP_CUDA = "cuda_core"` (defined but **currently unused** by any tier — vestigial; will recur on future silicon) |
| `unsupported` | `none` | `_CAP_NONE = "none"` |

The taxonomy comment block at `sizer/precision.py:96–108` is explicit that this surface is **orthogonal** to `peak_tops_<dtype>`: "*This is ORTHOGONAL to peak_tops_<dtype> which only asks 'is there any path.' For LLM-scale matmul, you almost always want tensor_core — cuda_core is acceptable for small CNN inference, crippling for 14B+.*"

**PAI's encoding deltas from keyhole that the ratchet must settle:**

1. **Location.** PAI: module-level function `tier_precision_capability(hw_name: str) -> dict[str, str]`, lookup by **tier-name string**. Keyhole: `capability_levels` field on the Hardware dataclass. The PAI shape correctly couples to stock identity via `hw.tier_lookup_name` (consistent with INPUT 25); the keyhole shape co-locates the data with the tier object. Both have merits.
2. **fp16/bf16 conflation surfaced into the precision dimension.** PAI keys are `"bf16/fp16"` / `"fp8"` / `"int8"` / `"q4_km"` — fp16 and bf16 share a single key, mirroring `_DTYPE_ATTR`'s field-routing conflation. Keyhole presumably keys them separately.
3. **Q4_K_M as a peer precision.** PAI's dict has a `"q4_km"` key — weight-only quant flavor that doesn't appear in `_DTYPE_ATTR` at all. Critical for `deployment_path_for_tier`: Q4_K_M support hinges on the underlying fp16 dequant path being tensor-core-supported. Keyhole's per-dtype dict may not have a Q4_K_M equivalent.
4. **String literals, not an enum.** PAI uses string values (`"tensor_core"` / `"tensor_compat"` / `"cuda_core"` / `"none"`) with leading-underscore-private constants as convenience aliases. The ratchet's `CapabilityLevel(Enum)` proposal is a tighter encoding.
5. **`tensor_compat` semantic collapse at one consumer.** PAI's `deployment_path_for_tier` (`sizer/precision.py:277–310`) **treats `tensor_compat` the same as `none`** for fast-path purposes — it only checks `== _CAP_TENSOR`. That's intentional for SM120 INT8 (vLLM CUTLASS LLM serving is blocked even though hardware works for pre-compiled TRT INT8), but it means the 4-level distinction collapses to binary at this downstream. UI consumer (`app.py:1078–1118`) honors all 4 levels via `capability_color()` / `capability_badge()` / `capability_label()` helpers.

**Existing PAI helpers around the function (re-exported via `app.py:32`):**

| Helper | File:line | Maps level → |
|---|---|---|
| `capability_badge(level)` | `sizer/precision.py:156` | One glyph: ✓ / ⚠︎ / ⚠︎ / ✗ |
| `capability_label(level)` | `sizer/precision.py:166` | Short text: "tensor-core" / "CUDA-core only" / "tensor-core (compat path)" / "not supported" |
| `capability_color(level)` | `sizer/precision.py:176` | Hex: #10b981 / #f59e0b / #f59e0b / #ef4444 |
| `deployment_path_for_tier(hw_name, model_quant_scheme)` | `sizer/precision.py:277` | Consumer — returns `"fp_native"` / `"weight_only"` / `"ptq"` |

**Real-world motivation for 4-level (unchanged from original draft).** Consumer Blackwell (SM120) at INT8 — hardware HAS INT8 tensor-core capability (ncu profiling shows non-zero tensor-pipe instruction counts with sm80 IMMA kernel names → Ampere binary-compat path engages real tensor cores), but ecosystems that compile fresh per-arch (vLLM's CUTLASS W8A8) refuse SM120 because SM120-specific templates don't exist yet. This is exactly the `tensor_compat` (binary-compat) vs `tensor_native` (native kernels) distinction. **PAI already encodes this** — RTX 5090's INT8 entry is `_CAP_COMPAT` (see `sizer/precision.py:131–136`), with an extensive comment block explaining the ncu probe finding.

**Ratchet recommendation.** 🟡 **Unify two existing 4-level surfaces into one canonical capability model** (downgraded from 🔶 — refactor cost is smaller than initially estimated because the taxonomy already exists in PAI; what's needed is consolidation, not introduction):

1. **Pick canonical location.** Either:
   - **(a) On Hardware** (keyhole's approach): add a `capability_levels: dict[str, CapabilityLevel] | None = None` field; the per-tier capability dict moves into each `Hardware(...)` constructor. Keeps capability co-located with tier definition; loses the tier-name-string lookup decoupling. Memory-upgrade clones inherit via `dataclasses.replace`.
   - **(b) Module-level function** (PAI's approach): keep `tier_precision_capability(hw_name)` as the canonical accessor; let it dispatch on `hw.tier_lookup_name`. Keeps the stock-identity coupling clean; downside is the data isn't introspectable from a Hardware instance alone.
   - **(c) Both, with one canonical source.** Hardware carries the dict; the module-level function becomes a thin convenience wrapper. Most permissive; slight duplication.
2. **Adopt `CapabilityLevel(Enum)`** with `__bool__` for backwards compat. PAI's string-literal constants migrate to enum members (`CapabilityLevel.TENSOR_NATIVE` etc.). Pick canonical names — recommend keyhole's `tensor_native` / `tensor_compat` / `cuda_core` / `unsupported` (PAI's `tensor_core` / `none` synonyms are less descriptive).
3. **Settle the schema dimension questions:**
   - fp16/bf16 — single combined key `"fp16/bf16"` or two separate keys with identical values? PAI's choice (combined) is more compact; keyhole's choice (separate) is more orthogonal. Recommend separate keys in the canonical schema; let `_DTYPE_ATTR` continue to do the field-routing conflation at the TOPS-lookup layer.
   - `q4_km` — peer precision (PAI's choice) or derived from `bf16/fp16` + a `is_weight_only_quant` flag at the model layer? Recommend keeping as peer precision — Q4_K_M's support story is silicon-specific (NPU Mid runs it via INT8 dequant, NPU High via fp16 dequant); a derived flag would mask that.
2. **Settle the `tensor_compat` consumer semantics.** Does `tensor_compat` count as "fast path available" for the deployment-path classifier? PAI's current code says NO (treats as `none`); UI rendering says it's amber, not red. Recommend documenting the rule: **"`tensor_compat` is fast for pre-compiled kernel workloads (TRT, ONNX Runtime, vendor SDKs) and slow/blocked for fresh-compile workloads (vLLM CUTLASS, TVM, JIT paths)."** Classifier behavior then becomes workload-aware rather than collapsing to binary.
3. **Keep `hw_supports_dtype()` and the new `capability_levels` distinct.** They answer different questions and both need to stay. `hw_supports_dtype()` stays binary (peak_tops > 0); the capability accessor returns the 4-level. Consumer chooses based on what they're asking. Document the orthogonality explicitly — it's the existing PAI comment block at `sizer/precision.py:96–108`, lifted into the engine package docstring.
4. **Coordinate with INPUT 24's measurement-attachment refactor.** Less of a concern than originally noted — capability levels are silicon-fixed metadata; measurement-attachment is runtime data. The two refactors touch different code paths.

---

## INPUT 27 — Custom tier construction

**Analyst finding.** Keyhole's UI builds `Hardware(...)` from sidebar inputs. Calibration constants left at defaults (over-optimistic projection risk). PAI has no custom tier support. Ratchet should provide `make_custom_tier()` factory with sensible defaults + explicit "calibration: default" warning when Custom tier is used.

**PAI-side perspective.** Confirmed PAI has zero custom-tier support — TIERS is a hardcoded module-level constant set at import. The compute-ceiling Phase 2 task in `REMEDIATION_PLAN.md` (KH-P1-003) is closely related — for sub-5-TOPS silicon, default calibration constants over-project by 5–10×. A user constructing a custom 2-TOPS NPU with default `compute_util_factor=0.45` and `llm_prefill_util_factor=0.10` would get wildly optimistic numbers.

**Ratchet recommendation.** 🟡 **Provide `make_custom_tier()` factory** with:

1. Mandatory args: `name`, `peak_tops_int8`, `mem_bandwidth_gbs`, `mem_capacity_gb`, `mem_bus_width_bits`, `mem_type`, `mem_data_rate_gtps`. (Same as Hardware's required fields.)
2. Optional args for FP path: `peak_tops_bf16=0.0`, `peak_tops_fp8=0.0` (default to INT8-only).
3. All calibration constants stay at Hardware's class defaults but get explicitly **tagged** by adding an immutable `_calibration_source: str | None = None` field — `make_custom_tier()` sets it to `"default — NOT calibrated for this silicon"`; canonical tiers set it to the calibration-source ref (e.g., `"[backend] 2026-04-29 13:17"`).
4. UI surfaces a 🔶 banner above the projection: "Custom tier — projection uses default calibration constants (`compute_util_factor=0.45`, `llm_prefill_util_factor=0.10`). For sub-5-TOPS silicon, expect 5–10× over-projection until calibrated against ground truth."
5. The 🔶 status bleeds through the source-banner cascade — any cell that pulls from a custom tier is marked `cross_class_custom` (more pessimistic than 🔴 `cross_class` because we don't even know the silicon's util factors).

**Sub-detail to settle:** how does the engine package surface "custom tier was used" to downstream sizer apps? Options:
- (a) A separate `make_custom_tier` import explicitly distinct from canonical tier construction.
- (b) The `_calibration_source` field is publicly readable; UI checks it.
- (c) Both — explicit factory + readable field.

Recommend (c).

---

## INPUT 28 — Tier registry composition

**Analyst finding.** Same set of canonical tier entities used by both (NPU_LOW_LP5_64BIT, NPU_LOW_LP5X, NPU_MID, NPU_HIGH, RTX_5090_REFERENCE). PAI exposes Low-LP4, Low-LP5-32 too. Keyhole exposes IMX95_MEASURED with ground-truth measurements. Registry holds the *union* of all canonical tiers as named entities. Each surface composes its own ladder from the registry. IMX95_MEASURED belongs in the registry.

**PAI-side perspective.** PAI's current `TIERS` dict has 7 entries:

```python
# npu_model.py:360-368
TIERS = {t.name: t for t in (
    NPU_LOW_LP4,
    NPU_LOW_LP5_32BIT,
    NPU_LOW_LP5_64BIT,
    NPU_LOW_LP5X,
    NPU_MID,
    NPU_HIGH,
    RTX_5090_REFERENCE,
)}
```

Plus a vestigial bare `RTX_5090` constant at `npu_model.py:181` that is defined but **NOT** registered — see §4 of `HARDWARE_TIER_EXTRACT.md`. Should be culled in the ratchet (or its sole consumer migrated to `RTX_5090_REFERENCE`).

PAI lacks `IMX95_MEASURED` because Skippy is LLM-only and the i.MX 95 deployment is vision-pipeline-anchored — the keyhole side owns that measurement.

**Ratchet recommendation.** 🟢 **Adopt the union-registry pattern.** Engine package exposes the full canonical set; each downstream sizer composes its visible ladder via a per-surface tuple:

```python
# engine/tiers.py
TIERS = {t.name: t for t in (
    NPU_LOW_LP4,
    NPU_LOW_LP5_32BIT,
    NPU_LOW_LP5_64BIT,
    NPU_LOW_LP5X,
    IMX95_MEASURED,           # ← from keyhole side; PAI hides it
    NPU_MID,
    NPU_HIGH,
    RTX_5090_REFERENCE,
)}

# personal-ai-assistant-sizer/sizer/__init__.py
from engine.tiers import TIERS
VISIBLE_TIERS = [TIERS[n] for n in (
    "NPU Low-LP4", "NPU Low-LP5-32bit", "NPU Low-LP5-64bit", "NPU Low-LP5X",
    "NPU Mid", "NPU High", "RTX 5090 (reference, measured)",
)]
# PAI sizer dropdown reads from VISIBLE_TIERS, not TIERS directly.

# keyhole-sizer composes its own VISIBLE_TIERS including IMX95_MEASURED.
```

The bare `RTX_5090` constant gets dropped in the ratchet.

---

## INPUT 29 — Module-level dtype dispatch pattern

**Analyst finding.** PAI: `_DTYPE_ATTR` dict + `hw_supports_dtype()` + `hw_peak_tops_for_dtype()` external helpers. Keyhole: dispatch dict inlined in `effective_tops()` and `capability_level()` (acknowledged as refactor target). Adopt PAI's pattern. External helpers, single source-of-truth dict.

**PAI-side perspective.** Confirmed PAI's pattern at `npu_model.py:1579–1604` (full source in §6 of `HARDWARE_TIER_EXTRACT.md`). One tweak worth flagging: PAI's `_DTYPE_ATTR` is leading-underscore-private. Downstream consumers (UI building a "supported dtypes" badge, JSON exporters, etc.) may want to introspect — the engine ratchet should expose it publicly.

PAI's own code at `npu_model.py:1981` already does:
```python
supported = [d for d in ("int8", "fp8", "bf16") if hw_supports_dtype(hw, d)]
```
That hardcoded tuple should derive from the canonical map's keys instead:
```python
supported = [d for d in DTYPE_ATTR_MAP if hw_supports_dtype(hw, d)]
```

**Ratchet recommendation.** 🟢 **Adopt PAI's external-helper pattern.** Rename `_DTYPE_ATTR` → `DTYPE_ATTR_MAP` (public). Both helpers stay as module-level functions taking `hw` first. PAI's two-function split (`hw_supports_dtype` returns bool, `hw_peak_tops_for_dtype` returns float) is good — different consumers want different return types; one function with a "return mode" arg would be worse.

Note: under INPUT 26's revised framing, `hw_supports_dtype` **stays binary** (orthogonal to the 4-level capability accessor — see INPUT 26's "PAI's encoding deltas" table and the existing PAI comment block at `sizer/precision.py:96–108`). The 4-level capability lives in a separate `hw_capability_levels()` accessor. The "external helpers, single source-of-truth dict" pattern stands for both.

---

## INPUT 30 — Vision-specific Hardware field

**Analyst finding.** Keyhole has `measured_edge_ms` (pipeline_key → resolution → ms). PAI has no vision measurement attachment. Include `measured_edge_ms` as an optional Hardware field. PAI surfaces leave it `None`.

**PAI-side perspective.** Confirmed — PAI sizer is LLM-only. Vision measurements on the PAI side **do** exist but they live in the anchor-secrets system (`CNNAnchor` typed dataclass loaded from `.streamlit/secrets.toml` via `load_cnn_anchor()`), not on Hardware. See `ANCHOR_SECRETS_LOADER_EXTRACT.md` §1.

So there are **two** vision-measurement paths candidate for unification:
- Keyhole's `measured_edge_ms` field on Hardware (public, source-tree-resident).
- PAI's `cnn_anchors.{tier_prec}.{cnn_key}` schema in private Streamlit secrets (private).

These are **not** the same thing — keyhole's are sourced from public bake-offs; PAI's are private silicon measurements. The ratchet should preserve the public/private distinction.

**Ratchet recommendation.** 🟡 **Add `measured_edge_ms` as an optional Hardware field**, structurally aligned with the unified `measured_llm` nested-dict shape from INPUT 24:

```python
measured_edge_ms: dict[str, dict[str, float]] | None = None
# {pipeline_key: {resolution: ms_per_inference}}
```

PAI's CNN anchor-secrets path stays separate (private overlay), exactly like the LLM anchor-secrets path stays separate from `measured_llm` today.

**Sub-detail to settle:** should the engine offer a single `measured` attachment field that holds *both* LLM and vision cells in a discriminated nested shape, or keep them as two distinct fields (`measured_llm`, `measured_edge_ms`)? See cross-input concern below.

---

## Cross-input concerns

### Concern 1 — Measurement-attachment story is tangled across 4 paths

INPUTs 24 + 30 + the anchor-secrets system together imply **four** measurement-attachment paths on the PAI side post-ratchet:

| Path | Scope | Privacy | Today's PAI location |
|---|---|---|---|
| 1. LLM bake-off catalog | per (tier, model, workload) | public | `Hardware.measured_llm` (5090 only, populated by `sizer/measured.py` from `sizer_bundle.json`) |
| 2. Tier-level LLM override (legacy) | per (tier, model), workload-agnostic | public | `Hardware.measured_decode_overrides` + `measured_prefill_overrides` (NPU_MID only) |
| 3. LLM private anchor (real silicon) | per (tier, precision, model) | **private** | `npu_anchors.load_llm_anchor()` → `LLMAnchor` dataclass |
| 4. Vision private anchor | per (tier, precision, cnn_key) | **private** | `npu_anchors.load_cnn_anchor()` → `CNNAnchor` dataclass |

Plus keyhole brings:
- 5. Vision bake-off catalog (`measured_edge_ms`, public).
- 6. Legacy flat fields (`measured_llm_q4_decode_tok_s`, public).

**A clean ratchet collapses (1) + (2) + (5) + (6) into a single Hardware-side measurement attachment** (a unified nested-dict schema discriminated by workload class). **Paths (3) and (4) stay separate** because they're runtime private-secrets overlays — they don't belong in source-tree-resident objects.

Recommended unified Hardware-side field:
```python
measured: dict[str, dict[str, dict[str, float]]] | None = None
# Outer key:  workload class — "llm" | "vision"
# Middle key: model_key or pipeline_key
# Inner key:  workload_id or resolution
# Innermost:  metric name → value
#
# Example:
#   measured["llm"]["qwen3-30b-a3b-q4-moe"]["chat_short"] = {
#       "decode_tok_s": 37.85, "prefill_tok_s": 2849.0, ...
#   }
#   measured["vision"]["yolov8n_w4"]["640x640"] = {
#       "ms_per_inference": 12.3, "fps": 81.3, ...
#   }
```

This subsumes INPUT 24's `measured_llm` + INPUT 30's `measured_edge_ms` into one shape. Anchor-secrets stays as a parallel private-overlay system (paths 3 + 4 retain their current architecture).

**Open question to reviewer:** is the unified `measured: dict[class, ...]` shape too cute? A flatter `measured_llm` + `measured_edge_ms` pair (two fields) is more obvious in IDE autocomplete and slightly less indirection. Trade-off: discoverability vs single-source-of-truth.

### Concern 2 — Sequencing of refactors

The 8 INPUTs aren't independent. Suggested order to minimize churn:

1. **First batch (low-risk lifts):** INPUTs 23, 25, 28, 29. These are shape-only or registry changes with limited call-site ripple.
2. **Second batch (medium-risk consolidation):** INPUT 24 (measurement attachment unification) + INPUT 30 (vision field addition). Do together — they share the nested-dict schema decision in Concern 1.
3. **Third batch (lifts + consolidation, lower risk than originally estimated):** INPUT 26 (4-level capability taxonomy). After locating PAI's existing 4-level surface in `sizer/precision.py:115–153`, this is consolidation work, not introduction. Still surfaces a new `CapabilityLevel` enum across both downstreams; sequence after the shape-only batches but doesn't have to wait for measurement-attachment.
4. **Fourth batch (new feature, not a refactor):** INPUT 27 (custom tier factory). Additive; safe to do anytime but easiest after INPUT 23's baseline is settled.

### Concern 3 — Backwards-compat during the migration

Both PAI sizer and keyhole-sizer ship as live Streamlit Cloud apps. The engine ratchet can't take both surfaces offline simultaneously. Two viable migration strategies:

- **(a) Vendor first, switch later.** Engine package gets developed in a new repo (`personal-ai-engine` or similar). Both sizers vendor it as a git submodule or PyPI install. They migrate to engine-backed Hardware one surface at a time, with the legacy `sizer/npu_model.py` kept as a thin re-export shim until both sides are switched.
- **(b) Co-develop in one repo.** Pick one sizer as the engine host (probably keyhole-sizer since it has more historical surface area), develop the unified Hardware there, and have the other (PAI) consume via import.

PAI-side preference: **(a)** — clean separation of concerns, easier to reason about the engine's own API as a package rather than as keyhole-sizer's internal module. Both sizers ratcheting to the same vendor source gives the canonical-spec discipline a real software artifact.

---

## Synthesized ratchet baseline

If reviewer accepts the recommendations above, the engine-package Hardware dataclass looks like:

```python
# engine/hardware.py (proposed)
from dataclasses import dataclass

@dataclass
class Hardware:
    """Generic compute-and-bandwidth spec for any NPU or GPU."""
    # ─── Required fields (lifted from PAI/keyhole consensus 20-field shape) ───
    name: str
    peak_tops_bf16: float
    peak_tops_int8: float
    peak_tops_fp8: float
    mem_bandwidth_gbs: float
    mem_capacity_gb: float
    mem_bus_width_bits: int
    mem_type: str
    mem_data_rate_gtps: float

    # ─── Calibration constants (consensus defaults) ───
    compute_efficiency: float = 0.65
    bandwidth_efficiency: float = 0.70
    tdp_watts: float = 0.0
    tier_family: str | None = None
    compute_util_factor: float = 0.45        # vision
    llm_prefill_util_factor: float = 0.10    # LLM
    llm_decode_bw_realization: float = 1.0
    compute_overhead_ms: float = 1.0
    npu_share_default: float = 0.75

    # ─── Stock identity (INPUT 25 — lifted from PAI) ───
    bw_projected: bool = False
    stock_mem_bandwidth_gbs: float | None = None
    stock_name: str | None = None

    # ─── Measurement attachment (INPUTs 24 + 30 — unified) ───
    # Option A: single discriminated dict
    measured: dict[str, dict[str, dict[str, float]]] | None = None
    #
    # Option B (alternative): two flat fields — more discoverable
    # measured_llm:       dict[str, dict[str, dict[str, float]]] | None = None
    # measured_edge_ms:   dict[str, dict[str, float]] | None = None

    # ─── Calibration provenance (INPUT 27 — new) ───
    _calibration_source: str | None = None

    # ─── Methods unchanged from PAI ───
    @property
    def effective_bandwidth_gbs(self) -> float: ...

    @property
    def tier_lookup_name(self) -> str: ...     # INPUT 25

    def effective_tops(self, dtype: str) -> float: ...

    def get_measured(self, workload_class: str, key: str, workload_id: str) -> dict | None: ...
    # ↑ signature change to accommodate discriminated `measured` dict


# Module-level (INPUT 29)
DTYPE_ATTR_MAP = {                              # ← renamed from _DTYPE_ATTR (public)
    "fp16": "peak_tops_bf16",
    "bf16": "peak_tops_bf16",
    "fp8":  "peak_tops_fp8",
    "int8": "peak_tops_int8",
}

# INPUT 26 — 4-level capability taxonomy (unifies PAI's tier_precision_capability +
# keyhole's capability_levels; lifts string constants to enum)
class CapabilityLevel(Enum):
    TENSOR_NATIVE = "tensor_native"   # PAI: _CAP_TENSOR  / keyhole: tensor_native
    TENSOR_COMPAT = "tensor_compat"   # PAI: _CAP_COMPAT  / keyhole: tensor_compat
    CUDA_CORE     = "cuda_core"       # PAI: _CAP_CUDA    / keyhole: cuda_core
    UNSUPPORTED   = "unsupported"     # PAI: _CAP_NONE    / keyhole: unsupported

    def __bool__(self) -> bool:
        return self is not CapabilityLevel.UNSUPPORTED

# Binary "is there ANY path" — stays as-is (orthogonal to capability_levels)
# Per PAI's sizer/precision.py:96-108 comment block.
def hw_supports_dtype(hw: Hardware, dtype: str) -> bool: ...
def hw_peak_tops_for_dtype(hw: Hardware, dtype: str) -> float: ...

# 4-level "what KIND of path" — canonical accessor lifted from PAI's
# sizer/precision.py:115 tier_precision_capability(). Schema reconciled:
#   - fp16/bf16 kept as SEPARATE keys (more orthogonal than PAI's combined "bf16/fp16")
#   - q4_km kept as a peer precision (PAI pattern; weight-only quant flavor)
# Return: {precision_key: CapabilityLevel}
def hw_capability_levels(hw: Hardware) -> dict[str, CapabilityLevel]: ...

# UI/classifier helpers — lifted from PAI's sizer/precision.py:156-183
def capability_badge(level: CapabilityLevel) -> str: ...   # one glyph
def capability_label(level: CapabilityLevel) -> str: ...   # short text
def capability_color(level: CapabilityLevel) -> str: ...   # hex

# Deployment-path classifier — lifted from PAI's sizer/precision.py:277-310
# Settles the "tensor_compat = fast path?" question with a workload arg:
def deployment_path_for_tier(
    hw: Hardware,
    model_quant_scheme: str = "fp16",
    workload_kernel_source: str = "fresh_compile",   # "fresh_compile" | "precompiled"
) -> str:
    """tensor_compat counts as fast path when workload_kernel_source='precompiled'
    (TRT, ONNX Runtime, vendor SDKs); blocks when 'fresh_compile' (vLLM CUTLASS,
    TVM, JIT paths)."""
    ...

# INPUT 27 — custom tier factory
def make_custom_tier(
    name: str,
    peak_tops_int8: float,
    mem_bandwidth_gbs: float,
    mem_capacity_gb: float,
    mem_bus_width_bits: int,
    mem_type: str,
    mem_data_rate_gtps: float,
    peak_tops_bf16: float = 0.0,
    peak_tops_fp8: float = 0.0,
) -> Hardware:
    """Construct a non-canonical Hardware with default calibration constants.
    UI must surface a 🔶 banner — these defaults will over-project sub-5-TOPS silicon."""
    return Hardware(
        name=name,
        peak_tops_int8=peak_tops_int8,
        peak_tops_bf16=peak_tops_bf16,
        peak_tops_fp8=peak_tops_fp8,
        mem_bandwidth_gbs=mem_bandwidth_gbs,
        mem_capacity_gb=mem_capacity_gb,
        mem_bus_width_bits=mem_bus_width_bits,
        mem_type=mem_type,
        mem_data_rate_gtps=mem_data_rate_gtps,
        _calibration_source="default — NOT calibrated for this silicon",
    )

# INPUT 28 — union registry, per-surface composition
TIERS = {t.name: t for t in (
    NPU_LOW_LP4,
    NPU_LOW_LP5_32BIT,
    NPU_LOW_LP5_64BIT,
    NPU_LOW_LP5X,
    IMX95_MEASURED,          # ← lifted from keyhole side
    NPU_MID,
    NPU_HIGH,
    RTX_5090_REFERENCE,
)}
```

The vestigial bare `RTX_5090` constant from PAI is **not** carried forward.

---

## Open questions back to reviewer

1. **INPUT 24 sub-detail.** Tier-level-but-workload-agnostic override — sentinel workload_id (`"_any"`), per-workload replication at construction, or strict workload-scoped only? PAI's current semantics are workload-agnostic; pure shape demands one of the three. Recommend the sentinel for minimum semantic change.
2. **Cross-input Concern 1.** Unified `measured: dict[class, ...]` (one field, discriminated) vs separate `measured_llm` + `measured_edge_ms` (two fields, flatter)? Both work; pick discoverability or single-source-of-truth.
3. **INPUT 26 — partially resolved, new sub-questions raised.** PAI's `tier_precision_capability` (`sizer/precision.py:115–153`) **already implements the 4-level taxonomy** the analyst recommended adopting from keyhole — schema-equivalent with naming deltas. The ratchet's job is unification, not introduction. Sub-questions:
   - **(3a)** Canonical location — Hardware field (keyhole pattern) vs module-level function (PAI pattern) vs both with one source? PAI-side preference: **keep module-level function as the canonical accessor, dispatching on `hw.tier_lookup_name`** — couples cleanly to stock-identity tracking (INPUT 25) and matches the existing PAI shape. Hardware doesn't need a field.
   - **(3b)** fp16/bf16 — single combined `"fp16/bf16"` key (PAI) or two separate keys with equal values (keyhole)? Recommend separate (more orthogonal).
   - **(3c)** `q4_km` as a peer precision (PAI) or derived flag at model layer (keyhole)? Recommend peer precision (silicon-specific support story).
   - **(3d)** `tensor_compat` semantic at the deployment-path classifier — is it "fast path" or "no fast path"? Recommend making the classifier workload-source-aware (`precompiled` → fast, `fresh_compile` → blocked) per the synthesized baseline above.
4. **Migration strategy.** (a) Engine in new repo + both sizers vendor, or (b) co-develop in one sizer + the other imports? PAI-side preference: (a). Reviewer's call.
5. **Backwards-compat shim duration.** During the ratchet, does `sizer/npu_model.py` stay as a thin re-export of `engine.hardware` for one release cycle, or does the migration happen atomically? Atomic is cleaner but requires both sizers to ship the same day.

---

*Drafted 2026-05-19 by PAI sizer Claude session in response to reviewer's cross-repo synthesis INPUTs 23–30. Source data: `ANCHOR_SECRETS_LOADER_EXTRACT.md` + `HARDWARE_TIER_EXTRACT.md` in this repo (both at tag `v1.0.0`). Recommendations are PAI-side input — engine package design is a joint call.*
