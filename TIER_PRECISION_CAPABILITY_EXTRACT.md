# PAI sizer — `tier_precision_capability()` extract

**Repo:** `personal-ai-assistant-sizer`
**Generated:** 2026-05-19 (against `c1d56f9` — engine-extraction prep docs landed)
**Purpose:** Settle whether PAI already has something equivalent to keyhole's `capability_levels` dict before designing the ratchet's unified capability model. Direct response to reviewer's single-function extraction request.
**Scope:** Source for one function + its supporting constants + four sibling helpers + one downstream classifier. All callers + return-shape expectations.

---

## Location heads-up

The function lives at **`sizer/precision.py:115–153`**, not `app.py`. The `app.py:1078` site only *calls* it — it's imported from `sizer.precision` at `app.py:32`.

```python
# app.py:30-37 (import surface)
from sizer.precision import (
    MEASURED_PRECISION_QUALITY, MEASURED_PRECISION_SPEED,
    tier_precision_capability, capability_badge, capability_label,
    capability_color, quality_badge_text, quality_color,
    RETARGETING_COSTS, deployment_path_for_tier, retargeting_cost_color,
    REGRESSION_RIGOR, gates_per_cycle, annualized_testing_cost,
    DEPLOYMENT_MODELS,
)
```

---

## 1. The function body

```python
# sizer/precision.py:115-153 (full function body)
def tier_precision_capability(hw_name: str) -> dict[str, str]:
    """Return {precision: capability_level} for a given tier name."""
    # Hardcoded because this is silicon-family knowledge, not derivable
    # from the perf/BW numbers we store on Hardware. Extend as we add tiers.
    if hw_name.startswith("RTX 5090"):
        # Consumer Blackwell SM120. Earlier framing said "tensor-core INT8
        # dropped, DP4A CUDA-core fallback" — [backend]'s 2026-04-24 ncu
        # probe falsified that. The real story: SM120 HARDWARE has INT8
        # tensor-core capability (sm80 IMMA via binary compat — 13M tensor-
        # pipe instructions measured on yolov8n-seg INT8 TRT engine), but
        # SM120-native CUTLASS INT8 kernel templates don't exist yet. So:
        #   - Pre-compiled TRT INT8 (YOLO): ✓ runs on tensor cores via sm80 compat
        #   - Fresh CUTLASS INT8 (vLLM W8A8): ✗ blocks — no SM120 templates
        # Marked as _CAP_COMPAT (tensor-core via binary compat) — amber, not
        # green, because LLM-class workloads (the sizer's focus) are blocked
        # today even though the hardware is capable.
        return {
            "bf16/fp16": _CAP_TENSOR,
            "fp8":       _CAP_TENSOR,
            "int8":      _CAP_COMPAT,   # tensor-core via sm80 binary compat; vLLM/CUTLASS blocked
            "q4_km":     _CAP_TENSOR,   # weight-only, runs on bf16 tensor cores
        }
    if hw_name == "NPU Low-LP4":
        return {"bf16/fp16": _CAP_NONE,  "fp8": _CAP_NONE, "int8": _CAP_TENSOR, "q4_km": _CAP_NONE}
    if hw_name in ("NPU Low-LP5-32bit", "NPU Low-LP5-64bit"):
        return {"bf16/fp16": _CAP_NONE,  "fp8": _CAP_NONE, "int8": _CAP_TENSOR, "q4_km": _CAP_NONE}
    if hw_name == "NPU Low-LP5X":
        return {"bf16/fp16": _CAP_TENSOR, "fp8": _CAP_TENSOR, "int8": _CAP_TENSOR, "q4_km": _CAP_TENSOR}
    if hw_name == "NPU Mid":
        # Per [docs] 2026-04-29 14:58 spec correction: actual Mid silicon
        # is INT8-only, no FP path. bf16 / fp8 / fp16 capabilities are
        # not present on the chip. INT8 stays tensor_native; Q4_K_M
        # stays tensor_native (weight-only quant runs via the INT8
        # dequant path — that's what Skippy's measured anchor proves).
        return {"bf16/fp16": _CAP_NONE, "fp8": _CAP_NONE, "int8": _CAP_TENSOR, "q4_km": _CAP_TENSOR}
    if hw_name == "NPU High":
        return {"bf16/fp16": _CAP_TENSOR, "fp8": _CAP_TENSOR, "int8": _CAP_TENSOR, "q4_km": _CAP_TENSOR}
    # Default conservative — unknown tier
    return {"bf16/fp16": _CAP_NONE, "fp8": _CAP_NONE, "int8": _CAP_NONE, "q4_km": _CAP_NONE}
```

**Dispatch shape:** if/elif chain on `hw_name` string. Hardcoded because the capability data is silicon-family knowledge, not derivable from the perf/BW numbers stored on Hardware. Adding a new tier means adding a new branch.

**Tier coverage:** 6 tier-name branches (`RTX 5090*`, `NPU Low-LP4`, `NPU Low-LP5-32bit`, `NPU Low-LP5-64bit`, `NPU Low-LP5X`, `NPU Mid`, `NPU High`) + 1 default-conservative fallback. The `startswith("RTX 5090")` prefix-match catches both the bare `RTX 5090` constant (unused in TIERS) and the registered `RTX 5090 (reference, measured)` tier with the same caps.

---

## 2. Supporting constants and helpers

### Capability-level constants (the canonical taxonomy)

```python
# sizer/precision.py:96-112 (capability-level constants + design rationale)
# Precision capability per silicon class. A tier can "support" a precision
# in two flavors:
#   - "tensor_core":  native tensor-core matmul (fast, preferred)
#   - "tensor_compat": tensor-core hardware accessed via binary compat
#                      (e.g. sm80 IMMA on SM120). Works for workloads with
#                      pre-compiled kernel libraries (TRT YOLO), blocked
#                      for workloads that compile fresh kernels per-arch
#                      (vLLM CUTLASS LLM serving)
#   - "cuda_core":    DP4A or equivalent (works, significantly slower)
#   - "none":         can't execute this precision at all
#
# This is ORTHOGONAL to peak_tops_<dtype> which only asks "is there any
# path." For LLM-scale matmul, you almost always want tensor_core —
# cuda_core is acceptable for small CNN inference, crippling for 14B+.
_CAP_TENSOR = "tensor_core"
_CAP_CUDA   = "cuda_core"
_CAP_COMPAT = "tensor_compat"   # tensor-core via binary compat (e.g. sm80 IMMA on SM120)
_CAP_NONE   = "none"
```

The four constants are leading-underscore-private (only consumed within `sizer/precision.py`). The comment block at `sizer/precision.py:96–108` is **load-bearing**: it explicitly calls out that this surface is *orthogonal* to PAI's binary `hw_supports_dtype(hw, dtype)` (`sizer/npu_model.py:1587`). Two different questions, two different surfaces.

### UI-side level → display helpers

```python
# sizer/precision.py:156-183 (three helpers — glyph / text / color)
def capability_badge(level: str) -> str:
    """One-glyph summary of capability level for UI."""
    return {
        _CAP_TENSOR: "✓",
        _CAP_CUDA:   "⚠︎",
        _CAP_COMPAT: "⚠︎",
        _CAP_NONE:   "✗",
    }.get(level, "?")


def capability_label(level: str) -> str:
    """Short text label for capability level."""
    return {
        _CAP_TENSOR: "tensor-core",
        _CAP_CUDA:   "CUDA-core only",
        _CAP_COMPAT: "tensor-core (compat path)",
        _CAP_NONE:   "not supported",
    }.get(level, "unknown")


def capability_color(level: str) -> str:
    """CSS color for capability level (green/amber/red)."""
    return {
        _CAP_TENSOR: "#10b981",   # green
        _CAP_CUDA:   "#f59e0b",   # amber
        _CAP_COMPAT: "#f59e0b",   # amber — same as cuda_core in UI
        _CAP_NONE:   "#ef4444",   # red
    }.get(level, "#6b7280")
```

All three are pure level-string → display-element maps. Unknown levels degrade gracefully (returns `?` / `unknown` / `#6b7280` neutral gray).

### Downstream classifier (the function's primary non-UI consumer)

```python
# sizer/precision.py:277-310 (deployment-path classifier — consumes the capability dict)
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
    if has_fp_path:
        return "fp_native"

    # Unknown / no supported path
    return "ptq"
```

Returns one of `"fp_native"` / `"weight_only"` / `"ptq"`. **Critical:** this classifier only checks `== _CAP_TENSOR` — it treats `_CAP_COMPAT`, `_CAP_CUDA`, `_CAP_NONE` as equivalent "no fast path." The 4-level distinction **collapses to binary** at this consumer.

---

## 3. What does this function return?

**A 4-level taxonomy — schema-equivalent to keyhole's `capability_levels`, with naming deltas:**

| Keyhole level | PAI level (string value) | PAI constant |
|---|---|---|
| `tensor_native` | `tensor_core` | `_CAP_TENSOR` |
| `tensor_compat` | `tensor_compat` | `_CAP_COMPAT` |
| `cuda_core` | `cuda_core` | `_CAP_CUDA` |
| `unsupported` | `none` | `_CAP_NONE` |

**Bottom line: PAI already has the 4-level taxonomy.** Just in a different location than expected (separate module, dispatch function rather than Hardware field) and with minor naming deltas.

### Return value structure

```python
# Example: tier_precision_capability("NPU Mid")
{
    "bf16/fp16": "none",        # _CAP_NONE  — NPU Mid is INT8-only
    "fp8":       "none",        # _CAP_NONE
    "int8":      "tensor_core", # _CAP_TENSOR
    "q4_km":     "tensor_core", # _CAP_TENSOR — weight-only via INT8 dequant
}

# Example: tier_precision_capability("RTX 5090 (reference, measured)")
{
    "bf16/fp16": "tensor_core",  # _CAP_TENSOR
    "fp8":       "tensor_core",  # _CAP_TENSOR
    "int8":      "tensor_compat",# _CAP_COMPAT — sm80 IMMA via binary compat
    "q4_km":     "tensor_core",  # _CAP_TENSOR
}
```

### Key schema differences vs keyhole (ratchet-design-relevant)

1. **PAI conflates fp16/bf16 into a single key `"bf16/fp16"`** — encoded at the precision-dimension layer, mirroring `_DTYPE_ATTR`'s field-routing conflation in `sizer/npu_model.py:1579`. Keyhole presumably keys them separately.

2. **PAI adds a `"q4_km"` key** — weight-only-quant flavor that does **not** appear in `_DTYPE_ATTR` (which has fp16/bf16/fp8/int8 only). Critical for the deployment-path classifier: Q4_K_M support hinges on whether the underlying fp16 dequant path is tensor-core-supported. On NPU Mid the Q4_K_M support flows through the INT8 dequant path; on NPU High through the fp16 dequant path.

3. **PAI looks up tiers by string match on `hw.tier_lookup_name`, not by Hardware instance.** Per INPUT 25's stock-identity tracking framing, this is the **correct** coupling — memory-upgrade overlays still resolve to stock silicon caps. Confirmed via the actual call site:
   ```python
   # app.py:1078
   _cap = tier_precision_capability(hw.tier_lookup_name)
   ```

4. **PAI uses string literals as capability values**, not an enum. The four `_CAP_*` module constants are convenience aliases. The ratchet's `CapabilityLevel(Enum)` proposal (per `ENGINE_RATCHET_PLAN.md` INPUT 26) is a tighter encoding.

5. **PAI's `_CAP_CUDA` is defined but currently unused** by any tier in the dict — vestigial. The 4-level shape supports CUDA-core fallback semantics, but no tier in PAI's current ladder is in that state. Could be removed without changing behavior; should be **kept** in the ratchet's enum because the case will recur on future silicon classes (e.g., GPU INT4/INT2 paths that fall back to non-tensor cores).

### What makes this clearly 4-level (not binary)

The RTX 5090 INT8 entry is `_CAP_COMPAT`, not `_CAP_TENSOR` and not `_CAP_NONE`. That's the load-bearing piece: PAI is actively encoding the *third* state ("tensor cores work via binary compat but vLLM CUTLASS LLM serving is blocked") — exactly the case keyhole's `tensor_compat` exists for. The extensive comment block at `sizer/precision.py:120–130` documents the [backend] 2026-04-24 ncu-probe finding that drove this encoding (13M tensor-pipe instructions measured on yolov8n-seg INT8 TRT engine, sm80 IMMA via binary compat).

---

## 4. Call-site shape expectations

Two consumers exist in this repo:

| Consumer | File:line | Shape expected | How the 4 levels are handled |
|---|---|---|---|
| Precision-capability UI grid | `app.py:1078–1118` | `dict[precision_key, capability_level_str]` with the 4 precision keys `bf16/fp16` / `fp8` / `int8` / `q4_km` | All 4 levels honored via `capability_color()` / `capability_badge()` / `capability_label()`. Defaults missing precisions to `"none"` via `.get(cap_key, "none")`. |
| Deployment-path classifier | `sizer/precision.py:277–310` (`deployment_path_for_tier`) | Same dict | Only checks `== _CAP_TENSOR`. **`_CAP_COMPAT`, `_CAP_CUDA`, `_CAP_NONE` collapse to "no fast path"** — the 4-level distinction reduces to binary here. |

### UI consumer detail (app.py:1078–1118)

```python
# app.py:1078-1118 (precision-capability rendering grid — 4 columns)
_cap = tier_precision_capability(hw.tier_lookup_name)
_precision_columns = [
    ("bf16/fp16", "fp16",  "fp16/bf16"),
    ("fp8",       "fp8",   "FP8"),
    ("int8",      "int8",  "INT8 (W8A8)"),
    ("q4_km",     "q4_km", "Q4_K_M"),
]
_cols = st.columns(4)
for col, (cap_key, quality_key, display) in zip(_cols, _precision_columns):
    cap_level = _cap.get(cap_key, "none")
    cap_color = capability_color(cap_level)
    cap_glyph = capability_badge(cap_level)
    cap_text  = capability_label(cap_level)
    # ... renders a colored card with glyph + level text
```

The 4-precision rendering grid (`bf16/fp16` / `fp8` / `int8` / `q4_km`) is locked to the dict's key set. Adding a precision key in `tier_precision_capability` would not automatically surface in the UI; the consumer's `_precision_columns` tuple would also need updating.

### Classifier consumer detail (deployment_path_for_tier)

The classifier's binary-collapse behavior is **intentional** for the SM120 INT8 case: vLLM CUTLASS LLM serving is blocked even though the hardware works for pre-compiled TRT INT8 — so for the deployment-path decision (which assumes fresh-compile LLM-serving workloads), `tensor_compat` correctly maps to "needs alternative path." But this means **the same dict serves two consumers with very different semantics** for `tensor_compat`.

This is the single most important sub-detail for the ratchet's unified capability model: **does `tensor_compat` count as a "fast path"?** PAI's current answer is "yes for UI, no for deployment classifier." A workload-source-aware classifier signature (the `workload_kernel_source: "precompiled" | "fresh_compile"` arg proposed in `ENGINE_RATCHET_PLAN.md` §INPUT 26) would resolve the ambiguity without losing either semantic.

### Caveats for the ratchet

- **The `_DTYPE_ATTR` map in `sizer/npu_model.py:1579` uses `fp16` / `bf16` as separate keys**, but `tier_precision_capability` uses the combined `"bf16/fp16"` key. Anyone reconciling these surfaces in the engine package needs to pick one keying convention (the recommendation in `ENGINE_RATCHET_PLAN.md` Q3b is to use separate keys at the canonical layer and let `_DTYPE_ATTR` do field-routing conflation).
- **`q4_km` is a peer precision in `tier_precision_capability` but absent from `_DTYPE_ATTR`.** Engine-package design must decide whether weight-only-quant variants are peer precisions (PAI's choice, surfaces silicon-specific support story) or derived flags at the model layer (alternative pattern).
- **Unknown tier names fall through to default-conservative `{all: "none"}`** at the function tail. A custom-tier construction (per `ENGINE_RATCHET_PLAN.md` INPUT 27) would surface as `{all: "none"}` unless `make_custom_tier()` is paired with a capability-registration mechanism. This is a real design gap.
- **The tier-name string-match dispatch is brittle to renaming.** "NPU Mid" appearing in the function as a hardcoded string means any rename in `npu_model.py` silently breaks capability lookup. The engine package should consider keying off a tier-id symbol (e.g., the Hardware constant identity or a `tier_id: str` field) rather than the display `name`.

---

## Bottom-line answer to the ratchet's capability-model question

**PAI already has the 4-level capability taxonomy** that the analyst recommended adopting from keyhole. INPUT 26's framing of "PAI is binary" was half-right (the `hw_supports_dtype()` surface is binary, but it's orthogonal to the capability question, per the explicit comment block at `sizer/precision.py:96–108`).

The ratchet's job for INPUT 26 is therefore **unification of two existing 4-level surfaces** (PAI's `tier_precision_capability` + keyhole's `capability_levels`), not introduction of a new taxonomy. Recommended shape per the revised `ENGINE_RATCHET_PLAN.md` §INPUT 26:

1. Lift the 4-level taxonomy to an enum (`CapabilityLevel.TENSOR_NATIVE` / `TENSOR_COMPAT` / `CUDA_CORE` / `UNSUPPORTED`). Adopt keyhole's level names (more descriptive than PAI's `tensor_core` / `none` synonyms).
2. Keep PAI's module-level function pattern (`hw_capability_levels(hw)` dispatching on `hw.tier_lookup_name`) as the canonical accessor — couples cleanly to INPUT 25 stock-identity tracking.
3. Keep `hw_supports_dtype()` as the binary "is there any path" accessor — they answer different questions; both stay.
4. Resolve schema deltas: separate `fp16`/`bf16` keys, keep `q4_km` as peer precision, lift the string constants to enum members.
5. Add a `workload_kernel_source` arg to `deployment_path_for_tier()` so `tensor_compat` correctly maps to fast-path-vs-blocked depending on workload kernel source.

See `ENGINE_RATCHET_PLAN.md` (committed at `c1d56f9`) for the full revised recommendation and the synthesized engine-package skeleton.

---

*Extract generated 2026-05-19. Repo state: `c1d56f9`. Lifts `sizer/precision.py:96–183` + `277–310` + `app.py:1078–1118`. Discipline: no anchor-secrets values (this function reads no Streamlit secrets — purely silicon-family metadata, all public).*
