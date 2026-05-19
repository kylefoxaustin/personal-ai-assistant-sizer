# PAI sizer — Hardware tier object extract

**Repo:** `personal-ai-assistant-sizer`
**Generated:** 2026-05-19 (against `7e34df0` / tag `v1.0.0`)
**Purpose:** Cross-repo comparison with keyhole-sizer's parallel extract.
**Source file:** `sizer/npu_model.py` (2280+ lines).
**Discipline:** No anchor-secrets values (`.streamlit/secrets.toml` content). Public canonical NPU spec values (peak TOPS, mem_bandwidth_gbs, mem_data_rate_gtps, etc.) are present — those are documented in `personal-ai-framework/docs/` and the deck.

---

## 1. The Hardware dataclass definition

**File:** `sizer/npu_model.py:22–177`
**Decorator:** `@dataclass` (plain — **not** `frozen`, **not** `slots`)
**Post-init:** none (no `__post_init__` method defined)
**Class-level constants:** none on the class body (no `DEFAULT_BW_SHARE = ...` class-attribute pattern). Module-level constants (`_DTYPE_ATTR`, `RUNTIME_OVERHEAD_BYTES`, `MEMORY_UPGRADE_OPTIONS`) are documented in §3 and §6 below.

```python
# sizer/npu_model.py:22-177 (full Hardware definition, fields + methods)
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
    # [...full comment block at lines 39-49...]
    tier_family: str | None = None

    compute_util_factor: float = 0.45       # vision compute-floor calibration
    llm_prefill_util_factor: float = 0.10   # LLM prefill compute-floor calibration
    llm_decode_bw_realization: float = 1.0  # LLM decode BW realization fraction
    compute_overhead_ms: float = 1.0        # kernel launch / sync overhead

    npu_share_default: float = 0.75         # default BW share to NPU on SoC

    measured_llm: dict[str, dict[str, dict[str, float]]] | None = None

    bw_projected: bool = False
    stock_mem_bandwidth_gbs: float | None = None
    stock_name: str | None = None

    measured_decode_overrides: dict[str, float] | None = None
    measured_prefill_overrides: dict[str, float] | None = None
```

### Field set (full table — 24 fields)

| # | Field | Type | Default | Mutability | Comment |
|---|---|---|---|---|---|
| 1 | `name` | `str` | (required) | mutable | Display name. Rewritten by `hw_with_memory()` to add the variant suffix (e.g., "NPU Mid (LPDDR6 @ 14 GT/s)"). |
| 2 | `peak_tops_bf16` | `float` | (required) | mutable | Raw peak BF16/FP16 TOPS. `0.0` means tier has no FP16 path (e.g., NPU Mid is INT8-only). |
| 3 | `peak_tops_int8` | `float` | (required) | mutable | Raw peak INT8 TOPS. |
| 4 | `peak_tops_fp8` | `float` | (required) | mutable | Raw peak FP8 TOPS. `0.0` on tiers without FP8 path. |
| 5 | `mem_bandwidth_gbs` | `float` | (required) | mutable | **Raw** peak DRAM BW in GB/s. Effective BW = this × `bandwidth_efficiency` × `npu_share`. Rewritten by `hw_with_memory()` (`= mem_bus_width_bits × mem_data_rate_gtps / 8`). |
| 6 | `mem_capacity_gb` | `float` | (required) | mutable | DRAM capacity in GB. |
| 7 | `mem_bus_width_bits` | `int` | (required) | mutable | Memory bus width (32 / 64 / 128 / 512). Used by `hw_with_memory()` to recompute BW under data-rate swaps. |
| 8 | `mem_type` | `str` | (required) | mutable | "LPDDR4" / "LPDDR5" / "LPDDR5X" / "LPDDR5T" / "LPDDR6" / "GDDR7". |
| 9 | `mem_data_rate_gtps` | `float` | (required) | mutable | DRAM I/O data rate in GT/s. The anchor-secrets overlay uses `abs(mem_data_rate_gtps - 8.4) > 0.05` to detect memory-upgrade variants and skip the hot-swap. |
| 10 | `compute_efficiency` | `float` | `0.65` | mutable | Effective TOPS multiplier (Hardware.effective_tops()). Per-tier defaults: 0.60 (Low), 0.65 (Mid), 0.70 (High / 5090). |
| 11 | `bandwidth_efficiency` | `float` | `0.70` | mutable | Effective BW multiplier (Hardware.effective_bandwidth_gbs property). 0.70 NPUs / 0.85 5090. |
| 12 | `tdp_watts` | `float` | `0.0` | mutable | Thermal envelope; informational only (no power modeling per remediation P0 — out of scope). |
| 13 | `tier_family` | `str \| None` | `None` | mutable | Memory-class taxonomy (Class 1–5 per [backend] 2026-04-29 13:07). Same-family anchors BW-scale within family (🟡); cross-family falls through to MAX(BW, compute) (🔴). |
| 14 | `compute_util_factor` | `float` | `0.45` | mutable | **Vision** compute-floor utilization. Per-tier: 0.19 Neutron / 0.45 Mid / 0.50 High / 0.85 5090. **Vision-only** — does not apply to LLM. |
| 15 | `llm_prefill_util_factor` | `float` | `0.10` | mutable | **LLM** prefill compute-floor utilization. Calibrated separately from vision because LLM prefill realizes 5–15% on edge NPUs. 0.10 Mid / 0.11 High. |
| 16 | `llm_decode_bw_realization` | `float` | `1.0` | mutable | LLM decode BW realization fraction. Default `1.0` (ceiling = active_params_GB / effective_BW). Carried at default on cross-class cells because realization is model-class-specific. |
| 17 | `compute_overhead_ms` | `float` | `1.0` | mutable | Per-tier kernel launch + sync overhead. Default 1.0 NPU / 0.3 GPU. |
| 18 | `npu_share_default` | `float` | `0.75` | mutable | Default fraction of memory BW available to NPU. 0.75 SoC NPUs (shared bus) / 1.0 5090 (dedicated VRAM). User-overridable via sidebar selector. |
| 19 | `measured_llm` | `dict[...] \| None` | `None` | mutable | Per-`(model_key, workload_id)` measured decode tok/s + TTFT. Populated **only on `RTX_5090_REFERENCE`** at import time by `sizer/measured.py` from `sizer_bundle.json`. Schema: `{model_key: {workload_id: {"decode_tok_s", "ttft_s", "prefill_tok_s", "host_ms"}}}`. |
| 20 | `bw_projected` | `bool` | `False` | mutable | `True` iff the Hardware was synthesized via `hw_with_memory()` (memory-upgrade what-if). UI marks BW-scaled LLM tok/s as "(BW-proj)" so users don't confuse a what-if with a vendor measurement. |
| 21 | `stock_mem_bandwidth_gbs` | `float \| None` | `None` | mutable | Snapshot of stock peak BW captured by `hw_with_memory()`. Lets `project_llm` hold prefill at stock under memory-upgrade overlays (prefill is compute-bound). |
| 22 | `stock_name` | `str \| None` | `None` | mutable | Snapshot of stock tier name captured by `hw_with_memory()`. Surfaced via `tier_lookup_name` property for silicon-intrinsic lookups (precision capability, deployment path) — those key off the **stock** tier regardless of memory variant. |
| 23 | `measured_decode_overrides` | `dict[str, float] \| None` | `None` | mutable | Tier-level measured decode tok/s override per `model_key`. Currently populated on `NPU_MID` only: `{"qwen3-30b-a3b-q4-moe": 37.85}` (Skippy MoE Q4_K_M bake-off on real INT8 silicon vs 5090-projection ~13.9). Tactical interim per [docs] 2026-04-29 12:34; Phase 2 compute clamp will replace. |
| 24 | `measured_prefill_overrides` | `dict[str, float] \| None` | `None` | mutable | Tier-level measured prefill tok/s override per `model_key`. Currently populated on `NPU_MID` only: `{"qwen3-30b-a3b-q4-moe": 2849.0}` (derived from 351 ms TTFT @ 1K prompt). Held at stock under memory upgrades. |

### Frozen / mutability summary

- The dataclass is **mutable end-to-end.** No `frozen=True` flag. The codebase relies on mutation at exactly **two** points:
  1. `sizer/measured.py:58, 114, 189` — `RTX_5090_REFERENCE.measured_llm = measured` (populated at import).
  2. `sizer/measured.py:131` — `cell_14b["decode_tok_s"] = ...` (Tier 3 14B refresh).
- All other paths construct new Hardware instances via `dataclasses.replace(...)` (in `hw_with_memory`) or define them at module top level.

### Post-init logic

**None.** Hardware has no `__post_init__` — every field is set from constructor arguments and defaults. Derived quantities (effective BW, effective TOPS, lookup name) are exposed as `@property` accessors or methods.

### Class-level constants

None on the dataclass. Module-level constants relevant to Hardware:

| Constant | File:line | Purpose |
|---|---|---|
| `RUNTIME_OVERHEAD_BYTES = 1_000_000_000` | `npu_model.py:1629` | 1 GB runtime overhead (llama-cpp + CUDA graphs + activation buffers). Used in feasibility checks. |
| `MEMORY_UPGRADE_OPTIONS` | `npu_model.py:388–392` | Ascending list of `(label, mem_type, mem_data_rate_gtps)` tuples consumed by the sidebar memory-upgrade selectbox. |
| `_DTYPE_ATTR` | `npu_model.py:1579–1584` | Dtype-name → Hardware-field-name map (see §6). |

---

## 2. All methods on the Hardware class

Four methods total: two `@property` accessors + two regular methods. No classmethods, no staticmethods.

### `effective_bandwidth_gbs` (property)

```python
# sizer/npu_model.py:143-145
@property
def effective_bandwidth_gbs(self) -> float:
    return self.mem_bandwidth_gbs * self.bandwidth_efficiency
```

- **Signature:** `hw.effective_bandwidth_gbs` (no args).
- **Docstring:** none.
- **What it returns:** raw peak BW × `bandwidth_efficiency`. Does **not** include `npu_share` — that composes downstream in `project_llm()`.
- **Called from:**
  - **Projection code:** `npu_model.py:1854, 2147, 2162` (decode + prefill BW-floor math in `project_llm` / `decode_tok_s_at_context`).
  - **UI code:** `app.py:308, 1308, 1569, 1625` (sidebar caption, performance tab, anchor section, memory-upgrade preview).
  - **Both** — most-touched derived field on Hardware.

### `tier_lookup_name` (property)

```python
# sizer/npu_model.py:147-154
@property
def tier_lookup_name(self) -> str:
    """Stock tier name for silicon-intrinsic lookups (precision
    capability, deployment path). Memory-only upgrades inherit silicon
    caps from the stock tier — `hw_with_memory()` rewrites `name` to
    surface the variant in display strings, but precision / dtype
    capabilities don't change."""
    return self.stock_name if self.stock_name is not None else self.name
```

- **Signature:** `hw.tier_lookup_name` (no args).
- **What it returns:** `self.stock_name` if set (memory-upgrade variant), else `self.name`. Silicon-intrinsic lookups (precision-capability table, deployment-path map) key off this — a memory swap doesn't change tensor-core support or dtype capability.
- **Called from:**
  - **Projection code:** `npu_model.py:1832, 2056, 2062` (anchor-tier matching in `project_llm` / `decode_tok_s_at_context`).
  - **UI code:** `app.py:693, 1078, 1614, 1830` (source banner, precision-capability lookup, memory-upgrade preview, deployment-path lookup).
  - **Both.**

### `effective_tops(dtype)` (method)

```python
# sizer/npu_model.py:156-163
def effective_tops(self, dtype: str) -> float:
    peak = {
        "int8": self.peak_tops_int8,
        "fp8": self.peak_tops_fp8,
        "bf16": self.peak_tops_bf16,
        "fp16": self.peak_tops_bf16,
    }.get(dtype.lower(), self.peak_tops_bf16)
    return peak * self.compute_efficiency
```

- **Signature:** `hw.effective_tops(dtype: str) -> float`.
- **Docstring:** none.
- **What it returns:** raw peak TOPS for the requested dtype × `compute_efficiency`. **fp16 ↔ bf16 conflation:** both route to `peak_tops_bf16` (per the inline dict — most edge SoCs treat fp16 and bf16 as the same tensor-core class).
- **Note:** an unrecognized dtype string silently falls back to `peak_tops_bf16` (no validation; not fail-loud).
- **Called from:**
  - **Projection code:** vision compute-floor calculations in `project_llm` (cross-class fallback for vision workloads). **Not** used for the LLM compute-floor — that path uses the module-level `hw_peak_tops_for_dtype()` (see §6) against raw peak without the `compute_efficiency` multiplier, because LLM prefill util factors were calibrated against raw peak.
  - **UI code:** none directly.

### `get_measured(model_key, workload_id)` (method)

```python
# sizer/npu_model.py:165-177
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
```

- **Signature:** `hw.get_measured(model_key: str, workload_id: str) -> dict | None`.
- **Docstring:** none (logic explained in the comment).
- **What it returns:** the `(model_key, workload_id)` cell from `measured_llm` (a dict like `{"decode_tok_s": ..., "ttft_s": ..., ...}`), or `None`. Falls through to the `measurement_alias` sibling on miss (catalog-level alias lookup via `MODELS`).
- **Note:** this is the **only** Hardware method that reads outside the dataclass — it indexes `MODELS` (module-level dict, see §5) to resolve aliases.
- **Called from:**
  - **Projection code only:** `npu_model.py:2044` inside `project_llm`. Returns the measured cell that the green-badge "🟢 measured_anchor" path consumes.
  - **UI code:** not called directly.

### Module-level functions that act on Hardware

Not methods, but operate on Hardware instances (call site of `hw` as first arg). See §6 for `hw_supports_dtype` and `hw_peak_tops_for_dtype`. Two more:

| Function | File:line | Signature | Purpose |
|---|---|---|---|
| `hw_with_memory` | `npu_model.py:395` | `(hw, mem_type, mem_data_rate_gtps, name_suffix=None) -> Hardware` | Memory-upgrade overlay (see §3). |
| `describe_hw` | `npu_model.py:1875` | `(hw) -> str` | Returns a one-line caption "*peak_tops* TOPS · *mem_bandwidth* GB/s (*effective* GB/s effective) · ..." for the sidebar. UI-only. |

---

## 3. Cloning / variant mechanisms

### `hw_with_memory()` — memory-upgrade overlay

```python
# sizer/npu_model.py:395-431 (full function)
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
    memory-only swap shouldn't move it. [...]

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
```

### Clone mechanism

- **Via `dataclasses.replace`** (imported at `npu_model.py:17` — `from dataclasses import dataclass, replace`).
- **NOT** via deep copy, **NOT** via fresh `Hardware(...)` construction.
- `replace` produces a new Hardware instance with the listed fields overridden and all others inherited from `hw`.

### Fields overridden in a clone

| Field | Source on clone | Notes |
|---|---|---|
| `name` | `f"{hw.name} ({name_suffix})"` if `name_suffix` else `hw.name` | UI surfaces variant suffix here |
| `mem_type` | new arg | e.g., "LPDDR5T" / "LPDDR6" |
| `mem_data_rate_gtps` | new arg | 11.2 / 12.0 / 14.0 per `MEMORY_UPGRADE_OPTIONS` |
| `mem_bandwidth_gbs` | `mem_bus_width_bits × mem_data_rate_gtps / 8` | recomputed from bus width |
| `bw_projected` | `True` | flag — distinguishes BW-projected from vendor-measured |
| `stock_mem_bandwidth_gbs` | `hw.mem_bandwidth_gbs` | snapshot of pre-swap BW |
| `stock_name` | `hw.name` | snapshot of pre-swap name (without parens) |

### Fields inherited unchanged from the stock tier

- `peak_tops_bf16`, `peak_tops_int8`, `peak_tops_fp8` — silicon-fixed.
- `mem_capacity_gb`, `mem_bus_width_bits` — physical bus width doesn't change with a faster chip.
- `compute_efficiency`, `bandwidth_efficiency` — silicon-fixed.
- `tdp_watts` — silicon-fixed (informational).
- **`tier_family`** — ✓ **preserved** on clones. A memory-upgrade overlay on `NPU_MID` (tier_family `"LP5X-8.4-128b"`) still reports `tier_family="LP5X-8.4-128b"` post-swap, so same-family anchor matching at `_find_same_family_anchor` still hits (`npu_model.py:2199–2230`).
- `compute_util_factor`, `llm_prefill_util_factor`, `llm_decode_bw_realization`, `compute_overhead_ms` — silicon-fixed calibration.
- `npu_share_default` — shared-bus characteristic; preserved.
- `measured_llm`, `measured_decode_overrides`, `measured_prefill_overrides` — measurement data is stock-tier indexed; preserved (the anchor on `NPU_MID` still applies after a memory swap because the model still runs on the same INT8 silicon).

### The `bw_projected` flag

- **Type:** `bool`, default `False`.
- **Set by:** `hw_with_memory()` (the only setter in the codebase).
- **Read by:**
  - `npu_model.py:1832` — `decode_tok_s_at_context`: refuses the "measured anchor on this tier" green-badge path if `hw.bw_projected` is True (a memory-upgrade variant must reproject, never reuse the stock anchor verbatim).
  - `npu_model.py:2063` — same logic inside `project_llm`.
  - `app.py:1614` — UI: appends "(BW-proj)" annotation to memory-upgrade tier displays.
- **Semantics:** "this Hardware is a what-if memory upgrade — anything BW-derived should be marked as a projection, not a vendor measurement."

### Is `tier_family` preserved on clones?

**Yes** — `dataclasses.replace` only overrides the listed kwargs; `tier_family` is inherited. The Hardware's docstring at lines 47–49 explicitly calls out memory-upgrade overlays as Class 4 in the taxonomy, projecting within the stock family. `_find_same_family_anchor` (`npu_model.py:2199–2230`) is documented to handle the memory-upgrade case by walking `bw_projected` chains back to the stock tier — confirmed at `npu_model.py:2204–2206`.

---

## 4. The TIERS registry

### Tier definitions

Eight Hardware constants are defined at module top level. Seven are registered in `TIERS`; one (`RTX_5090`, line 181) is **defined but not registered** — kept for backwards compatibility and bake-off-script imports.

```python
# sizer/npu_model.py:181-368 (abbreviated — full constructors below)
RTX_5090 = Hardware(...)               # line 181  (NOT in TIERS dict)
NPU_LOW_LP4 = Hardware(...)            # line 198
NPU_LOW_LP5_32BIT = Hardware(...)      # line 210
NPU_LOW_LP5_64BIT = Hardware(...)      # line 222
NPU_LOW_LP5X = Hardware(...)           # line 234
NPU_MID = Hardware(...)                # line 251  (carries measured_*_overrides)
NPU_HIGH = Hardware(...)               # line 305
RTX_5090_REFERENCE = Hardware(...)     # line 343  (measured_llm populated at import)

# Registry (line 360-368)
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
```

**Construction style:** hardcoded module-level constants. **No YAML loading**, no environment-variable lookups, no dynamic tier discovery. Bake-off measurements are layered onto `RTX_5090_REFERENCE.measured_llm` at import time by `sizer/measured.py` (see §5), but the tier *constants themselves* are pure Python.

### Per-tier key field values

| Tier | name | BF16 TOPS | INT8 TOPS | FP8 TOPS | BW (GB/s) | Cap (GB) | Bus (bits) | mem_type | Rate (GT/s) | comp_eff | bw_eff | TDP (W) | tier_family | compute_util_factor | llm_prefill_util_factor | npu_share_default |
|---|---|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|---|--:|--:|--:|
| `NPU_LOW_LP4` | "NPU Low-LP4" | 0.0 | 2.0 | 0.0 | 17.1 | 8.0 | 32 | LPDDR4 | 4.266 | 0.60 | 0.70 | 10.0 | `Neutron-32-LP4` | 0.19 | 0.10 | 0.75 |
| `NPU_LOW_LP5_32BIT` | "NPU Low-LP5-32bit" | 0.0 | 2.0 | 0.0 | 25.6 | 16.0 | 32 | LPDDR5 | 6.4 | 0.60 | 0.70 | 10.0 | `Neutron-32-LP5` | 0.19 | 0.10 | 0.75 |
| `NPU_LOW_LP5_64BIT` | "NPU Low-LP5-64bit" | 0.0 | 2.0 | 0.0 | 51.2 | 16.0 | 64 | LPDDR5 | 6.4 | 0.60 | 0.70 | 10.0 | `Neutron-64-LP5` | 0.19 | 0.10 | 0.75 |
| `NPU_LOW_LP5X` | "NPU Low-LP5X" | 50.0 | 100.0 | 100.0 | 67.2 | 16.0 | 64 | LPDDR5X | 8.4 | 0.60 | 0.70 | 10.0 | `LP5X-8.4-64b` | 0.45 | 0.10 | 0.75 |
| `NPU_MID` | "NPU Mid" | **0.0** | 200.0 | **0.0** | 134.4 | 24.0 | 128 | LPDDR5X | 8.4 | 0.65 | 0.70 | 25.0 | `LP5X-8.4-128b` | 0.45 | 0.10 | 0.75 |
| `NPU_HIGH` | "NPU High" | 200.0 | 400.0 | 400.0 | 134.4 | 32.0 | 128 | LPDDR5X | 8.4 | 0.70 | 0.70 | 40.0 | `LP5X-8.4-128b` | 0.50 | 0.11 | 0.75 |
| `RTX_5090_REFERENCE` | "RTX 5090 (reference, measured)" | 209.0 | 419.0 | 419.0 | 1792.0 | 32.0 | 512 | GDDR7 | 28.0 | 0.70 | 0.85 | 575.0 | `GDDR7-28` | 0.85 | 0.10 | **1.0** |

Bolded values are noteworthy:
- **NPU Mid is INT8-only** — `peak_tops_bf16 = 0.0` and `peak_tops_fp8 = 0.0` per the [docs] 2026-04-29 14:58 spec correction. FP-capable silicon is NPU High.
- **NPU Mid + NPU High share the same memory class** (`tier_family="LP5X-8.4-128b"`, BW = 134.4, same bus). The tier difference is **compute-only.**
- **RTX 5090 has `npu_share_default = 1.0`** — dedicated VRAM, no shared-bus contention.

### Per-tier overrides on `NPU_MID` only

```python
# sizer/npu_model.py:291-302 (inside NPU_MID constructor)
measured_decode_overrides={
    "qwen3-30b-a3b-q4-moe": 37.85,  # Skippy MoE Q4_K_M bake-off on real INT8 silicon
},
measured_prefill_overrides={
    "qwen3-30b-a3b-q4-moe": 2849.0, # derived from 351 ms TTFT @ 1K prompt
},
```

These are **NOT** anchor-secrets values — they're public deck/spec numbers from the [backend] 2026-04-29 Mid bake-off. The values:
- `37.85 tok/s` decode — Skippy MoE Q4_K_M on NPU Mid INT8 silicon
- `2849.0 tok/s` prefill — derived from the measured 351 ms TTFT @ 1K

are documented in the deck (Slide 19 / 21) and reviewer-closed.

### HW_SLUGS

URL/state-key-safe slug per tier. Example: `"NPU Mid" → "npu_mid"`, `"RTX 5090 (reference, measured)" → "rtx_5090_reference_measured"`. Generated at `npu_model.py:370–371`.

---

## 5. Integration points

### Hardware ↔ LLM catalog (`MODELS`)

| Connection | Where | How |
|---|---|---|
| `get_measured()` resolves `measurement_alias` | `npu_model.py:174` | Reads `MODELS[model_key]["measurement_alias"]` to find architecture-sibling measurements (e.g., Thinking-2507 → qwen3-30b-a3b-q4-moe). |
| Required dtype lookup | `npu_model.py:1980, 2033` (in `project_llm`) | Reads `MODELS[model_key]["compute_dtype"]` ("int8", "fp16", "bf16", "fp8") and gates feasibility via `hw_supports_dtype()`. |
| Active params, bytes/param, num_layers, GQA ratio | `npu_model.py:1607–1623` | `model_active_bytes_per_token()` / `kv_cache_bytes_per_token()` consume `active_params`, `bytes_per_param`, `num_layers`, `hidden_dim`, `num_kv_heads`, `num_attention_heads` from MODELS to compute BW-bound floors. |

**Hardware itself never imports MODELS at class-definition time** (no forward references, no `TYPE_CHECKING` block). MODELS is referenced **only inside `get_measured()`** (line 174) and module-level helpers below the MODELS dict. This means the Hardware dataclass + tier constants are usable in isolation if a downstream needs them without the catalog.

### Hardware ↔ anchor-secrets system

**No direct connection.** The anchor-secrets loader (`sizer/npu_anchors.py`) returns `LLMAnchor` / `CNNAnchor` typed dataclass instances; the overlay function `_maybe_anchor_overlay` in `app.py:543–576` mutates a **copy of the projection result dict** post-`project_llm`. Hardware objects are **never mutated** by the anchor-secrets path.

Connection mediated through:
- `_ANCHOR_MODEL_KEY_MAP` (in `app.py:526–540`) — alias rewrite for PAI keys → spec keys.
- Spec tier-precision routing matrix (in `_maybe_anchor_overlay`):
  ```
  NPU Mid + int8     → mid_int8
  NPU High + int8    → high_int8
  NPU High + fp16    → high_fp
  ```
  uses `tier_name` (the Hardware `name`) + `MODELS[model_key]["compute_dtype"]` — does **not** read any other Hardware field.
- LPDDR memory-upgrade skip — overlay short-circuits if `abs(hw.mem_data_rate_gtps - 8.4) > 0.05`.

See `ANCHOR_SECRETS_LOADER_EXTRACT.md` §3 for the full overlay path.

### Hardware ↔ bake-off measurements (`sizer_bundle.json`)

```python
# sizer/measured.py:34-58 (abbreviated)
def attach_measurements_to_reference() -> dict:
    """Populate `RTX_5090_REFERENCE.measured_llm` from the bundle."""
    bundle = load_bundle()  # reads sizer/sizer_bundle.json
    measured = {}
    for model_canonical, workloads in bundle["models"].items():
        canonical_model = model_canonical
        if canonical_model not in MODELS:
            continue
        for workload_id, cell in workloads.items():
            measured.setdefault(canonical_model, {})[workload_id] = cell
    RTX_5090_REFERENCE.measured_llm = measured   # ← mutation at import time
```

- **Single target tier:** only `RTX_5090_REFERENCE` gets a populated `measured_llm`. Every other tier's `measured_llm` stays `None` for life.
- **Cascading enrichment helpers** at `sizer/measured.py`:
  - `_attach_perf_reference_anchors()` (line 70) — Tier 3 perf-reference rows (Q5/Q8/W8A8 variants).
  - `_override_14b_q4_5090_with_fresh_eval()` (line 120) — Tier 3 14B refresh.
  - `_attach_cross_family_5090_anchors()` (line 142) — Tier 3 cross-family bases (Llama-3.1 / Mistral).
- All three run at module import (lines 67–197) — eager loading, no `@st.cache`.

### Hardware ↔ workload definitions

Workload definitions live in a `WORKLOAD_DEFAULTS` dict (in `npu_model.py` further down — not in scope for this extract). Hardware connects only via `get_measured(model_key, workload_id)`, where `workload_id` is the second dict-key axis on `measured_llm`. Workload IDs are strings like `"chat_short"`, `"chat_medium"`, etc., and don't appear as fields on Hardware.

### Field-population sources summary

| Field | Source |
|---|---|
| All required fields (1–9) and silicon-fixed defaults (10–18) | Hardcoded in each tier's `Hardware(...)` constructor at module load. |
| `tier_family` (13) | Hardcoded per tier; assigned by [backend]'s 2026-04-29 taxonomy. |
| `measured_llm` (19) | Populated **only on `RTX_5090_REFERENCE`** at module import time by `sizer/measured.py` reading `sizer_bundle.json`. |
| `bw_projected`, `stock_mem_bandwidth_gbs`, `stock_name` (20–22) | Set **only by `hw_with_memory()`** when synthesizing memory-upgrade overlays. |
| `measured_decode_overrides`, `measured_prefill_overrides` (23–24) | Hardcoded in `NPU_MID` constructor only. |

---

## 6. The `_DTYPE_ATTR` map and dtype compatibility

### `_DTYPE_ATTR`

```python
# sizer/npu_model.py:1575-1584
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
```

- **fp16/bf16 conflation:** present and explicit — both keys route to `peak_tops_bf16`. The comment justifies it: "fp16 maps to bf16 tensor class on most SoCs."
- **Module-level constant**, leading-underscore convention (treated as private).
- Read by `hw_supports_dtype` and `hw_peak_tops_for_dtype`.

### `hw_supports_dtype(hw, dtype)`

```python
# sizer/npu_model.py:1587-1591
def hw_supports_dtype(hw: "Hardware", dtype: str) -> bool:
    attr = _DTYPE_ATTR.get(dtype.lower())
    if attr is None:
        return False
    return getattr(hw, attr, 0.0) > 0.0
```

- **Module-level function**, not a method. Hardware passed in as first arg.
- **Returns** `True` iff `getattr(hw, _DTYPE_ATTR[dtype.lower()], 0.0) > 0.0`.
- **Unknown dtype string** → `False` (safe deny).
- **Called from:**
  - `npu_model.py:1980, 2089` — `project_llm` feasibility gate (the `dtype_mismatch` source classification fires here).
  - `npu_model.py:1981` — same site, used to build the "supported dtypes" list for the error banner.

### `hw_peak_tops_for_dtype(hw, dtype)`

```python
# sizer/npu_model.py:1594-1604
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
```

- **Module-level function**, not a method.
- **Returns** raw peak TOPS for the requested dtype, **without** the `compute_efficiency` multiplier.
- **Why two functions?** `Hardware.effective_tops()` applies `compute_efficiency` (used by vision cross-class floor); `hw_peak_tops_for_dtype()` skips it (used by LLM cross-class floor because `llm_prefill_util_factor` was calibrated against raw peak). The "raw vs effective" distinction is encoded in the **method/function naming** rather than separate field pairs (see §7 not-present row on raw-vs-effective).
- **Called from:** `npu_model.py:1853, 2033, 2089–2090` (LLM compute-floor math inside `project_llm`).

---

## 7. Explicit "not present" checklist

For cross-repo apples-to-apples comparison with keyhole-sizer's parallel extract.

| Feature | Status | Reference |
|---|:-:|---|
| **`tier_family` field** | ✓ Present | `npu_model.py:50`. Taxonomy values: `Neutron-32-LP4`, `Neutron-32-LP5`, `Neutron-64-LP5`, `LP5X-8.4-64b`, `LP5X-8.4-128b`, `GDDR7-28`. Preserved on memory-upgrade clones. |
| **`bw_projected` flag** | ✓ Present | `npu_model.py:111`. Set to `True` only by `hw_with_memory()` (line 428). Read by projection (`1832, 2063`) + UI (`app.py:1614`). |
| **`mem_data_rate_gtps` / `mem_bus_width_bits` fields** | ✓ Present | `npu_model.py:31, 33`. Both required, no defaults. Used by `hw_with_memory()` to recompute BW under data-rate swaps (`new_bw = bus_width × data_rate / 8` at line 420). |
| **`peak_tops_int8` / `peak_tops_bf16` / `peak_tops_fp8` separately** | ✓ Present | `npu_model.py:26-28`. Three separate fields, all required, all `float`. NPU Mid uses 0.0 / 200.0 / 0.0 to encode INT8-only. |
| **`compute_efficiency` / `bandwidth_efficiency` as plain floats** | ✓ Present | `npu_model.py:35-36`. Defaults `0.65` / `0.70`. NPU High: `0.70` / `0.70`. 5090: `0.70` / `0.85`. |
| **`compute_util_factor` (vision) AND `llm_prefill_util_factor` (LLM) AS SEPARATE FIELDS** | ✓ Present | `npu_model.py:58, 65`. **Explicitly separate by design** (per [backend] 2026-04-29 13:17): vision util factor was calibrated against effective_tops (with `compute_efficiency` multiplier); LLM prefill util factor was calibrated against raw peak. Different code paths consume each. |
| **`llm_decode_bw_realization` field** | ✓ Present | `npu_model.py:76`. Default `1.0` (ceiling). Held at default across tiers because realization is model-class-specific (MoE 0.66 vs dense unknown), and using a measured-MoE value in cross-class cells would over-pessimize dense without justification. |
| **`compute_overhead_ms` field** | ✓ Present | `npu_model.py:81`. Default `1.0` NPU / `0.3` 5090. |
| **`npu_share_default` field** | ✓ Present | `npu_model.py:96`. Default `0.75` SoC NPUs / `1.0` 5090. User-overridable via sidebar selector. |
| **`measured_llm` dict on Hardware** | ✓ Present | `npu_model.py:104`. Type `dict[str, dict[str, dict[str, float]]] | None`. Schema: `{model_key: {workload_id: {decode_tok_s, ttft_s, prefill_tok_s, host_ms}}}`. Populated **only on `RTX_5090_REFERENCE`** by `sizer/measured.py` at import. |
| **`measured_decode_overrides` / `measured_prefill_overrides` tier-level dicts** | ✓ Present | `npu_model.py:134, 141`. Both `dict[str, float] | None`. Populated **only on `NPU_MID`**: `{"qwen3-30b-a3b-q4-moe": 37.85}` decode, `{"qwen3-30b-a3b-q4-moe": 2849.0}` prefill. Documented as tactical-interim per [docs] 2026-04-29 12:34 (Path C / Phase 2 compute clamp will replace). |
| **`stock_name` / `tier_lookup_name` string fields for canonical lookup** | ✓ Present | `stock_name` is a field (`npu_model.py:123`, `str \| None`). `tier_lookup_name` is a `@property` (`npu_model.py:147–154`) returning `stock_name if set else name`. Used by silicon-intrinsic lookups (precision capability, deployment path) so memory-upgrade variants still resolve to the stock tier's capabilities. |
| **Any "raw vs effective" TOPS distinction in field naming** | ⚠️ Partial | Fields store **raw** values (`peak_tops_int8`, etc.). "Effective" is exposed via methods/properties:<br>• `effective_tops(dtype)` → raw × `compute_efficiency` (vision use)<br>• `hw_peak_tops_for_dtype(hw, dtype)` → raw (LLM use, calibrated against raw)<br>• `effective_bandwidth_gbs` property → raw × `bandwidth_efficiency` (no `npu_share`)<br>**No `effective_*` fields** — the distinction is encoded in method/function naming. |
| **Memory-upgrade overlay support (`hw_with_memory` or equivalent)** | ✓ Present | `npu_model.py:395–431`. Single entry point: `hw_with_memory(hw, mem_type, mem_data_rate_gtps, name_suffix=None)`. Uses `dataclasses.replace`. Sets `bw_projected=True`, snapshots `stock_mem_bandwidth_gbs` + `stock_name`. Driven by `MEMORY_UPGRADE_OPTIONS` (`npu_model.py:388–392`): LPDDR5T @ 11.2 / LPDDR6 @ 12 / LPDDR6 @ 14. |
| **Custom-tier UI construction support** | ✗ Not present | Tiers are module-level constants registered in `TIERS` at import. **No** UI flow constructs new Hardware instances at runtime, **no** `Hardware(**custom_dict)` parsing from sidebar inputs, **no** YAML/JSON tier-spec loader. The only runtime tier-shape mutation is the memory-upgrade clone via `hw_with_memory`. |
| **Any per-tier vendor metadata (`vendor_name`, `soc_codename`, etc.)** | ✗ Not present | No `vendor_name`, no `soc_codename`, no `vendor_url`, no `release_year`, no `process_node`, no `package` field. The `name` field carries the only display string ("NPU Mid" / "RTX 5090 (reference, measured)") and the `mem_type` field carries the only memory-vendor-adjacent string ("LPDDR4" / "LPDDR5X" / "LPDDR6" / "GDDR7"). |

### Additional load-bearing facts not in the request

| Feature | Status | Reference |
|---|:-:|---|
| `@dataclass(frozen=True)` | ✗ Not frozen | `npu_model.py:22` — plain `@dataclass`, all fields mutable. |
| `__post_init__` | ✗ Not present | No init-time derived fields; all derivations are method/property-based. |
| `__slots__` | ✗ Not present | Plain dataclass; instance `__dict__` exists. |
| `MODELS` dict | ✓ Present | `npu_model.py:440–1555`, 20 model entries (1 PROD + 7 FT + 6 BASE + 6 PERF per v1.0.0 checkpoint memory). |
| `measurement_alias` mechanism | ✓ Present | At the catalog layer (`MODELS[model_key]["measurement_alias"]`), consumed by `Hardware.get_measured()` (line 174). PAI sizer uses it for Thinking-2507, MoE FT v1 (router-v1, full-v1) → qwen3-30b-a3b-q4-moe. |
| `compute_dtype` per model | ✓ Present | `MODELS[model_key]["compute_dtype"]` ∈ `{"int8", "fp16", "bf16", "fp8"}`. Gated against tier via `hw_supports_dtype()`. |
| `RUNTIME_OVERHEAD_BYTES` constant | ✓ Present | `npu_model.py:1629`. Single global 1 GB allocation budget. |
| Phase 2 anchor invariant checks at import | ✓ Present | `_assert_invariants()` (`npu_model.py:2232`) + `_assert_phase2_anchors()` (`npu_model.py:2258`). Fail-loud at module load if anchor data is malformed or Phase 2 invariants violated. |

---

## Appendix — Hardware import surface

```python
# What downstream consumers import from sizer.npu_model:
from sizer.npu_model import (
    Hardware,                           # the dataclass
    RTX_5090, RTX_5090_REFERENCE,       # tier constants
    NPU_LOW_LP4, NPU_LOW_LP5_32BIT, NPU_LOW_LP5_64BIT, NPU_LOW_LP5X,
    NPU_MID, NPU_HIGH,
    TIERS, HW_SLUGS,                    # registry dicts
    MEMORY_UPGRADE_OPTIONS, hw_with_memory,  # memory-upgrade overlay
    MODELS,                             # catalog dict
    project_llm, decode_tok_s_at_context,    # projection entry points
    describe_hw,                        # UI caption helper
    hw_supports_dtype, hw_peak_tops_for_dtype,  # dtype helpers
    model_active_bytes_per_token, kv_cache_bytes_per_token,
    RUNTIME_OVERHEAD_BYTES,
    PRODUCTION_REFERENCE_KEY,           # "qwen25-7b-v4-q4-dense"
    # (... plus what-if helpers, projection sub-functions, etc.)
)
```

Hardware is **the central engine object** — every projection helper takes `hw: Hardware` as its first or second argument; every UI rendering of compute/BW/feasibility reads off Hardware fields or methods.

---

*Extract generated 2026-05-19. Repo state: `7e34df0` (tag `v1.0.0`). No anchor-secrets values included — only public canonical NPU spec (deck Slide 11) values.*
