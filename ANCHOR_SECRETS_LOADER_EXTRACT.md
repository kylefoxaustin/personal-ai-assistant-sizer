# PAI sizer — anchor-secrets loader extract

**Repo:** `personal-ai-assistant-sizer`
**Generated:** 2026-05-19 (against `7e34df0` / tag `v1.0.0`)
**Purpose:** Cross-repo comparison with keyhole-sizer's parallel extract.
**Canonical spec:** `~/Documents/GitHub/personal-ai-framework/docs/private_anchor_secrets_spec.md` (commit `65bf89c`, schema-locked 2026-05-14).
**Discipline:** All measured values <REDACTED> — schema is public, values are credentials.

---

## 1. The loader module

**File:** `sizer/npu_anchors.py` (137 lines, full file).

```python
# sizer/npu_anchors.py — lines 1-137 (entire module)
"""Private NPU + CNN anchor loader.

Numbers live in Streamlit secrets (.streamlit/secrets.toml locally; Cloud
Secrets in production). This module exposes typed accessors with graceful
fallback when secrets aren't set (returns None → app falls back to
projection or shows 'not measured').

Bandwidth derivation: stored peak_bw_gbps × bw_share_frac × bw_efficiency_frac
gives the achieved bandwidth used to back out bytes-per-token. The
share_frac is overridable at call time so the UI's 100/75/50/25%
share-selector can re-derive on the fly without re-reading secrets.

Spec: personal-ai-framework `docs/private_anchor_secrets_spec.md`
(commit 65bf89c, schema-locked 2026-05-14). Mirrored on PAI sizer side
per [docs] 13:24 + 13:31 bus messages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import streamlit as st


# Badge color by source — matches keyhole-sizer's _render_source_banner convention.
BADGE_FOR_SOURCE = {
    "measured":     "🟢",
    "vendor_spec":  "🟡",
    "projected":    "🟠",
    # unknown / missing source → no badge
}


@dataclass(frozen=True)
class LLMAnchor:
    tokps: float
    prefill_tokps: float
    mem_gb: float
    seqlen: int
    source: str
    measured_date: str
    peak_bw_gbps: float
    bw_share_frac: float
    bw_efficiency_frac: float
    notes: str = ""

    @property
    def badge(self) -> str:
        return BADGE_FOR_SOURCE.get(self.source, "")

    def achieved_bw_gbps(self, share_override: Optional[float] = None) -> float:
        """BW available to NPU, applying any UI share override."""
        share = share_override if share_override is not None else self.bw_share_frac
        return self.peak_bw_gbps * share * self.bw_efficiency_frac

    def bytes_per_token(self, share_override: Optional[float] = None) -> float:
        """Memory bytes moved per decoded token (BW-bound decode model)."""
        if self.tokps <= 0:
            return 0.0
        return self.achieved_bw_gbps(share_override) * 1e9 / self.tokps


@dataclass(frozen=True)
class CNNAnchor:
    ms_per_inference: float
    fps: float
    mem_mb: float
    input_res: str
    source: str
    measured_date: str
    peak_bw_gbps: float
    bw_share_frac: float
    bw_efficiency_frac: float
    notes: str = ""

    @property
    def badge(self) -> str:
        return BADGE_FOR_SOURCE.get(self.source, "")

    def achieved_bw_gbps(self, share_override: Optional[float] = None) -> float:
        share = share_override if share_override is not None else self.bw_share_frac
        return self.peak_bw_gbps * share * self.bw_efficiency_frac


def _try_get(section: str, sub: str, key: str) -> Optional[dict]:
    """Defensive .get-chain for st.secrets — returns None on any miss."""
    try:
        return dict(st.secrets[section][sub][key])
    except Exception:
        return None


def load_llm_anchor(tier: str, precision: str, model_key: str) -> Optional[LLMAnchor]:
    """tier in {'mid','high'}, precision in {'int8','fp'}, model_key e.g. 'qwen3_30b_a3b_moe'.

    Returns None if the entry isn't in secrets — caller falls back to projection.
    """
    sub = f"{tier}_{precision}"
    data = _try_get("npu_llm_anchors", sub, model_key)
    if data is None or data.get("tokps", 0) <= 0:
        return None
    return LLMAnchor(
        tokps=float(data["tokps"]),
        prefill_tokps=float(data.get("prefill_tokps", 0.0)),
        mem_gb=float(data.get("mem_gb", 0.0)),
        seqlen=int(data.get("seqlen", 0)),
        source=str(data.get("source", "")),
        measured_date=str(data.get("measured_date", "")),
        peak_bw_gbps=float(data.get("peak_bw_gbps", 0.0)),
        bw_share_frac=float(data.get("bw_share_frac", 0.75)),
        bw_efficiency_frac=float(data.get("bw_efficiency_frac", 0.70)),
        notes=str(data.get("notes", "")),
    )


def load_cnn_anchor(tier: str, precision: str, cnn_key: str) -> Optional[CNNAnchor]:
    """tier in {'mid','high'}, precision in {'int8','fp'}, cnn_key e.g. 'resnet50_w4'."""
    sub = f"{tier}_{precision}"
    data = _try_get("cnn_anchors", sub, cnn_key)
    if data is None or data.get("ms_per_inference", 0) <= 0:
        return None
    fps = float(data.get("fps", 0.0))
    if fps <= 0 and data.get("ms_per_inference", 0) > 0:
        fps = 1000.0 / float(data["ms_per_inference"])
    return CNNAnchor(
        ms_per_inference=float(data["ms_per_inference"]),
        fps=fps,
        mem_mb=float(data.get("mem_mb", 0.0)),
        input_res=str(data.get("input_res", "")),
        source=str(data.get("source", "")),
        measured_date=str(data.get("measured_date", "")),
        peak_bw_gbps=float(data.get("peak_bw_gbps", 0.0)),
        bw_share_frac=float(data.get("bw_share_frac", 0.75)),
        bw_efficiency_frac=float(data.get("bw_efficiency_frac", 0.70)),
        notes=str(data.get("notes", "")),
    )
```

### Module surface

| Symbol | Lines | Visibility | Purpose |
|---|---:|---|---|
| `BADGE_FOR_SOURCE` | 26–31 | public dict | `source` string → emoji (🟢 measured / 🟡 vendor_spec / 🟠 projected) |
| `LLMAnchor` | 34–60 | public frozen dataclass | Typed container for an LLM cell + `achieved_bw_gbps()` and `bytes_per_token(share_override)` derivations |
| `CNNAnchor` | 63–82 | public frozen dataclass | Typed container for a CNN cell + `achieved_bw_gbps()` derivation |
| `_try_get` | 85–90 | private | Defensive 3-level `st.secrets[section][sub][key]` dict cast — returns `None` on any `KeyError`/`AttributeError`/etc. |
| `load_llm_anchor(tier, precision, model_key)` | 93–113 | public loader | Lookup → guard zero-`tokps` → typed `LLMAnchor` or `None` |
| `load_cnn_anchor(tier, precision, cnn_key)` | 116–136 | public loader | Lookup → guard zero-`ms_per_inference` → typed `CNNAnchor` or `None`; auto-derives `fps = 1000/ms` if absent |

### Disk-to-Hardware flow

1. `.streamlit/secrets.toml` (gitignored locally; pasted into Streamlit Cloud Secrets at deploy time) is read by Streamlit at startup and exposed via `st.secrets`.
2. `_try_get(section, sub, key)` indexes `st.secrets["npu_llm_anchors"|"cnn_anchors"]["mid_int8"|"high_int8"|"high_fp"]["<model_key>"]` and casts to dict.
3. `load_llm_anchor` / `load_cnn_anchor` guard against the placeholder-zero case (`tokps <= 0` / `ms_per_inference <= 0` → return `None`) and pack a frozen typed dataclass.
4. The dataclass is **not attached to the `Hardware` tier object.** Callers pass it through an overlay function (`_maybe_anchor_overlay` in `app.py`, see §3) that mutates a *copy* of the projection-result dict.

---

## 2. The expected secrets.toml schema

`.streamlit/secrets.toml.example` (230 lines, verbatim, real-value-redacted — placeholder zeros are already public):

```toml
# Rename to secrets.toml and set the password. On Streamlit Cloud, set
# this via the Secrets UI instead of committing the file.
PASSWORD = "change-me"


# ─────────────────────────────────────────────────────────────────────
# Private NPU + CNN anchor measurements
#
# Spec: personal-ai-framework `docs/private_anchor_secrets_spec.md`
# (commit 65bf89c, schema-locked 2026-05-14).
#
# Numbers here are PLACEHOLDER ZEROS for schema reference. Real measured
# values live ONLY in your local .streamlit/secrets.toml (gitignored)
# and in Streamlit Cloud Secrets — never in chat, git, or Drive.
#
# Loader: sizer/npu_anchors.py — load_llm_anchor() / load_cnn_anchor()
# return None when tokps/ms_per_inference is zero, so this example file
# is safe to leave with zeros (caller falls back to projection).
# ─────────────────────────────────────────────────────────────────────

# LLM anchor measurements — 3 NPU tier-precision cells × 3 models
#
#  Tier-precision cells:
#    mid_int8   = NPU Mid (INT8-only)
#    high_int8  = NPU High at INT8 (400 eTOPS)
#    high_fp    = NPU High at FP  (200 eTOPS)
#
#  Model keys:
#    qwen3_30b_a3b_moe   — Qwen3 30B-A3B Instruct (MoE, 3B active)
#    qwen25_32b_dense    — Qwen 2.5 32B Instruct (dense)
#    qwen25_7b_dense     — Qwen 2.5 7B Instruct (dense)

[npu_llm_anchors.mid_int8.qwen3_30b_a3b_moe]
tokps                 = <REDACTED>
prefill_tokps         = <REDACTED>
mem_gb                = <REDACTED>
seqlen                = 2048
source                = "measured"
measured_date         = "2026-05-14"
peak_bw_gbps          = 134.4         # 8.4 GT/s × 128-bit LPDDR5X ÷ 8 = 134.4 GB/s
bw_share_frac         = 0.75          # default share; UI may override
bw_efficiency_frac    = 0.70          # matches keyhole BW-efficiency methodology
notes                 = ""

# [...8 more LLM cells with identical field set — see canonical schema...]
# Full set: npu_llm_anchors.{mid_int8,high_int8,high_fp}.{qwen3_30b_a3b_moe,qwen25_32b_dense,qwen25_7b_dense}
# = 3 tier-precision × 3 models = 9 LLM cells


# ─────────────────────────────────────────────────────────────────────
# CNN anchor measurements — 2 tier-precision cells × 3 CNN variants
#
#  CNN keys (confirmed 2026-05-14):
#    resnet50_w4   — ResNet-50, 4-bit weights, 224×224 input
#    yolov8n_w4    — YOLOv8n, 4-bit weights, 640×640 input
#    yolov8n_w8    — YOLOv8n, 8-bit weights, 640×640 input
#
#  CNN measured INT-only — no high_fp.* sections.
# ─────────────────────────────────────────────────────────────────────

[cnn_anchors.mid_int8.resnet50_w4]
ms_per_inference      = <REDACTED>
fps                   = <REDACTED>
mem_mb                = <REDACTED>
input_res             = "224x224"
source                = "measured"
measured_date         = "2026-05-14"
peak_bw_gbps          = 134.4
bw_share_frac         = 0.75
bw_efficiency_frac    = 0.70
notes                 = ""

# [...5 more CNN cells with identical field set — see canonical schema...]
# Full set: cnn_anchors.{mid_int8,high_int8}.{resnet50_w4,yolov8n_w4,yolov8n_w8}
# = 2 tier-precision × 3 CNN variants = 6 CNN cells
```

### Key naming scheme

- **Top-level sections:** `npu_llm_anchors`, `cnn_anchors` (two flat namespaces; no `[default]`, no nesting beyond these two roots).
- **Tier-precision sub-key** (level 2): `f"{tier}_{precision}"` — `mid_int8`, `high_int8`, `high_fp`. CNN omits `high_fp` (INT-only).
- **Model / CNN sub-key** (level 3): snake_case (`qwen3_30b_a3b_moe`, `resnet50_w4`). Hyphens forbidden — TOML headers `[a.b.c]` require bare keys.
- Underscores in catalog model_keys (`qwen3-30b-a3b-q4-moe` in MODELS) → an alias map in `app.py` (see §3) rewrites them to the spec's snake_case form.

### Field set (required vs optional)

| Field | LLM | CNN | Required? | Loader behavior on absence |
|---|:-:|:-:|:-:|---|
| `tokps` | ✓ | — | **required**, > 0 | `<= 0` → loader returns `None` (cell treated as not measured) |
| `ms_per_inference` | — | ✓ | **required**, > 0 | `<= 0` → loader returns `None` |
| `prefill_tokps` | ✓ | — | optional | default `0.0` |
| `fps` | — | ✓ | optional | if missing/zero and `ms_per_inference > 0`, auto-derived as `1000 / ms_per_inference` |
| `mem_gb` / `mem_mb` | ✓ | ✓ | optional | default `0.0` |
| `seqlen` | ✓ | — | optional | default `0` (canonical schema uses 2048) |
| `input_res` | — | ✓ | optional | default `""` |
| `source` | ✓ | ✓ | optional | default `""` (no badge); spec values: `"measured"` / `"vendor_spec"` / `"projected"` |
| `measured_date` | ✓ | ✓ | optional | default `""` |
| `peak_bw_gbps` | ✓ | ✓ | optional | default `0.0` (schema canonical: 134.4 on both NPU Mid + NPU High) |
| `bw_share_frac` | ✓ | ✓ | optional | default `0.75` |
| `bw_efficiency_frac` | ✓ | ✓ | optional | default `0.70` |
| `notes` | ✓ | ✓ | optional | default `""` |

Unknown extra keys: silently ignored — the loader only reads named fields via `.get()`. No schema validator complains about extras.

---

## 3. Where the loader is called

Three call sites, all in `app.py`. Import: `app.py:23 — from sizer.npu_anchors import load_llm_anchor, load_cnn_anchor`.

### Call site 1 — headline-tile hot-swap (`_maybe_anchor_overlay`)

```python
# app.py:526-576
_ANCHOR_MODEL_KEY_MAP = {
    "qwen3-30b-a3b-q4-moe":        "qwen3_30b_a3b_moe",
    "qwen3-30b-a3b-q4-moe-fp":     "qwen3_30b_a3b_moe",
    "qwen2.5-32b-q4-dense":        "qwen25_32b_dense",
    "qwen2.5-7b-q4-dense":         "qwen25_7b_dense",
    "qwen2.5-32b-q4-dense-int8":   "qwen25_32b_dense",
    "qwen2.5-7b-q4-dense-int8":    "qwen25_7b_dense",
}

def _maybe_anchor_overlay(r, model_key, hw, tier_name, decode_tokens):
    if r is None or r.get("source") in ("wont_fit", "dtype_mismatch"):
        return r
    # Anchors measured at stock LPDDR5X 8.4 GT/s — skip hot-swap on memory upgrades.
    if abs(hw.mem_data_rate_gtps - 8.4) > 0.05:
        return r
    spec_model = _ANCHOR_MODEL_KEY_MAP.get(model_key)
    if spec_model is None:
        return r
    dtype = MODELS.get(model_key, {}).get("compute_dtype", "")
    if tier_name == "NPU Mid" and dtype == "int8":
        spec_tier, spec_prec = "mid", "int8"
    elif tier_name == "NPU High" and dtype == "int8":
        spec_tier, spec_prec = "high", "int8"
    elif tier_name == "NPU High" and dtype == "fp16":
        spec_tier, spec_prec = "high", "fp"
    else:
        return r
    anchor = load_llm_anchor(spec_tier, spec_prec, spec_model)         # ← call site
    if anchor is None or anchor.source != "measured" or anchor.tokps <= 0:
        return r
    r2 = dict(r)
    r2["decode_tok_s"] = anchor.tokps
    r2["decode_s"] = decode_tokens / anchor.tokps
    r2["total_s"] = r.get("ttft_s", 0.0) + r2["decode_s"]
    r2["source"] = "measured_silicon_anchor"
    r2["_silicon_anchor_meta"] = {
        "measured_date": anchor.measured_date,
        "spec_tier_precision": f"{spec_tier}_{spec_prec}",
        "spec_model_key": spec_model,
    }
    return r2
```

Invocation chain (`app.py:579–588`):
```python
r = project_llm(model_key, hw, workload_id, ...)
r = _maybe_anchor_overlay(r, model_key, hw, tier_name, decode_tokens)
```

The overlay replaces `decode_tok_s` + `decode_s` + `total_s` on the projection-result dict if (and only if) a measured anchor exists for the current `(tier, dtype, model)` cell. TTFT/prefill/feasibility/regime are preserved from the projection — anchors don't always carry those fields.

### Call site 2 — LLM anchor expander (Performance tab → "📡 Measured silicon anchors")

```python
# app.py:1179-1218
with st.expander("📡 Measured silicon anchors (private)", expanded=False):
    st.markdown("**LLM throughput**")
    _llm_tier_rows = [
        ("NPU Mid INT8",  "mid",  "int8"),
        ("NPU High INT8", "high", "int8"),
        ("NPU High FP",   "high", "fp"),
    ]
    _llm_model_cols = [
        ("Qwen3 30B-A3B MoE",  "qwen3_30b_a3b_moe"),
        ("Qwen 2.5 32B dense", "qwen25_32b_dense"),
        ("Qwen 2.5 7B dense",  "qwen25_7b_dense"),
    ]
    for _tier_label, _tier_key, _prec_key in _llm_tier_rows:
        st.markdown(f"*{_tier_label}*")
        _cols = st.columns(3)
        for _col, (_model_label, _model_key) in zip(_cols, _llm_model_cols):
            _anchor = load_llm_anchor(_tier_key, _prec_key, _model_key)   # ← call site
            with _col:
                if _anchor is None:
                    st.metric(f"⏸ {_model_label}", "not measured")
                else:
                    _bpt = _anchor.bytes_per_token(share_override=npu_share)
                    st.metric(
                        f"{_anchor.badge} {_model_label}",
                        f"{_anchor.tokps:.1f} tok/s",
                        delta=f"{_bpt/1e6:.0f} MB/tok @ {int(npu_share*100)}% BW share",
                        delta_color="off",
                    )
```

Renders a 3-tier-row × 3-model-col grid (9 LLM cells). Each cell either shows `⏸ not measured` (loader returned `None`) or `🟢 N.N tok/s ▴ M MB/tok @ X% BW share` (typed anchor). The `share_override=npu_share` passes the sidebar's BW-share selector into `bytes_per_token()` so the derived metric updates without re-reading secrets.

### Call site 3 — CNN anchor expander (same expander, below LLM block)

```python
# app.py:1223-1247
st.markdown("**CNN latency**")
_cnn_tier_rows = [
    ("NPU Mid INT8",  "mid",  "int8"),
    ("NPU High INT8", "high", "int8"),
]
_cnn_model_cols = [
    ("ResNet-50 W4",  "resnet50_w4",  "224×224"),
    ("YOLOv8n W4",    "yolov8n_w4",   "640×640"),
    ("YOLOv8n W8",    "yolov8n_w8",   "640×640"),
]
for _tier_label, _tier_key, _prec_key in _cnn_tier_rows:
    st.markdown(f"*{_tier_label}*")
    _cols = st.columns(3)
    for _col, (_cnn_label, _cnn_key, _res) in zip(_cols, _cnn_model_cols):
        _anchor = load_cnn_anchor(_tier_key, _prec_key, _cnn_key)         # ← call site
        with _col:
            if _anchor is None:
                st.metric(f"⏸ {_cnn_label}", "not measured")
            else:
                st.metric(
                    f"{_anchor.badge} {_cnn_label}",
                    f"{_anchor.ms_per_inference:.2f} ms",
                    delta=f"{_anchor.fps:.1f} FPS · {_res}",
                    delta_color="off",
                )
```

2 tier-rows × 3 CNN-cols = 6 CNN cells. CNN intentionally has no `high_fp` row — the spec confirms CNN was measured INT-only on NPU High.

### Lifecycle

- **When:** every Streamlit rerun (default behavior). Streamlit reruns the entire `app.py` script on each user input change (sidebar selectbox, sliders, etc.).
- **Caching:** **none.** No `@st.cache_data` or `@st.cache_resource` decorator on the loaders or the dataclasses. Each call re-reads `st.secrets` (which is itself a Streamlit-managed read-only mapping; the underlying TOML is parsed once at app boot).
- **Per-tier cost:** 3 × `_try_get` per LLM block + 6 × `_try_get` per CNN block + 1 × `_try_get` in `_maybe_anchor_overlay` = 10 dict-lookups per rerun. No performance impact.
- **Hardware mutation:** **none.** The `Hardware` dataclass (`sizer/npu_model.py:22–96`) is constructed once at import and never modified by the anchor loader. The overlay path mutates a *copy* of the projection result dict (`r2 = dict(r)`), not the tier object or `MODELS` dict.

---

## 4. The data shape

### LLMAnchor instance (example, values redacted)

```python
LLMAnchor(
    tokps=<REDACTED>,                # measured decode tok/s
    prefill_tokps=<REDACTED>,
    mem_gb=<REDACTED>,
    seqlen=2048,
    source="measured",
    measured_date="2026-05-14",
    peak_bw_gbps=134.4,
    bw_share_frac=0.75,
    bw_efficiency_frac=0.70,
    notes="",
)
# Derived:
# anchor.badge                       == "🟢"
# anchor.achieved_bw_gbps()          == 134.4 * 0.75 * 0.70 = 70.56 GB/s
# anchor.achieved_bw_gbps(0.50)      == 134.4 * 0.50 * 0.70 = 47.04 GB/s
# anchor.bytes_per_token()           == achieved_bw_gbps() * 1e9 / tokps
```

### CNNAnchor instance (example, values redacted)

```python
CNNAnchor(
    ms_per_inference=<REDACTED>,
    fps=<REDACTED>,                  # auto-derived as 1000/ms_per_inference if absent
    mem_mb=<REDACTED>,
    input_res="640x640",
    source="measured",
    measured_date="2026-05-14",
    peak_bw_gbps=134.4,
    bw_share_frac=0.75,
    bw_efficiency_frac=0.70,
    notes="",
)
```

### After hot-swap — projection-result dict shape

When `_maybe_anchor_overlay` fires, the projection-result dict gains:

```python
r2 = {
    # ...all existing fields from project_llm: ttft_s, prefill_s, host_ms,
    # decode_s, total_s, source, feasibility, regime, ...
    "decode_tok_s":  anchor.tokps,                # overwritten
    "decode_s":      decode_tokens / anchor.tokps,# recomputed
    "total_s":       r["ttft_s"] + r2["decode_s"],# recomputed
    "source":        "measured_silicon_anchor",   # upgraded from projected/measured_anchor
    "_silicon_anchor_meta": {                     # NEW field
        "measured_date":        "2026-05-14",
        "spec_tier_precision":  "mid_int8",
        "spec_model_key":       "qwen3_30b_a3b_moe",
    },
}
```

Downstream projection-consumer code reads `r["decode_tok_s"]` / `r["total_s"]` / `r["source"]` unchanged. The badge color is selected upstream in `app.py:602+` by inspecting `r["source"]` against a known set: `wont_fit`, `dtype_mismatch`, `measured`, `measured_silicon_anchor`, `same_class_anchor`, `cross_class`, `projected`, etc.

The `LLMAnchor` / `CNNAnchor` dataclasses themselves never escape the call sites in `app.py` — only `.tokps`, `.badge`, `.bytes_per_token()`, and `.measured_date` are surfaced into user-facing UI.

---

## 5. Error handling + fallback

### What the loader itself swallows

```python
def _try_get(section: str, sub: str, key: str) -> Optional[dict]:
    try:
        return dict(st.secrets[section][sub][key])
    except Exception:
        return None
```

A blanket `except Exception` — covers:
- `secrets.toml` absent entirely (Streamlit returns an empty `Secrets` object; `[section]` raises `KeyError`).
- Section missing (`npu_llm_anchors` not defined).
- Tier-precision sub-section missing (`high_fp` not declared).
- Model key missing.
- Value not coercible to dict (corrupt TOML structure, type mismatch).

All collapse to `None`.

### Two additional guards inside the loader bodies

1. `data.get("tokps", 0) <= 0` (LLM) / `data.get("ms_per_inference", 0) <= 0` (CNN) → return `None`.
   This is how placeholder zeros in `secrets.toml.example` are treated as "not measured" without raising. **A cell with `tokps = 0.0` is indistinguishable from "this row is absent"** by design.
2. Malformed individual fields: each scalar is coerced via `float(...)` / `int(...)` / `str(...)`. A non-numeric `tokps` would raise inside the constructor — uncaught at the loader level; bubbles up to the call site. **Not defended against today.**

### Caller-side fallback cascade

In `_maybe_anchor_overlay` (`app.py:543–576`), the anchor path is skipped if **any** of:

| Condition | What kicks in |
|---|---|
| `r is None` (projection itself failed) | Caller-level `st.warning(...)` + `st.stop()` |
| `r["source"] in ("wont_fit", "dtype_mismatch")` | Feasibility banner (no decode-rate to overlay) |
| `abs(hw.mem_data_rate_gtps - 8.4) > 0.05` | LPDDR memory-upgrade variant — anchor was at stock 8.4 GT/s; skip |
| `_ANCHOR_MODEL_KEY_MAP.get(model_key) is None` | Model not in the alias table → no spec cell exists for it |
| `MODELS[model_key]["compute_dtype"]` not in {int8, fp16} | No tier-precision routing rule |
| `(tier_name, dtype)` not in the int8/int8/fp16 routing matrix | No tier-precision routing rule |
| `anchor is None` (loader returned None) | Fall through to projection |
| `anchor.source != "measured"` | Reject vendor_spec / projected (require real measurement) |
| `anchor.tokps <= 0` | Belt-and-suspenders zero-check |

In **every** skip case the original projection-result dict `r` is returned unmodified — the projection-layer's own source classification stands (🟢 `measured_anchor` from sizer_bundle.json, 🟡 `same_class_anchor` BW-scaled within `tier_family`, 🔴 `cross_class` MAX(BW, compute) physics).

### Cascade order — full picture

Anchor-secrets is **layer 1 of 4** in the source-classification cascade:

```
1. measured_silicon_anchor      ← THIS LOADER (LLM only; CNN not in headline path)
   ↓ (none for this cell)
2. measured_anchor              ← sizer_bundle.json bake-off catalog (5090 source-of-truth)
   ↓ (none for this model_key, with measurement_alias resolution attempted)
3. same_class_anchor (🟡)       ← BW-scaled within tier_family in projection layer
   ↓ (no anchor in tier_family)
4. cross_class (🔴)             ← MAX(BW, compute) two-floor physics
```

The anchor-secrets loader participates **only at layer 1** for LLM (Performance-tab headline tile) and runs **standalone** for CNN (no headline hot-swap path for CNN today). Layers 2–4 live in `sizer/measured.py` + `sizer/npu_model.py.project_llm()` and never see the `st.secrets` source.

---

## 6. Connection to canonical spec

The canonical spec is `personal-ai-framework/docs/private_anchor_secrets_spec.md` (commit `65bf89c`, dated 2026-05-14). PAI sizer implements it **verbatim**, with two minor extensions noted below.

### Spec compliance audit

| Spec element | PAI sizer status | Reference |
|---|---|---|
| Two top-level sections (`npu_llm_anchors`, `cnn_anchors`) | ✓ implemented | `secrets.toml.example:33,159` |
| 9 LLM cells (3 tier-precision × 3 models) | ✓ all 9 cells present | `secrets.toml.example:33–145` |
| 6 CNN cells (2 tier-precision × 3 CNN variants, INT-only) | ✓ all 6 cells present | `secrets.toml.example:159–229` |
| `peak_bw_gbps = 134.4` on both Mid + High (same LPDDR5X bus) | ✓ canonical value | every cell |
| `bw_share_frac` default 0.75 / `bw_efficiency_frac` default 0.70 | ✓ canonical defaults | `npu_anchors.py:110–111, 133–134` |
| `LLMAnchor` / `CNNAnchor` frozen dataclasses | ✓ exact field set | `npu_anchors.py:34–82` |
| `achieved_bw_gbps(share_override)` method | ✓ implemented | `npu_anchors.py:51, 80` |
| `bytes_per_token(share_override)` method | ✓ implemented (LLM only — spec specifies LLM-only) | `npu_anchors.py:56` |
| `_try_get` defensive 3-level dict cast | ✓ implemented | `npu_anchors.py:85–90` |
| `load_llm_anchor(tier, precision, model_key) -> Optional[LLMAnchor]` | ✓ exact signature | `npu_anchors.py:93` |
| `load_cnn_anchor(tier, precision, cnn_key) -> Optional[CNNAnchor]` | ✓ exact signature | `npu_anchors.py:116` |
| Tokps-zero / ms-zero short-circuit (`<= 0` returns `None`) | ✓ implemented | `npu_anchors.py:100, 120` |
| `BADGE_FOR_SOURCE` dict (🟢 measured / 🟡 vendor_spec / 🟠 projected) | ✓ implemented | `npu_anchors.py:26–31` |
| Gitignore `.streamlit/secrets.toml` | ✓ present | confirmed in checkpoint memory |
| `secrets.toml.example` safe-to-commit with zeros | ✓ committed | repo root |

### Extensions beyond the spec

1. **`_ANCHOR_MODEL_KEY_MAP` (app.py:526–540) — PAI-specific alias table.** The spec defines spec-side model_keys (`qwen3_30b_a3b_moe`, `qwen25_32b_dense`, `qwen25_7b_dense`). PAI's catalog has 6 PAI-side keys that map to these 3 spec keys — including the int8-routed variants (`qwen2.5-32b-q4-dense-int8`, `qwen2.5-7b-q4-dense-int8` from commit `ee41def`) and the fp16-routed MoE variant (`qwen3-30b-a3b-q4-moe-fp`). This mapping is **outside** the spec and lives in `app.py`, not `npu_anchors.py`. Sister project may have a different mapping.
2. **`_silicon_anchor_meta` field on the overlay result dict (app.py:571–575) — PAI-specific.** Carries `{measured_date, spec_tier_precision, spec_model_key}` for the source banner to surface in the UI. The spec doesn't prescribe how the overlay metadata is propagated — PAI puts it in a leading-underscore key on `r2`.
3. **LPDDR memory-upgrade skip (app.py:547–548).** PAI catalogs LPDDR5T-11.2 / LPDDR6-12 / LPDDR6-14 memory-upgrade variants on top of NPU Mid/High. The spec's `peak_bw_gbps = 134.4` only applies to the stock LPDDR5X 8.4 GT/s case, so the overlay short-circuits whenever `hw.mem_data_rate_gtps != 8.4`. The variant tier projects via the normal cross-tier BW-scaling layer instead. **Sister project may not have memory-upgrade variants** — if so, this guard is unnecessary.

### Deferred items (spec mentions; PAI doesn't ship)

- **BW-share-selector UI (100/75/50/25%) explicitly below tier dropdown.** PAI's `app.py:326–348` implements the share selector — the `npu_share` value is passed through to `_anchor.bytes_per_token(share_override=npu_share)` at every render. **PAI matches.** (Sister project should too, but verify.)
- No spec-mentioned deferred items remain outstanding on the PAI side.

---

## 7. Explicit "not present" checklist

For cross-repo comparison with keyhole-sizer's extract.

| Feature | Status | Lines / notes |
|---|:-:|---|
| **Per-cell measurement anchors (model_key × workload_id/quant)** | ⚠️ Partial | ✓ per-`(tier, precision, model_key)` cells (9 LLM + 6 CNN). ✗ No `workload_id` dimension — anchors are workload-agnostic. Workload-specific bake-off measurements live separately in `sizer_bundle.json` consumed by `sizer/measured.py`. |
| **Tier-level anchor overrides (whole-tier `measurement_decode_overrides` dict)** | ✗ Not present | The anchor-secrets loader has no tier-level override concept. `Hardware` has a `measured_llm` field (`sizer/npu_model.py:165–177`) but it's populated from `sizer_bundle.json` at startup, not from `st.secrets`. |
| **`measurement_alias` resolution** | ✗ Not in anchor-secrets loader | `measurement_alias` exists in the catalog at `sizer/npu_model.py:165–177` and `sizer/measured.py:218–245`, used by `Hardware.get_measured(model_key, workload_id)` — fully separate from the secrets loader. The anchor-secrets-side analog is `_ANCHOR_MODEL_KEY_MAP` in `app.py:526–540` (6 PAI keys → 3 spec keys), but it's not a real alias mechanism — it's a static rewrite table living in the call site. |
| **`bw_share_frac` / `bw_efficiency_frac` fields** | ✓ Present | `npu_anchors.py:43–44` (LLMAnchor), `73–74` (CNNAnchor). Defaults `0.75` / `0.70`. Composed in `achieved_bw_gbps()` at `npu_anchors.py:51–54, 80–82`. UI overrides via `share_override=npu_share` at every render. |
| **`source` / `measured_date` metadata** | ✓ Present | `npu_anchors.py:40–41, 70–71`. Surfaced via `LLMAnchor.badge` / `CNNAnchor.badge` (🟢🟡🟠 mapping at `npu_anchors.py:26–31`) and propagated into `r2["_silicon_anchor_meta"]` at `app.py:571–575`. |
| **Cross-tier scaling within `tier_family`** | ✗ Not in anchor-secrets loader | `tier_family` is a `Hardware` field at `sizer/npu_model.py:50` and the "🟡 same_class_anchor" classification fires in the projection layer (`project_llm` / `_projection_path` in `npu_model.py`). The anchor-secrets layer never reads `tier_family`. |
| **Anchor-secrets validation at load time (schema check)** | ✗ Not present | `_try_get` (`npu_anchors.py:85–90`) is a permissive `except Exception` dict-cast. No type-checker, no schema validator, no required-key enforcement. Unknown extra keys silently ignored. Malformed scalars (non-numeric `tokps`) would raise inside the dataclass constructor and bubble to the call site. |
| **Logging when a value comes from anchor vs projection** | ✗ Not present | No `logging` calls in `npu_anchors.py`. Source attribution flows through `r["source"]` strings (e.g., `"measured_silicon_anchor"`) and the `BADGE_FOR_SOURCE` emoji map. No structured log emission, no audit trail at runtime. |
| **A separate "vision anchors" loader (CNN cells)** | ✓ Present | `load_cnn_anchor(tier, precision, cnn_key)` at `npu_anchors.py:116–136`. `CNNAnchor` dataclass at `npu_anchors.py:63–82`. 6 CNN cells in the secrets schema. CNN-only feature: auto-derive `fps = 1000 / ms_per_inference` if `fps` absent (`npu_anchors.py:122–124`). |
| **Loader mutates Hardware OR returns value overlaid post-projection** | **Returns value, overlay applied post-projection** | The loader returns `Optional[LLMAnchor|CNNAnchor]` typed dataclasses. **`Hardware` objects are never mutated** by the anchor path. The post-projection overlay (`_maybe_anchor_overlay` in `app.py:543–576`) mutates a **copy** (`r2 = dict(r)`) of the projection result dict, never the tier object or the `MODELS` dict. |
| **Anything else load-bearing** | ✓ — | (1) `bytes_per_token(share_override)` recomputes implied bytes/tok at any share fraction without re-reading secrets — used by the LLM anchor expander to update live with the sidebar share selector. (2) `frozen=True` on both dataclasses — immutable by construction. (3) `BADGE_FOR_SOURCE` is a single source-of-truth for the 3-color source taxonomy (🟢🟡🟠), surfaced as a property. (4) Auto-fps derivation in `load_cnn_anchor` is the only field-level computation the loader performs; everything else is pure I/O. |

---

## Appendix — Hardware tier object shape (referenced by §3)

```python
# sizer/npu_model.py:22-96 (abbreviated, field set only)
@dataclass
class Hardware:
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

    tier_family: str | None = None         # cross-tier BW-scaling taxonomy
    compute_util_factor: float = 0.45      # vision compute-floor calibration
    llm_prefill_util_factor: float = 0.10  # LLM prefill compute-floor calibration
    llm_decode_bw_realization: float = 1.0 # LLM decode BW realization fraction
    compute_overhead_ms: float = 1.0       # kernel launch overhead
    npu_share_default: float = 0.75        # default BW share to NPU
    # ...
    measured_llm: dict | None = None       # populated from sizer_bundle.json
                                           # — NOT from secrets.toml
```

**Critical:** `Hardware.measured_llm` and the anchor-secrets loader operate on **disjoint data sources**. `measured_llm` reads `sizer_bundle.json` (committed, public, 5090 bake-offs); the anchor-secrets loader reads `st.secrets[...]` (gitignored, private, NPU silicon measurements). The two participate in different layers of the source-classification cascade (see §5).

---

*Extract generated 2026-05-19. Repo state: `7e34df0` (tag `v1.0.0`). All measured values redacted per discipline rule (KEY-not-VALUE for anchor-secrets data).*
