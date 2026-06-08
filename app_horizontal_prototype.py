"""Skippy NPU sizer — HORIZONTAL LAYOUT PROTOTYPE (Step 2, 2026-06-07).

A throwaway layout mockup for Kyle to judge against the live vertical-sidebar
app. Run side-by-side:

    streamlit run app.py                       # current (tall left sidebar)
    streamlit run app_horizontal_prototype.py  # this (top control strip, wide)

Mirrors keyhole-sizer's horizontal prototype (its `app_horizontal_prototype.py`,
origin/main d215b3e) on PAI's domain. PAI is LLM-only, so this is the
single-workload version of keyhole's Vision/LLM/VLA shell. Uses the REAL engine
(live numbers). Demonstrates: no left sidebar; controls in a horizontal top
strip (tier pills + Model/Workload pickers + Memory/BW-share popovers + a ⚙
Settings popover for the rarely-touched knobs); results + charts fill the full
page width; KPIs visible onscreen (not download-only); each section's verbose
depth-tabs tuck into a collapsible "🔎 detail" expander so headline metrics stay
on a short first paint. If you like it, Step 3 is migrating app.py onto this
shell.
"""
from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ratchet.anchors import load_llm_anchor
from sizer.npu_model import (
    MODELS, TIERS, project_llm, describe_hw, hw_supports_dtype,
    hw_with_memory, MEMORY_UPGRADE_OPTIONS, hw_with_precision,
    decode_tok_s_at_context, PRODUCTION_REFERENCE_KEY, CATEGORY_LABELS,
)

st.set_page_config(page_title="Skippy NPU Sizer · horizontal prototype",
                   layout="wide", initial_sidebar_state="collapsed")

# Highlight the picker popovers (Model / Workload) in green so the "what am I
# configuring" control pops out from the neutral popovers around it. A green
# border + translucent fill reads correctly in BOTH light and dark browser
# themes (the fill tints whatever's behind it); button text is left
# theme-inherited so contrast is never broken. (Ported from keyhole prototype.)
st.markdown("""
<style>
.st-key-pop_model button, .st-key-pop_quant button, .st-key-pop_work button {
    border: 1.5px solid #22A06B !important;
    background-color: rgba(34, 160, 107, 0.14) !important;
}
.st-key-pop_model button:hover, .st-key-pop_quant button:hover, .st-key-pop_work button:hover {
    border-color: #1B7E54 !important;
    background-color: rgba(34, 160, 107, 0.24) !important;
}
</style>
""", unsafe_allow_html=True)

_ACCENT = "#7C3AED"   # Skippy violet — highlight the selected tier in charts

# ── short tier label → TIERS key (the comparison ladder, in order) ──
_TIER_MAP = {
    "Low-LP4":     "NPU Low-LP4",
    "Low-LP5-32":  "NPU Low-LP5-32bit",
    "Low-LP5-64":  "NPU Low-LP5-64bit",
    "Low-LP5X":    "NPU Low-LP5X",
    "Mid":         "NPU Mid",
    "High":        "NPU High",
    "RTX 5090":    "RTX 5090 (reference, measured)",
}
_SHARE_MAP = {"100%": 1.0, "75%": 0.75, "50%": 0.5, "25%": 0.25}

# ── Per-workload token shapes (mirrors app.py::WORKLOAD_DEFAULTS) ──
WORKLOAD_DEFAULTS = {
    "short_chat":            {"prompt_tokens": 750,   "decode_tokens": 150,  "label": "Short chat — no RAG, conversational"},
    "rag_qa":                {"prompt_tokens": 4800,  "decode_tokens": 400,  "label": "RAG Q&A — retrieved context 5–10 chunks"},
    "long_decode":           {"prompt_tokens": 2800,  "decode_tokens": 2500, "label": "Long-decode doc gen — spec / proposal"},
    "meeting_summarization": {"prompt_tokens": 12700, "decode_tokens": 800,  "label": "Meeting summarization — 30–60 min transcript"},
    "agentic_roundtrip":     {"prompt_tokens": 2500,  "decode_tokens": 200,  "label": "Agentic roundtrip — single tool-loop iteration"},
}

# ── Measured-silicon anchor overlay (ported from app.py) ──
# When a real measured NPU anchor exists for the current (tier, model) cell,
# override decode_tok_s with the measurement and upgrade the source to
# "measured_silicon_anchor" (🟢) so the prototype's headline matches the live
# app instead of showing a cross-class/same-class projection.
_ANCHOR_MODEL_KEY_MAP = {
    "qwen3-30b-a3b-q4-moe":      "qwen3_30b_a3b_moe",
    "qwen3-30b-a3b-q4-moe-fp":   "qwen3_30b_a3b_moe",
    "qwen2.5-32b-q4-dense":      "qwen25_32b_dense",
    "qwen2.5-7b-q4-dense":       "qwen25_7b_dense",
    "qwen2.5-32b-q4-dense-int8": "qwen25_32b_dense",
    "qwen2.5-7b-q4-dense-int8":  "qwen25_7b_dense",
}


def _maybe_anchor_overlay(r, model_key, hw, tier_name, decode_tokens):
    if r is None or r.get("source") in ("wont_fit", "dtype_mismatch"):
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
    anchor = load_llm_anchor(spec_tier, spec_prec, spec_model)
    if anchor is None or anchor.source != "measured" or anchor.tokps <= 0:
        return r
    bw_ratio = 1.0
    if getattr(hw, "bw_projected", False) and hw.stock_mem_bandwidth_gbs:
        bw_ratio = hw.mem_bandwidth_gbs / hw.stock_mem_bandwidth_gbs
    decode = anchor.tokps * bw_ratio
    r2 = dict(r)
    r2["decode_tok_s"] = decode
    r2["decode_s"] = decode_tokens / decode
    r2["total_s"] = r.get("ttft_s", 0.0) + r2["decode_s"]
    r2["source"] = "measured_silicon_anchor"
    r2["_silicon_anchor_meta"] = {
        "measured_date": anchor.measured_date,
        "spec_tier_precision": f"{spec_tier}_{spec_prec}",
        "spec_model_key": spec_model,
    }
    return r2


# ── source badge taxonomy (compact, mirrors app.py's five states) ──
def _source_badge(r):
    s = r["source"]
    if s == "measured_silicon_anchor":
        m = r.get("_silicon_anchor_meta", {})
        return ("🟢", "measured silicon",
                f"real NPU measurement — `{m.get('spec_tier_precision','?')}` × "
                f"`{m.get('spec_model_key','?')}`, {m.get('measured_date','?')}")
    if s == "measured":
        return ("🟢", "measured", "direct RTX 5090 bake-off baseline")
    if s == "measured_anchor":
        return ("🟢", "decode-anchored", "Skippy bake-off measurement on this tier")
    if s == "same_class_anchor":
        mk = " · ×BW-upgrade" if getattr(r.get("_hw"), "bw_projected", False) else ""
        return ("🟡", "same-class projection" + mk,
                "BW-scaled from a measured anchor in this tier-family")
    if s == "cross_class":
        return ("🟠", "cross-class (what-if)",
                "two-floor MAX(BW, compute) physics — no anchor; directional")
    return ("⚪", s, "projection state not recognized")


# ── Quant pill (mirrors keyhole's Quant ▾). PAI bakes quant into the catalog
# key (no quant param in project_llm), so the pill REMAPS to the q4/q5/q8 sibling
# catalog row when one exists — works for the qwen2.5-7b / qwen2.5-32b dense
# families today; every other model is Q4-only (Q5/Q8 disabled). p_model stays
# the family anchor; p_quant overlays the quant on top (stable across reruns). ──
_QUANT_PILL = [("Q4_K_M", "q4"), ("Q5_K_M", "q5"), ("Q8_0", "q8")]
_QUANT_TOK = dict(_QUANT_PILL)


def _model_quant(key: str) -> str | None:
    """The quant token baked into a catalog key, e.g. 'q4' from
    'qwen2.5-7b-q4-dense'. None if the key carries no -qN- token."""
    m = re.search(r"-q(\d+)-", key)
    return f"q{m.group(1)}" if m else None


def _quant_sibling(key: str, quant_tok: str) -> str | None:
    """The sibling catalog key at `quant_tok` (e.g. 'q8'), or None if no such
    row exists. Swaps the first -qN- token; identity if already that quant."""
    if _model_quant(key) is None:
        return None
    cand = re.sub(r"-q\d+-", f"-{quant_tok}-", key, count=1)
    return cand if cand in MODELS else None


def _model_role(k: str) -> tuple[str, str]:
    if k == PRODUCTION_REFERENCE_KEY:
        return "PROD", "🚀"
    m = MODELS[k]
    if m.get("perf_reference_only"):
        return "PERF", "⚙️"
    if str(m.get("training", "")).startswith("skippy_"):
        return "FT", "🔬"
    return "BASE", "📚"


def _per_tier_bar(values: dict[str, float], y_title: str, selected: str):
    """Horizontal-friendly bar of a metric across the tier ladder; the selected
    tier is accented. `values` keyed by short tier label."""
    fig = go.Figure(go.Bar(
        x=list(values), y=list(values.values()),
        marker_color=[_ACCENT if k == selected else "#9AA7BD" for k in values],
        text=[f"{v:.1f}" for v in values.values()], textposition="outside",
    ))
    fig.update_layout(
        template="plotly_white", height=240,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title=y_title, showlegend=False,
    )
    return fig


# ───────────────────────── TOP CONTROL STRIP ─────────────────────────
st.markdown("### 🧠 Skippy NPU Sizer  ·  &nbsp;_horizontal-layout prototype_",
            unsafe_allow_html=True)

with st.container(border=True):
    # ── Row 1: the frequently-touched controls. NPU tier as horizontal pills
    # (not a sidebar dropdown); Model + Workload as GREEN pickers (the "what am
    # I sizing" controls) immediately to its right. ──
    r1 = st.columns([3.2, 1.2, 1.2, 1.5, 2.7])
    with r1[0]:
        tier_label = st.segmented_control(
            "NPU tier", options=list(_TIER_MAP), default="High", key="p_tier",
            help="Silicon target — horizontal pills, not a sidebar dropdown.",
        ) or "High"
    base_tier = tier_label
    tier_name = _TIER_MAP[tier_label]
    hw = TIERS[tier_name]

    with r1[1]:
        with st.popover("Model ▾", use_container_width=True, key="pop_model"):
            # role-prefixed labels; production default. (Tier-aware 🔴 marker so
            # dtype-incompatible models read at a glance, like the live app.)
            def _fmt_model(k: str) -> str:
                role, badge = _model_role(k)
                bad = "" if hw_supports_dtype(hw, MODELS[k]["compute_dtype"]) else "🔴 "
                return f"{bad}{badge} {role} · {MODELS[k]['display_name']}"
            mkeys = list(MODELS)
            st.selectbox("LLM model", mkeys,
                         index=mkeys.index(PRODUCTION_REFERENCE_KEY),
                         format_func=_fmt_model, key="p_model")
            st.caption("🚀 production · 🔬 fine-tune · 📚 stock base · "
                       "⚙️ perf ref · 🔴 won't run on this tier")

    # ── Quant ▾ — remaps to the q4/q5/q8 sibling catalog row when one exists.
    # p_model is the family anchor; this overlays the quant on top of it. ──
    _base_model = st.session_state.get("p_model", PRODUCTION_REFERENCE_KEY)
    _cur_q = _model_quant(_base_model) or "q4"
    _avail = {tok: (tok == _cur_q or _quant_sibling(_base_model, tok) is not None)
              for _, tok in _QUANT_PILL}
    with r1[2]:
        with st.popover("Quant ▾", use_container_width=True, key="pop_quant"):
            _labels = [lbl for lbl, _ in _QUANT_PILL]
            _cur_lbl = next(lbl for lbl, t in _QUANT_PILL if t == _cur_q)
            sel_q_lbl = st.radio(
                "Quantization", _labels, index=_labels.index(_cur_lbl),
                format_func=lambda l: l if _avail[_QUANT_TOK[l]] else f"{l} ⊘",
                key="p_quant",
                help="Q4_K_M ≈ 0.57 B/param · Q5_K_M ≈ 0.71 · Q8_0 ≈ 1.06. "
                     "PAI encodes quant in the catalog, so this swaps to the "
                     "model's q4/q5/q8 sibling row. ⊘ = no sibling at that "
                     "quant (this model ships Q4 only).")
            _sel_tok = _QUANT_TOK[sel_q_lbl]
            if not _avail[_sel_tok]:
                st.caption(f"⊘ **{MODELS[_base_model]['display_name']}** ships "
                           f"{_cur_q.upper()} only — staying on {_cur_q.upper()}.")
    # Effective model = the quant sibling if available + different, else anchor.
    _sel_tok = _QUANT_TOK.get(st.session_state.get("p_quant", _cur_lbl), _cur_q)
    if _avail.get(_sel_tok) and _sel_tok != _cur_q:
        model_key = _quant_sibling(_base_model, _sel_tok) or _base_model
    else:
        model_key = _base_model
    # Effective quant = the resolved model's own quant (NOT the requested one —
    # they differ when an unavailable quant fell back to the anchor row).
    eff_quant = (_model_quant(model_key) or _cur_q).upper()

    with r1[3]:
        with st.popover("Workload ▾", use_container_width=True, key="pop_work"):
            st.selectbox("Workload profile", list(WORKLOAD_DEFAULTS),
                         format_func=lambda k: WORKLOAD_DEFAULTS[k]["label"],
                         index=0, key="p_work")

    # ── Row 2: the tuning knobs. All POPOVERS so their buttons share one
    # baseline (Memory / BW-share / Settings). ──
    r2 = st.columns([1.1, 1.1, 1.1, 5.7])
    with r2[0]:
        if tier_label in ("Mid", "High"):
            with st.popover("Memory ▾", use_container_width=True):
                opts = ["Stock"] + [o[0] for o in MEMORY_UPGRADE_OPTIONS]
                mc = st.radio("Memory upgrade", opts, index=0, key="p_mem")
                if mc != "Stock":
                    o = next(o for o in MEMORY_UPGRADE_OPTIONS if o[0] == mc)
                    hw = hw_with_memory(hw, o[1], o[2],
                                        name_suffix=f"{o[1]}-{o[2]:.0f}")
        else:
            st.popover("Memory ▾", use_container_width=True, disabled=True,
                       help="Memory upgrades apply to Mid / High only.")
    with r2[1]:
        with st.popover("BW share ▾", use_container_width=True):
            share_label = st.segmented_control(
                "NPU BW share", options=list(_SHARE_MAP), default="75%",
                key="p_share",
                help="Fraction of peak DRAM bandwidth available to the NPU "
                     "(shared with display / camera / audio / CPU). Affects "
                     "decode (BW-bound); TTFT is compute-bound and unaffected.",
            ) or "75%"
    npu_share = _SHARE_MAP[share_label]

    # ⚙ Settings — the "rarely touched" controls home (Kyle's call: a settings
    # button, not a sidebar). Token-shape override + compiler quality + KPI
    # toggle. precision_base_hw is captured here (memory applied, no precision).
    precision_base_hw = hw
    with r2[2]:
        with st.popover("⚙ Settings", use_container_width=True):
            wl = st.session_state.get("p_work", "short_chat")
            wd = WORKLOAD_DEFAULTS[wl]
            st.caption("**Token shape** — overrides the workload profile default")
            prompt_tokens = st.number_input("Prompt tokens", 10, 128000,
                                            wd["prompt_tokens"], 100, key=f"p_pt_{wl}")
            decode_tokens = st.number_input("Decode tokens", 1, 8000,
                                            wd["decode_tokens"], 50, key=f"p_dt_{wl}")
            st.divider()
            compiler_quality = st.slider(
                "Compiler quality", 0.5, 1.0, 1.0, 0.05, key="p_cq",
                help="Target NPU compiler maturity vs NVIDIA CUDA. 1.0 = parity. "
                     "Only applied to projected (non-measured) tiers.")
            show_kpis = st.toggle("Show KPI table onscreen", value=True,
                                  key="p_kpi_on")

workload_id = st.session_state.get("p_work", "short_chat")

st.caption(describe_hw(hw))
st.divider()

# ───────────────────────── PROJECT THE CELL ─────────────────────────
try:
    r = project_llm(model_key, hw, workload_id,
                    prompt_tokens=prompt_tokens, decode_tokens=decode_tokens,
                    compiler_quality=compiler_quality, npu_share=npu_share)
    r = _maybe_anchor_overlay(r, model_key, hw, tier_name, decode_tokens)
    r["_hw"] = hw
    err = None
except ValueError as e:
    r, err = None, str(e)

_model = MODELS[model_key]
_prod = MODELS[PRODUCTION_REFERENCE_KEY]
_is_production = (model_key == PRODUCTION_REFERENCE_KEY)

st.markdown(f"#### 🤖 {_model['display_name']}")

if err:
    st.warning(f"**No measurement yet for this cell.** {err}")
    st.stop()

# ── Feasibility gates first (won't load / can't execute the compute path) ──
if r["source"] == "wont_fit":
    f = r["feasibility"]
    st.error(f"🔴 **Won't fit** — needs {f['required_gb']} GB, "
             f"{hw.name} has {f['available_gb']} GB "
             f"(short by {-f['headroom_gb']:.1f} GB). Pick a smaller model or a "
             f"higher-capacity tier.")
    st.stop()
if r["source"] == "dtype_mismatch":
    d = r["dtype_detail"]
    st.error(f"🔴 **Dtype incompatible** — model needs **{d['model_needs']}**; "
             f"{hw.name} supports **{', '.join(d['hw_supports'])}**. Re-quantize "
             f"to W8A8 or move to an FP-capable tier (Low-LP5X+).")
    st.stop()

# ── Source badge + headline metrics (onscreen, full width) ──
_bi, _bl, _bt = _source_badge(r)
_regime = ("BW-bound" if r.get("regime") == "bw_bound"
           else "compute-bound" if r.get("regime") else "?")
st.caption(f"{_bi} **{_bl}** · `{tier_name}` · **{eff_quant}** · decode "
           f"regime **{_regime}** · {prompt_tokens:,} prompt / "
           f"{decode_tokens:,} decode — {_bt}")

_share_mk = "" if abs(npu_share - 1.0) < 1e-6 else f" @ {int(npu_share*100)}%"
_proj_mk = " (BW-proj)" if getattr(hw, "bw_projected", False) else ""
_ttft_s = max(r["total_s"] - r["decode_s"], 0.0)
_ttft_val = f"{_ttft_s*1000:.0f} ms" if _ttft_s < 0.1 else f"{_ttft_s:.2f} s"

m = st.columns([1.1, 1.1, 1.1, 1.1, 3.6])  # cluster 4 metrics left; spacer eats rest
m[0].metric(f"Decode tok/s{_proj_mk}{_share_mk}", f"{r['decode_tok_s']:.1f}")
m[1].metric("TTFT", _ttft_val,
            delta=f"prefill {r['prefill_tok_s']:.0f} tok/s", delta_color="off")
m[2].metric(f"End-to-end{_proj_mk}{_share_mk}", f"{r['total_s']:.2f} s")
m[3].metric("Memory fit",
            "✓ fits" if r["feasibility"]["verdict"] != "wont_fit" else "✗ spills",
            delta=f"{r['feasibility']['required_gb']} GB", delta_color="off")

# ── 2-up: per-tier decode ladder + precision what-if (Mid/High) ──
g1, g2 = st.columns([2, 3])
with g1:
    per_tier = {}
    for lbl, key in _TIER_MAP.items():
        try:
            tr = project_llm(model_key, TIERS[key], workload_id,
                             prompt_tokens=prompt_tokens, decode_tokens=decode_tokens,
                             compiler_quality=compiler_quality, npu_share=npu_share)
            tr = _maybe_anchor_overlay(tr, model_key, TIERS[key], key, decode_tokens)
            per_tier[lbl] = tr.get("decode_tok_s", 0.0) if tr else 0.0
        except ValueError:
            per_tier[lbl] = 0.0
    st.plotly_chart(_per_tier_bar(per_tier, "decode tok/s", base_tier),
                    use_container_width=True, key="l_tier")
    st.caption("Decode tok/s across the silicon ladder — near-flat across NPU "
               "classes (decode is BW-bound, not compute-bound); selection in "
               "violet. Bars at 0 = the model won't run on that tier.")
with g2:
    # The precision what-if compare (Mid/High) — PAI's validated feature, in its
    # full-width home now instead of buried in a sidebar.
    if base_tier in ("Mid", "High"):
        st.caption("**🎛️ Precision what-if — if this NPU added FP8 / FP4**")
        mat = st.radio("FP4 runtime", ["Immature (edge)", "Mature (vLLM/TRT)"],
                       index=0, horizontal=True, key="p_fp4mat")
        mat = "immature" if mat.startswith("Immature") else "mature"
        rc = st.columns(3)
        base_ttft = None
        for col, (lab, ps) in zip(rc, [("INT-only", "int8"),
                                       ("+FP8", "int8_fp8"),
                                       ("+FP8+FP4", "int8_fp8_fp4")]):
            _mt = mat if ps == "int8_fp8_fp4" else "mature"
            pr = project_llm(model_key, hw_with_precision(precision_base_hw, ps),
                             workload_id, prompt_tokens=prompt_tokens,
                             decode_tokens=decode_tokens,
                             compiler_quality=compiler_quality, npu_share=npu_share,
                             fp4_runtime_maturity=_mt)
            tt = (pr.get("ttft_s") or 0.0) * 1000
            if ps == "int8":
                base_ttft = tt
            sp = (f" · {base_ttft/tt:.1f}× vs INT8"
                  if base_ttft and ps != "int8" and tt and tt < base_ttft else "")
            col.metric(lab, f"{tt:.0f} ms", delta="prefill", delta_color="off")
            col.caption(f"decode {pr['decode_tok_s']:.0f} tok/s{sp}")
        st.caption("INT8 & FP8 both buy ~2× prefill over naive fp16; FP8's edge "
                   "is accuracy recovery (same speed). FP4 adds ~2× more — mature "
                   "runtime only. 🟠 zero edge-NPU FP4 silicon anchors (low conf).")
    else:
        st.info("**🎛️ Precision what-if** is a Mid / High feature — those tiers "
                "sit in the FP-capable LPDDR5X memory class where positing an "
                "FP8 / FP4 tensor engine is meaningful. Select **Mid** or "
                "**High** to compare INT-only → +FP8 → +FP8+FP4 prefill.")

# ── scoped depth tabs (Accuracy / Precision / Performance / Timing), wrapped in
# a collapsible "detail" expander so the section can MINIMIZE. Tabs are created
# INSIDE the expander; the with-blocks below fill them (Streamlit binds each
# tab's container at creation, so output still lands inside the expander even
# though the blocks sit outside the with-stmt). Any inner expander in a tab body
# becomes a popover — expander-inside-expander is illegal. ──
with st.expander("🔎 LLM detail — accuracy · precision · performance · timing",
                 expanded=False):
    t_acc, t_prec, t_perf, t_tim = st.tabs(
        ["Accuracy", "Precision", "Performance", "Timing"])

with t_acc:
    if "pass_rate" not in _model or _model.get("perf_reference_only"):
        st.info(f"**{_model['display_name']}** — perf-reference variant (no "
                "standalone Skippy v2+RAG eval). Pick a production / fine-tune / "
                "base row for accuracy.")
    else:
        _d = (_model["pass_rate"] - _prod["pass_rate"]) * 100.0
        st.markdown(
            f"**{_model['display_name']}** — {_model['pass_rate']*100:.1f}% pass "
            f"({_model['pass_n_passes']}/{_model['pass_n_total']}, v2+RAG)"
            + ("  ·  production reference" if _is_production
               else f"  ·  {'+' if _d >= 0 else ''}{_d:.1f}pp vs production"))
        rows = []
        for _k, _mm in MODELS.items():
            if "pass_rate" not in _mm:
                continue
            _rd = (_mm["pass_rate"] - _prod["pass_rate"]) * 100.0
            rows.append({
                "Model": (("➤ " if _k == model_key else "")
                          + _mm["display_name"]),
                "Base": _mm.get("base_model", "—"),
                "Pass": f"{_mm['pass_rate']*100:.1f}%",
                "Δ vs prod": ("— (ref)" if _k == PRODUCTION_REFERENCE_KEY
                              else f"{'+' if _rd >= 0 else ''}{_rd:.1f}pp"),
                "n": f"{_mm['pass_n_passes']}/{_mm['pass_n_total']}",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        _cd = _model.get("category_deltas") or {}
        if any(isinstance(v, dict) for v in _cd.values()):
            _pc = _prod.get("category_deltas") or {}
            st.markdown("**Per-category** (raw rates; Δ vs production):")
            for cat, data in _cd.items():
                if not isinstance(data, dict):
                    continue
                lab = CATEGORY_LABELS.get(cat, cat)
                p, n, rate = data["pass"], data["n"], data["rate"]
                pcat = _pc.get(cat)
                if isinstance(pcat, dict) and not _is_production:
                    dl = p - pcat["pass"]
                    st.markdown(f"- {lab}: **{p}/{n}** ({rate:.0%}) — "
                                f"Δ {'+' if dl >= 0 else ''}{dl}")
                else:
                    st.markdown(f"- {lab}: **{p}/{n}** ({rate:.0%})")
        # popover (not expander) so it doesn't illegally nest inside the
        # collapsible "LLM detail" expander wrapping these tabs.
        with st.popover("📐 Eval methodology — Finding 4 (Qwen-family format bias)"):
            st.markdown(
                "Headline uses **semantic grading** (GPT-4o binary, 132-sample "
                "v2-RAG, temp=0). The production model's substring lift eroded "
                "across five successive cross-checks (substring +3.1 → semantic "
                "−4.6pp, sign reversal). Production decision unaffected — Skippy "
                "ships on the three-gate framework (capability + voice + safety); "
                "substring was never load-bearing.")

with t_prec:
    if base_tier in ("Mid", "High"):
        st.subheader("Precision-set benefit — same model, three precision rungs")
        st.caption(
            f"**{_model['display_name']}** on the **{tier_name}** memory class @ "
            f"{prompt_tokens:,}-token prompt. Each rung posits an FP-capable "
            f"tensor engine and runs the matmul at that precision. Prefill / TTFT "
            f"is the headline compute benefit; decode is BW-bound (held by the "
            f"4-bit weight stream). FP4 is 🟠 modeled — zero edge-NPU anchors.")
        _rungs = [("INT-only", "int8", "W8A8 — 2× prefill vs fp16",
                   "🔴 −3.8pp (W8A8 cliff)"),
                  ("INT + FP8", "int8_fp8", "== INT8 speed, near-lossless",
                   "🟢 ≈0pp (recovers cliff)"),
                  ("INT + FP8 + FP4", "int8_fp8_fp4", "2× the 8-bit prefill (mature)",
                   "🟢 ≈0pp (NVFP4)")]
        _mat = st.session_state.get("p_fp4mat", "Immature (edge)")
        _mat = "immature" if str(_mat).startswith("Immature") else "mature"
        cols = st.columns(3)
        base_ttft = None
        for col, (lab, ps, speed, acc) in zip(cols, _rungs):
            mt = _mat if ps == "int8_fp8_fp4" else "mature"
            rr = project_llm(model_key, hw_with_precision(precision_base_hw, ps),
                             workload_id, prompt_tokens=prompt_tokens,
                             decode_tokens=decode_tokens,
                             compiler_quality=compiler_quality, npu_share=npu_share,
                             fp4_runtime_maturity=mt)
            ttms = (rr.get("ttft_s") or 0.0) * 1000
            if ps == "int8":
                base_ttft = ttms
            if ps == "int8_fp8_fp4":
                if mt == "immature":
                    speed = "immature → no prefill win (bf16 floor)"
                    acc = "🟡 ≈0pp acc, ADR-016 'no win'"
                lab = f"{lab} · {mt}"
            sp = (f" ({base_ttft/ttms:.1f}× vs INT8)"
                  if ttms and base_ttft and ps != "int8" and ttms < base_ttft else "")
            with col:
                st.markdown(f"**{lab}**")
                st.metric("Prefill / TTFT", f"{ttms:.0f} ms", delta=speed,
                          delta_color="off")
                st.caption(f"Decode: **{rr['decode_tok_s']:.1f}** tok/s{sp}")
                st.caption(f"Accuracy: {acc}")
        _wgb = r["feasibility"].get("breakdown", {}).get("weights_gb")
        st.caption(
            "↑ INT8 **and** FP8 both buy ~2× prefill over a naive fp16 run; FP8's "
            "edge over INT8 is the accuracy recovery (same speed). FP4 adds "
            "another ~2× prefill on a mature runtime only. "
            + (f"**Weight RAM ≈ {_wgb} GB** is fixed by this model's Q4 "
               f"quantization — orthogonal to the compute rung."
               if _wgb is not None else
               "Weight RAM is set by the model's quantization, not the rung."))
    else:
        st.info("Precision-set compare is a Mid / High feature (the FP-capable "
                "LPDDR5X memory class). Select **Mid** or **High**.")

with t_perf:
    st.markdown("**Decode tok/s vs context length** — decode is BW-bound, so the "
                "curve flattens to the anchor value once a same-class anchor "
                "takes over (only the 5090 reference shows prompt-length shape).")
    ctx_grid = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    ys, src = [], "?"
    for ctx in ctx_grid:
        try:
            d = decode_tok_s_at_context(model_key, hw, ctx,
                                        compiler_quality=compiler_quality,
                                        npu_share=npu_share)
            ys.append(d["decode_tok_s"])
            src = d.get("source", src)
        except ValueError:
            ys.append(0.0)
    fig = go.Figure(go.Scatter(
        x=ctx_grid, y=ys, mode="lines+markers",
        line=dict(color=_ACCENT, width=3), marker=dict(size=9)))
    fig.update_layout(template="plotly_white", height=320,
                      margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="Context length (tokens)", xaxis_type="log",
                      yaxis_title=f"decode tok/s on {hw.name}")
    st.plotly_chart(fig, use_container_width=True, key="l_ctx")
    st.caption(f"Source: `{src}`. Active-param weight streaming dominates decode "
               f"BW; context length barely moves it on MoE / dense alike.")

with t_tim:
    sp_ms = _ttft_s * 1000
    sd_ms = r["decode_s"] * 1000
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Prefill / TTFT", x=["This workload"], y=[sp_ms],
                         marker_color="#F59E0B",
                         text=[f"{sp_ms:.0f} ms"], textposition="auto"))
    fig.add_trace(go.Bar(name="Decode", x=["This workload"], y=[sd_ms],
                         marker_color=_ACCENT,
                         text=[f"{sd_ms:.0f} ms"], textposition="auto"))
    fig.update_layout(barmode="stack", template="plotly_white", height=300,
                      margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title="Per-answer latency (ms)",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True, key="l_tim")
    st.caption(f"TTFT = **{_ttft_val}** (host + prefill compute, compute-bound) · "
               f"decode = **{r['decode_tok_s']:.1f} tok/s** over "
               f"{decode_tokens:,} tokens = **{r['decode_s']:.2f} s**. "
               "TTFT held at stock under any memory upgrade (prefill isn't "
               "BW-bound); decode scales with effective BW × NPU-share.")

st.divider()

# ───────────────────────── KPIs ONSCREEN ─────────────────────────
if st.session_state.get("p_kpi_on", True):
    st.markdown("#### 📊 KPIs — current configuration")
    kpi = {
        "Model": _model["display_name"],
        "Quant": eff_quant,
        "Tier": tier_name,
        "Source": _bl,
        "Decode tok/s": round(r["decode_tok_s"], 1),
        "TTFT": _ttft_val,
        "End-to-end (s)": round(r["total_s"], 2),
        "Mem required (GB)": r["feasibility"]["required_gb"],
        "Fits": "✓" if r["feasibility"]["verdict"] != "wont_fit" else "✗",
    }
    df = pd.DataFrame([kpi])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("⬇ Export KPIs (CSV)", df.to_csv(index=False),
                       "skippy_kpis.csv", "text/csv", key="p_kpi_dl")

    # ── hardware config (reference detail, collapsed) ──
    with st.expander("🔬 Hardware config — selected tier (incl. overlays)"):
        st.json({
            "name": hw.name,
            "mem_type": hw.mem_type,
            "mem_data_rate_gtps": hw.mem_data_rate_gtps,
            "mem_bandwidth_gbs_theoretical": round(hw.mem_bandwidth_gbs, 1),
            "mem_bandwidth_gbs_effective": round(hw.effective_bandwidth_gbs, 1),
            "mem_capacity_gb": hw.mem_capacity_gb,
            "peak_tops_int8": hw.peak_tops_int8,
            "peak_tops_bf16": getattr(hw, "peak_tops_bf16", None),
            "peak_tops_fp8": getattr(hw, "peak_tops_fp8", None),
            "npu_share_applied": npu_share,
            "bw_projected": getattr(hw, "bw_projected", False),
        })

st.divider()
st.caption("⬑ **Prototype** — top control strip + full-width results/charts + "
           "onscreen KPIs, no sidebar; verbose tabs tuck into the collapsible "
           "detail expander. Tier across all silicon highlighted in violet. If "
           "the layout works, Step 3 ports the live app.py onto this shell.")
