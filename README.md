# Skippy NPU sizer (personal-ai-assistant-sizer)

[![Version](https://img.shields.io/badge/version-v2.0.0-3b82f6?style=flat-square)](https://github.com/kylefoxaustin/personal-ai-assistant-sizer/releases/tag/v2.0.0)
[![Live app](https://img.shields.io/badge/Live_app-personal--ai--assistant--sizer.streamlit.app-10b981?style=flat-square&logo=streamlit)](https://personal-ai-assistant-sizer.streamlit.app/)

Interactive sizing tool that projects Skippy's performance (the personal AI
assistant in `personal-ai-framework`) across NPU tiers — from NPU Low-LP4
through NPU High — using measured RTX 5090 baselines and (when secrets
populated) real silicon-anchor measurements.

Companion repo to [keyhole-sizer](https://github.com/…/keyhole-sizer). Same
pattern, different domain: LLM-first (no vision pipelines).

**Live:** <https://personal-ai-assistant-sizer.streamlit.app/> — password-gated
(password is in the Streamlit Cloud secrets; ask Kyle for access). The URL
is public but the app itself is gated — intended audience is internal
reviewers evaluating feasibility, not end users.

## What it does

- **Pick from 20 catalog models** — role-classified in the dropdown via
  badges: 🚀 PROD (production Skippy 7B v4) · 🔬 FT (Skippy fine-tune
  experiments) · 📚 BASE (stock public bases — Qwen 2.5 / Qwen3 / Llama /
  Mistral) · ⚙️ PERF (perf-reference variants for alternate compute paths
  or quantizations). 🔴-prefixed models can't execute on the currently-
  selected NPU tier (dtype mismatch) and sort to the bottom.
- **Pick an NPU tier** — Low-LP4 / Low-LP5* / Low-LP5X / Mid / High / RTX
  5090. Optional memory upgrade overlays (LPDDR5T 11.2 / LPDDR6-12 / 14).
- **Pick a workload** — short chat / RAG Q&A / long-decode / meeting
  summarization / agentic roundtrip.
- **Tune NPU_share** — 100/75/50/25% of bandwidth available to the NPU
  (the rest reserved for camera DMA, codecs, other masters).
- **See the headline tile** — decode tok/s, TTFT, end-to-end latency,
  decode duration. When a real measured silicon anchor exists for the
  selected (tier, model) cell, the headline tile uses the measured value
  with a 🟢 "Measured on real NPU silicon" banner; otherwise it shows the
  BW-projected value with 🟡/🟠 source classification.
- **Read KPIs onscreen** — the KPI table renders inline (not download-only),
  with a current-config CSV plus cross-tier + cross-model tables and an uber
  XLSX (both sheets) for the chosen tier.
- **Drill into the `🔎 detail` expander** — the verbose depth tabs
  (`Accuracy · Precision · Performance · Timing`) tuck into a collapsible
  per-section expander so the headline metrics stay on a short first paint.

## Key insight the sizer surfaces

On decode throughput, the "bigger" MoE 30B-A3B **runs 3–15× faster than the
"smaller" dense 14B**. Because MoE activates ~3B params per token vs dense's
~14B, bandwidth-bound decode favors the MoE. The gap widens at long context
(15× at 13K prefill).

The role-classified dropdown + tier-aware compatibility filter + role badges
make this kind of cross-model / cross-tier comparison visually scannable
instead of requiring users to know each model's compute path.

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. If `.streamlit/secrets.toml` has `PASSWORD=...`,
a password gate appears.

`app.py` is the **horizontal layout** (promoted to primary at v2.0.0,
2026-06-11): no left sidebar — a top horizontal **control strip** (NPU-tier
pills + Model / Quant / Workload pickers + Memory / BW-share popovers + a ⚙
Settings popover), full-width results, onscreen KPIs, and each section's verbose
depth-tabs tucked into a collapsible "🔎 detail" expander so headline metrics
stay on a short first paint. The **Quant ▾** pill remaps to the q4/q5/q8 sibling
catalog row when one exists (PAI bakes quant into the catalog key — there is no
quant param in `project_llm`). Mirrors keyhole-sizer's horizontal go-live
(its v2.0.0) on PAI's LLM-only domain.

### Legacy vertical-sidebar layout

```bash
streamlit run app_vertical_legacy.py
```

The prior layout (the live app through v1.2.0): a tall left **sidebar** drives
model + tier + memory upgrade + NPU_share + workload selection, and the results
fill a 7-tab main page (`Overview · Accuracy · Precision · Performance · KPIs ·
Cost · Data`). Preserved for reference / rollback; it shares the same engine, so
projections are identical to the horizontal `app.py`. Roll back to it by
re-pointing the Streamlit Cloud entrypoint, or check out tag `v1.2.0`.

## Data sources — two layers

1. **RTX 5090 reference bake-offs** populate `sizer/sizer_bundle.json` via
   `personal-ai-framework/eval/build_sizer_bundle.py`. Every non-anchor cell
   BW-projects from these. The bundle's `meta.methodology_version` stamp
   surfaces in the Accuracy tab footnote and in the About expander.
2. **Private NPU silicon anchors** live in Streamlit secrets
   (`.streamlit/secrets.toml` locally; Settings → Secrets in Streamlit
   Cloud). 9 LLM cells (3 tier-precision × 3 Qwen models) + 6 CNN cells (2
   tier-precision × 3 CNN variants). When a measured anchor matches the
   selected (tier, model) cell, the headline decode tile uses the anchor
   value directly. Spec:
   `personal-ai-framework/docs/private_anchor_secrets_spec.md`.

**Discipline rule:** real measurement values never enter chat, git, or
Drive. Refer by KEY (`npu_llm_anchors.mid_int8.qwen3_30b_a3b_moe.tokps`)
not VALUE. The `.gitignore` excludes `.streamlit/secrets.toml`.

See [REPRODUCE.md](REPRODUCE.md) for how to regenerate the bundle from
fresh bake-offs.

## Architecture

- `app.py` — Streamlit UI, **horizontal layout** (primary since v2.0.0). No
  sidebar; top control strip drives model + tier + memory upgrade + NPU_share
  + workload selection; onscreen KPIs; verbose depth-tabs in a collapsible
  `🔎 detail` expander. See *Quickstart*.
- `app_vertical_legacy.py` — the prior vertical-sidebar layout (live through
  v1.2.0). 7-tab main page; sidebar-driven controls. Same engine, identical
  projections; preserved for reference / rollback.
- `sizer/npu_model.py` — PAI's visible 7-tier ladder (`TIERS`, composed from
  ratchet's registry) + MODELS catalog (20 entries) + BW-bound `project_llm()`
  + the NPU precision-set engine (`hw_with_precision`). Hardware tiers,
  `hw_with_memory`, and dtype gates come from `ratchet`.
- `sizer/measured.py` — loads `sizer_bundle.json` → attaches to
  `RTX_5090_REFERENCE.measured_llm` at import; exposes `get_bundle_summary()`
  for the About expander and the `methodology_version` footnote.
- Silicon anchors load via `ratchet.anchors` (`load_llm_anchor` /
  `load_cnn_anchor`) — the former local `sizer/npu_anchors.py` was lifted into
  ratchet at v1.1.0. Returns `None` on missing/zero values so callers fall back
  to projection.
- `sizer/precision.py` — precision capability tables (sourced from ratchet's
  canonical capability tables) + retargeting cost model + annualized lifecycle
  cost computation.
- `sizer/sizer_bundle.json` — vendored measurements from Skippy bake-offs
  (5090 reference).

## Deploying to Streamlit Cloud

Push to GitHub, connect repo, `app.py` is the entrypoint. Put the password
in Streamlit Cloud's secrets UI. For private silicon anchors, paste the
contents of your local `.streamlit/secrets.toml` (gitignored) into the same
Secrets UI — Streamlit encrypts at rest and injects at runtime.

**Reboot rule**: changes under `sizer/*.py` require a manual reboot
(share.streamlit.io → Manage app → Reboot) because Streamlit Cloud's auto-
reload is `app.py`-only — `sys.modules` caches the stale `sizer/*` module
otherwise.

## Cross-app coordination

PAI sizer and [keyhole-sizer](https://github.com/…/keyhole-sizer) are sister
Streamlit apps maintained in lockstep. They share:

- Same 7-tab UX structure (`Overview · Accuracy · Precision · Performance ·
  KPIs · Cost · Data` on PAI; keyhole adds `Stream scaling · Duty-cycle`
  before `KPIs · Detail` for its vision domain).
- Same role-classification icon scheme (🚀/🔬/📚/⚙️/🔴).
- Same anchor-secrets spec
  (`personal-ai-framework/docs/private_anchor_secrets_spec.md`).
- Same `methodology_version` stamp pattern on bundle metadata.
- Same eval methodology: semantic-graded pass rates (GPT-4o binary judge,
  132-sample v2-RAG) — see the Accuracy tab's Finding 4 surface.
- Same **horizontal layout** — both apps promoted it to primary `app.py` at
  v2.0.0 (keyhole 2026-06-10, PAI 2026-06-11), preserving the prior layout as
  `app_vertical_legacy.py`. Shared collapsible-detail "minimize" pattern plus a
  parity KPI-export surface (cross-tier + cross-model tables + uber XLSX). PAI
  is the LLM-only cut; keyhole generalizes it across Vision / LLM / VLA
  workloads.

## Version history

| Version | Date | Highlights |
|---|---|---|
| **v2.0.0** | 2026-06-11 | **Horizontal layout go-live** — promoted `app_horizontal_prototype.py` to primary `app.py` (no sidebar, top control strip, onscreen KPIs, collapsible `🔎 detail` expander); prior vertical-sidebar layout preserved as `app_vertical_legacy.py`. Mirrors keyhole-sizer v2.0.0. |
| **v1.2.0** | 2026-06-11 | Rollback point — last vertical-sidebar `app.py` before the horizontal go-live. |
| **v1.1.0** | 2026-06-06 | Retrofit onto **ratchet** (the shared SoC sizing engine, Option C). Hardware tiers, anchor loader, memory-upgrade + dtype-gate helpers, and precision-capability tables now come from ratchet; PAI keeps its own projection math. Adds the NPU precision-set selector (Stock / INT-only / INT+FP8 / INT+FP8+FP4) for Mid/High (ratchet v0.2.7 / ADR 017). |
| **v1.0.0** | 2026-05-18 | First tagged release. Recovery point before engine-extraction. Captures: 20-model catalog with semantic-graded eval (Finding 4), private NPU + CNN anchor-secrets mechanism with hot-swap into headline tile, 7-tab main-page UX, role-classified model dropdown with tier-aware compatibility filter, `methodology_version` stamp surfaced in UI, cross-app convergence with keyhole-sizer. |

## Roadmap

- **✅ Engine-extraction (shipped v1.1.0)** — the shared LLM/anchor/bundle
  layers were extracted into **ratchet**, the common SoC sizing engine consumed
  by PAI sizer, keyhole-sizer, and the upcoming drone-repo sizer. PAI adopted it
  Option-C (lightest): ratchet owns hardware tiers + anchor loader + capability
  tables; PAI keeps its own projection math (consolidation onto ratchet's
  `project_llm` is a deliberate later pass — rule of three).
- **v2.x (future)** — continuous /metrics instrumentation on production
  Skippy (model_name labels, prefill/decode split, RAG latency) so the
  bundle regenerates from live production traffic — see
  `export_skippy_metrics_for_sizer.py` pattern offered by [backend]
  2026-04-22.
- **v3.x (future)** — machine-to-machine agentic load simulation — two-
  Skippy HTTP dialogue, sustained throughput measurement; replaces synthetic
  workload profiles with measured production patterns.
