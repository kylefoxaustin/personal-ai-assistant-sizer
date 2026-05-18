# Skippy NPU sizer (personal-ai-assistant-sizer)

[![Version](https://img.shields.io/badge/version-v1.0.0-3b82f6?style=flat-square)](https://github.com/kylefoxaustin/personal-ai-assistant-sizer/releases/tag/v1.0.0)
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
- **Drill into 7 main-page tabs** — `Overview · Accuracy · Precision ·
  Performance · KPIs · Cost · Data`.

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

- `app.py` — Streamlit UI (~2200 lines). 7-tab main page; sidebar drives
  model + tier + memory upgrade + NPU_share + workload selection.
- `sizer/npu_model.py` — Hardware dataclass + MODELS catalog (20 entries)
  + TIERS dict + BW-bound `project_llm()` + Phase 2 source-state
  classification.
- `sizer/measured.py` — loads `sizer_bundle.json` → attaches to
  `RTX_5090_REFERENCE.measured_llm` at import; exposes `get_bundle_summary()`
  for the About expander and the `methodology_version` footnote.
- `sizer/npu_anchors.py` — typed `LLMAnchor` / `CNNAnchor` loader for the
  private silicon anchor secrets. Returns `None` on missing/zero values so
  callers fall back to projection.
- `sizer/precision.py` — precision capability tables + retargeting cost
  model + annualized lifecycle cost computation.
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

## Version history

| Version | Date | Highlights |
|---|---|---|
| **v1.0.0** | 2026-05-18 | First tagged release. Recovery point before engine-extraction. Captures: 20-model catalog with semantic-graded eval (Finding 4), private NPU + CNN anchor-secrets mechanism with hot-swap into headline tile, 7-tab main-page UX, role-classified model dropdown with tier-aware compatibility filter, `methodology_version` stamp surfaced in UI, cross-app convergence with keyhole-sizer. |

## Roadmap

- **v1.x (next)** — engine-extraction: refactor shared LLM/anchor/bundle
  layers into a common 'engine' package consumed by PAI sizer, keyhole-
  sizer, and the upcoming drone-repo sizer. The 4-sizer ecosystem
  (Skippy / Keyhole / Drone / future-4th) needs a shared core so methodology
  decisions propagate cleanly.
- **v2.x (future)** — continuous /metrics instrumentation on production
  Skippy (model_name labels, prefill/decode split, RAG latency) so the
  bundle regenerates from live production traffic — see
  `export_skippy_metrics_for_sizer.py` pattern offered by [backend]
  2026-04-22.
- **v3.x (future)** — machine-to-machine agentic load simulation — two-
  Skippy HTTP dialogue, sustained throughput measurement; replaces synthetic
  workload profiles with measured production patterns.
