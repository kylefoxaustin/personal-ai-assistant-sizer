# Skippy NPU sizer (personal-ai-assistant-sizer)

[![Live app](https://img.shields.io/badge/Live_app-personal--ai--assistant--sizer.streamlit.app-10b981?style=flat-square&logo=streamlit)](https://personal-ai-assistant-sizer.streamlit.app/)

Interactive sizing tool that projects Skippy's performance (the personal AI
assistant in `personal-ai-framework`) across NPU tiers — from NPU Low-LP4
through NPU High — using measured RTX 5090 baselines.

Companion repo to [keyhole-sizer](https://github.com/…/keyhole-sizer). Same
pattern, different workload: LLM-first (no vision pipelines).

**Live:** <https://personal-ai-assistant-sizer.streamlit.app/> — password-gated
(password is in the Streamlit Cloud secrets; ask Kyle for access). The URL
is public but the app itself is gated — intended audience is internal
reviewers evaluating feasibility, not end users.

## What it does

- Pick a model: Qwen 2.5 14B (dense) or Qwen3-30B-A3B (MoE)
- Pick an NPU tier, or the RTX 5090 reference
- Pick a workload: short chat / RAG Q&A / long-decode doc gen / meeting
  summarization / agentic roundtrip
- Tune the compiler-quality slider
- See decode tok/s, prefill throughput, per-stage breakdown, and cross-tier
  comparison — with measured (🟢) or BW-projected (🟡) badging per cell

## Key insight the sizer surfaces

On decode throughput, the "bigger" MoE 30B-A3B **runs 3–15× faster than the
"smaller" dense 14B**. Because MoE activates ~3B params per token vs dense's
~14B, bandwidth-bound decode favors the MoE. The gap widens dramatically at
long context (15× at 13K prefill).

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. If `.streamlit/secrets.toml` has `PASSWORD=...`,
a password gate appears.

## Data source

`sizer/sizer_bundle.json` is vendored from the `personal-ai-framework` repo's
`eval/build_sizer_bundle.py` output. See [REPRODUCE.md](REPRODUCE.md) for how
to regenerate it from fresh bake-offs.

## Architecture

- `app.py` — Streamlit UI
- `sizer/npu_model.py` — Hardware + MODELS dicts, BW-bound `project_llm()`
- `sizer/measured.py` — loads `sizer_bundle.json` → attaches to
  `RTX_5090_REFERENCE.measured_llm` at import
- `sizer/sizer_bundle.json` — vendored measurements from Skippy bake-offs

## Deploying to Streamlit Cloud

Push to GitHub, connect repo, app.py is the entrypoint. Put the password in
Streamlit Cloud's secrets UI. **Reboot rule**: changes under `sizer/*.py`
require a manual reboot (share.streamlit.io → Manage app → Reboot) because
Streamlit Cloud's auto-reload is `app.py`-only — `sys.modules` caches the
stale `sizer/*` module otherwise.

## Roadmap

- **Phase 1 (current)**: synthetic LLM-only profiles, 5090 measured baselines,
  BW projection to edge tiers.
- **Phase 2**: continuous /metrics instrumentation on Skippy (model_name
  labels, prefill/decode split, RAG latency) so the bundle regenerates from
  live production traffic — see `export_skippy_metrics_for_sizer.py` pattern
  offered by [backend] 2026-04-22.
- **Phase 3**: machine-to-machine agentic load simulation — two-Skippy HTTP
  dialogue, sustained throughput measurement.
