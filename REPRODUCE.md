# Reproducing the Skippy sizer bundle

The `sizer/sizer_bundle.json` vendored in this repo is the output of a
bake-off pipeline run against a live Skippy instance. The source scripts
live in the `personal-ai-framework` repo, not this one.

## Prerequisites

- `personal-ai-framework` cloned at `~/Documents/GitHub/personal-ai-framework`
- Skippy running: `./run.sh start` (docker compose up)
- `SKIPPY_USER` and `SKIPPY_PASSWORD` exported (Skippy auth)

## Full regeneration workflow

```bash
cd ~/Documents/GitHub/personal-ai-framework
source ~/.skippy-env   # or: export SKIPPY_USER=... SKIPPY_PASSWORD=...

# 1. Bake off MoE (currently loaded per pipeline/config.yaml)
python3 eval/run_sizer_bakeoffs.py --name kyle-30b-moe-$(date +%Y%m%d) \
    --samples 5 --profiles short_chat,rag_qa,long_decode,meeting_summarization,agentic_roundtrip

# 2. Swap config to dense
#    pipeline/config.yaml model.path → kyle-14b-v3-q4_k_m.gguf
./run.sh stop && ./run.sh start

# 3. Bake off dense
python3 eval/run_sizer_bakeoffs.py --name kyle-14b-dense-$(date +%Y%m%d) \
    --samples 5 --profiles short_chat,meeting_summarization,agentic_roundtrip
# (long_decode currently stalls on dense at 2500 decode tokens — see
#  project memory project_personal_ai_sizer.md for investigation notes)

# 4. Restore MoE config, restart
#    pipeline/config.yaml model.path → kyle-30b-a3b-v1-q4_k_m.gguf
./run.sh stop && ./run.sh start

# 5. Build the bundle
python3 eval/build_sizer_bundle.py \
    --bakeoff eval/results/sizer_bakeoff_kyle-30b-moe-*_*.json \
    --bakeoff eval/results/sizer_bakeoff_kyle-14b-dense-*_*.json \
    --out eval/results/sizer_bundle.json

# 6. Vendor into this repo
cp eval/results/sizer_bundle.json \
   ~/Documents/GitHub/personal-ai-assistant-sizer/sizer/sizer_bundle.json

# 7. Push — Streamlit Cloud auto-deploys.
#    (Rebooting from share.streamlit.io not strictly needed since only JSON
#    changed, but the sizer/ module load path may cache — reboot if the
#    projections look stale.)
```

## Bake-off knobs to be aware of

- `/generate` accepts two sizer-specific flags (added in personal-ai-framework
  `pipeline/llm_server.py`):
  - `include_telemetry: bool` — makes /generate stream internally and return
    `{host_ms, prefill_ms, decode_ms, prompt_tokens, completion_tokens,
    prefill_tok_per_s, decode_tok_per_s, model_name, ...}`
  - `skip_agent_loop: bool` — bypasses `_run_agent_loop()` tool-detection so
    the telemetry reflects pure LLM perf (without up-to-5 extra LLM calls
    for tool detection)
- Profiles are defined in `eval/sizer_workload_profiles.json`. Add new
  workloads by extending that file and the script picks them up.

## Known data caveats (as of 2026-04-22)

- **KV cache reuse**: llama-cpp-python reuses prefill KV cache on repeated
  identical prompts. Samples 2-5 of the same prompt show ~30ms prefill
  instead of the cold ~3500ms. p95 prefill values are the honest cold
  number; p50 is contaminated.
- **Streaming artifact on tiny completions**: rare, visible as decode_ms
  near-zero. Filter decode_tok_s > 10K as non-physical if aggregating
  post-hoc.
- **Dense long_decode stalls**: 2500-token generation on 14B dense takes
  >10min — context-dependent decode slowdown. Profile currently skipped
  for dense in v3 bundles.
- **ChromaDB persistence**: until the fix (`IS_PERSISTENT=TRUE` in
  docker-compose.yaml, applied 2026-04-22 late session), `rag_qa` data
  could be lost on container restart. Present in MoE v1 bundle only.
