"""Schema-v2 bundle: real-silicon board anchors + pre-roofline feasibility gates.

Vendored from personal-ai-framework `eval/results/sizer_bundle_v2.json`
(origin/main @ 60b4f48). v2 is ADDITIVE: `models` / `workloads` are the v1
content and are still consumed by `measured.py`. This module owns only the
two NEW sections, plus the one engine rule they carry:

  ⚠ FEASIBILITY IS A GATE, NOT A COST.
  A model whose ops the toolchain cannot place does not run "slowly" — it
  does not run. Folding that into a roofline yields a confident wrong
  number ([95emulator], carried by [docs] 2026-09-17). So the predicate
  below is designed to be called BEFORE any cost model, and it returns a
  verdict, never a rate.

  ⚠ A CONTEXT CAP IS A GATE TOO.
  iq9's NPU accepts <=256 input tokens; beyond that the workload does not
  get slower on the NPU, it LANDS ON A DIFFERENT BACKEND (CPU llama.cpp,
  ~19 tok/s). That is a discontinuity, not a curve.

Both sections are keyed and provenance-tagged; nothing here invents a
number. Every value returned carries the bundle's own `tag`.
"""
from __future__ import annotations

import json
from pathlib import Path

_V2_PATH = Path(__file__).parent / "sizer_bundle_v2.json"


def load_v2() -> dict:
    """Load the v2 bundle. Returns {} if absent — v2 is additive, so its
    absence must degrade to v1 behaviour rather than raising."""
    if not _V2_PATH.exists():
        return {}
    d = json.loads(_V2_PATH.read_text())
    if d.get("meta", {}).get("schema_version") != 2:
        return {}
    return d


_V2 = load_v2()


# ─────────────────────────── boards ───────────────────────────

def boards() -> dict:
    """Real measured silicon, keyed by board_id. Sibling of the abstract
    `tiers_measured` presets — NOT a replacement (open item 1)."""
    return _V2.get("boards_measured", {})


def board_perf(board_id: str, model_key: str, precision: str | None = None):
    """Measured perf cell for (board, model[, precision]), or None.

    Returns the bundle cell verbatim including its `tag` and `date`; callers
    must not strip provenance."""
    per_model = boards().get(board_id, {}).get("perf", {}).get(model_key)
    if per_model is None:
        return None
    if precision is None:
        return per_model
    return per_model.get(precision)


# ───────────────────── context cap (a GATE) ─────────────────────

def context_gate(board_id: str, prompt_tokens: int) -> dict | None:
    """Does this prompt fit the board's NPU input cap?

    Returns None when the board declares no cap. Otherwise a verdict:
      {"verdict": "npu" | "over_cap", "cap": N, ...}

    On "over_cap" the returned `backend`/`decode_tps` describe the FALLBACK
    path, because exceeding the cap changes which engine runs the model.
    This is deliberately not expressed as a slowdown factor."""
    cap_block = boards().get(board_id, {}).get("context_cap")
    if not cap_block:
        return None
    cap = cap_block.get("npu_max_input_tokens")
    if cap is None:
        return None
    if prompt_tokens <= cap:
        return {"verdict": "npu", "cap": cap, "board": board_id}
    beyond = dict(cap_block.get("beyond_cap", {}))
    beyond.update({"verdict": "over_cap", "cap": cap, "board": board_id,
                   "note": cap_block.get("note")})
    return beyond


# ─────────────── feasibility gate (runs BEFORE roofline) ───────────────

def gates() -> dict:
    return _V2.get("feasibility_gates", {})


def gate_for(part: str | None = None, toolchain: str | None = None,
             version: str | None = None) -> dict | None:
    """Look up a gate by (part, toolchain, version). Version may be omitted
    when the bundle records the gate as stable across versions."""
    for key, g in gates().items():
        if part and part not in (g.get("part"), key.split("__")[0]):
            continue
        if toolchain and toolchain != g.get("toolchain"):
            continue
        if version and version != g.get("version"):
            stable = (g.get("version_stability") or {}).get("stable_across", [])
            if version not in stable:
                continue
        return g
    return None


def placement_verdict(gate: dict | None, model_ops: set[str] | None) -> dict:
    """PRE-ROOFLINE predicate. Returns a verdict, never a rate.

    `model_ops` is the POST-FUSION op set for the model on this toolchain.
    When it is unknown we return "unknown" rather than guessing placeable —
    an unproven placement must not silently become a cost.
    """
    if gate is None:
        return {"verdict": "no_gate", "reason": "no feasibility data for this (part, toolchain)"}
    if not model_ops:
        return {"verdict": "unknown",
                "reason": "post-fusion op set not supplied; placement unproven",
                "gate_key": gate.get("part")}
    blocking = []
    for op in gate.get("gating_ops", []):
        name = op.get("op")
        if name in model_ops:
            blocking.append({"op": name, "outcome": op.get("outcome"),
                             "excludes": op.get("excludes"), "tag": op.get("tag")})
    if blocking:
        return {"verdict": "unplaceable", "blocking_ops": blocking,
                "provenance": gate.get("provenance", {})}
    return {"verdict": "placeable", "checked_against": [o.get("op") for o in gate.get("gating_ops", [])],
            "provenance": gate.get("provenance", {})}

# ─────────────── bundle-key ↔ catalog-key mapping ───────────────
# v2 names the production model `skippy-7b-v4-q4-dense`; PAI's catalog has
# the same model (geometry verified identical: hidden_dim/layers/heads/
# kv_heads/vocab/family all match) as `qwen25-7b-v4-q4-dense`, which is
# PRODUCTION_REFERENCE_KEY. Without this map the loader's `if key not in
# MODELS: continue` silently drops every v2 board row for the production
# model — a skip that reads as "no data" rather than "name mismatch".
#
# Same transitional shape as npu_model._ANCHOR_MODEL_KEY_MAP (PAI hyphenated
# -> ratchet snake_case); a canonical-key migration across surfaces is the
# real fix and is out of scope here.
_BUNDLE_TO_CATALOG = {
    "skippy-7b-v4-q4-dense": "qwen25-7b-v4-q4-dense",
}


def catalog_key(bundle_key: str) -> str:
    """Map a v2 bundle model key onto PAI's catalog key (identity if unmapped)."""
    return _BUNDLE_TO_CATALOG.get(bundle_key, bundle_key)


def unresolved_model_keys(catalog: dict) -> list[str]:
    """v2 model keys that resolve to nothing in PAI's catalog.

    Call this at load time: a non-empty list means the bundle carries data
    the engine will silently ignore. Returning it is the point — the failure
    mode this guards against is a skip nobody sees."""
    out = []
    for section in ("models", "workloads"):
        for k in _V2.get(section, {}):
            if section == "workloads":
                continue
            if catalog_key(k) not in catalog:
                out.append(k)
    for b in boards().values():
        for k in b.get("perf", {}):
            if catalog_key(k) not in catalog and k not in out:
                out.append(k)
    return sorted(out)
