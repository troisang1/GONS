"""Shared GONS run helper: write config -> run FSCIL -> score via official OS-HM.

Every GONS run (smoke, main, ablation) flows through `run_gons_cfg`, which scores
through the ONE official entrypoint
`gons.evaluation.aggregate_openset.full_metrics_from_run`.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import yaml

from gons.evaluation.aggregate_openset import full_metrics_from_run
from gons.runners.fscil import run_fscil_experiment

REPO = Path(__file__).resolve().parent
CFG_SCRATCH = REPO / "artifacts" / "_cfg_scratch"


def run_gons_cfg(cfg: dict, tag: str, *, cleanup: bool = True) -> dict:
    """Run one GONS config and return full open-set metrics.

    Returns a dict with status + os_hm/ccr/tur/f1_unknown/macro_f1_with_unknown
    (None on failure) and elapsed seconds.
    """

    CFG_SCRATCH.mkdir(parents=True, exist_ok=True)
    cfg_path = CFG_SCRATCH / f"{tag}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, default_flow_style=False))

    t0 = time.time()
    try:
        # `seed` here is the MODEL rng, not the protocol seed -- it mirrors
        # `app.py run-experiment --seed`, which defaulted to 42 on every published
        # run. The protocol seed travels in cfg["base_class_seed"].
        result = run_fscil_experiment(cfg_path, repo_root=REPO, seed=cfg["model"]["seed"])
    except Exception as exc:  # never let one cell kill a sweep
        return {
            "tag": tag,
            "status": "failed",
            "err": f"{type(exc).__name__}: {exc}"[:400],
            "elapsed": round(time.time() - t0, 1),
        }
    elapsed = round(time.time() - t0, 1)
    run_dir = Path(result.run_dir)
    fm = full_metrics_from_run(run_dir)
    out = {
        "tag": tag,
        "status": "ok" if fm else "no_metrics",
        "elapsed": elapsed,
        "run_id": result.run_id,
    }
    if fm:
        out.update(fm)
    if cleanup and fm:
        # the run dir is disposable scratch once metrics are captured.
        shutil.rmtree(run_dir, ignore_errors=True)
    else:
        out["run_dir"] = str(run_dir)
    return out
