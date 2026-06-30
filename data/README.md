# Data — GONS reproduction

## Bundled (ready to run)

| Dir | Dataset | Splits | Used by |
|---|---|---|---|
| `processed/nbaiot_5k/` | N-BaIoT (5k subsampled) | train / calibration / test | smoke test + main + ablation |
| `processed/nsl_kdd_ta/` | NSL-KDD (processed) | train / calibration / test | main + ablation (demo) |

Each split is a CSV with a `label` column (the multiclass attack/category label)
and numeric feature columns. The FSCIL protocol splits classes into a base set
and an ordered sequence of novel classes; un-admitted novel classes act as
true-Unknown at each session.

## Full 5-dataset main result (prepare yourself)

The headline main result uses 5 datasets. Three are not bundled because their
processed splits are larger:

| Name | Source (public) |
|---|---|
| ToN-IoT (`toniot`) | UNSW ToN-IoT — https://research.unsw.edu.au/projects/toniot-datasets |
| CICIDS2018 (`cicids2018`) | CSE-CIC-IDS2018 — https://www.unb.ca/cic/datasets/ids-2018.html |
| 5G-NIDD (`5g_nidd`) | 5G-NIDD — https://ieee-dataport.org/documents/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless |

### Preparation recipe — `prepare_dataset.py` (turnkey)

The bundled splits were produced by a single deterministic recipe, shipped as
`prepare_dataset.py` at the repo root. Point it at a raw labeled CSV (one label
column + numeric feature columns) and it writes the exact `train.csv` /
`calibration.csv` / `test.csv` the runner reads:

```bash
# ToN-IoT: 10000/class cap
uv run python prepare_dataset.py --input raw/toniot.csv  --name toniot     --cap 10000
# Other IDS datasets: 5000/class cap
uv run python prepare_dataset.py --input raw/cicids.csv  --name cicids2018 --cap 5000
uv run python prepare_dataset.py --input raw/5gnidd.csv  --name 5g_nidd    --cap 5000
```

What the recipe does (deterministic, seed 42): normalize labels to lowercase
(`-`/space → `_`); per class take `min(cap, available)` rows (sampled without
replacement, then shuffled); split each class **60 / 20 / 20** into
train / calibration / test. The output `--name` must match the `data/processed/<name>/`
path in `gons_configs.py` (`DATASETS`).

Then run `uv run python run_gons_main.py` — datasets whose `train.csv` is present
are included automatically; missing ones are skipped with a message.

> **Note on exact reproduction.** Re-prepared splits may differ *slightly* from the
> bundled ones. The public datasets are periodically re-released and several are
> available from more than one source/version, so the raw row set you download can
> differ from the one used here. Preparation is otherwise fully deterministic (fixed
> seed 42, fixed 60/20/20 split), so results reproduce up to this data-source
> variation — expect small per-cell deltas, not different conclusions.

> The exact per-class caps, seeds, and label maps used for the paper's numbers
> are documented in the paper's experimental appendix. The bundled `nbaiot_5k`
> and `nsl_kdd_ta` are provided so the pipeline is runnable end-to-end without
> any external download.
