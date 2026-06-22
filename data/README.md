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

### Preparation recipe

For each dataset we produce a balanced, category-level processed subset and split
it per class into train / calibration / test:

1. **Label** at the category level (benign + attack categories); normalize label
   strings to lowercase.
2. **Subsample** per class (the bundled subsets use ~5k rows/class cap for the
   IDS datasets; ToN-IoT uses the 10k-per-class balanced subset).
3. **Split** per class: train / calibration / test (the runner reads three CSVs).
4. Write `train.csv`, `calibration.csv`, `test.csv` under
   `data/processed/<name>/` matching the paths in `gons_configs.py` (`DATASETS`).

Then run `python run_gons_main.py` — datasets whose `train.csv` is present are
included automatically; missing ones are skipped with a message.

> The exact per-class caps, seeds, and label maps used for the paper's numbers
> are documented in the paper's experimental appendix. The bundled `nbaiot_5k`
> and `nsl_kdd_ta` are provided so the pipeline is runnable end-to-end without
> any external download.
