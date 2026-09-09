# Data — GONS reproduction

## Bundled (ready to run)

| Dir | Dataset | Splits | Used by |
|---|---|---|---|
| `processed/nbaiot_5k/` | N-BaIoT (5k subsampled) | train / calibration / test | smoke test + main + ablation |
| `processed/nsl_kdd_ta/` | NSL-KDD (processed) | train / calibration / test | main + ablation (demo) |

The other five datasets of the benchmark are prepared below. Both runners skip a
dataset whose `train.csv` is absent, so the repo is runnable end-to-end on these
two alone.

Each split is a CSV with a `label` column (the multiclass attack/category label)
and numeric feature columns. The FSCIL protocol splits classes into a base set
and an ordered sequence of novel classes; un-admitted novel classes act as
true-Unknown at each session.

## Full 7-dataset benchmark (prepare yourself)

The reported benchmark is 7 datasets. Five are not bundled because their
processed splits are larger:

| Name | Per-class cap | Source (public) |
|---|---|---|
| ToN-IoT (`toniot`) | 10000 | UNSW ToN-IoT — https://research.unsw.edu.au/projects/toniot-datasets |
| CICIDS2018 (`cicids2018`) | 5000 | CSE-CIC-IDS2018 — https://www.unb.ca/cic/datasets/ids-2018.html |
| 5G-NIDD (`5g_nidd`) | 5000 | 5G-NIDD — https://ieee-dataport.org/documents/5g-nidd-comprehensive-network-intrusion-detection-dataset-generated-over-5g-wireless |
| Edge-IIoTset (`edge_iiotset`) | 5000 | Edge-IIoTset — https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot |
| CICIoT2023, 7-category (`ciciot2023_cat`) | 10000 | CICIoT2023 — https://www.unb.ca/cic/datasets/iotdataset-2023.html |

**Edge-IIoTset uses the ports-retaining variant.** Port numbers are service
identifiers, not host identity, so they are kept; every identity-bearing column
is dropped (`ip.src_host`, `ip.dst_host`, `arp.*.proto_ipv4`, URIs, payloads,
timestamps). Retaining ports while dropping host identity is what the reported
numbers were measured on — a variant that also drops ports, or one that keeps
host identity, will not reproduce them.

**CICIoT2023 is the 7-category granularity**, not the 34-label fine granularity.

### Preparation recipe — `prepare_dataset.py` (turnkey)

The bundled splits were produced by a single deterministic recipe, shipped as
`prepare_dataset.py` at the repo root. Point it at a raw labeled CSV (one label
column + numeric feature columns) and it writes the exact `train.csv` /
`calibration.csv` / `test.csv` the runner reads:

```bash
# 10000/class cap
uv run python prepare_dataset.py --input raw/toniot.csv    --name toniot                  --cap 10000
uv run python prepare_dataset.py --input raw/ciciot23.csv  --name ciciot2023_category_10k --cap 10000
# 5000/class cap
uv run python prepare_dataset.py --input raw/cicids.csv    --name cicids2018_5k           --cap 5000
uv run python prepare_dataset.py --input raw/5gnidd.csv    --name 5g_nidd_5k              --cap 5000
uv run python prepare_dataset.py --input raw/edge_iiot.csv --name edge_iiotset_5k_ports   --cap 5000
```

> **`--name` is the output DIRECTORY, not the dataset key.** It writes
> `data/processed/<name>/`, and that path must be exactly what the dataset's entry
> in `gons_configs.py` (`DATASETS`) points at — for most datasets the two differ
> (key `cicids2018` → dir `cicids2018_5k`). Get it wrong and the runner reports the
> dataset as "not found" and silently skips it. The mapping:
>
> | `DATASETS` key | `--name` (directory) |
> |---|---|
> | `toniot` | `toniot` |
> | `nbaiot` | `nbaiot_5k` *(bundled)* |
> | `cicids2018` | `cicids2018_5k` |
> | `5g_nidd` | `5g_nidd_5k` |
> | `nsl_kdd` | `nsl_kdd_ta` *(bundled)* |
> | `edge_iiotset` | `edge_iiotset_5k_ports` |
> | `ciciot2023_cat` | `ciciot2023_category_10k` |

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
