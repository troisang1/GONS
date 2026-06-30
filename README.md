# GONS — Geometric Open-set Null Space

Reproduction package for **GONS (Geometric Open-set Null Space)**, a method for
open-set Few-Shot Class-Incremental Learning (FSCIL). GONS is a **non-deep** (no
neural backbone, no `torch`) null-space classifier with a calibrated open-set
reject rule. It runs on **CPU only** in seconds-to-minutes per dataset.

This repo contains the GONS method in isolation so it can be audited end-to-end.
The competing baselines' **code** is not bundled (this package ships GONS only),
but every baseline's **per-dataset result array** is bundled under `results/`
(e.g. `results_oshm_matrix_5ds.csv` carries EWC-NCM, NC-FSCIL, OpenMax, MSP, DOC,
BiC, RFS, and IKNDA, tuned and default), so every number in the paper's tables is
checkable against a shipped CSV.

> ### 📄 Full proofs → `paper/supplementary.pdf`
> **All theory is proved in the supplementary material.** `paper/supplementary.pdf`
> gives the complete assumptions, statements, and **full proofs** of every result in
> the paper: **Proposition 1, Theorems 1–6, Lemmas 1–4**, plus the **null-space
> conditioning analysis**. The main paper states each result's claim (its theory
> section and Table 2); the supplementary carries the proofs in full. **For anything
> theory-related, read the supplementary first** — it is the authoritative reference
> for the method's guarantees (calibrated open-set admission/rejection, null-space
> concentration, and the finite-sample threshold bound).

---

## 1. What you can reproduce

| Artifact | Script | What it produces |
|---|---|---|
| **Smoke test** | `uv run python smoke_test.py` | One fast GONS fit (N-BaIoT, seed 42) end-to-end; asserts a valid OS-HM (seed-42 single-run = **0.7029**; the 10-seed N-BaIoT mean is 0.770). |
| **Main result** | `uv run python run_gons_main.py` | GONS, 5 datasets × 10 seeds, headline **OS-HM** + per-dataset table. |
| **Ablation** | `uv run python run_gons_ablation.py` | The 4-component ablation on the GONS tuned config (refresh / min-distance scoring / capacity-512 / quantile). |

All three score through the **single official scorer**
`gons.evaluation.aggregate_openset.full_metrics_from_run` — there is no second
metric implementation in this repo (a parity test in `tests/` asserts this).

### Headline numbers (what to expect)

**Main result** — GONS FIXED config (one global config), mean OS-HM over the
open-set sessions, averaged across 10 seeds (backing CSV:
`results_oshm_matrix_5ds.csv`, `full_metric_5ds_mean.csv`):

| Dataset    | OS-HM |
|------------|------:|
| ToN-IoT    | 0.800 |
| N-BaIoT    | 0.770 |
| CICIDS2018 | 0.732 |
| 5G-NIDD    | 0.821 |
| NSL-KDD    | 0.643 |
| **Mean**   | **0.7532** |

**Ablation** — OS-HM drop when each GONS component is removed (positive = the
component helps; backing CSV: `ablation_per_ds.csv`):

| Removed component                  | Δ OS-HM |
|------------------------------------|--------:|
| consolidated refresh               | +0.067  |
| min-distance scoring (→ energy)    | +0.146  |
| RFF capacity 512 (→ 128)           | +0.053  |
| quantile (NCDR) classification     | +0.0003 (INERT) |

The quantile/NCDR arm is reported as **inert** — it is part of the model but does
not move the headline metric on this benchmark.

### Runtime

CPU-only. Per (dataset, seed) GONS fit + eval: roughly **17 s** (small datasets
like N-BaIoT / NSL-KDD) to **~292 s** (ToN-IoT, the largest split). The smoke test
is a single small fit (~75 s on N-BaIoT). The full main result is 5 × 10 = 50 fits;
the ablation is 5 × 5 arms × 10 seeds = 250 fits. Both runners are **resumable**
(they skip cells already recorded in their output `.jsonl`).

---

## 2. Install

Requires Python 3.11 or 3.12. **No GPU, no `torch`.**

### With `uv` (recommended)

```bash
uv sync                 # or: uv venv --python 3.12 && uv pip install -e .
```

### With `pip`

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
# or: pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scikit-learn`, `scipy`, `pydantic`, `pyyaml`,
`rich`. That is the whole stack.

---

## 3. Data setup

Two small, public processed datasets are bundled under `data/processed/` so the
smoke test and a runnable demo work out of the box:

- `data/processed/nbaiot_5k/` — N-BaIoT (5k subsampled), used by the smoke test.
- `data/processed/nsl_kdd_ta/` — NSL-KDD (processed), a second runnable demo.

Each contains `train.csv`, `calibration.csv`, `test.csv` with a `label` column.

The **full 5-dataset** main result additionally needs ToN-IoT, CICIDS2018, and
5G-NIDD splits. These are larger and not bundled; see
[`data/README.md`](data/README.md) for the public sources, then run the turnkey
preparer **`prepare_dataset.py`** (per-class subsampling, label normalization,
60/20/20 train/cal/test split — the exact recipe behind the bundled splits) to
write `data/processed/<name>/` matching `gons_configs.py` (`DATASETS`), and re-run.

> **Note on exact reproduction.** Re-prepared splits may differ *slightly* from the
> bundled ones: the public datasets are periodically re-released and several are
> available from more than one source/version, so the raw row set can differ. The
> preparation itself is fully deterministic (fixed seed 42), so numbers reproduce up
> to this data-source variation — expect small per-cell deltas, not different
> conclusions.

If you only have the two bundled datasets, both runners will still execute and
score those two — the other three rows will simply be skipped with a clear
message.

---

## 4. Table / figure → artifact map

Every number in the paper is backed by a shipped CSV under `results/`:

| Paper element | Backing artifact |
|---|---|
| Headline OS-HM; CCR/TUR breakdown (Tables 5, 6) | `results_oshm_matrix_5ds.csv`, `full_metric_5ds_mean.csv`, `per_dataset_full_metrics.csv` |
| Significance (Δ, paired-t, Wilcoxon) | `significance_vs_ewc_ncm_tuned.csv` |
| Component ablation (Table 7) | `ablation_per_ds.csv` |
| Forgetting / BWT (Table 10) | `forgetting_bwt_summary.csv`, `forgetting_per_session.csv` |
| Projection ablation +41.3/+36.0 (Table 11) | `projection_ablation.csv` |
| K + class-order robustness (Table 12) | `k_sensitivity.csv`, `class_order_sensitivity.csv` |
| Per-seed robustness (Fig 4) | `results_ours_per_seed.csv` |
| Cross-session slack η_t + γ_min stress (Fig 5) | `eta_drift.csv`, `gamma_min_stress.csv` |
| Reject ROC / AUROC 0.778–0.941 (Fig 3) | `roc_summary.csv` |
| Margin-feasibility audit (0 of 1,230) | `projected_unknown_variance.csv` |

The per-seed array (`results_ours_per_seed.csv`) is the lowest-level artifact:
one row per `(dataset, config, seed)` with `os_hm`/`ccr`/`tur`/`f1_unknown`/
`macro_f1_with_unknown`. The aggregate CSVs above are derived from it.

---

## 5. Run

> All commands use `uv run python …`, which runs inside the environment created by
> `uv sync` (§2). If you installed with `pip install -e .` into an **activated**
> virtualenv instead, drop the `uv run` prefix and just use `python …`.

### Smoke test (do this first)

```bash
uv run python smoke_test.py
```

Fits GONS on N-BaIoT seed 42 end-to-end and asserts a valid OS-HM. The seed-42
single run scores **OS-HM 0.7029** (CCR 0.7152, TUR 0.7127) — bit-exact with the
reference value for that cell; the headline 0.770 is the 10-seed mean
(per-seed range ~0.69–0.84). If this passes, your environment + data are wired
correctly. Runtime ~75 s on N-BaIoT (the largest bundled split).

### Main result

```bash
uv run python run_gons_main.py            # all available datasets, seeds 42..51
uv run python run_gons_main.py --datasets nbaiot nsl_kdd --seeds 42 43   # quick subset
```

The exact per-dataset hyperparameters (the single **fixed** headline config and
the per-dataset **tuned** configs) are defined in `gons_configs.py`
(`GONS_FIXED` + `GONS_TUNED_PER_DS`) and exported as human-readable YAML records
under `configs/fixed/<dataset>.yaml` and `configs/tuned/<dataset>.yaml`. The
FSCIL runner itself is the library module `gons.runners.fscil` (driven by
`run_gons_main.py`).

Outputs:
- `artifacts/reports/gons_main.jsonl` — one row per (dataset, seed) with full
  metrics (os_hm, ccr, tur, f1_unknown, macro_f1_with_unknown).
- A printed per-dataset table (mean ± std OS-HM over seeds) at the end. These
  reproduce the per-seed arrays in `results/`.

### Ablation

```bash
uv run python run_gons_ablation.py        # 5 arms × all available datasets × seeds 42..51
```

Outputs:
- `artifacts/reports/gons_ablation.jsonl` — one row per (dataset, arm, seed).
- A printed per-dataset ablation table (Δ OS-HM of each arm vs the FULL config).

---

## 6. How to read the outputs

Each `.jsonl` row carries the headline `os_hm` plus the companions `ccr`, `tur`,
`f1_unknown`, `macro_f1_with_unknown`, the `dataset`/`seed` (and `arm` for the
ablation), and the elapsed wall time. The per-run OS-HM is the **mean over the
open-set sessions** of that run (the final all-admitted session has no
true-unknowns and is excluded by the scorer). The printed summary tables
aggregate across seeds (mean ± std) and, for the ablation, report the signed Δ
vs the FULL config so positive means "this component helps".

---

## 7. What GONS is (one paragraph)

GONS fits class-conditional null-space prototypes over an explicit Random Fourier
Feature (RFF) map of the input, calibrates a per-session rejection threshold, and
classifies a query by minimum prototype distance — rejecting to `unknown` when the
score exceeds the calibrated threshold. New classes are admitted few-shot and the
prototype bank is consolidated by a refresh step between sessions. There is no
deep network and no gradient training; everything is closed-form linear algebra
on CPU.

---

## 8. Repository layout

```
gons-release/
├── README.md
├── LICENSE                       # MIT (anonymous: "GONS authors")
├── paper/
│   └── supplementary.pdf         # FULL PROOFS — Prop 1, Thm 1–6, Lemmas 1–4, null-space conditioning
├── pyproject.toml                # non-deep deps only (no torch)
├── requirements.txt
├── .gitattributes                # binary handling for .pdf/.png (prevents corruption on commit)
├── smoke_test.py                 # fast single-fit sanity check
├── run_gons_main.py              # 5×10 main-result runner
├── run_gons_ablation.py          # 4-component ablation runner
├── prepare_dataset.py            # turnkey raw-CSV → train/cal/test split preparer
├── gons_configs.py               # GONS FIXED + per-dataset TUNED config builders
├── gons_run.py                   # shared run helper (single official scorer)
├── configs/
│   ├── data/                     # dataset YAML configs the runner reads
│   ├── fixed/                    # exported FIXED headline config, per dataset
│   └── tuned/                    # exported per-dataset TUNED config (ablation base)
├── data/
│   ├── README.md                 # sources + prep for the full 5 datasets
│   └── processed/                # bundled small datasets (nbaiot_5k, nsl_kdd_ta)
├── results/                      # per-seed + per-table result arrays (§4)
├── src/gons/                     # the GONS package
│   ├── runners/fscil.py          # the FSCIL runner
│   ├── fit.py                    # GONS fit
│   ├── projection/               # null-space projection core
│   ├── maps/                     # RFF map
│   ├── calibration/              # rejection-threshold calibration
│   ├── prototype/                # prototype bank
│   ├── continual/                # session inference + refresh
│   ├── localization/             # mutual-rNN support store
│   ├── evaluation/               # open-set eval + the official OS-HM scorer
│   └── ...                       # state, preprocess, data loaders, config, runtime
└── tests/                        # parity / property unit tests
```

The python package is named `gons`.

### Notes on retained modules

- **`localization/` is retained** — it provides the mutual-rNN support store used
  by the GONS support/refresh path.
- **`baselines/` is an inert placeholder.** The competing methods' code is not
  shipped (GONS is the only method here); the package retains the baseline config
  *schema* only because `gons.evaluation.config` imports its dataclasses. Any
  baseline code path raises rather than running silently. The baselines' result
  arrays are bundled as CSVs (see §4) so their numbers remain checkable.
```
