# GONS — Geometric Open-set Null Space

**Official code and supplementary material for the ISPEC 2026 paper**
*"GONS: Self-Calibrated Open-Set Rejection for Few-Shot Class-Incremental
Intrusion Detection"* — Xuan Ha Nguyen, Kim-Hung Le, Nhien-An Le-Khac.
International Conference on Information Security Practice and Experience
(ISPEC 2026). **Accepted.**

Reproduction package for **GONS (Geometric Open-set Null Space)**, a method for
open-set Few-Shot Class-Incremental Learning (FSCIL). GONS is a **non-deep** (no
neural backbone, no `torch`) null-space classifier with a calibrated open-set
reject rule. It runs on **CPU only** in seconds-to-minutes per dataset.

This repo contains the GONS method in isolation so it can be audited end-to-end.
The competing baselines' **code** is not bundled (this package ships GONS only),
but every baseline's **per-dataset result array** is bundled under `results/`
(e.g. `results/oshm_matrix_7ds.csv` carries EWC-NCM, NC-FSCIL, OpenMax, MSP, DOC,
BiC and RFS, each per-dataset tuned), so every number in the paper's tables is
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
| **Smoke test** | `uv run python smoke_test.py` | One fast GONS fit (N-BaIoT, seed 42) end-to-end; asserts **OS-HM 0.6565** to ±0.002 (10-seed N-BaIoT mean 0.7704). ~70 s. |
| **Main result** | `uv run python run_gons_main.py` | GONS FIXED, 7 datasets × 10 seeds, headline **OS-HM** + per-dataset table. |
| **Tuned arm** | `uv run python run_gons_main.py --arm tuned` | Each dataset's own argmax config + bandwidth rule — the other column of the fixed-vs-tuned table. |
| **Ablation** | `uv run python run_gons_ablation.py` | The 4-component ablation on the GONS **FIXED** config (refresh / min-distance scoring / capacity-512 / low-support tie-break). |

All of them score through the **single official scorer**
`gons.evaluation.aggregate_openset.full_metrics_from_run` — there is no second
metric implementation in this repo (a parity test in `tests/` asserts this).

### Headline numbers (what to expect)

**Main result** — GONS FIXED config (one global config; the kernel bandwidth is the
uniform kNN-20 rule, evaluated on each dataset's own base-session training split),
mean OS-HM over the open-set sessions, averaged across 10 seeds (backing CSV:
`results/oshm_matrix_7ds.csv`):

| Dataset                 | OS-HM |
|-------------------------|------:|
| 5G-NIDD                 | 0.8494 |
| ToN-IoT                 | 0.8025 |
| N-BaIoT                 | 0.7704 |
| CICIDS2018              | 0.7102 |
| Edge-IIoTset            | 0.6170 |
| CICIoT2023 (7-category) | 0.6072 |
| NSL-KDD                 | 0.6005 |
| **Mean**                | **0.7082** |

Against the strongest per-dataset-tuned baseline (EWC-NCM, 0.5127) that is
**+0.1955**, winning **49 of 49** method-dataset cells, **44** surviving
Holm correction (`results/significance_all_baselines_7ds.csv`).

**Fixed vs tuned** — what shipping ONE global config costs, against giving GONS the
same per-dataset tuning every baseline gets (backing CSV:
`results/headline_fixed_vs_tuned_7ds.csv`; reproduce with `run_gons_main.py` and
`run_gons_main.py --arm tuned`):

| | Fixed (shipped) | Tuned per dataset | Gap |
|---|---:|---:|---:|
| 7-dataset mean OS-HM | **0.7082** | 0.7312 | +0.0230 |

The headline uses the **fixed** column throughout. Note that the tuned arm selects
the kernel bandwidth *rule* as well as the grid cell, so its configs are not simply
the fixed config with different flags (§5).

**Ablation** — OS-HM when each GONS component is removed, seven datasets, on the
**FIXED** configuration (so the "Complete method" row is the headline row above)
(backing CSV: `results/component_ablation_7ds.csv`):

| Variant                            | OS-HM | Δ vs complete |
|------------------------------------|------:|--------------:|
| Complete method                    | 0.7082 |  —      |
| Without low-support tie-break      | 0.7094 | +0.0012 |
| Feature map capacity 128           | 0.6678 | −0.0404 |
| Without prototype refresh          | 0.6171 | −0.0911 |
| Energy score instead of distance   | 0.5120 | −0.1962 |

The low-support tie-break is **inert** on this benchmark (+0.0012, median cell
difference +5e-5).

### Runtime

CPU-only. Per (dataset, seed) GONS fit + eval: roughly **17 s** (small datasets
like N-BaIoT / NSL-KDD) to **~292 s** (ToN-IoT, the largest split). The smoke test
is a single small fit (~70 s on N-BaIoT). The full main result is 7 × 10 = 70 fits;
the ablation is 7 datasets × 5 arms × 10 seeds = 350 fits. Both runners are
**resumable** (they skip cells already recorded in their output `.jsonl`).

With only the two bundled datasets, that is 20 fits and 100 fits respectively.

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

The **full seven-dataset** benchmark additionally needs ToN-IoT, CICIDS2018,
5G-NIDD, Edge-IIoTset and CICIoT2023 (7-category) splits. These are larger and not
bundled; see [`data/README.md`](data/README.md) for the public sources and the
per-class caps, then run the turnkey preparer **`prepare_dataset.py`** (per-class
subsampling, label normalization, 60/20/20 train/cal/test split — the exact recipe
behind the bundled splits) and re-run.

> `prepare_dataset.py --name` is the output **directory**, which for most datasets
> is not the dataset key (key `cicids2018` → directory `cicids2018_5k`). The exact
> key→directory mapping is tabulated in [`data/README.md`](data/README.md); a
> mismatch makes the runner skip the dataset as "not found".

> **Note on exact reproduction.** Re-prepared splits may differ *slightly* from the
> bundled ones: the public datasets are periodically re-released and several are
> available from more than one source/version, so the raw row set can differ. The
> preparation itself is fully deterministic (fixed seed 42), so numbers reproduce up
> to this data-source variation — expect small per-cell deltas, not different
> conclusions.

If you only have the two bundled datasets, all runners still execute and score
those two — the other five rows are skipped with a clear message.

---

## 4. Table / figure → artifact map

Every number in the paper is backed by a shipped CSV under `results/`:

| Paper element | Backing artifact |
|---|---|
| Headline OS-HM, fixed vs tuned | `headline_fixed_vs_tuned_7ds.csv` |
| GONS against every tuned baseline | `oshm_matrix_7ds.csv` |
| Significance (paired-t, Wilcoxon, Holm) | `significance_all_baselines_7ds.csv` |
| Per-dataset, per-method with SD | `per_dataset_all_methods_7ds.csv` |
| Component ablation | `component_ablation_7ds.csv` |
| Projection ablation (null-space vs PCA vs random) | `projection_ablation_7ds.csv` |
| Forgetting / BWT | `forgetting_bwt_summary_7ds.csv`, `forgetting_per_session_7ds.csv` |
| Per-session CCR / TUR trajectory | `per_session_metrics_7ds.csv` |
| Shot count + class-order robustness | `k_sensitivity_7ds.csv`, `class_order_sensitivity_7ds.csv` |
| Kernel bandwidth rule selection | `bandwidth_rule.csv` |
| Where the fixed cell ranks in the grid | `config_selection.csv` |
| Per-seed values, every method | `per_seed_all_methods_7ds.csv` |

The per-seed array (`per_seed_all_methods_7ds.csv`) is the lowest-level artifact:
one row per `(dataset, seed, method)`. The aggregate CSVs above are derived from it.

---

## 5. Run

> All commands use `uv run python …`, which runs inside the environment created by
> `uv sync` (§2). If you installed with `pip install -e .` into an **activated**
> virtualenv instead, drop the `uv run` prefix and just use `python …`.

### Smoke test (do this first)

```bash
uv run python smoke_test.py
```

Fits GONS on N-BaIoT seed 42 end-to-end. The seed-42 single run scores
**OS-HM 0.6565** (CCR 0.6197, TUR 0.7191); the 10-seed N-BaIoT mean is 0.7704. If
this passes, your environment and data wiring are correct. Runtime ~70 s.

The assertion is tight (±0.002), so it is the first thing to run if a
re-implementation disagrees with anything else in this package.

### Main result

```bash
uv run python run_gons_main.py            # all available datasets, seeds 42..51
uv run python run_gons_main.py --datasets nbaiot nsl_kdd --seeds 42 43   # quick subset
```

The exact hyperparameters are defined in `gons_configs.py` (`GONS_FIXED` +
`GONS_TUNED_PER_DS`) and exported as human-readable YAML records by
`export_configs.py`: one `configs/fixed/gons_fixed.yaml` (the config is global, so
there is one record, not one per dataset) and one `configs/tuned/<dataset>.yaml`
per dataset. The FSCIL runner itself is the library module `gons.runners.fscil`
(driven by `run_gons_main.py`).

**The bandwidth rule is part of the tuned selection.** The FIXED config holds it at
the uniform kNN-20 rule on every dataset; the tuned arm selects the rule alongside
the grid cell, and five of the seven winners sit on a different rule (kNN-5, kNN-10,
kNN-50 or the median heuristic). Each `GONS_TUNED_PER_DS` entry therefore carries
its own `gamma_rule`.

### Two seeds, and only one of them is "the seed"

This trips up re-implementations, so it is worth stating plainly:

| Field | Value | What it controls |
|---|---|---|
| `base_class_seed` | **42…51** — the protocol seed | The base/novel class split and which few-shot support rows are drawn. This is what "10 seeds" means in the paper. |
| `model.seed` | **42 on every cell** | The model rng — in practice the RFF draw. It is a **constant**, not the protocol seed. |

Every published run was produced through a CLI whose `--seed` option defaults to
42 and is applied to the model config unconditionally, so `model.seed` was 42 on
all seventy cells while `base_class_seed` swept 42…51. `gons_configs.MODEL_SEED`
pins this.

Letting `model.seed` follow the protocol seed instead is a silent failure: seed 42
still matches exactly (42 == 42), so a single-seed smoke test passes, while seeds
43…51 each redraw the RFF and drift in both directions by up to ~0.03 OS-HM per
cell. `tests/test_gons_release.py` asserts the separation.

The kernel bandwidth is resolved per **(dataset, protocol seed)**, because it is
computed on that seed's base-session training split — so `map_gamma` does vary
across the ten seeds even though `model.seed` does not.

Outputs:
- `artifacts/reports/gons_main.jsonl` — one row per (dataset, seed) with full
  metrics (os_hm, ccr, tur, f1_unknown, macro_f1_with_unknown).
- A printed per-dataset table (mean ± std OS-HM over seeds) at the end. These
  reproduce the per-seed arrays in `results/`.

### Tuned arm (the fixed-vs-tuned comparison)

```bash
uv run python run_gons_main.py --arm tuned
```

The paper's claim is that GONS ships **one** configuration while every baseline is
tuned per dataset, so the honest cost of that constraint is `tuned − fixed`. This
arm measures it. Running both arms reproduces both columns of
`results/headline_fixed_vs_tuned_7ds.csv` (7-dataset means: fixed **0.7082**,
tuned **0.7312**, gap **+0.0230**).

### Ablation

```bash
uv run python run_gons_ablation.py        # 5 arms × all available datasets × seeds 42..51 (FIXED base)
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
GONS/
├── README.md
├── LICENSE                       # MIT
├── CITATION.cff                  # machine-readable citation metadata
├── paper/
│   └── supplementary.pdf         # FULL PROOFS — Prop 1, Thm 1–6, Lemmas 1–4, null-space conditioning
├── pyproject.toml                # non-deep deps only (no torch)
├── requirements.txt
├── .gitattributes                # binary handling for .pdf/.png (prevents corruption on commit)
├── smoke_test.py                 # fast single-fit sanity check
├── run_gons_main.py              # 7×10 main-result runner (--arm fixed | tuned)
├── run_gons_ablation.py          # 4-component ablation runner (FIXED base)
├── prepare_dataset.py            # turnkey raw-CSV → train/cal/test split preparer
├── gons_configs.py               # GONS FIXED + per-dataset TUNED config builders
├── gons_run.py                   # shared run helper (single official scorer)
├── export_configs.py             # regenerates the configs/ YAML records from gons_configs.py
├── configs/
│   ├── data/                     # dataset YAML configs the runner reads
│   ├── fixed/                    # exported FIXED headline config (one global record)
│   └── tuned/                    # exported per-dataset TUNED config, all 7 datasets
├── data/
│   ├── README.md                 # sources + prep for the full 7 datasets
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

### Tests

```bash
uv run pytest -q
```

Config invariants (the fixed cell, the tuned coverage, the seed separation, the
bandwidth-rule anchor), scorer properties, a single-scorer parity check, and an
end-to-end parity run against the N-BaIoT seed-42 reference cell.

### Notes on retained modules

- **`localization/` is retained** — it provides the mutual-rNN support store used
  by the GONS support/refresh path.
- **`baselines/` is an inert placeholder.** The competing methods' code is not
  shipped (GONS is the only method here); the package retains the baseline config
  *schema* only because `gons.evaluation.config` imports its dataclasses. Any
  baseline code path raises rather than running silently. The baselines' result
  arrays are bundled as CSVs (see §4) so their numbers remain checkable.
```

---

## 9. Citation

```bibtex
@inproceedings{nguyen2026gons,
  title     = {{GONS}: Self-Calibrated Open-Set Rejection for Few-Shot
               Class-Incremental Intrusion Detection},
  author    = {Nguyen, Xuan Ha and Le, Kim-Hung and Le-Khac, Nhien-An},
  booktitle = {Information Security Practice and Experience (ISPEC 2026)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff). Page numbers, DOI
and volume are added here once the proceedings are published.

## 10. License and contact

Code and result arrays: MIT (see [`LICENSE`](LICENSE)). `paper/supplementary.pdf`
is the authors' own manuscript, distributed here for reference alongside the code.

The bundled data are redistributions of public research datasets under their
original terms; cite the original dataset papers if you use them (sources in
[`data/README.md`](data/README.md)).

Questions and issues: open a GitHub issue, or contact
Xuan Ha Nguyen (`xuan.h.nguyen@ucdconnect.ie`).
