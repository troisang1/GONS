# GONS — Geometric Open-set Null Space

Reproduction package for **GONS (Geometric Open-set Null Space)**, a method for
open-set Few-Shot Class-Incremental Learning (FSCIL). GONS is a **non-deep** (no
neural backbone, no `torch`) null-space classifier with a calibrated open-set
reject rule. It runs on **CPU only** in seconds-to-minutes per dataset.

This repo contains the GONS method in isolation so it can be audited end-to-end.
The competing baselines (EWC-NCM, NC-FSCIL, OpenMax, MSP, DOC, BiC, RFS) are
evaluated elsewhere and are not bundled here.

---

## 1. What you can reproduce

| Artifact | Script | What it produces |
|---|---|---|
| **Smoke test** | `python smoke_test.py` | One fast GONS fit (N-BaIoT, seed 42) end-to-end; asserts a valid OS-HM (seed-42 single-run = **0.7029**; the 10-seed N-BaIoT mean is 0.770). |
| **Main result** | `python run_gons_main.py` | GONS gate-OFF, 5 datasets × 10 seeds, headline **OS-HM** + per-dataset table. |
| **Ablation** | `python run_gons_ablation.py` | The 4-component ablation on the GONS tuned config (refresh / min-distance scoring / capacity-512 / quantile). |

All three score through the **single official scorer**
`bcmrnfst.evaluation.aggregate_openset.open_set_session_metrics` — there is no
second metric implementation in this repo.

### Headline numbers (what to expect)

**Main result** — GONS FIXED config (one global config, gate-OFF), mean OS-HM over
the open-set sessions, averaged across 10 seeds:

| Dataset    | OS-HM |
|------------|------:|
| ToN-IoT    | 0.800 |
| N-BaIoT    | 0.770 |
| CICIDS2018 | 0.732 |
| 5G-NIDD    | 0.821 |
| NSL-KDD    | 0.643 |
| **Mean**   | **0.7532** |

**Ablation** — OS-HM drop when each GONS component is removed (positive = the
component helps):

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
is a single small fit (~15-30 s). The full main result is 5 × 10 = 50 fits; the
ablation is 5 × 5 arms × 10 seeds = 250 fits. Both runners are **resumable** (they
skip cells already recorded in their output `.jsonl`).

---

## 2. Install

Requires Python 3.11 or 3.12. **No GPU, no `torch`.**

### With `uv` (recommended)

```bash
uv venv --python 3.12
uv pip install -e .
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
[`data/README.md`](data/README.md) for the public sources and the exact
preparation recipe (per-class subsampling, label normalization, train/cal/test
split). After preparing them, place them under the `data/processed/<name>/`
paths referenced in `run_gons_main.py` (`DATASETS` dict) and re-run.

If you only have the two bundled datasets, both runners will still execute and
score those two — the other three rows will simply be skipped with a clear
message.

---

## 4. Run

### Smoke test (do this first)

```bash
python smoke_test.py
```

Fits GONS on N-BaIoT seed 42 end-to-end and asserts a valid OS-HM. The seed-42
single run scores **OS-HM 0.7029** (CCR 0.7152, TUR 0.7127) — bit-exact with the
reference value for that cell; the headline 0.770 is the 10-seed mean
(per-seed range ~0.69-0.84). If this passes, your environment + data are wired
correctly. Runtime ~75 s on N-BaIoT (the largest bundled split).

### Main result

```bash
python run_gons_main.py            # all available datasets, seeds 42..51
python run_gons_main.py --datasets nbaiot nsl_kdd --seeds 42 43   # quick subset
```

Outputs:
- `artifacts/reports/gons_main.jsonl` — one row per (dataset, seed) with full
  metrics (os_hm, ccr, tur, f1_unknown, macro_f1_with_unknown).
- A printed per-dataset table (mean ± std OS-HM over seeds) at the end.

### Ablation

```bash
python run_gons_ablation.py        # 5 arms × all available datasets × seeds 42..51
```

Outputs:
- `artifacts/reports/gons_ablation.jsonl` — one row per (dataset, arm, seed).
- A printed per-dataset ablation table (Δ OS-HM of each arm vs the FULL config).

---

## 5. How to read the outputs

Each `.jsonl` row carries the headline `os_hm` plus the companions `ccr`, `tur`,
`f1_unknown`, `macro_f1_with_unknown`, the `dataset`/`seed` (and `arm` for the
ablation), and the elapsed wall time. The per-run OS-HM is the **mean over the
open-set sessions** of that run (the final all-admitted session has no
true-unknowns and is excluded by the scorer). The printed summary tables
aggregate across seeds (mean ± std) and, for the ablation, report the signed Δ
vs the FULL config so positive means "this component helps".

`gate_active` is recorded on every row and must be `false` for every GONS run —
the optional gate mechanism is disabled in GONS.

---

## 6. What GONS is (one paragraph)

GONS fits class-conditional null-space prototypes over an explicit Random Fourier
Feature (RFF) map of the input, calibrates a per-session rejection threshold, and
classifies a query by minimum prototype distance — rejecting to `unknown` when the
score exceeds the calibrated threshold. New classes are admitted few-shot and the
prototype bank is consolidated by a refresh step between sessions. There is no
deep network and no gradient training; everything is closed-form linear algebra
on CPU. The optional gate mechanism referred to throughout is an extra whitening
stage that is **off** in GONS.

---

## 7. Repository layout

```
gons-release/
├── README.md
├── LICENSE                       # MIT
├── pyproject.toml                # non-deep deps only (no torch)
├── requirements.txt
├── smoke_test.py                 # fast single-fit sanity check
├── run_gons_main.py             # 5×10 main-result runner (GONS only)
├── run_gons_ablation.py         # 4-component ablation runner (GONS only)
├── configs/data/                 # dataset YAML configs the runner reads
├── data/
│   ├── README.md                 # sources + prep for the full 5 datasets
│   └── processed/                # bundled small datasets (nbaiot_5k, nsl_kdd_ta)
├── src/bcmrnfst/                 # the GONS package (import closure of the gate-OFF path)
│   ├── runners/track_a_fscil.py  # the FSCIL runner
│   ├── fit.py                    # GONS fit (gate-OFF)
│   ├── projection/               # null-space projection core
│   ├── maps/explicit.py          # RFF map
│   ├── calibration/              # rejection-threshold calibration
│   ├── prototype/                # prototype bank
│   ├── continual/                # session inference + refresh
│   ├── evaluation/               # open-set eval + the official OS-HM scorer
│   └── ...                       # state, preprocess, data loaders, config, runtime
└── tests/                        # parity / property unit tests
```

The python package is named `bcmrnfst`.

### A note on the (inactive) gate code

`fit.py`, `continual/refresh.py`, `continual/inference.py`, and
`projection/dvmad_*.py` still contain the optional whitening-gate branches
(`enable_dvmad_gate` / `enable_ewm` / `enable_nsd_gate`). This is the optional
gate mechanism and it is **never executed in this release**: every GONS config
sets all three flags to `False` (the gate-OFF invariant, asserted in `tests/` and
guarded at runtime in `gons_run.py`). The branches are retained only because
`fit.py` hard-imports two symbols from `projection/dvmad_gate.py` at module load
— they are not cleanly removable without rewriting the fit pipeline, and removing
them would risk perturbing the numbers being reproduced. The entrypoints used to
*turn the gate on* are not shipped.
