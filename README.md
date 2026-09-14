# EV IDM Battery Digital Twin

> Research repository accompanying the paper **“Battery-Lifespan-Aware Calibration of the Intelligent Driver Model for Electric Vehicles: A Closed-Loop Digital-Twin Study.”**


A Python pipeline that calibrates the **Intelligent Driver Model (IDM)** for
battery-electric vehicles and integrates a **semi-empirical battery
degradation model** into every stage of an EV digital twin. The pipeline
runs end-to-end from raw trajectory CSVs to optimised driver parameters,
sensitivity analysis, and external validation on the Vehicle Energy Dataset
(VED).

## Relationship to the Paper

This repository contains the computational implementation underlying the study. The code is used for
data preprocessing, IDM calibration, EV digital-twin simulation, multi-objective optimisation,
closed-loop driver-profile experiments, sensitivity analysis, VED comparison, and figure generation.
The manuscript reports results generated from the files stored under `results/`; these JSON outputs are
the primary traceability layer between reported numbers and executable code.

For publication, the manuscript should point to a fixed Git tag or commit corresponding to the exact
code/results state used to generate the submitted figures and tables.

---

## What This Repository Contains

This repository contains **only the code, the input data placeholders, and
the result JSON files** needed to run the pipeline. It does **not** contain
the manuscript, the figures, or any LaTeX artefacts — those live in a
separate bundle. extra raw data used are in "extra data"

### Code Modules

| Folder | Purpose | Key Files |
|--------|---------|-----------|
| `preprocessing/` | Reads raw trajectory CSVs (leader + follower), aligns them on a 10 Hz grid, computes acceleration from the speed signal, and integrates the bumper-to-bumper gap from relative speed. | `preprocess.py` |
| `idm/` | Implements the Intelligent Driver Model (IDM) with Numba-JIT kernels, plus a three-stage calibration pipeline (Differential Evolution + Nelder-Mead + L-BFGS-B). | `idm_model.py`, `jit_kernels.py`, `calibration.py` |
| `digital_twin/` | The EV digital twin: longitudinal vehicle dynamics + pack-level battery electrical model + a Wang et al. (2011) / Schmalstieg et al. (2014) semi-empirical degradation sub-model. | `ev_twin.py`, `battery_degradation.py` |
| `optimization/` | Multi-objective Pareto scan that trades energy consumption against battery capacity fade over a convex weight grid. | `multi_objective_optimization.py` |
| `experiments/` | Closed-loop evaluation of five driver archetypes (Aggressive, Sporty, Average, Cautious, Eco) and external validation on the VED dataset. | `closed_loop.py`, `ved_validation.py` |
| `analysis/` | One-at-a-time (OAT) sensitivity analysis around the joint Pareto optimum, computing per-parameter elasticities. | `sensitivity.py` |
| `figures/` | Generates every figure used in the study (data distributions, IDM fit, Pareto front, closed-loop trajectories, sensitivity, VED validation). | `make_figures.py` |
| `tests/` | Five IDM sanity tests (free-road convergence, equilibrium gap, stopped-leader no-collision, hand-computed acceleration, emergency braking cap). | `test_idm.py` |

### Data Folders

| Path | Description | Format |
|------|-------------|--------|
| `data/raw/LEADCAR/` | Leader-car trajectory CSVs from the one-pedal driving experiment. **Empty by default** — drop the real CSVs here (see "Adding Real Data" below). | CSV: `Record, Time, Latitude, Longitude, Altitude, Speed, GForceX, GForceY, GForceZ, Lap` |
| `data/raw/FLWCAR1/` | Follower-car #1 trajectory CSVs. **Empty by default.** | Same as above |
| `data/raw/FLWCAR2/` | Follower-car #2 trajectory CSVs. **Empty by default.** | Same as above |
| `data/raw/FLWCAR3/` | Follower-car #3 trajectory CSVs. **Empty by default.** | Same as above |
| `data/ved_subset/` | Vehicle Energy Dataset (VED) weekly CSVs. **Empty by default** — download from <https://github.com/gsoh/VED>. | CSV: `DayNum, VehId, Trip, Timestamp(ms), Latitude[deg], Longitude[deg], Vehicle Speed[km/h], HV Battery Current[A], HV Battery SOC[%], HV Battery Voltage[V], ...` |
| `data/processed_segments.csv` | Pre-processed segments used for IDM calibration. Bundled with this repo so the pipeline runs out-of-the-box. | CSV: `segment_id, t, v, dv, gap, a` |
| `data/segment_manifest.json` | Summary statistics and a provenance flag (`"raw:data/raw"` when real data is used). | JSON |

### Result Files

The pipeline persists every numerical result as JSON so each claim is
traceable to a file.

| File | Contents |
|------|----------|
| `results/calibration_results.json` | Calibrated IDM parameters, acceleration RMSE, speed RMSE, Pearson r. |
| `results/pareto_front.json` | Pareto front across five weight ratios `w_B/w_E ∈ {0.1, 0.3, 1.0, 3.0, 10.0}`. |
| `results/closed_loop_results.json` | Per-driver metrics (energy, degradation, regen fraction, max cell current) for the five archetypes. |
| `results/sensitivity_analysis.json` | OAT sensitivity curves and per-parameter elasticities for both objectives. |
| `results/ved_validation.json` | Per-trip predicted vs reference energy and degradation for the 11 VED trips; degradation reference values are synthetic unless measured ageing labels are supplied. |

### Config & Orchestration

| File | Purpose |
|------|---------|
| `run_all.sh` | One-shot bash orchestrator that runs the full pipeline (9 steps). |
| `run_all.py` | Equivalent Python orchestrator for platforms without bash. |
| `requirements.txt` | Python dependencies: `numpy`, `scipy`, `numba`, `matplotlib`, `pandas`. |
| `.gitignore` | Ignores Python caches, LaTeX intermediates, and large data CSVs. |
| `.github/workflows/ci.yml` | GitHub Actions workflow that runs the IDM sanity tests on every push. |
| `LICENSE` | MIT license. |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/sahandd80-stack/EV_IDM_Battery_DT_Project.git
cd EV_IDM_Battery_DT_Project

# 2. Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline end-to-end
bash run_all.sh
# or:  python run_all.py
```

**Expected runtime:** about 9 minutes on a single CPU core. The heaviest
step is the differential-evolution IDM calibration (~4 minutes).

**Expected output:** every JSON file under `results/` is regenerated, the
sanity tests print five `PASS` lines, and the figure PDFs/PNGs are written
to `figures_output/` (which is created on the fly).

---

## What Each Step Does

When you run `bash run_all.sh`, the script executes nine steps in order:

1. **Preprocessing** — `python -m preprocessing.preprocess`
   Reads raw CSVs from `data/raw/` (if present) or falls back to the
   bundled `data/processed_segments.csv`. Writes a fresh
   `data/segment_manifest.json` with summary statistics and a provenance
   flag.

2. **IDM Sanity Tests** — `python tests/test_idm.py`
   Runs five sanity checks on the IDM implementation. All five must pass
   before calibration can be trusted.

3. **IDM Calibration** — `python -m idm.calibration`
   Calibrates the six IDM parameters `(v0, T, s0, a, b, delta)` by
   minimising the mean-squared acceleration error. Uses Differential
   Evolution (seed 42) followed by Nelder-Mead and L-BFGS-B polishes.
   Writes `results/calibration_results.json`.

4. **Multi-Objective Optimisation** — `python -m optimization.multi_objective_optimization`
   Scans the Pareto front across five weight ratios. Writes
   `results/pareto_front.json`.

5. **Closed-Loop Experiments** — `python -m experiments.closed_loop`
   Runs the five driver archetypes (Aggressive, Sporty, Average, Cautious,
   Eco) over a 20-minute mixed highway/urban trace. Writes
   `results/closed_loop_results.json`.

6. **Sensitivity Analysis** — `python -m analysis.sensitivity`
   Sweeps each IDM parameter ±15 % around the joint optimum and records
   the induced change in energy and degradation. Writes
   `results/sensitivity_analysis.json`.

7. **External VED Validation** — `python -m experiments.ved_validation`
   Loads the Kia Soul EV trips from `data/ved_subset/` (if present),
   runs the twin on each trip, and reports Pearson r, MAPE, and bias.
   Writes `results/ved_validation.json`.

8. **Figures** — `python -m figures.make_figures`
   Generates 10 PDF + 10 PNG figures in `figures_output/`.

9. **Manuscript Compilation** (optional) — only runs if `tectonic` or
   `pdflatex` is installed on the system. Skipped silently otherwise
   because the manuscript source is in a separate bundle.

---

## Adding Real Data

The repository ships with a pre-processed `data/processed_segments.csv`
(2 MB, 27 643 samples, 32 segments) so the pipeline runs out of the
box. To re-run the full pipeline on the **original raw dataset**, drop
the CSVs into the indicated folders:

### One-Pedal Driving Experiment (from the reference study)

```
data/raw/LEADCAR/leadingcar1.csv ... leadingcar12.csv
data/raw/FLWCAR1/followingcar_1_1.csv ... followingcar_1_12.csv
data/raw/FLWCAR2/followingcar_2_1.csv ... followingingcar_2_12.csv
data/raw/FLWCAR3/followingcar_3_1.csv ... followingingcar_3_12.csv
```

Each CSV must have the columns:
`Record, Time, Latitude, Longitude, Altitude, Speed, GForceX, GForceY, GForceZ, Lap`

where `Speed` is in km/h. The loader automatically detects the CSVs,
time-aligns each leader-follower pair, converts speed to m/s, computes
acceleration as the central finite difference of the speed signal, and
integrates the gap from the relative speed.

### Vehicle Energy Dataset (VED)

```
data/ved_subset/VED_YYYYMMDD_week.csv
```

Download from <https://github.com/gsoh/VED>. The loader automatically
identifies the EV vehicles (VehIds 10, 455, 541) inside the weekly CSVs
and groups rows by `(VehId, Trip)` to extract per-trip speed and HV-battery
power traces.

### After Adding Data

```bash
bash run_all.sh
```

The script will detect the real CSVs, replace the bundled
`data/processed_segments.csv` with a freshly pre-processed version, and
update the `provenance` flag in `data/segment_manifest.json` from
`"synthetic_ngsim_like"` to `"raw:data/raw"`.

---

## Key Results

The current results (with the bundled pre-processed data, which originates
from the real one-pedal driving experiment) are:

| Metric | Value | Source |
|--------|-------|--------|
| Number of training segments | 32 | `data/segment_manifest.json` |
| Number of training samples | 27 643 | `data/segment_manifest.json` |
| Mean speed | 18.24 m/s (~ 65 km/h) | `data/segment_manifest.json` |
| IDM acceleration-domain RMSE | 8.79 m/s² | `results/calibration_results.json` |
| IDM speed-domain RMSE | 3.48 m/s | `results/calibration_results.json` |
| IDM speed-domain Pearson r | 0.824 | `results/calibration_results.json` |
| Pareto weight ratios | {0.1, 0.3, 1.0, 3.0, 10.0} | `results/pareto_front.json` |
| Aggressive driver energy per trip | 7.73 kWh | `results/closed_loop_results.json` |
| Eco driver energy per trip | 2.89 kWh (-62.6 %) | `results/closed_loop_results.json` |
| Aggressive driver degradation per trip | 0.256 % | `results/closed_loop_results.json` |
| Eco driver degradation per trip | 0.144 % (-43.7 %) | `results/closed_loop_results.json` |
| VED external-evaluation trips | 11 VED trips | `results/ved_validation.json` |

The relatively high IDM RMSE (compared to the 0.93 m/s typical for NGSIM
highway data) is expected because the bundled data comes from a one-pedal
driving experiment in which the follower relies heavily on regenerative
braking. The standard IDM does not natively model regenerative braking,
so it under-fits the observed acceleration patterns. The speed-domain
correlation r = 0.824 confirms that the calibrated model still captures
the dominant kinematic trend, which is sufficient for the relative
energy and degradation comparisons reported above.

---

## Reproducibility

All random seeds are fixed:

| Step | Seed |
|------|------|
| Preprocessing | 42 |
| IDM calibration (Differential Evolution) | 42 |
| Multi-objective optimisation (Nelder-Mead) | 7 |

Re-running `bash run_all.sh` with the documented software environment, the same input data,
and the fixed random seeds is intended to reproduce the recorded results closely. Exact
bit-for-bit identity is not guaranteed across different operating systems, CPU/BLAS stacks,
or library versions.

---

## Running Individual Steps

Each step can be run independently for debugging or iteration:

```bash
# Set PYTHONPATH so the local modules are importable
export PYTHONPATH="$(pwd)"

# 1. Preprocessing
python -m preprocessing.preprocess \
    --raw-dir data/raw \
    --output   data/processed_segments.csv \
    --manifest data/segment_manifest.json

# 2. IDM sanity tests
python tests/test_idm.py

# 3. IDM calibration
python -m idm.calibration \
    --input  data/processed_segments.csv \
    --output results/calibration_results.json

# 4. Pareto scan
python -m optimization.multi_objective_optimization \
    --calibration results/calibration_results.json \
    --output     results/pareto_front.json

# 5. Closed-loop experiments
python -m experiments.closed_loop \
    --calibration results/calibration_results.json \
    --output     results/closed_loop_results.json

# 6. Sensitivity analysis
python -m analysis.sensitivity \
    --baseline results/pareto_front.json \
    --output   results/sensitivity_analysis.json

# 7. VED validation
python -m experiments.ved_validation \
    --ved-dir data/ved_subset \
    --output  results/ved_validation.json

# 8. Figures
python -m figures.make_figures
```

---

## Running the Tests

```bash
# Direct execution
PYTHONPATH=. python tests/test_idm.py

# Or with pytest
pip install pytest
PYTHONPATH=. python -m pytest tests/test_idm.py -v
```

Five sanity tests cover:

1. **Free-road convergence** — when there is no leader, the follower
   speed asymptotes to `v0`.
2. **Equilibrium gap** — at steady state with `v_leader < v0`, the
   follower's gap equals `(s0 + v_lead * T) / sqrt(1 - (v_lead/v0)^delta)`.
3. **Stopped-leader no-collision** — when the leader is permanently
   stopped, the follower's gap stays above `s0/2`.
4. **Hand-computed acceleration** — the JIT-compiled kernel matches a
   pure-Python reference to 1e-12.
5. **Emergency braking cap** — when the leader decelerates hard, the
   IDM deceleration stays finite and bounded.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | >= 1.26 | Numerical arrays |
| `scipy` | >= 1.11 | Optimisation (DE, Nelder-Mead, L-BFGS-B) |
| `numba` | >= 0.59 | JIT compilation of the IDM kernels |
| `matplotlib` | >= 3.7 | Figure generation |
| `pandas` | >= 2.0 | CSV / Excel I/O |

Install all at once:

```bash
pip install -r requirements.txt
```

Tested on Python 3.12.14. Python 3.10+ should work.

---

## Data Sources and Provenance

This repository combines author-provided processing code and derived data with externally sourced datasets.
The repository is intended to make the computational workflow transparent and reproducible; it does not
automatically grant redistribution rights for third-party raw datasets.

| Dataset / source | Role in this study | Main signals / fields | Reproducibility note |
|---|---|---|---|
| One-pedal driving experiment | IDM calibration | Time, speed, GPS, G-force, leader/follower traces | Raw CSVs may be supplied locally under `data/raw/`; derived segments are bundled |
| Vehicle Energy Dataset (VED) | External EV-trip evaluation | Vehicle speed, HV battery current, SOC, voltage, GPS/OBD signals | Obtain the original VED data from the upstream repository and place the required weekly files under `data/ved_subset/` |
| Additional trajectory / driving datasets | Supporting experiments / cross-checks | Dataset-specific trajectory and vehicle-state variables | Keep provenance and original licensing with the source dataset; do not redistribute restricted raw files without permission |

### What Is Actually Evaluated?

- **Trajectory processing / IDM calibration:** based on the real one-pedal driving trajectories and the retained 32 segments.
- **VED energy comparison:** uses real VED vehicle-trip measurements, but the current implementation compares different energy conventions/sign definitions; this should be described as an external evaluation/comparison rather than a successful accuracy validation.
- **Battery degradation:** the current pipeline uses a semi-empirical model. The VED degradation reference values are synthetic unless measured capacity-fade/SOH labels are supplied.

The distinction between measured data, derived data, model output, and synthetic reference data is intentional and should be preserved in future releases.

## Reproducibility and Release Practice

For publication, use a tagged repository release (for example `v1.0.0`) corresponding to the exact code/results state used to generate the submitted manuscript. Do not rely only on the moving `main` branch.

Recommended release metadata:

- Python version and operating system used for the reported run
- exact dependency versions (ideally a lock file or environment export)
- input-data provenance and dataset versions
- fixed random seeds
- Git tag / commit hash used for manuscript figures and tables

## Citation

If you use this repository or its derived results in academic work, please cite the associated paper and the
upstream datasets from their original publications/repositories.

A `CITATION.cff` file is recommended for the public release so GitHub can expose structured citation metadata.

## Research Use and Scope

This repository implements an offline computational digital-twin workflow. It should not be interpreted as a validated online
vehicle controller or a production battery-lifetime estimator. In particular, calibration feasibility, physical current limits,
energy-definition consistency, and independent battery-ageing validation should be checked before using the outputs for deployment
or lifetime prediction.

## License

MIT — see `LICENSE`.

---

## Repository Layout

```
EV_IDM_Battery_DT_Project/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── run_all.sh
├── run_all.py
├── .github/workflows/ci.yml
│
├── preprocessing/
│   ├── __init__.py
│   └── preprocess.py
├── idm/
│   ├── __init__.py
│   ├── idm_model.py
│   ├── jit_kernels.py
│   └── calibration.py
├── digital_twin/
│   ├── __init__.py
│   ├── ev_twin.py
│   └── battery_degradation.py
├── optimization/
│   ├── __init__.py
│   └── multi_objective_optimization.py
├── experiments/
│   ├── __init__.py
│   ├── closed_loop.py
│   └── ved_validation.py
├── analysis/
│   ├── __init__.py
│   └── sensitivity.py
├── figures/
│   ├── __init__.py
│   └── make_figures.py
├── tests/
│   ├── __init__.py
│   └── test_idm.py
│
├── data/
│   ├── segment_manifest.json
│   ├── processed_segments.csv
│   ├── raw/
│   │   ├── LEADCAR/.gitkeep
│   │   ├── FLWCAR1/.gitkeep
│   │   ├── FLWCAR2/.gitkeep
│   │   └── FLWCAR3/.gitkeep
│   └── ved_subset/.gitkeep
│
└── results/
    ├── calibration_results.json
    ├── pareto_front.json
    ├── closed_loop_results.json
    ├── sensitivity_analysis.json
    └── ved_validation.json
```

Total repository size: about 2.3 MB (without the raw CSVs).
