# Myasthenia Gravis Ocular Signal Classification

This project processes ocular signal data from patients to classify visits as `Definite MG` or `Healthy control`. Raw saccade recordings (position) are turned into kinematic feature sequences (velocity, speed, gain for left/right/average at each drive frequency) and fed to a recurrent model (GRU, with an LSTM variant available).

All Python runs with **CWD = repo root** (imports are `dataset.*`, `models.*`, `tools.*`; paths are relative like `data/raw`, `out/dataset`).

## Repository structure

- `datasets/`
  - `dataset.py` — `MGDataset`: indexes and loads the raw CSVs from `data/raw` (utf-16-le). This is the raw-data reader.
  - `preprocesing/` — `filters.py` (`denoise`, `median_filter`, `remove_spikes`, `clamp`) and `cycles.py` (`get_cycles`). Note the dir is spelled `preprocesing`; do not rename it.
  - `explore.py` — Interactive exploration script for analyzing individual patient data with kernel comparisons.
- `models/`
  - `accession_dataset.py` — `AccessionDataset` + `ocular_collate_fn`: reads the generated JSON, extracts window-level features, handles variable-length sequences.
  - `gru.py` — `OcularStatefulGRU` (used by `train_gru.py`).
  - `lstm.py` — `OcularStatefulLSTM` (legacy/alternative model).
- `tools/` — analysis and dataset utilities: `datasets/generate_dataset.py` (CLI dataset generator script), `accession.py` (per-visit kinematics), `accessions_db.py`, `data_loader.py`, `stats.py`, `recording_analysis.py`, `recording_visualisers.py`, `visualize.py`, `jsonl_util.py`.
- `notebooks/`
  - `01-dataset_generation.ipynb` — generates the dataset from raw (entrypoint for step 1 below).
  - `02-statistics_test.ipynb`, `01a - dataset_with_dataset.ipynb` — exploratory data analysis / filtering.
  - `03-training.ipynb` — training notebook (preferred to use `train_gru.py`).
- `data/raw/{Definite MG, Healthy control}/<YYYY-MM-DD <patient name>>/*.csv` — raw recordings (utf-16-le, spaces + Korean characters in paths — always quote).
- `out/dataset/<label>/<NN-NN>.json` — generated per-patient dataset (what training reads). `out/cv_config.json` — cross-validation folds. `out/explore/` — exploration output.
- `train_gru.py` — training entrypoint. `draw_validation.py` — plots validation folds. `notification.py` — signed webhook (reads a gitignored `.env`; never commit it).
- `TODO.md` — pending tasks.
- `pyproject.toml` — Python project configuration with dependencies (managed by `uv`).

## Workflow (in order)

Run everything from the repo root inside the venv.

**Setup with `uv`:**

```bash
# Install dependencies and create virtual environment
uv sync

# Run commands using uv (automatically activates venv)
uv run python datasets/explore.py --patient 0
uv run python train_gru.py
```

Or activate the venv manually and run directly:

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
python datasets/explore.py --patient 0
```

### 1. Generate the dataset from raw

You can generate the profile dataset JSONL directly using the standalone CLI script [`tools/datasets/generate_dataset.py`](file:///workspace/MG/tools/datasets/generate_dataset.py):

```bash
uv run python tools/datasets/generate_dataset.py --raw-dir data/raw --output data/dataset/dataset.jsonl
```

**Flags:**
- `--raw-dir` — path to raw CSV accessions folder (default: `data/raw`)
- `--output` — output `.jsonl` file path (default: `data/dataset/dataset.jsonl`)

The raw CSVs are read from `data/raw/{Healthy control, Definite MG}`, and per-visit kinematics (velocity/speed/gain, baseline translation, percentile clamping, and episode micro-jumps/plateaus profiling) are written to `data/dataset/dataset.jsonl`.

Smoke-test the raw reader (indexing, CSV loading, cycle detection, filtering, and figure rendering):

```bash
python -m dataset.dataset   # run from repo root
```

### Format the dataset (required before training)

The previous step only writes a single flat JSONL (`data/dataset/dataset.jsonl`). Training does **not** read that file directly — `AccessionDataset` expects the grouped-per-patient layout `out/dataset/<label>/<file>.json`. You must format the JSONL into that layout:

```bash
uv run python datasets/format_dataset.py --input data/dataset/dataset.jsonl --output out/dataset
```

**Flags:**
- `--input` — the JSONL produced in step 1 (default: `data/dataset/dataset.jsonl`)
- `--output` — destination folder for the grouped JSONs (default: `data/dataset/formatted`). **Override this to `out/dataset`**, otherwise training (`DATASET_PATH = out/dataset`) will fail with a "missing label folders" error.

This writes one JSON per patient visit under `out/dataset/{Definite MG, Healthy control}/` plus a `manifest.json`. Skip this step and `train_gru.py` will abort at `ensure_cv_config` because `out/dataset` has no `Definite MG` / `Healthy control` folders.

### 1.5 Explore individual patient data

Use `datasets/explore.py` to analyze a single patient's visit with kernel comparison plots and cycle analysis:

```bash
# With uv (recommended)
uv run python datasets/explore.py --patient 0 --group "Definite MG" --visit 0

# Or activate venv and run directly
source .venv/bin/activate
python datasets/explore.py --patient 0 --group "Definite MG" --visit 0
```

**Flags:**
- `--patient` — patient index (default: 0)
- `--group` — group name (default: "Definite MG")
- `--visit` — visit index (default: 0)

Output is saved to `out/explore/`.

### 2. Train

`train_gru.py` builds/reuses `out/cv_config.json` (5 stratified folds, seed 42) and trains `OcularStatefulGRU`:

```bash
python train_gru.py         # defaults: epochs=15, batch_size=8
```

Delete `out/cv_config.json` to regenerate the folds. `notebooks/03-training.ipynb` is the interactive equivalent.

> Note: if you do not yet have a `cv_config.json`, or want to regenerate it, run the dedicated tool:
>
> ```bash
> uv run python tools/datasets/cross_validation.py
> ```
>
> It lists the samples under `data/raw` (or whatever `DATASET_PATH` is set to inside the script), shuffles with seed 42, and writes 5 stratified folds to `results_dir/cv_config.json` (`results_dir` is `data/dataset` by default inside that script). Confirm that the script's `RESULTS_DIR` matches where your training pipeline reads the config from.

## Feature extraction summary

- Each frequency series is divided into fixed-size windows (window size = `240 Hz / frequency`: 480 for 0.5 Hz, 360 for 0.75 Hz, 240 for 1.0 Hz).
- Each window is split into two halves; for each half we extract max, min, index-of-max, index-of-min (8 features).
- Using all three frequencies (`freq_0.5`, `freq_0.75`, `freq_1.0`) yields a 24-feature time step (8 × 3).

## Data pipeline (model input)

1. `AccessionDataset` loads the generated JSON files from `out/dataset`.
2. Each file is processed per `visit`.
3. Window-feature sequences are extracted per configured frequency; shorter sequences are zero-padded per step to align to the longest.
4. Batches are formed with `ocular_collate_fn`, producing `sequences` `(batch, max_steps, features)`, `labels`, and `masks`.
5. The model reads the **last valid** step per sequence (padding is handled by the masks), then a linear layer classifies.

## Training configuration

- `DATASET_PATH` = `out/dataset`; `RESULTS_DIR` = `out`; folds in `out/cv_config.json`.
- Label is inferred from the file path substring: `"Healthy control"` → 0, `"Definite MG"` → 1 (`train_gru.resolve_paths` falls back `Probable MG` → `Definite MG`).
- `AccessionDataset` default feature key is `speed_horizontalAVG`; `frequency_key='all'` gives the 24-feature step.
- Device: CUDA if available, else CPU.

## Debug notes

- If you see a shape error, ensure the dataset returns 24-feature steps when all three frequencies are used.
- Missing frequencies are padded with zeros to preserve shape.
- After editing anything in `dataset/`, `models/`, or `tools/`, restart Jupyter kernels and rerun all cells.
- `main.py` is a placeholder (`Hello from mg!`), not an entrypoint.

## Notes

`data/` and `out/` are gitignored and not in git — the raw data and generated dataset must be present before training. The `.env` used by `notification.py` (webhook secret) is gitignored and must never be committed.
