# AGENTS.md

## Setup & verification
- Python 3.14 (`.python-version`); `.venv` exists; deps managed with `uv` (root `uv.lock`). Install local package: `pip install -e ./mg_scripts` (pulls torch, numpy, pandas).
- All scripts must run with CWD = repo root (they use `mg_scripts.*` imports and relative paths like `out/dataset`, `data/raw`).
- No test suite, no lint/format/typecheck config, no CI. Verify by running the script/notebook and checking output.

## Code map (trust `mg_scripts/`, not the README)
- `mg_scripts/mg_dataset/` — `MGDataset` (reads raw CSVs) + `preprocesing/` (cycles, filters). Note the dir is misspelled `preprocesing`; do not rename it.
- `mg_scripts/mg_models/` — `accession_dataset.py` (PyTorch dataset + collate), `lstm.py`, `gru.py`.
- `mg_scripts/mg_tools/` — stats, loaders, visualization helpers.
- `train_gru.py` — current training entrypoint (GRU). `draw_validation.py` — plots validation folds from `data/raw` + `out/cv_config.json`. `notification.py` — signed webhook util (needs `python-dotenv`, `requests`; reads `.env`).
- `models/`, `tools/`, `dataset/` at root contain only stale `__pycache__` — the real code moved into `mg_scripts/`.
- README is partially stale: `mg_data/`, `tools/analysis/visualize_raw_samples.py`, and `export/` no longer exist. `notebooks/03-training.ipynb` is also stale (imports `models.*` and points at `../exports/dataset/v2`) — use `train_gru.py` to train.

## Data
- Raw: `data/raw/{Definite MG, Healthy control}/<YYYY-MM-DD <patient name>>/*.csv`. CSVs are **utf-16-le** encoded. Patient names are Korean — paths contain spaces and non-ASCII chars; always quote them.
- Processed (what training reads): `out/dataset/<label>/<NN-NN>.json` → `visits → {freq_0.5, freq_0.75, freq_1.0} → {speed_horizontalAVG, gain_verticalAVG, ...}` (lists of floats). Some JSONs are utf-16; `AccessionDataset` already tries utf-16-le/utf-16/utf-8 — keep that fallback if you touch it.
- `data/` and `out/` are gitignored and NOT in git — regenerate `out/dataset` via `notebooks/01-dataset_generation.ipynb` if missing (needs `data/raw`, which may also be absent).
- `notification.py`'s `.env` (webhook secret) is gitignored; never commit it.

## Training specifics
- 240 Hz sampling; window sizes: 480 (freq_0.5), 360 (freq_0.75), 240 (freq_1.0). Per window: split in two halves, extract max/min/argmax/argmin (8 features). All 3 frequencies → 24 features/timestep; shorter sequences zero-padded per timestep.
- `AccessionDataset` label is inferred by substring match on the file path: `"Healthy control"`→0, `"Definite MG"`→1. `train_gru.resolve_paths` falls back `Probable MG` → `Definite MG` per sample.
- Default feature key is `speed_horizontalAVG`; `frequency_key='all'` gives the 24-feature step.
- `train_gru.py` reuses `out/cv_config.json` if present (5 folds, stratified by label, seed 42); delete it to regenerate folds. Defaults: epochs=15, batch_size=8.

## Gotchas
- Restart Jupyter kernels / rerun all cells after editing anything in `mg_scripts/`.
- Variable-length sequences are zero-padded, not dropped; the model reads the **last valid** step — padding affects which step is "last" only via masks.
- `main.py` is a placeholder (`Hello from mg!`), not an entrypoint.
