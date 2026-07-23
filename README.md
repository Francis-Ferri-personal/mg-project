# Myasthenia Gravis Ocular Signal Classification

This project processes ocular signal data from patients to classify visits as `Definite MG` or `Healthy control` using frequency-based windows and an LSTM model.

## Repository structure

- notebooks/
  - `01-dataset-generation.ipynb` - Notebook to generate the dataset and prepare the input JSONL files.
  - `02-statistics_test.ipynb` - Notebook for exploratory data analysis and feature validation.
  - `03-training.ipynb` - Training notebook that builds the dataset, creates folds, and trains the model.
- `data/` - Raw data organized in folders by label and date.
- `export/` - Generated outputs, including the processed dataset and `cv_config.json`.
- `models/`
  - `accession_dataset.py` - Custom PyTorch dataset that reads JSON files, extracts window-level features, and handles variable-length sequences.
  - `lstm.py` - `OcularStatefulLSTM` model that processes temporal feature sequences with an LSTM.
- `tools/` - Utility modules for loading data, statistics, and analysis.
- `TODO.md` - List of pending tasks and improvements.

## Goal

Extract temporal features from ocular signal series at multiple frequencies and train a binary classifier.

### Feature extraction summary

- Each frequency is divided into fixed-size windows.
- Each window is split into two halves.
- For each half we extract:
  - maximum value
  - minimum value
  - index of the maximum
  - index of the minimum
- This produces 8 features per window.
- Using all three frequencies (`freq_0.5`, `freq_0.75`, `freq_1.0`) yields a 24-feature time step.

## Data pipeline

1. The dataset loads JSON files from `out/dataset`.
2. Each file is processed for each available `visit` entry.
3. Window sequences are extracted for each configured frequency.
4. Missing frequency windows are padded with zeros to normalize sequence length.
5. Batches are created with `ocular_collate_fn`, producing:
   - `sequences` shaped `(batch, max_steps, features)`
   - `labels`
   - `masks` for variable-length sequences

## Model

- `OcularStatefulLSTM` uses a PyTorch LSTM with `batch_first=True`.
- It packs padded sequences with `pack_padded_sequence` to handle variable lengths.
- The model selects the last valid output from each sequence for classification.
- The final classifier is a linear layer over the hidden output.

## Usage

1. Activate the virtual environment:

```bash
source .venv/bin/activate
```

2. Open the training notebook:

```bash
jupyter notebook 03-training.ipynb
```

3. Run the cells in order:
   - Define paths and parameters.
   - Generate cross-validation folds.
   - Execute `train_with_json_splits`.

## Training configuration

- `DATASET_PATH` should point to `out/dataset`.
- `RESULTS_DIR` should point to `out`.
- The notebook creates `cv_config.json` with cross-validation folds.
- The default training settings use `batch_size=8` and `epochs=15`.

## Debug notes

- If you see a shape error in the model, ensure the dataset returns 24-feature time steps when all three frequencies are used.
- If some visits are missing frequencies, the dataset pads with zeros to preserve the expected shape.
- If an outdated module version is loaded in Jupyter, restart the kernel and rerun all cells.

## Potential improvements

- Normalize features per window or across the dataset.
- Visualize window extraction with plots.
- Add a `requirements.txt` file with required packages.
- Add explicit missing-data validation and clearer reporting.

## Notes

This README documents the project structure and main components of the MG classification pipeline built on ocular signal features.
