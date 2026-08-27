# Myasthenia Gravis (MG) Profile Dataset

This directory contains the processed profile dataset generated from raw eye-tracking / kinematic movement recordings.

## Dataset Files

- **`dataset.jsonl`**: The generated dataset in JSON Lines (`.jsonl`) format, where each line represents a Patient Visit containing all available frequencies (`0.5`, `0.75`, `1.0` Hz) and their cycle feature profiles.
- **`README.md`**: This documentation file explaining the dataset structure and 27-feature schema per frequency.

---

## Dataset Structure

Each line in `dataset.jsonl` has the following top-level schema:

```json
{
  "visit_id": "Definite MG/Patient 01/2023-05-10/horizontal",
  "pathology_class": "Definite MG",
  "patient_name": "Patient 01",
  "visit_date": "2023-05-10",
  "axis": "horizontal",
  "frequencies": {
    "0.5": {
      "cycles": [
        {
          "L": { ... 27 features ... },
          "R": { ... 27 features ... },
          "cross_channel_stats": {
            "median_difference": float,
            "amplitude_ratio": float,
            "lag": int
          }
        }
      ]
    },
    "0.75": { ... },
    "1.0": { ... }
  }
}
```

---

## 27 Features per Channel (`L` and `R`) per Frequency

For each channel (`L` and `R`) in each cycle, the following 27 features are extracted:

### 1. Position & Target State Features (12)
- **Overall Cycle Position**: `pos_min`, `pos_max`, `pos_mean`, `pos_std`, `pos_skewness`, `pos_energy`
- **Low Target State**: `low_state_amplitude`, `low_state_std`, `low_state_mean`
- **High Target State**: `high_state_amplitude`, `high_state_std`, `high_state_mean`

### 2. Symmetry Ratio (1)
- `symmetry_ratio`: Balance ratio between positive and negative state amplitudes (`abs(high_state_amplitude / low_state_amplitude)`).

### 3. Movement Gain / IoU (1)
- `movement_gain_iou`: Overall tracking quality IoU between target square wave and eye position signal.

### 4. Relative Offset (1)
- `relative_offset`: Residual baseline offset / drift after calibration correction `(high_state_mean + low_state_mean) / 2`.

### 5. Episode Kinematics (12)
- **Rise Transition** (6):
  - `rise_mean_vel`, `rise_max_vel`, `rise_std_vel` (Velocity)
  - `rise_mean_acc`, `rise_max_acc`, `rise_std_acc` (Acceleration)
- **Fall Transition** (6):
  - `fall_mean_vel`, `fall_max_vel`, `fall_std_vel` (Velocity)
  - `fall_mean_acc`, `fall_max_acc`, `fall_std_acc` (Acceleration)

---

## How to Regenerate the Dataset

To regenerate `dataset.jsonl` from raw CSV files in `data/raw`, run:

```bash
uv run python tools/datasets/generate_dataset.py --raw-dir data/raw --output data/dataset/dataset.jsonl --denoise
```

### Command Flags:
- `--raw-dir`: Directory containing raw accessions (default: `data/raw`).
- `--output`: Output JSONL file path (default: `data/dataset/dataset.jsonl`).
- `--denoise`: Apply signal filtering (`clamp`, `median_filter`, `remove_spikes`) from `datasets.preprocesing.filters`.
