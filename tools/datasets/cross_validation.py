
import os
import sys
import random
import json
import numpy as np
from functools import partial


SELECTED_FREQUENCY = '1.0'
DATASET_PATH = 'data/raw'
RESULTS_DIR = 'data/dataset'

SAMPLING_RATE = 240

FREQUENCIES_WINDOWS = {
    "0.5": int(SAMPLING_RATE * 2),
    "0.75": int(SAMPLING_RATE * 1.5),
    "1.0": int(SAMPLING_RATE * 1.0),
}

window_size = FREQUENCIES_WINDOWS[SELECTED_FREQUENCY]



def main():
    mg_samples = os.listdir(os.path.join(DATASET_PATH, "Definite MG"))
    healthy_samples = os.listdir(os.path.join(DATASET_PATH, "Healthy control"))

    print("MG len", len(mg_samples))
    print("Healthy len", len(healthy_samples))

    seed = 42
    random.seed(seed)

    n_folds = 5

    healthy_shuffled = healthy_samples.copy()
    mg_shuffled = mg_samples.copy()

    random.shuffle(healthy_shuffled)
    random.shuffle(mg_shuffled)

    folds = []
    for fold_idx in range(n_folds):
        val_healthy = healthy_shuffled[fold_idx::n_folds]
        train_healthy = [s for i, s in enumerate(healthy_shuffled) if i % n_folds != fold_idx]

        val_mg = mg_shuffled[fold_idx::n_folds]
        train_mg = [s for i, s in enumerate(mg_shuffled) if i % n_folds != fold_idx]

        folds.append({
            "fold": fold_idx,
            "train": {
                "healthy": train_healthy,
                "mg": train_mg,
            },
            "val": {
                "healthy": val_healthy,
                "mg": val_mg,
            },
        })

    config = {
        "seed": seed,
        "n_folds": n_folds,
        "dataset_path": DATASET_PATH,
        "folds": folds,
    }

    with open(os.path.join(RESULTS_DIR, "cv_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("Configuration of 5-fold saved in cv_config.json")




if __name__ == "__main__":
    main()