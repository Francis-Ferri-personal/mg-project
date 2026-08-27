import os
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from models.accession_dataset import AccessionDataset, ocular_collate_fn
from models.gru import OcularStatefulGRU

SELECTED_FREQUENCY = "1.0"
DEFAULT_DATASET_CANDIDATES = [
    os.path.join("data", "dataset", "formatted"),
    os.path.join("out", "dataset"),
]
DATASET_PATH = next((p for p in DEFAULT_DATASET_CANDIDATES if os.path.isdir(p)), "out/dataset")
RESULTS_DIR = "out"
SPLITS_JSON = os.path.join(RESULTS_DIR, "cv_config.json")

SAMPLING_RATE = 240
FREQUENCIES_WINDOWS = {
    "0.5": int(SAMPLING_RATE * 2),
    "0.75": int(SAMPLING_RATE * 1.5),
    "1.0": int(SAMPLING_RATE * 1.0),
}
window_size = FREQUENCIES_WINDOWS[SELECTED_FREQUENCY]


def ensure_cv_config(dataset_path, results_dir, n_folds=5, seed=42):
    os.makedirs(results_dir, exist_ok=True)
    config_path = os.path.join(results_dir, "cv_config.json")

    label_dirs = [
        os.path.join(dataset_path, "Definite MG"),
        os.path.join(dataset_path, "Healthy control"),
    ]
    if not all(os.path.isdir(d) for d in label_dirs):
        raise FileNotFoundError(
            f"Dataset directory {dataset_path} is missing label folders: "
            + ", ".join(os.path.basename(d) for d in label_dirs)
        )

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if (
                existing.get("dataset_path") == dataset_path
                and existing.get("n_folds") == n_folds
                and existing.get("seed") == seed
                and isinstance(existing.get("folds"), list)
                and len(existing["folds"]) > 0
            ):
                print(f"Using existing cross-validation config: {config_path}")
                return existing
        except Exception:
            pass

        print(f"Discarding stale cross-validation config: {config_path}")

    mg_samples = sorted(
        name for name in os.listdir(os.path.join(dataset_path, "Definite MG"))
        if name.lower().endswith(".json")
    )
    healthy_samples = sorted(
        name for name in os.listdir(os.path.join(dataset_path, "Healthy control"))
        if name.lower().endswith(".json")
    )

    if len(mg_samples) == 0 or len(healthy_samples) == 0:
        raise ValueError(
            f"Dataset {dataset_path} has no JSON samples for one of the classes. "
            f"MG={len(mg_samples)}, Healthy={len(healthy_samples)}"
        )

    random.seed(seed)
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
            "train": {"healthy": train_healthy, "mg": train_mg},
            "val": {"healthy": val_healthy, "mg": val_mg},
        })

    config = {
        "seed": seed,
        "n_folds": n_folds,
        "dataset_path": dataset_path,
        "folds": folds,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved new cross-validation config to {config_path}")
    return config


def resolve_paths(base_dataset_dir, split_section):
    paths = []

    for f_name in split_section.get("healthy", []):
        candidate = os.path.join(base_dataset_dir, "Healthy control", f_name)
        if os.path.exists(candidate):
            paths.append(candidate)
        else:
            alt = os.path.join(base_dataset_dir, "healthy", f_name)
            if os.path.exists(alt):
                paths.append(alt)

    for f_name in split_section.get("mg", []):
        definite_path = os.path.join(base_dataset_dir, "Definite MG", f_name)
        probable_path = os.path.join(base_dataset_dir, "Probable MG", f_name)
        if os.path.exists(definite_path):
            paths.append(definite_path)
        elif os.path.exists(probable_path):
            paths.append(probable_path)
        else:
            legacy_path = os.path.join(base_dataset_dir, f_name)
            if os.path.exists(legacy_path):
                paths.append(legacy_path)

    return paths


def run_epoch(model, loader, optimizer, criterion, device, is_train=True):
    if is_train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    total_loss = 0.0
    correct = 0
    total = 0

    for sequences, labels, masks in loader:
        sequences = sequences.to(device)
        labels = labels.to(device).long()
        masks = masks.to(device)

        outputs = model(sequences, masks)
        loss = criterion(outputs, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return total_loss / total, 100 * correct / total


def train_with_json_splits(splits_json_path, base_dataset_dir, epochs=40, batch_size=8, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join("out", "training_results")
    os.makedirs(out_dir, exist_ok=True)

    log_file_path = os.path.join(out_dir, "training_log.txt")
    log_file = open(log_file_path, "w", encoding="utf-8")

    def log_print(message):
        print(message)
        log_file.write(str(message) + "\n")
        log_file.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_print(f"Using device: {device}")

    with open(splits_json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    folds_config = config.get("folds", [])
    log_print(f"Loaded config containing {len(folds_config)} pre-configured folds.")

    fold_train_accs, fold_val_accs = [], []
    summary_data = {
        "dataset_path": base_dataset_dir,
        "selected_frequency": SELECTED_FREQUENCY,
        "window_size": window_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "folds": []
    }

    for fold_data in folds_config:
        fold_idx = fold_data["fold"]
        log_print("\n" + "=" * 50)
        log_print(f" TRAINING PREDEFINED FOLD {fold_idx + 1}/{len(folds_config)} ")
        log_print("=" * 50)

        train_paths = resolve_paths(base_dataset_dir, fold_data["train"])
        val_paths = resolve_paths(base_dataset_dir, fold_data["val"])

        train_dataset = AccessionDataset(train_paths, window_size=window_size, frequency_key="all")
        val_dataset = AccessionDataset(val_paths, window_size=window_size, frequency_key="all")

        num_features = train_dataset[0][0].shape[1] if len(train_dataset) > 0 else 324
        log_print(f"Model input feature size: {num_features}")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=ocular_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=ocular_collate_fn)

        model = OcularStatefulGRU(input_size=num_features, num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        best_val_acc = 0.0
        fold_history = []

        for epoch in range(epochs):
            train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device, is_train=True)
            val_loss, val_acc = run_epoch(model, val_loader, None, criterion, device, is_train=False)

            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }
            fold_history.append(epoch_log)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save the model state dict to CPU to avoid CUDA dependency issues on load
                model_path = os.path.join(out_dir, f"best_model_fold_{fold_idx + 1}.pt")
                torch.save(model.state_dict(), model_path)

            log_print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        log_print(f">> Fold {fold_idx + 1} Done. Best Validation Accuracy: {best_val_acc:.2f}%")
        fold_val_accs.append(best_val_acc)
        fold_train_accs.append(train_acc)

        summary_data["folds"].append({
            "fold": fold_idx + 1,
            "best_val_accuracy": float(best_val_acc),
            "final_train_accuracy": float(train_acc),
            "history": fold_history
        })

    mean_train_acc = float(np.mean(fold_train_accs))
    mean_val_acc = float(np.mean(fold_val_accs))
    std_val_acc = float(np.std(fold_val_accs))

    log_print("\n" + "═" * 50)
    log_print(" FINAL PREDEFINED CROSS VALIDATION METRICS SUMMARY")
    log_print("═" * 50)
    log_print(f"Mean Training Accuracy: {mean_train_acc:.2f}%")
    log_print(f"Mean Validation Accuracy: {mean_val_acc:.2f}% (+/- {std_val_acc:.2f}%)")

    summary_data["mean_train_accuracy"] = mean_train_acc
    summary_data["mean_val_accuracy"] = mean_val_acc
    summary_data["val_accuracy_std"] = std_val_acc

    # Save summary.json
    summary_json_path = os.path.join(out_dir, "summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as sf:
        json.dump(summary_data, sf, indent=2)

    log_print(f"\nSaved training summary and weights to: {out_dir}")
    log_file.close()


if __name__ == "__main__":
    ensure_cv_config(DATASET_PATH, RESULTS_DIR, n_folds=5, seed=42)
    train_with_json_splits(SPLITS_JSON, DATASET_PATH, batch_size=8)

