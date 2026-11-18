"""Evaluate the saved models on the same held-out split the notebooks use and write results/metrics.json.

    python -m app.ml.evaluate

The web UI reads this file to show per-digit reports and confusion matrices. It needs train.csv
(the 20% test split is re-created with test_size=0.2, random_state=42, stratify=y) and the model files.
SVM and kNN can take a few minutes on 8,400 images.
"""
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split

from . import config
from .registry import ModelRegistry


def main():
    path = config.find("train.csv", config.DATA_DIRS)
    if path is None:
        sys.exit("train.csv not found. Put it in the data/ folder (or set DIGITS_DATA_DIR).")

    df = pd.read_csv(path)
    y = df["label"].to_numpy()
    x = df.drop("label", axis=1).to_numpy(dtype="uint8")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    registry = ModelRegistry()
    registry.load_all()
    if not registry.keys():
        sys.exit("No models could be loaded. Copy the .pkl / .keras files into models/ first.")

    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": {"train": int(len(x_train)), "test": int(len(x_test)), "test_size": 0.2, "random_state": 42},
        "class_counts": np.bincount(y, minlength=config.N_CLASSES).tolist(),
        "models": {},
    }

    for key in registry.keys():
        print(f"[eval] {config.MODEL_SPECS[key]['label']} on {len(x_test)} images ...", flush=True)
        started = time.time()
        pred = registry.predict_batched(key, x_test)
        acc = accuracy_score(y_test, pred)
        out["models"][key] = {
            "accuracy": acc,
            "precision": precision_score(y_test, pred, average="weighted"),
            "recall": recall_score(y_test, pred, average="weighted"),
            "f1": f1_score(y_test, pred, average="weighted"),
            "error_rate": 1 - acc,
            "correct": int((pred == y_test).sum()),
            "incorrect": int((pred != y_test).sum()),
            "total": int(len(y_test)),
            "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
            "confusion": confusion_matrix(y_test, pred).tolist(),
            "seconds": round(time.time() - started, 1),
        }
        print(f"       accuracy {acc:.4f}", flush=True)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = config.RESULTS_DIR / "metrics.json"
    target.write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {target}")


if __name__ == "__main__":
    main()
