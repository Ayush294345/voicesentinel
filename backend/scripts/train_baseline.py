"""
train_baseline.py
------------------
Trains the baseline real-vs-synthetic voice classifier used by
core/spoof_detector.py.

Usage (run from the project root):
    python scripts/train_baseline.py

Reads every .wav file from data/real/ (label 0) and data/fake/ (label
1), extracts features, trains a GradientBoosting classifier, prints a
held-out evaluation report, and saves the model to
models/baseline_model.pkl.
"""

import glob
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Make `core` importable when running this script directly from scripts/
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.feature_extraction import extract_features  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
REAL_DIR = os.path.join(ROOT, "data", "real")
FAKE_DIR = os.path.join(ROOT, "data", "fake")
MODEL_OUT = os.path.join(ROOT, "models", "baseline_model.pkl")


def load_dataset():
    X, y = [], []
    real_files = sorted(glob.glob(os.path.join(REAL_DIR, "*.wav")))
    fake_files = sorted(glob.glob(os.path.join(FAKE_DIR, "*.wav")))

    if not real_files or not fake_files:
        print("No training data found.")
        print(f"  real_dir: {REAL_DIR} ({len(real_files)} files)")
        print(f"  fake_dir: {FAKE_DIR} ({len(fake_files)} files)")
        print("\nRun `python scripts/make_demo_data.py` first, or drop your")
        print("own labeled .wav files into those two folders.")
        sys.exit(1)

    print(f"Loading {len(real_files)} real + {len(fake_files)} fake clips...")
    for f in real_files:
        X.append(extract_features(f))
        y.append(0)
    for f in fake_files:
        X.append(extract_features(f))
        y.append(1)

    return np.array(X), np.array(y)


def main():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\n--- Held-out evaluation ---")
    print(classification_report(y_test, y_pred, target_names=["real", "fake/cloned"]))
    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {auc:.3f}")
    except ValueError:
        pass

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    print(f"\nModel saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
