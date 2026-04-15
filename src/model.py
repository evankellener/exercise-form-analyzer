"""Multi-label per-rep classifier (forward_lean, shallow_depth, knee_cave).

Each fault is a separate binary head trained on the engineered feature vector.
Candidate estimators: LogisticRegression (linear, interpretable) and RandomForest
(non-linear, captures interactions). Dataset is tiny (~40 reps), so we:

  - Use leave-one-video-out CV (LOVO) to prevent rep leakage between train/test.
  - Augment the TRAINING fold only (never the held-out video).
  - Evaluate on raw (un-augmented) held-out reps.

Models are persisted to models/squat_multilabel_<estimator>.pkl and include a
StandardScaler fit on training features + metadata for later inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES

LABEL_NAMES = ["forward_lean", "shallow_depth", "knee_cave"]
MODEL_DIR = Path("models")


def _make_estimator(kind: str):
    if kind == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", C=1.0, solver="liblinear",
            )),
        ])
    if kind == "rf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_leaf=2,
                class_weight="balanced", random_state=0,
            )),
        ])
    raise ValueError(f"unknown estimator: {kind}")


@dataclass
class MultiLabelModel:
    estimators: list                # one per label
    feature_names: list[str]
    label_names: list[str]
    kind: str

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 3) int array of predicted fault flags."""
        out = np.zeros((X.shape[0], len(self.estimators)), dtype=np.int8)
        for j, est in enumerate(self.estimators):
            if est is None:
                continue  # constant-label column, predict 0
            out[:, j] = est.predict(X).astype(np.int8)
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (N, 3) float32 probability array for the positive class."""
        out = np.zeros((X.shape[0], len(self.estimators)), dtype=np.float32)
        for j, est in enumerate(self.estimators):
            if est is None:
                continue
            out[:, j] = est.predict_proba(X)[:, 1]
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "MultiLabelModel":
        return joblib.load(path)


def train(X: np.ndarray, Y: np.ndarray, kind: str = "rf") -> MultiLabelModel:
    """Fit one binary estimator per label. Constant-label columns get no estimator."""
    estimators = []
    for j in range(Y.shape[1]):
        yj = Y[:, j]
        if len(np.unique(yj)) < 2:
            estimators.append(None)
            continue
        est = _make_estimator(kind)
        est.fit(X, yj)
        estimators.append(est)
    return MultiLabelModel(
        estimators=estimators,
        feature_names=list(FEATURE_NAMES),
        label_names=list(LABEL_NAMES),
        kind=kind,
    )


def train_and_save(
    dataset: dict,
    kind: str = "rf",
    out_path: str | Path = None,
) -> Path:
    X, Y = dataset["X"], dataset["Y"]
    model = train(X, Y, kind=kind)
    out_path = Path(out_path or MODEL_DIR / f"squat_multilabel_{kind}.pkl")
    model.save(out_path)
    return out_path
