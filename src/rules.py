"""Rule-based per-rep classifier.

Takes a RepFeatures vector and applies interpretable thresholds to produce
three-way fault labels: [forward_lean, shallow_depth, knee_cave].
Thresholds match coaching consensus and can be tuned against the labeled data.

This is the v1 baseline the ML model in model.py must beat.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FEATURE_NAMES


# ---- tunable thresholds (set by inspection of labeled feature stats) -------
TRUNK_LEAN_DEG = 60.0
SHALLOW_HIP_BELOW_KNEE_FRAC = 0.15
SHALLOW_MIN_KNEE_DEG = 70.0
KNEE_CAVE_INSIDE_FRAC = 0.30
VIEW_FRONT_THRESH = 0.70         # above this, treat as clean front view for cave detection


def _f(values: np.ndarray, name: str) -> float:
    return float(values[FEATURE_NAMES.index(name)])


def classify(values: np.ndarray) -> dict:
    """Return dict with keys forward_lean, shallow_depth, knee_cave (0/1) + diag."""
    mean_lean_bottom = _f(values, "mean_trunk_lean_bottom")
    min_knee = _f(values, "min_knee_angle")
    hip_below_knee = _f(values, "hip_below_knee_frac")
    cave_delta = _f(values, "knee_cave_delta")
    inside_frac = _f(values, "knee_inside_ankle_frac")
    view_front = _f(values, "view_front_score")

    forward_lean = int(mean_lean_bottom > TRUNK_LEAN_DEG)
    shallow = int(
        hip_below_knee < SHALLOW_HIP_BELOW_KNEE_FRAC and min_knee > SHALLOW_MIN_KNEE_DEG
    )
    # Knee cave: only trust the signal in front view. In side view the knees
    # overlap in 2D so the spread ratio is pure noise.
    if view_front > VIEW_FRONT_THRESH:
        knee_cave = int(inside_frac > KNEE_CAVE_INSIDE_FRAC)
    else:
        knee_cave = 0

    return {
        "forward_lean": forward_lean,
        "shallow_depth": shallow,
        "knee_cave": knee_cave,
        "counted_valid": int(not (forward_lean or shallow or knee_cave)),
        "diag": {
            "mean_lean_bottom": round(mean_lean_bottom, 2),
            "min_knee_angle": round(min_knee, 2),
            "hip_below_knee_frac": round(hip_below_knee, 3),
            "knee_cave_delta": round(cave_delta, 3),
            "knee_inside_ankle_frac": round(inside_frac, 3),
            "view_front_score": round(view_front, 3),
        },
    }


def classify_batch(X: np.ndarray) -> np.ndarray:
    """Return (N, 3) int array of [lean, shallow, cave] for a batch of feature vectors."""
    out = np.zeros((X.shape[0], 3), dtype=np.int8)
    for i in range(X.shape[0]):
        r = classify(X[i])
        out[i] = [r["forward_lean"], r["shallow_depth"], r["knee_cave"]]
    return out
