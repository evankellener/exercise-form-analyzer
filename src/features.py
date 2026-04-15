"""Per-rep biomechanical feature extraction.

Given a (T, 33, 4) landmark sequence spanning one rep, produce a fixed-length
feature vector capturing depth, forward lean, knee tracking, tempo, and symmetry.

These features are consumed by:
 - rules.py      : rule-based classifier (interpretable thresholds)
 - model.py      : ML multi-label classifier (LogReg / RandomForest)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pose import (
    L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
)


FEATURE_NAMES = [
    "min_knee_angle",
    "mean_knee_angle_bottom",
    "max_trunk_lean",
    "mean_trunk_lean_bottom",
    "min_hip_height",           # smaller y == higher on screen; min here is LOWEST hip
    "hip_below_knee_frac",      # fraction of bottom-phase frames where hip y > knee y
    "knee_cave_delta",          # (start_spread - bottom_spread) / start_spread (pos = caving)
    "knee_inside_ankle_frac",   # fraction of bottom frames with knees narrower than ankles
    "descent_seconds",
    "ascent_seconds",
    "bottom_hold_seconds",
    "total_seconds",
    "knee_angle_lr_diff",       # asymmetry at bottom
    "view_front_score",         # heuristic: 1.0 = likely front view, 0.0 = likely side
    "mean_knee_visibility",
]


@dataclass
class RepFeatures:
    values: np.ndarray          # shape (len(FEATURE_NAMES),)
    meta: dict                  # diagnostic info (fps, T, etc.)

    def as_dict(self) -> dict:
        return dict(zip(FEATURE_NAMES, map(float, self.values)))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba, axis=-1) + 1e-8
    nc = np.linalg.norm(bc, axis=-1) + 1e-8
    cos = np.clip(np.sum(ba * bc, axis=-1) / (na * nc), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _trunk_lean(shoulder: np.ndarray, hip: np.ndarray) -> np.ndarray:
    """Angle from vertical of the shoulder-hip vector, across time."""
    vec = shoulder - hip
    n = np.linalg.norm(vec, axis=-1) + 1e-8
    # Up vector in image coords: y decreases upward, so up = (0, -1).
    cos = np.clip(-vec[:, 1] / n, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def extract_features(seq: np.ndarray, fps: float) -> RepFeatures:
    """Extract a fixed-length feature vector from a per-rep landmark sequence."""
    xy = seq[..., :2]
    vis = seq[..., 3]
    T = seq.shape[0]

    # Per-time joint positions (average of left/right for robustness).
    shoulder = 0.5 * (xy[:, L_SHOULDER] + xy[:, R_SHOULDER])
    hip = 0.5 * (xy[:, L_HIP] + xy[:, R_HIP])
    knee = 0.5 * (xy[:, L_KNEE] + xy[:, R_KNEE])
    ankle = 0.5 * (xy[:, L_ANKLE] + xy[:, R_ANKLE])

    # Knee angles (left / right / mean).
    ka_l = _angle(xy[:, L_HIP], xy[:, L_KNEE], xy[:, L_ANKLE])
    ka_r = _angle(xy[:, R_HIP], xy[:, R_KNEE], xy[:, R_ANKLE])
    ka = 0.5 * (ka_l + ka_r)

    # Trunk lean.
    trunk = _trunk_lean(shoulder, hip)

    # Bottom phase = frames in the deepest 25% of the signal.
    thresh = np.nanpercentile(ka, 25)
    bottom_mask = ka <= thresh
    if not bottom_mask.any():
        bottom_mask = np.zeros_like(ka, dtype=bool)
        bottom_mask[np.nanargmin(ka)] = True

    # Depth: hip_y vs. knee_y. In image coords, LARGER y is LOWER on screen.
    hip_below_knee = (hip[:, 1] > knee[:, 1])
    hip_below_knee_frac = float(hip_below_knee[bottom_mask].mean()) if bottom_mask.any() else 0.0

    # Knee cave indicators (front view only reliable).
    knee_spread = np.abs(xy[:, L_KNEE, 0] - xy[:, R_KNEE, 0])
    ankle_spread = np.abs(xy[:, L_ANKLE, 0] - xy[:, R_ANKLE, 0]) + 1e-4
    # Delta: how much knees narrowed from top of rep to bottom. Positive = caving.
    top_k = 5
    start_spread = float(np.nanmean(knee_spread[:top_k])) if len(knee_spread) >= top_k else float(knee_spread[0])
    bottom_spread_series = knee_spread  # we'll take the deepest-phase mean below
    # knee_inside_ankle: in front view, knee-x should be >= ankle-x spread. When caving,
    # each knee crosses inside its ipsilateral ankle.
    l_inside = xy[:, L_KNEE, 0] > xy[:, L_ANKLE, 0]
    r_inside = xy[:, R_KNEE, 0] < xy[:, R_ANKLE, 0]
    knee_inside_ankle = (l_inside & r_inside)  # both knees caved inward

    # View heuristic: in front view, L/R shoulders are far apart horizontally; in side
    # view they overlap (distance small relative to shoulder-hip vertical).
    sh_x_spread = np.abs(xy[:, L_SHOULDER, 0] - xy[:, R_SHOULDER, 0])
    torso_height = np.abs(hip[:, 1] - shoulder[:, 1]) + 1e-4
    view_front_score = float(np.clip(np.nanmedian(sh_x_spread / torso_height), 0.0, 1.0))

    # Tempo: descent = first arg-to-minimum, ascent = from minimum to end.
    bottom_idx = int(np.nanargmin(ka))
    descent_frames = bottom_idx
    ascent_frames = T - bottom_idx - 1
    bottom_hold_frames = int(bottom_mask.sum())

    bottom_spread_mean = float(np.nanmean(bottom_spread_series[bottom_mask])) if bottom_mask.any() else float(knee_spread[bottom_idx])
    knee_cave_delta = (start_spread - bottom_spread_mean) / (start_spread + 1e-4)

    values = np.array([
        float(np.nanmin(ka)),
        float(np.nanmean(ka[bottom_mask])),
        float(np.nanmax(trunk)),
        float(np.nanmean(trunk[bottom_mask])),
        float(np.nanmax(hip[:, 1])),          # lowest hip (largest y)
        hip_below_knee_frac,
        float(knee_cave_delta),
        float(knee_inside_ankle[bottom_mask].mean()) if bottom_mask.any() else 0.0,
        descent_frames / max(fps, 1.0),
        ascent_frames / max(fps, 1.0),
        bottom_hold_frames / max(fps, 1.0),
        T / max(fps, 1.0),
        float(np.nanmean(np.abs(ka_l - ka_r)[bottom_mask])) if bottom_mask.any() else 0.0,
        view_front_score,
        float(np.nanmean(vis[:, [L_KNEE, R_KNEE]])),
    ], dtype=np.float32)

    return RepFeatures(values=values, meta={"T": T, "fps": fps, "bottom_idx": bottom_idx})


def build_dataset(
    label_rows: list[dict],
    pose_cache: dict[str, dict],
    augment_copies: int = 0,
    seed: int = 0,
) -> dict:
    """Assemble (X, Y, meta) arrays from labels + pose caches.

    label_rows     : list of dicts loaded from data/labels/rep_labels.csv
    pose_cache     : map video_path -> pose extraction result (from extract_pose_sequence)
    augment_copies : number of augmented copies per real rep (0 = no augmentation)

    Returns:
        X        : (N, F) float32 feature matrix
        Y        : (N, 3) int labels [forward_lean, shallow_depth, knee_cave]
        valid    : (N,) int — counted_valid column
        source   : (N,) str  — source video path (used for LOVO-CV grouping)
        is_aug   : (N,) bool — True for augmented rows, False for originals
    """
    from .augment import augment_one
    rng = np.random.default_rng(seed)

    X_rows, Y_rows, valid_rows, src_rows, aug_rows = [], [], [], [], []

    for row in label_rows:
        video = row["video"]
        pose = pose_cache.get(video)
        if pose is None:
            continue
        seq = pose["landmarks"][int(row["start_frame"]): int(row["end_frame"]) + 1]
        if seq.shape[0] < 4:
            continue

        feats = extract_features(seq, pose["fps"]).values
        label = np.array([
            int(row["forward_lean"]), int(row["shallow_depth"]), int(row["knee_cave"]),
        ], dtype=np.int8)

        X_rows.append(feats)
        Y_rows.append(label)
        valid_rows.append(int(row["counted_valid"]))
        src_rows.append(video)
        aug_rows.append(False)

        # Augmented copies inherit the same label.
        view = "front" if "front" in video else "side"
        for _ in range(augment_copies):
            aug_seq, _ = augment_one(seq, rng, view=view)
            aug_feats = extract_features(aug_seq, pose["fps"]).values
            X_rows.append(aug_feats)
            Y_rows.append(label)
            valid_rows.append(int(row["counted_valid"]))
            src_rows.append(video)
            aug_rows.append(True)

    return {
        "X": np.array(X_rows, dtype=np.float32),
        "Y": np.array(Y_rows, dtype=np.int8),
        "valid": np.array(valid_rows, dtype=np.int8),
        "source": np.array(src_rows),
        "is_aug": np.array(aug_rows, dtype=bool),
        "feature_names": FEATURE_NAMES,
        "label_names": ["forward_lean", "shallow_depth", "knee_cave"],
    }
