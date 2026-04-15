"""Rep segmentation from a landmark time series.

A squat rep = one descent + ascent of the hips. We track knee-flexion angle
(smaller = deeper squat) as the 1-D signal, smooth it, and find local minima
corresponding to the bottom of each rep. Each rep spans from the previous
standing frame to the next standing frame bracketing a minimum.

Returns a list of (start_frame, bottom_frame, end_frame) tuples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from .pose import L_HIP, L_KNEE, L_ANKLE, R_HIP, R_KNEE, R_ANKLE


@dataclass
class Rep:
    index: int             # 0-based rep index within the clip
    start: int             # frame index where descent begins
    bottom: int            # frame index of deepest point
    end: int               # frame index where ascent finishes (standing again)

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Vectorized angle at vertex b, across time. a, b, c are (T, 2)."""
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba, axis=1) + 1e-8
    nc = np.linalg.norm(bc, axis=1) + 1e-8
    cos = np.clip(np.sum(ba * bc, axis=1) / (na * nc), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def knee_angle_series(landmarks: np.ndarray, side: str = "both") -> np.ndarray:
    """Return (T,) knee-flexion angle in degrees. 180 = straight leg, ~70 = deep squat.

    side: "left", "right", or "both" (mean of the two). "both" is robust to
    side/front views where one leg may be partially occluded.
    """
    xy = landmarks[:, :, :2]  # (T, 33, 2) normalized
    left = _angle(xy[:, L_HIP], xy[:, L_KNEE], xy[:, L_ANKLE])
    right = _angle(xy[:, R_HIP], xy[:, R_KNEE], xy[:, R_ANKLE])
    if side == "left":
        return left
    if side == "right":
        return right
    return np.nanmean(np.stack([left, right], axis=1), axis=1)


def smooth_signal(x: np.ndarray, fps: float) -> np.ndarray:
    """Savitzky-Golay smoothing with a window scaled to frame rate (~0.25s)."""
    win = max(5, int(round(fps * 0.25)))
    if win % 2 == 0:
        win += 1
    win = min(win, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if win < 5:
        return x.astype(np.float32)
    return savgol_filter(x, window_length=win, polyorder=2).astype(np.float32)


def detect_reps(
    landmarks: np.ndarray,
    valid: np.ndarray,
    fps: float,
    standing_angle: float = 155.0,
    min_depth_drop: float = 25.0,
    min_rep_seconds: float = 1.0,
) -> list[Rep]:
    """Detect squat reps from the knee-angle signal.

    Parameters:
        standing_angle: angle above which we consider the person standing
        min_depth_drop: minimum required (standing - min) angle for a real rep
        min_rep_seconds: minimum time between reps, filters out jitter
    """
    angle = knee_angle_series(landmarks, side="both")
    # Fill invalid frames by forward-fill then nan-safe smoothing.
    angle = np.where(valid, angle, np.nan)
    if np.all(np.isnan(angle)):
        return []
    # Fill nans with the last valid value; if none yet, use the mean.
    mean_angle = float(np.nanmean(angle))
    filled = np.copy(angle)
    last = mean_angle
    for i in range(len(filled)):
        if np.isnan(filled[i]):
            filled[i] = last
        else:
            last = filled[i]
    smooth = smooth_signal(filled, fps)

    # Find minima = peaks of -signal.
    min_distance = max(1, int(round(fps * min_rep_seconds)))
    peaks, props = find_peaks(
        -smooth,
        distance=min_distance,
        prominence=min_depth_drop / 2.0,  # prominence in same units as signal
    )

    reps: list[Rep] = []
    for idx, bottom in enumerate(peaks):
        # Walk backwards to the previous standing crossing.
        start = bottom
        while start > 0 and smooth[start] < standing_angle:
            start -= 1
        # Walk forwards to the next standing crossing.
        end = bottom
        while end < len(smooth) - 1 and smooth[end] < standing_angle:
            end += 1

        # Sanity: ensure depth drop is real.
        depth_drop = max(smooth[start], smooth[end]) - smooth[bottom]
        if depth_drop < min_depth_drop:
            continue

        # Avoid overlapping reps (trim end of previous if needed).
        if reps and start <= reps[-1].end:
            start = reps[-1].end + 1
        if start >= bottom:
            continue

        reps.append(Rep(index=len(reps), start=int(start), bottom=int(bottom), end=int(end)))

    return reps


def summarize(reps: list[Rep], fps: float) -> str:
    if not reps:
        return "No reps detected."
    lines = [f"Detected {len(reps)} reps (fps={fps:.1f})"]
    for r in reps:
        t0 = r.start / fps
        t1 = r.end / fps
        lines.append(f"  Rep {r.index + 1}: frames {r.start}-{r.end}  ({t0:.2f}s - {t1:.2f}s)")
    return "\n".join(lines)
