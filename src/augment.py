"""Landmark-sequence augmentations for per-rep training data.

All functions operate on a numpy array of shape (T, 33, 4) = (frames, landmarks, xyzv)
where x, y, z are normalized and v is visibility. Transforms leave visibility untouched
unless noted. They are designed to be label-preserving for the three squat fault types
(forward lean, shallow depth, knee cave) — rotations are kept tiny for that reason.
"""
from __future__ import annotations

import numpy as np

from .pose import LR_PAIRS, N_LANDMARKS


def mirror_lr(seq: np.ndarray) -> np.ndarray:
    """Horizontal flip: x -> 1-x, and swap left/right landmark pairs."""
    out = seq.copy()
    out[..., 0] = 1.0 - out[..., 0]
    for l, r in LR_PAIRS:
        out[:, [l, r]] = out[:, [r, l]]
    return out


def time_warp(seq: np.ndarray, factor: float) -> np.ndarray:
    """Resample the sequence to `factor` times its original length (0.5 .. 2.0).
    Labels are preserved — we just stretch or compress time uniformly."""
    T = seq.shape[0]
    new_T = max(4, int(round(T * factor)))
    src = np.linspace(0, T - 1, new_T)
    idx_lo = np.floor(src).astype(int)
    idx_hi = np.clip(idx_lo + 1, 0, T - 1)
    frac = (src - idx_lo).reshape(-1, 1, 1).astype(np.float32)
    return (1 - frac) * seq[idx_lo] + frac * seq[idx_hi]


def jitter(seq: np.ndarray, sigma: float = 0.004, rng: np.random.Generator | None = None) -> np.ndarray:
    """Additive Gaussian noise on x, y (in normalized coords). Default ~2-3 px for 720p."""
    rng = rng or np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=seq[..., :2].shape).astype(np.float32)
    out = seq.copy()
    out[..., :2] += noise
    return out


def translate(seq: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = seq.copy()
    out[..., 0] += dx
    out[..., 1] += dy
    return out


def scale(seq: np.ndarray, s: float, center: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    out = seq.copy()
    out[..., 0] = center[0] + (out[..., 0] - center[0]) * s
    out[..., 1] = center[1] + (out[..., 1] - center[1]) * s
    return out


def rotate(seq: np.ndarray, deg: float, center: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
    """Rotate landmarks in the XY plane by `deg` degrees about `center`.

    Keep angles small (<= ~5 deg) or forward-lean labels become unreliable.
    """
    rad = np.radians(deg)
    c, s = np.cos(rad), np.sin(rad)
    cx, cy = center
    out = seq.copy()
    x = out[..., 0] - cx
    y = out[..., 1] - cy
    out[..., 0] = cx + c * x - s * y
    out[..., 1] = cy + s * x + c * y
    return out


def dropout_frames(seq: np.ndarray, p: float = 0.05, rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomly drop frames and replace with linear interpolation from neighbors."""
    rng = rng or np.random.default_rng()
    T = seq.shape[0]
    mask = rng.random(T) < p
    if not mask.any() or mask.all():
        return seq.copy()
    keep = np.where(~mask)[0]
    out = seq.copy()
    for t in np.where(mask)[0]:
        # Nearest kept frames on either side.
        left = keep[keep < t]
        right = keep[keep > t]
        if len(left) == 0:
            out[t] = seq[right[0]]
        elif len(right) == 0:
            out[t] = seq[left[-1]]
        else:
            a, b = left[-1], right[0]
            alpha = (t - a) / max(b - a, 1)
            out[t] = (1 - alpha) * seq[a] + alpha * seq[b]
    return out


def augment_one(
    seq: np.ndarray,
    rng: np.random.Generator,
    view: str = "side",
    enable_mirror: bool = True,
) -> tuple[np.ndarray, dict]:
    """Apply a randomized combination of label-preserving transforms.

    Returns (augmented_sequence, metadata_dict_describing_what_was_applied).
    """
    meta = {}
    out = seq.copy()

    if enable_mirror and rng.random() < 0.5:
        out = mirror_lr(out)
        meta["mirror"] = True

    # Time warp within +/- 15%
    f = float(rng.uniform(0.85, 1.15))
    out = time_warp(out, f)
    meta["time_warp"] = round(f, 3)

    # Small translation within +/- 3%
    dx = float(rng.uniform(-0.03, 0.03))
    dy = float(rng.uniform(-0.03, 0.03))
    out = translate(out, dx, dy)
    meta["translate"] = (round(dx, 4), round(dy, 4))

    # Scale within +/- 5%
    s = float(rng.uniform(0.95, 1.05))
    out = scale(out, s)
    meta["scale"] = round(s, 3)

    # Tiny rotation (up to +/- 3 deg).
    rot = float(rng.uniform(-3.0, 3.0))
    out = rotate(out, rot)
    meta["rotate"] = round(rot, 2)

    # Landmark jitter.
    out = jitter(out, sigma=0.004, rng=rng)
    meta["jitter_sigma"] = 0.004

    # Occasional frame dropout.
    if rng.random() < 0.3:
        out = dropout_frames(out, p=0.05, rng=rng)
        meta["dropout"] = 0.05

    return out, meta


def generate_augmented(
    seq: np.ndarray,
    n: int,
    view: str = "side",
    seed: int | None = None,
) -> list[tuple[np.ndarray, dict]]:
    """Produce `n` augmented copies of a single rep sequence."""
    rng = np.random.default_rng(seed)
    return [augment_one(seq, rng, view=view) for _ in range(n)]
