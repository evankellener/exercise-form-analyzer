"""Pose extraction: run MediaPipe over a video, cache per-frame landmarks.

Output format: (T, 33, 4) numpy array — T frames, 33 landmarks, (x, y, z, visibility).
Coordinates are NORMALIZED (x, y in [0, 1], z relative to hips). This lets us work
view/resolution-independent for downstream features and augmentation.

Cache key is the video path; results go to data/interim/poses/<stem>.npz.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import cv2
import numpy as np

from .utils import SuppressStderr, mp_pose
from .video_io import RotatingCapture


CACHE_DIR = Path("data/interim/poses")
N_LANDMARKS = 33  # MediaPipe Pose outputs 33 keypoints


def _cache_path(video_path: str | Path) -> Path:
    video_path = Path(video_path)
    # Stem plus short hash of full path so identical filenames in different folders don't collide.
    digest = hashlib.md5(str(video_path.resolve()).encode()).hexdigest()[:8]
    return CACHE_DIR / f"{video_path.stem}__{digest}.npz"


def extract_pose_sequence(
    video_path: str | Path,
    use_cache: bool = True,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> dict:
    """Run MediaPipe over a video and return landmark sequence + metadata.

    Returns dict with keys:
        landmarks: (T, 33, 4) np.float32 — (x, y, z, visibility) in normalized coords
        valid:     (T,) bool — True where pose was detected
        fps:       float
        width:     int
        height:    int
        n_frames:  int
    """
    video_path = Path(video_path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(video_path)

    if use_cache and cache.exists():
        data = np.load(cache, allow_pickle=False)
        return {
            "landmarks": data["landmarks"],
            "valid": data["valid"],
            "fps": float(data["fps"]),
            "width": int(data["width"]),
            "height": int(data["height"]),
            "n_frames": int(data["n_frames"]),
            "video_path": str(video_path),
            "from_cache": True,
        }

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = RotatingCapture(str(video_path))
    fps = cap.fps
    width = cap.width
    height = cap.height

    landmarks_list: list[np.ndarray] = []
    valid_list: list[bool] = []

    pose_obj = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    with pose_obj as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with SuppressStderr():
                res = pose.process(rgb)

            if res.pose_landmarks:
                arr = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark],
                    dtype=np.float32,
                )
                landmarks_list.append(arr)
                valid_list.append(True)
            else:
                landmarks_list.append(np.zeros((N_LANDMARKS, 4), dtype=np.float32))
                valid_list.append(False)

    cap.release()

    if not landmarks_list:
        raise RuntimeError(f"No frames read from {video_path}")

    landmarks = np.stack(landmarks_list, axis=0)
    valid = np.array(valid_list, dtype=bool)

    np.savez_compressed(
        cache,
        landmarks=landmarks,
        valid=valid,
        fps=np.float32(fps),
        width=np.int32(width),
        height=np.int32(height),
        n_frames=np.int32(len(landmarks_list)),
    )

    return {
        "landmarks": landmarks,
        "valid": valid,
        "fps": fps,
        "width": width,
        "height": height,
        "n_frames": len(landmarks_list),
        "video_path": str(video_path),
        "from_cache": False,
    }


# ---- landmark index helpers (33-point MediaPipe Pose) --------------------
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_WRIST, R_WRIST = 15, 16
L_ELBOW, R_ELBOW = 13, 14

# Left/right mirror swap pairs (for horizontal-flip augmentation).
# All non-listed indices map to themselves.
LR_PAIRS: list[tuple[int, int]] = [
    (1, 4), (2, 5), (3, 6), (7, 8),        # face
    (9, 10),                                # mouth
    (11, 12), (13, 14), (15, 16),           # arms
    (17, 18), (19, 20), (21, 22),           # hands
    (23, 24), (25, 26), (27, 28),           # hips/knees/ankles
    (29, 30), (31, 32),                     # feet
]


def batch_extract(video_paths: list[str | Path], use_cache: bool = True) -> dict[str, dict]:
    """Convenience: extract poses for many videos, return {path_str: result}."""
    out = {}
    for p in video_paths:
        out[str(p)] = extract_pose_sequence(p, use_cache=use_cache)
    return out
