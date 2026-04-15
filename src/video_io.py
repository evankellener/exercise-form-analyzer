"""Video I/O helpers.

iPhone / Android .mov and .mp4 files store display rotation in metadata
(e.g. 90 / 180 / 270 degrees). OpenCV's VideoCapture does NOT auto-apply
this rotation, so frames come back sideways or upside-down. We read the
rotation metadata once per file and rotate every frame before returning it.
"""
from __future__ import annotations

import cv2
import numpy as np


def _get_rotation(cap: cv2.VideoCapture) -> int:
    """Return clockwise rotation in degrees (0, 90, 180, 270)."""
    if not hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
        return 0
    try:
        r = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0)
    except Exception:
        return 0
    return r % 360


def _apply_rotation(frame: np.ndarray, rot: int) -> np.ndarray:
    if rot == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


class RotatingCapture:
    """Wraps cv2.VideoCapture and auto-rotates frames per the video's metadata.

    Exposes the subset of the VideoCapture API we actually use.
    """

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.rotation = _get_rotation(self.cap)
        # On rotation, reported width/height stay as stored; we want displayed dims.
        self._w0 = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h0 = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        ret, frame = self.cap.read()
        if ret and self.rotation:
            frame = _apply_rotation(frame, self.rotation)
        return ret, frame

    @property
    def fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def width(self) -> int:
        return self._h0 if self.rotation in (90, 270) else self._w0

    @property
    def height(self) -> int:
        return self._w0 if self.rotation in (90, 270) else self._h0

    def set_frame(self, idx: int) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def release(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.release()
        return False
