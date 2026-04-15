"""Side-by-side annotated video output.

Left pane:  raw frame.
Right pane: same frame + MediaPipe skeleton + live HUD (rep count, current rep
            verdict, active fault flags).

Rep verdict is set when the rep finishes: uses the trained multi-label ML
model on the rep's features to decide counted / not-counted and which faults
are active. During a rep (between start and end), the HUD shows "analyzing...".
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .features import extract_features
from .model import MultiLabelModel, LABEL_NAMES
from .pose import extract_pose_sequence, N_LANDMARKS
from .segment import detect_reps, Rep
from .utils import mp_drawing, mp_pose
from .video_io import RotatingCapture


FAULT_COLOR = (0, 0, 255)       # BGR red
GOOD_COLOR = (0, 200, 0)        # BGR green
NEUTRAL_COLOR = (200, 200, 200)
BANNER_H = 110


def _landmark_proto_from_array(arr: np.ndarray):
    """Rebuild a mediapipe NormalizedLandmarkList from a (33, 4) numpy array so
    mp_drawing.draw_landmarks can render it without re-running pose."""
    from mediapipe.framework.formats import landmark_pb2
    lm_list = landmark_pb2.NormalizedLandmarkList()
    for i in range(arr.shape[0]):
        lm = lm_list.landmark.add()
        lm.x, lm.y, lm.z, lm.visibility = float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2]), float(arr[i, 3])
    return lm_list


def _draw_banner(img: np.ndarray, text_lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    h = img.shape[0]
    cv2.rectangle(img, (0, 0), (img.shape[1], BANNER_H), (20, 20, 20), -1)
    y = 32
    for text, color in text_lines:
        cv2.putText(img, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 32


def annotate_video(
    video_path: str | Path,
    model: Optional[MultiLabelModel],
    out_path: str | Path,
) -> dict:
    """Write a side-by-side annotated video. Returns per-rep verdicts."""
    video_path = Path(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pose_data = extract_pose_sequence(video_path)
    reps: list[Rep] = detect_reps(
        pose_data["landmarks"], pose_data["valid"], pose_data["fps"]
    )

    # Pre-compute per-rep verdicts (done once, used for every frame in that rep).
    rep_verdicts: dict[int, dict] = {}
    for r in reps:
        seq = pose_data["landmarks"][r.start:r.end + 1]
        if seq.shape[0] < 4:
            continue
        feats = extract_features(seq, pose_data["fps"]).values
        if model is not None:
            pred = model.predict(feats[None, :])[0]
            proba = model.predict_proba(feats[None, :])[0]
        else:
            from .rules import classify_batch
            pred = classify_batch(feats[None, :])[0]
            proba = pred.astype(np.float32)
        faults = [LABEL_NAMES[j] for j in range(len(LABEL_NAMES)) if pred[j]]
        rep_verdicts[r.index] = {
            "rep": r,
            "faults": faults,
            "proba": proba.tolist(),
            "counted": int(len(faults) == 0),
        }

    # Map frame -> active rep index (or None for between-rep frames).
    frame_rep_idx: list[Optional[int]] = [None] * pose_data["n_frames"]
    for r in reps:
        for t in range(r.start, r.end + 1):
            if 0 <= t < len(frame_rep_idx):
                frame_rep_idx[t] = r.index

    cap = RotatingCapture(str(video_path))
    w, h = cap.width, cap.height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, pose_data["fps"], (w * 2, h + BANNER_H))

    counted_so_far = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw = frame.copy()
        annotated = frame.copy()

        if frame_idx < pose_data["landmarks"].shape[0] and pose_data["valid"][frame_idx]:
            lm_proto = _landmark_proto_from_array(pose_data["landmarks"][frame_idx])
            mp_drawing.draw_landmarks(annotated, lm_proto, mp_pose.POSE_CONNECTIONS)

        # Recompute the counted-so-far tally: a rep adds to count only after its end frame.
        counted_so_far = sum(
            1 for r in reps if r.end < frame_idx and rep_verdicts.get(r.index, {}).get("counted")
        )

        active = frame_rep_idx[frame_idx] if frame_idx < len(frame_rep_idx) else None
        top = (f"counted reps: {counted_so_far}", GOOD_COLOR)

        if active is not None and active in rep_verdicts:
            v = rep_verdicts[active]
            if v["counted"]:
                mid = (f"rep {active + 1}: COUNTED", GOOD_COLOR)
            else:
                mid = (f"rep {active + 1}: NOT COUNTED — {', '.join(v['faults'])}", FAULT_COLOR)
            bot_text = "probs: " + "  ".join(
                f"{n}={v['proba'][j]:.2f}" for j, n in enumerate(LABEL_NAMES)
            )
            bot = (bot_text, NEUTRAL_COLOR)
        else:
            mid = ("(between reps)", NEUTRAL_COLOR)
            bot = ("", NEUTRAL_COLOR)

        # Compose output: two panes + banner on top of annotated pane.
        canvas = np.zeros((h + BANNER_H, w * 2, 3), dtype=np.uint8)
        canvas[BANNER_H:, :w] = raw
        canvas[BANNER_H:, w:] = annotated
        _draw_banner(canvas, [top, mid, bot])
        # Label each pane.
        cv2.putText(canvas, "RAW", (14, BANNER_H + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "ANNOTATED", (w + 14, BANNER_H + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(canvas)
        frame_idx += 1

    cap.release()
    writer.release()

    final_counted = sum(1 for v in rep_verdicts.values() if v["counted"])
    return {
        "video": str(video_path),
        "out_path": str(out_path),
        "fps": pose_data["fps"],
        "n_reps_detected": len(reps),
        "n_reps_counted": final_counted,
        "per_rep": [
            {
                "rep": rv["rep"].index + 1,
                "start": rv["rep"].start,
                "end": rv["rep"].end,
                "faults": rv["faults"],
                "counted": rv["counted"],
                "proba": rv["proba"],
            }
            for rv in rep_verdicts.values()
        ],
    }
