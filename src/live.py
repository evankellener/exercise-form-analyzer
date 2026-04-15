"""Live webcam mode.

Streams the default camera through MediaPipe Pose, segments reps online via
a knee-angle state machine, and classifies each rep the moment it ends using
the trained multi-label model. Draws a live HUD on top of the webcam feed.

Usage:
    python -m src.live                          # default webcam, rf model
    python -m src.live --model logreg
    python -m src.live --camera 1               # different camera index
    python -m src.live --record session.mp4     # save annotated session to disk
    python -m src.live --no-mirror              # show raw feed (default is mirrored)

Press q to quit.
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .features import extract_features
from .model import MultiLabelModel, LABEL_NAMES, MODEL_DIR
from .pose import N_LANDMARKS
from .utils import SuppressStderr, mp_drawing, mp_pose


# ---- state machine --------------------------------------------------------
STATE_STANDING = "standing"
STATE_DESCENDING = "descending"
STATE_ASCENDING = "ascending"

STANDING_ANGLE = 160.0   # knee angle above this = standing
BOTTOM_ENTRY_ANGLE = 115.0  # need to get below this before counting as a real rep
MIN_REP_FRAMES = 15      # ignore reps shorter than this (jitter)
BUFFER_FRAMES = 600      # ~20s at 30fps, more than enough for one rep
VERDICT_HOLD_SECONDS = 2.5


def _knee_angle(landmarks: np.ndarray) -> float:
    """Compute mean-left-right knee angle from a (33, 4) landmark array."""
    from .features import _angle
    xy = landmarks[None, :, :2]  # (1, 33, 2)
    l_angle = _angle(xy[:, 23], xy[:, 25], xy[:, 27])[0]
    r_angle = _angle(xy[:, 24], xy[:, 26], xy[:, 28])[0]
    return float((l_angle + r_angle) / 2.0)


class LiveRepTracker:
    """Online rep segmenter. Feed it one (landmarks, frame_index) per frame;
    when a rep completes, `pop_completed_rep()` returns its (start, bottom, end)
    frame indices and the list of landmark arrays for feature extraction."""

    def __init__(self):
        self.state = STATE_STANDING
        self.rep_start = None
        self.rep_bottom_frame = None
        self.rep_bottom_angle = 180.0
        self.went_below_threshold = False
        self.completed: list[dict] = []  # queue of finished reps
        self.buffer: deque = deque(maxlen=BUFFER_FRAMES)  # (frame_idx, landmarks)

    def update(self, frame_idx: int, landmarks: np.ndarray | None) -> tuple[str, float | None]:
        """Return (state, current_knee_angle_or_None)."""
        if landmarks is None:
            return self.state, None

        self.buffer.append((frame_idx, landmarks.copy()))
        angle = _knee_angle(landmarks)

        if self.state == STATE_STANDING:
            if angle < STANDING_ANGLE - 5:
                self.state = STATE_DESCENDING
                self.rep_start = frame_idx
                self.rep_bottom_angle = angle
                self.rep_bottom_frame = frame_idx
                self.went_below_threshold = False
        elif self.state == STATE_DESCENDING:
            if angle < self.rep_bottom_angle:
                self.rep_bottom_angle = angle
                self.rep_bottom_frame = frame_idx
            if angle < BOTTOM_ENTRY_ANGLE:
                self.went_below_threshold = True
            # transition to ascending when angle starts rising again
            if angle > self.rep_bottom_angle + 5:
                self.state = STATE_ASCENDING
        elif self.state == STATE_ASCENDING:
            if angle > STANDING_ANGLE:
                # rep ended
                rep_len = frame_idx - (self.rep_start or frame_idx)
                if rep_len >= MIN_REP_FRAMES and self.went_below_threshold:
                    # Gather the buffered landmarks for this rep
                    rep_frames = [
                        lm for (fi, lm) in self.buffer if self.rep_start <= fi <= frame_idx
                    ]
                    self.completed.append({
                        "start": self.rep_start,
                        "bottom": self.rep_bottom_frame,
                        "end": frame_idx,
                        "landmarks": np.stack(rep_frames) if rep_frames else None,
                    })
                # reset for next rep
                self.state = STATE_STANDING
                self.rep_start = None
                self.rep_bottom_frame = None
                self.rep_bottom_angle = 180.0
                self.went_below_threshold = False

        return self.state, angle

    def pop_completed_rep(self) -> dict | None:
        return self.completed.pop(0) if self.completed else None


# ---- HUD drawing ----------------------------------------------------------
def draw_hud(
    frame: np.ndarray,
    counted: int,
    detected: int,
    state: str,
    angle: float | None,
    last_verdict: dict | None,
    verdict_hold_until: float,
) -> None:
    h, w = frame.shape[:2]
    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.putText(frame, f"Counted: {counted}  (detected: {detected})",
                (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
    state_text = f"State: {state}" + (f"   knee: {angle:.0f}°" if angle is not None else "")
    cv2.putText(frame, state_text, (16, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # Verdict banner (bottom, held for a few seconds after rep end)
    if last_verdict is not None and time.time() < verdict_hold_until:
        color = (0, 200, 0) if last_verdict["counted"] else (0, 0, 255)
        label = "COUNTED" if last_verdict["counted"] else "NOT COUNTED"
        rep_num = last_verdict["rep"]
        text = f"Rep {rep_num}: {label}"
        faults = last_verdict.get("faults", [])
        if faults:
            text += "  —  " + ", ".join(faults)
        cv2.rectangle(frame, (0, h - 90), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, text, (16, h - 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        proba = last_verdict.get("proba", [0, 0, 0])
        ptxt = "  ".join(f"{n}={proba[j]:.2f}" for j, n in enumerate(LABEL_NAMES))
        cv2.putText(frame, ptxt, (16, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


# ---- main loop ------------------------------------------------------------
def run(
    model: MultiLabelModel,
    camera_index: int = 0,
    mirror: bool = True,
    record_path: str | None = None,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    writer = None
    if record_path:
        Path(record_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            record_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    tracker = LiveRepTracker()
    counted = 0
    detected = 0
    last_verdict: dict | None = None
    verdict_hold_until = 0.0

    window = "SquatCoach — live"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    pose_obj = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,   # fastest for live
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0
    try:
        with pose_obj as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with SuppressStderr():
                    res = pose.process(rgb)

                landmarks = None
                if res.pose_landmarks:
                    landmarks = np.array(
                        [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark],
                        dtype=np.float32,
                    )
                    mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                state, angle = tracker.update(frame_idx, landmarks)

                # Classify any newly completed rep
                done = tracker.pop_completed_rep()
                while done is not None:
                    detected += 1
                    if done["landmarks"] is not None and done["landmarks"].shape[0] >= 4:
                        feats = extract_features(done["landmarks"], fps).values
                        pred = model.predict(feats[None, :])[0]
                        proba = model.predict_proba(feats[None, :])[0]
                        faults = [LABEL_NAMES[j] for j in range(3) if pred[j]]
                        is_counted = int(not faults)
                        if is_counted:
                            counted += 1
                        last_verdict = {
                            "rep": detected,
                            "counted": bool(is_counted),
                            "faults": faults,
                            "proba": proba.tolist(),
                        }
                        verdict_hold_until = time.time() + VERDICT_HOLD_SECONDS
                    done = tracker.pop_completed_rep()

                draw_hud(frame, counted, detected, state, angle,
                         last_verdict, verdict_hold_until)

                if writer is not None:
                    writer.write(frame)
                cv2.imshow(window, frame)

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print(f"\nSession: detected {detected} reps, counted {counted}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rf", "logreg"], default="rf")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-mirror", action="store_true",
                    help="Disable horizontal flip of webcam feed.")
    ap.add_argument("--record", default=None,
                    help="Save the annotated session to this path (e.g. results/session.mp4).")
    args = ap.parse_args()

    model_path = MODEL_DIR / f"squat_multilabel_{args.model}.pkl"
    if not model_path.exists():
        raise SystemExit(
            f"No trained model at {model_path}. Run `python -m src.batch --train` first."
        )
    model = MultiLabelModel.load(model_path)

    run(
        model=model,
        camera_index=args.camera,
        mirror=not args.no_mirror,
        record_path=args.record,
    )


if __name__ == "__main__":
    main()
