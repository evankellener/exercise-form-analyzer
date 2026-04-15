import os

import cv2
import numpy as np

from .utils import (
    JOINTS, ISSUE_FORM_ACCEPTABLE, ISSUE_POSE_UNRELIABLE,
    SuppressStderr, get_landmark_xy, angle_between_points, torso_lean_angle,
    mp_drawing, mp_pose,
)

# Squat analysis thresholds
SQUAT_MAX_TORSO_LEAN = 35.0
SQUAT_MIN_KNEE_BEND = 90.0
SQUAT_MAX_KNEE_OFFSET = 0.08

# Only evaluate form when knee angle is below this (i.e., actively squatting)
SQUAT_ACTIVE_THRESHOLD = 160.0

# Issue type constants
ISSUE_FORWARD_LEAN = "Forward torso lean"
ISSUE_SHALLOW_DEPTH = "Shallow squat depth"
ISSUE_KNEE_ALIGNMENT = "Knee alignment issue"

# Rep counting thresholds (knee angle)
REP_STANDING_ANGLE = 160.0   # knee angle when standing
REP_BOTTOM_ANGLE = 110.0     # knee angle at bottom of squat


def extract_squat_features(landmarks, w, h):
    shoulder = get_landmark_xy(landmarks, JOINTS["left_shoulder"], w, h)
    hip = get_landmark_xy(landmarks, JOINTS["left_hip"], w, h)
    knee = get_landmark_xy(landmarks, JOINTS["left_knee"], w, h)
    ankle = get_landmark_xy(landmarks, JOINTS["left_ankle"], w, h)

    torso = torso_lean_angle(shoulder, hip)
    knee_angle = angle_between_points(hip, knee, ankle)

    mid_ha = 0.5 * (hip[0] + ankle[0])
    knee_offset_norm = (knee[0] - mid_ha) / w

    return {
        "torso_angle_deg": torso,
        "knee_angle_deg": knee_angle,
        "knee_offset_norm": knee_offset_norm,
    }


def evaluate_squat_frame(features):
    torso = features["torso_angle_deg"]
    knee_angle = features["knee_angle_deg"]
    knee_offset = features["knee_offset_norm"]

    if torso is None or knee_angle is None:
        return {"valid": False, "issues": [ISSUE_POSE_UNRELIABLE]}

    # Skip form evaluation when person is standing/transitioning
    if knee_angle > SQUAT_ACTIVE_THRESHOLD:
        return {"valid": False, "issues": []}

    issues = []

    if torso > SQUAT_MAX_TORSO_LEAN:
        issues.append(ISSUE_FORWARD_LEAN)

    if knee_angle > SQUAT_MIN_KNEE_BEND:
        issues.append(ISSUE_SHALLOW_DEPTH)

    if abs(knee_offset) > SQUAT_MAX_KNEE_OFFSET:
        issues.append(ISSUE_KNEE_ALIGNMENT)

    if not issues:
        issues.append(ISSUE_FORM_ACCEPTABLE)

    return {"valid": True, "issues": issues}


class RepCounter:
    """Counts squat reps using knee angle transitions."""
    def __init__(self):
        self.state = "up"  # "up" or "down"
        self.count = 0

    def update(self, knee_angle):
        if knee_angle is None:
            return self.count
        if self.state == "up" and knee_angle < REP_BOTTOM_ANGLE:
            self.state = "down"
        elif self.state == "down" and knee_angle > REP_STANDING_ANGLE:
            self.state = "up"
            self.count += 1
        return self.count


def analyze_squat_video(path, draw=False, save_debug=False,
                        debug_path="results/squat_debug.mp4", show_popup=False):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    total = 0
    valid = 0
    lean = 0
    shallow = 0
    knee_issue = 0
    rep_counter = RepCounter()

    out = None
    os.makedirs("results", exist_ok=True)

    pose_obj = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    if show_popup:
        cv2.namedWindow("Squat Form Analysis", cv2.WINDOW_NORMAL)

    with pose_obj as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total += 1
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with SuppressStderr():
                results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                feats = extract_squat_features(lm, w, h)
                eval_res = evaluate_squat_frame(feats)

                rep_counter.update(feats["knee_angle_deg"])

                if eval_res["valid"]:
                    valid += 1
                    issues = eval_res["issues"]

                    if ISSUE_FORWARD_LEAN in issues:
                        lean += 1
                    if ISSUE_SHALLOW_DEPTH in issues:
                        shallow += 1
                    if ISSUE_KNEE_ALIGNMENT in issues:
                        knee_issue += 1

                if draw or save_debug or show_popup:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                    )

            if show_popup:
                cv2.putText(frame, f"Frame: {total}  Reps: {rep_counter.count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Squat Form Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if save_debug and out is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(debug_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

            if save_debug and out is not None:
                out.write(frame)

        cap.release()
        if out is not None:
            out.release()
        if show_popup:
            cv2.destroyAllWindows()

    if valid == 0:
        return {
            "video_path": path,
            "total_frames": total,
            "valid_frames": 0,
            "reps": 0,
            "message": "No valid pose frames detected",
        }

    return {
        "video_path": path,
        "total_frames": total,
        "valid_frames": valid,
        "reps": rep_counter.count,
        "forward_lean": lean / valid,
        "shallow_depth": shallow / valid,
        "knee_alignment": knee_issue / valid,
        "summary": build_summary(lean, shallow, knee_issue, valid),
    }


def build_summary(lean, shallow, knee, valid):
    issues = []
    if lean / valid > 0.2:
        issues.append("Frequent forward torso lean")
    if shallow / valid > 0.3:
        issues.append("Squat depth often shallow")
    if knee / valid > 0.2:
        issues.append("Knee alignment issues")
    return issues if issues else ["Overall form appears acceptable"]


def print_summary(res):
    print("\n=== Squat Analysis Summary ===")
    print(f"Video:            {res['video_path']}")
    print(f"Frames analyzed:  {res['total_frames']}")
    print(f"Valid pose frames: {res['valid_frames']}")
    print(f"Reps detected:    {res['reps']}")
    if "message" in res:
        print(f"Message:          {res['message']}")
    else:
        print(f"Forward lean:     {res['forward_lean']:.2f}")
        print(f"Shallow depth:    {res['shallow_depth']:.2f}")
        print(f"Knee alignment:   {res['knee_alignment']:.2f}")
        print(f"Issues:           {', '.join(res['summary'])}")
