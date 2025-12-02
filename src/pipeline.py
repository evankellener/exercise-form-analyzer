# Suppress Mediapipe + TensorFlow Lite logging BEFORE any imports
import os
import sys
import warnings

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MP_VERBOSE"] = "0"

# Suppress Python warnings
warnings.filterwarnings("ignore")

# Redirect stderr at file descriptor level to suppress all MediaPipe/TensorFlow warnings
# Keep it suppressed globally since we only print to stdout
_null_fd = os.open(os.devnull, os.O_WRONLY)
_original_stderr_fd = os.dup(sys.stderr.fileno())
os.dup2(_null_fd, sys.stderr.fileno())

import argparse
import cv2
import numpy as np
import mediapipe as mp

# Keep stderr suppressed - we only use stdout for output

class SuppressStderr:
    """Context manager to suppress stderr during MediaPipe operations"""
    def __enter__(self):
        os.dup2(_null_fd, sys.stderr.fileno())
        return self
    
    def __exit__(self, *args):
        os.dup2(_original_stderr_fd, sys.stderr.fileno())
        return False

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

POSE_LANDMARKS = mp_pose.PoseLandmark

JOINTS = {
    "left_shoulder": POSE_LANDMARKS.LEFT_SHOULDER,
    "right_shoulder": POSE_LANDMARKS.RIGHT_SHOULDER,
    "left_hip": POSE_LANDMARKS.LEFT_HIP,
    "right_hip": POSE_LANDMARKS.RIGHT_HIP,
    "left_knee": POSE_LANDMARKS.LEFT_KNEE,
    "right_knee": POSE_LANDMARKS.RIGHT_KNEE,
    "left_ankle": POSE_LANDMARKS.LEFT_ANKLE,
    "right_ankle": POSE_LANDMARKS.RIGHT_ANKLE,
    "left_elbow": POSE_LANDMARKS.LEFT_ELBOW,
    "right_elbow": POSE_LANDMARKS.RIGHT_ELBOW,
    "left_wrist": POSE_LANDMARKS.LEFT_WRIST,
    "right_wrist": POSE_LANDMARKS.RIGHT_WRIST,
}

# Squat analysis thresholds
SQUAT_MAX_TORSO_LEAN = 35.0
SQUAT_MIN_KNEE_BEND = 80.0
SQUAT_MAX_KNEE_OFFSET = 0.08

# Pushup analysis thresholds
PUSHUP_MAX_ELBOW_ANGLE_DIFF = 15.0  # Elbows should be roughly symmetric
PUSHUP_MIN_BODY_ALIGNMENT = 160.0  # Body should be close to straight (180 deg)
PUSHUP_MAX_HIP_SAG = 0.05  # Max downward hip deviation
PUSHUP_MAX_HIP_PIKE = -0.05  # Max upward hip deviation (negative = hips too high)

# Issue type constants
ISSUE_FORM_ACCEPTABLE = "Form acceptable"
ISSUE_POSE_UNRELIABLE = "Pose landmarks unreliable"
# Squat issues
ISSUE_FORWARD_LEAN = "Forward torso lean"
ISSUE_SHALLOW_DEPTH = "Shallow squat depth"
ISSUE_KNEE_ALIGNMENT = "Knee alignment issue"
# Pushup issues
ISSUE_ASYMMETRIC_ARMS = "Asymmetric arm position"
ISSUE_HIP_SAG = "Hip sag detected"
ISSUE_HIP_PIKE = "Hip pike detected"
ISSUE_BODY_ALIGNMENT = "Body alignment issue"


def get_landmark_xy(landmarks, idx, w, h):
    lm = landmarks[idx.value]
    return np.array([int(lm.x * w), int(lm.y * h)], dtype=np.float32)


def angle_between_points(a, b, c):
    ba = a - b
    bc = c - b
    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return None

    ba_u = ba / np.linalg.norm(ba)
    bc_u = bc / np.linalg.norm(bc)

    cosang = np.clip(np.dot(ba_u, bc_u), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def torso_lean_angle(shoulder, hip):
    vec = shoulder - hip
    if np.linalg.norm(vec) == 0:
        return None

    vertical = np.array([0, -1.0], dtype=np.float32)
    vec_u = vec / np.linalg.norm(vec)

    cosang = np.clip(np.dot(vec_u, vertical), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def extract_squat_features_from_frame(landmarks, w, h):
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


def extract_pushup_features_from_frame(landmarks, w, h):
    """Extract pushup-relevant features from pose landmarks."""
    left_shoulder = get_landmark_xy(landmarks, JOINTS["left_shoulder"], w, h)
    right_shoulder = get_landmark_xy(landmarks, JOINTS["right_shoulder"], w, h)
    left_elbow = get_landmark_xy(landmarks, JOINTS["left_elbow"], w, h)
    right_elbow = get_landmark_xy(landmarks, JOINTS["right_elbow"], w, h)
    left_wrist = get_landmark_xy(landmarks, JOINTS["left_wrist"], w, h)
    right_wrist = get_landmark_xy(landmarks, JOINTS["right_wrist"], w, h)
    left_hip = get_landmark_xy(landmarks, JOINTS["left_hip"], w, h)
    right_hip = get_landmark_xy(landmarks, JOINTS["right_hip"], w, h)
    left_ankle = get_landmark_xy(landmarks, JOINTS["left_ankle"], w, h)
    right_ankle = get_landmark_xy(landmarks, JOINTS["right_ankle"], w, h)

    # Calculate elbow angles (shoulder-elbow-wrist)
    left_elbow_angle = angle_between_points(left_shoulder, left_elbow, left_wrist)
    right_elbow_angle = angle_between_points(right_shoulder, right_elbow, right_wrist)

    # Calculate body alignment (shoulder-hip-ankle should be straight)
    mid_shoulder = (left_shoulder + right_shoulder) / 2
    mid_hip = (left_hip + right_hip) / 2
    mid_ankle = (left_ankle + right_ankle) / 2
    body_alignment_angle = angle_between_points(mid_shoulder, mid_hip, mid_ankle)

    # Calculate hip sag/pike (deviation from straight line)
    # Check if hips are too high (pike) or too low (sag)
    shoulder_ankle_vec = mid_ankle - mid_shoulder
    shoulder_hip_vec = mid_hip - mid_shoulder
    
    # Project hip onto shoulder-ankle line to find expected position
    # Use a small epsilon to avoid floating-point precision issues
    vec_dot = np.dot(shoulder_ankle_vec, shoulder_ankle_vec)
    if vec_dot > 1e-8:
        t = np.dot(shoulder_hip_vec, shoulder_ankle_vec) / vec_dot
        expected_hip = mid_shoulder + t * shoulder_ankle_vec
        hip_deviation = (mid_hip[1] - expected_hip[1]) / h  # Normalized vertical deviation
    else:
        hip_deviation = 0.0

    return {
        "left_elbow_angle_deg": left_elbow_angle,
        "right_elbow_angle_deg": right_elbow_angle,
        "body_alignment_angle_deg": body_alignment_angle,
        "hip_deviation_norm": hip_deviation,
    }


def evaluate_pushup_frame(features):
    """Evaluate pushup form based on extracted features."""
    left_elbow = features["left_elbow_angle_deg"]
    right_elbow = features["right_elbow_angle_deg"]
    body_alignment = features["body_alignment_angle_deg"]
    hip_deviation = features["hip_deviation_norm"]

    if left_elbow is None or right_elbow is None or body_alignment is None:
        return {"valid": False, "issues": [ISSUE_POSE_UNRELIABLE]}

    issues = []

    # Check elbow symmetry
    if abs(left_elbow - right_elbow) > PUSHUP_MAX_ELBOW_ANGLE_DIFF:
        issues.append(ISSUE_ASYMMETRIC_ARMS)

    # Check body alignment (should be close to 180 degrees for straight line)
    if body_alignment < PUSHUP_MIN_BODY_ALIGNMENT:
        if hip_deviation > PUSHUP_MAX_HIP_SAG:
            issues.append(ISSUE_HIP_SAG)
        elif hip_deviation < PUSHUP_MAX_HIP_PIKE:
            issues.append(ISSUE_HIP_PIKE)
        else:
            issues.append(ISSUE_BODY_ALIGNMENT)

    if not issues:
        issues.append(ISSUE_FORM_ACCEPTABLE)

    return {"valid": True, "issues": issues}


def analyze_pushup_video(path, draw=False, save_debug=False, debug_path="results/pushup_debug.mp4", show_popup=False):
    """Analyze pushup form from video."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    total = 0
    valid = 0
    asymmetric = 0
    hip_sag = 0
    hip_pike = 0
    alignment_issue = 0

    out = None
    os.makedirs("results", exist_ok=True)

    pose_obj = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    if show_popup:
        cv2.namedWindow("Pushup Form Analysis", cv2.WINDOW_NORMAL)

    with pose_obj as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total += 1
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                feats = extract_pushup_features_from_frame(lm, w, h)
                eval_res = evaluate_pushup_frame(feats)

                if eval_res["valid"]:
                    valid += 1
                    issues = eval_res["issues"]

                    if ISSUE_ASYMMETRIC_ARMS in issues:
                        asymmetric += 1
                    if ISSUE_HIP_SAG in issues:
                        hip_sag += 1
                    if ISSUE_HIP_PIKE in issues:
                        hip_pike += 1
                    if ISSUE_BODY_ALIGNMENT in issues:
                        alignment_issue += 1

                if draw or save_debug or show_popup:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

            if show_popup:
                # Add text overlay with current frame info
                cv2.putText(frame, f"Frame: {total}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Pushup Form Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
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
            "message": "No valid pose frames detected"
        }

    return {
        "video_path": path,
        "total_frames": total,
        "valid_frames": valid,
        "asymmetric_arms": asymmetric / valid,
        "hip_sag": hip_sag / valid,
        "hip_pike": hip_pike / valid,
        "alignment_issues": alignment_issue / valid,
        "summary": build_pushup_summary(asymmetric, hip_sag, hip_pike, alignment_issue, valid)
    }


def build_pushup_summary(asymmetric, hip_sag, hip_pike, alignment, valid):
    """Build summary of pushup form issues."""
    issues = []
    if asymmetric / valid > 0.2:
        issues.append("Frequent asymmetric arm position")
    if hip_sag / valid > 0.2:
        issues.append("Frequent hip sag")
    if hip_pike / valid > 0.2:
        issues.append("Frequent hip pike")
    if alignment / valid > 0.2:
        issues.append("Body alignment issues")

    return issues if issues else ["Overall form appears acceptable"]


def print_pushup_summary(res):
    """Print pushup analysis summary."""
    print("\n=== Pushup Analysis Summary ===")
    print(f"Video:            {res['video_path']}")
    print(f"Frames analyzed:  {res['total_frames']}")
    print(f"Valid pose frames: {res['valid_frames']}")
    if "message" in res:
        print(f"Message:          {res['message']}")
    else:
        print(f"Asymmetric arms:  {res['asymmetric_arms']:.2f}")
        print(f"Hip sag:          {res['hip_sag']:.2f}")
        print(f"Hip pike:         {res['hip_pike']:.2f}")
        print(f"Alignment issues: {res['alignment_issues']:.2f}")
        print(f"Issues:           {', '.join(res['summary'])}")


def analyze_squat_video(path, draw=False, save_debug=False, debug_path="results/squat_debug.mp4", show_popup=False):
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

    out = None
    os.makedirs("results", exist_ok=True)

    pose_obj = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
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

            results = pose.process(rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                feats = extract_squat_features_from_frame(lm, w, h)
                eval_res = evaluate_squat_frame(feats)

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
                        mp_pose.POSE_CONNECTIONS
                    )

            if show_popup:
                # Add text overlay with current frame info
                cv2.putText(frame, f"Frame: {total}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Squat Form Analysis", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
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
            "message": "No valid pose frames detected"
        }

    return {
        "video_path": path,
        "total_frames": total,
        "valid_frames": valid,
        "forward_lean": lean / valid,
        "shallow_depth": shallow / valid,
        "knee_alignment": knee_issue / valid,
        "summary": build_summary(lean, shallow, knee_issue, valid)
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


def print_clean_summary(res):
    print("\n=== Squat Analysis Summary ===")
    print(f"Video:            {res['video_path']}")
    print(f"Frames analyzed:  {res['total_frames']}")
    print(f"Valid pose frames: {res['valid_frames']}")
    if "message" in res:
        print(f"Message:          {res['message']}")
    else:
        print(f"Forward lean:     {res['forward_lean']:.2f}")
        print(f"Shallow depth:    {res['shallow_depth']:.2f}")
        print(f"Knee alignment:   {res['knee_alignment']:.2f}")
        print(f"Issues:           {', '.join(res['summary'])}")


def main():
    parser = argparse.ArgumentParser(description="Exercise form analyzer")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--exercise", choices=["squat", "pushup"], default="squat",
                       help="Type of exercise to analyze (default: squat)")
    parser.add_argument("--no-debug-video", action="store_true",
                       help="Skip saving debug video")
    parser.add_argument("--show-popup", action="store_true",
                       help="Show real-time popup window with landmarks overlay")
    args = parser.parse_args()

    debug = not args.no_debug_video
    show_popup = args.show_popup

    if args.exercise == "squat":
        debug_path = "results/squat_debug.mp4"
        res = analyze_squat_video(
            path=args.video,
            draw=debug,
            save_debug=debug,
            debug_path=debug_path,
            show_popup=show_popup
        )
        print_clean_summary(res)
    elif args.exercise == "pushup":
        debug_path = "results/pushup_debug.mp4"
        res = analyze_pushup_video(
            path=args.video,
            draw=debug,
            save_debug=debug,
            debug_path=debug_path,
            show_popup=show_popup
        )
        print_pushup_summary(res)

    if debug and os.path.exists(debug_path):
        print(f"\nDebug video saved to: {debug_path}")


if __name__ == "__main__":
    main()
