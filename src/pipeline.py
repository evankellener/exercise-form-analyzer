import argparse
import cv2
import mediapipe as mp
import numpy as np
import os

from utils import (
    angle_between_points,
    angle_from_vertical,
    get_landmark_xy,
    save_features_to_csv,
    generate_output_filename,
)

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


def extract_squat_features_from_frame(landmarks, image_width, image_height):
    """
    Extract minimal squat-related features from MediaPipe landmarks.
    Currently uses the left side only.
    """
    left_shoulder = get_landmark_xy(landmarks, JOINTS["left_shoulder"], image_width, image_height)
    left_hip = get_landmark_xy(landmarks, JOINTS["left_hip"], image_width, image_height)
    left_knee = get_landmark_xy(landmarks, JOINTS["left_knee"], image_width, image_height)
    left_ankle = get_landmark_xy(landmarks, JOINTS["left_ankle"], image_width, image_height)

    torso_angle = angle_from_vertical(left_shoulder, left_hip)
    knee_angle = angle_between_points(left_hip, left_knee, left_ankle)

    hip_x = left_hip[0]
    knee_x = left_knee[0]
    ankle_x = left_ankle[0]
    mid_ha_x = 0.5 * (hip_x + ankle_x)
    knee_offset_norm = (knee_x - mid_ha_x) / image_width

    return {
        "torso_angle_deg": torso_angle,
        "knee_angle_deg": knee_angle,
        "knee_offset_norm": knee_offset_norm,
    }


def evaluate_squat_frame(features):
    """
    Apply simple rules to basic squat features for one frame.
    Returns a dict of boolean flags and messages.
    """
    issues = []

    torso_angle = features.get("torso_angle_deg")
    knee_angle = features.get("knee_angle_deg")
    knee_offset = features.get("knee_offset_norm")

    MAX_TORSO_LEAN = 35.0       # degrees
    MIN_KNEE_BEND_FOR_DEPTH = 80.0  # degrees, >80 => shallow
    MAX_KNEE_OFFSET = 0.08      # normalized

    if torso_angle is None or knee_angle is None:
        return {
            "valid": False,
            "issues": ["Pose landmarks too unreliable in this frame."]
        }

    if torso_angle > MAX_TORSO_LEAN:
        issues.append("Excessive forward torso lean.")

    if knee_angle > MIN_KNEE_BEND_FOR_DEPTH:
        issues.append("Squat depth appears shallow (knee not flexed enough).")

    if abs(knee_offset) > MAX_KNEE_OFFSET:
        issues.append("Possible knee misalignment (valgus/varus).")

    if not issues:
        issues.append("Form looks acceptable for this frame (based on simple rules).")

    return {
        "valid": True,
        "issues": issues
    }


def extract_pushup_features_from_frame(landmarks, image_width, image_height):
    """
    Extract pushup-related features from MediaPipe landmarks.
    Focuses on arm position, body alignment, and depth.
    """
    left_shoulder = get_landmark_xy(landmarks, JOINTS["left_shoulder"], image_width, image_height)
    left_hip = get_landmark_xy(landmarks, JOINTS["left_hip"], image_width, image_height)
    left_elbow = get_landmark_xy(landmarks, JOINTS["left_elbow"], image_width, image_height)
    left_wrist = get_landmark_xy(landmarks, JOINTS["left_wrist"], image_width, image_height)
    left_ankle = get_landmark_xy(landmarks, JOINTS["left_ankle"], image_width, image_height)

    # Elbow angle (for depth detection)
    elbow_angle = angle_between_points(left_shoulder, left_elbow, left_wrist)

    # Body alignment (shoulder to ankle should be relatively straight)
    body_angle = angle_from_vertical(left_shoulder, left_ankle)

    # Hip sag/pike detection (hip should be in line with shoulder-ankle)
    shoulder_ankle_mid_y = 0.5 * (left_shoulder[1] + left_ankle[1])
    hip_deviation_norm = (left_hip[1] - shoulder_ankle_mid_y) / image_height

    return {
        "elbow_angle_deg": elbow_angle,
        "body_angle_deg": body_angle,
        "hip_deviation_norm": hip_deviation_norm,
    }


def evaluate_pushup_frame(features):
    """
    Apply simple rules to pushup features for one frame.
    Returns a dict of boolean flags and messages.
    """
    issues = []

    elbow_angle = features.get("elbow_angle_deg")
    body_angle = features.get("body_angle_deg")
    hip_deviation = features.get("hip_deviation_norm")

    MIN_ELBOW_ANGLE_BOTTOM = 90.0  # degrees, should be ~90 at bottom
    MAX_HIP_DEVIATION = 0.05       # normalized, hip shouldn't sag or pike too much

    if elbow_angle is None or body_angle is None:
        return {
            "valid": False,
            "issues": ["Pose landmarks too unreliable in this frame."]
        }

    # If elbow angle is too large, might not be going deep enough
    if elbow_angle > 160.0:
        issues.append("Arms appear too extended (at top of movement).")
    elif elbow_angle > MIN_ELBOW_ANGLE_BOTTOM:
        issues.append("Pushup depth may be insufficient (elbows not bent enough).")

    if abs(hip_deviation) > MAX_HIP_DEVIATION:
        if hip_deviation > 0:
            issues.append("Hip appears to be sagging (lower back arched).")
        else:
            issues.append("Hip appears to be piked (hips too high).")

    if not issues:
        issues.append("Form looks acceptable for this frame (based on simple rules).")

    return {
        "valid": True,
        "issues": issues
    }


def analyze_squat_video(video_path, draw=False, save_debug_video=False, debug_output_path="results/squat_debug.mp4", save_csv=False, csv_output_path=None):
    """
    Load a video, run pose estimation frame-by-frame, compute basic squat features,
    and apply simple rules. Returns an aggregate summary.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = 0
    valid_frames = 0
    forward_lean_frames = 0
    shallow_depth_frames = 0
    knee_alignment_issue_frames = 0
    frame_features_list = []

    out_writer = None

    os.makedirs("results", exist_ok=True)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            image_height, image_width = frame.shape[:2]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                features = extract_squat_features_from_frame(landmarks, image_width, image_height)
                evaluation = evaluate_squat_frame(features)

                if evaluation["valid"]:
                    valid_frames += 1
                    msgs = evaluation["issues"]
                    if any("forward torso lean" in m for m in msgs):
                        forward_lean_frames += 1
                    if any("Squat depth appears shallow" in m for m in msgs):
                        shallow_depth_frames += 1
                    if any("knee misalignment" in m for m in msgs):
                        knee_alignment_issue_frames += 1

                # Save frame features for CSV export
                if save_csv:
                    frame_data = {
                        "frame_number": frame_count,
                        **features,
                        "valid": evaluation["valid"],
                        "issues": "; ".join(evaluation["issues"]),
                    }
                    frame_features_list.append(frame_data)

                if draw or save_debug_video:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    )

            if save_debug_video and out_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(
                    debug_output_path,
                    fourcc,
                    cap.get(cv2.CAP_PROP_FPS),
                    (image_width, image_height),
                )

            if save_debug_video and out_writer is not None:
                out_writer.write(frame)

        cap.release()
        if out_writer is not None:
            out_writer.release()

    # Save CSV if requested
    if save_csv and frame_features_list:
        if csv_output_path is None:
            csv_output_path = generate_output_filename(video_path, "_squat_features", "csv")
        save_features_to_csv(frame_features_list, csv_output_path, exercise_type="squat")

    if valid_frames == 0:
        return {
            "video_path": video_path,
            "total_frames": frame_count,
            "valid_frames": 0,
            "message": "No valid pose frames detected; cannot evaluate squat form.",
            "csv_path": csv_output_path if save_csv else None,
        }

    forward_lean_pct = forward_lean_frames / valid_frames
    shallow_depth_pct = shallow_depth_frames / valid_frames
    knee_issue_pct = knee_alignment_issue_frames / valid_frames

    summary_issues = []
    if forward_lean_pct > 0.2:
        summary_issues.append("Frequent forward torso lean detected.")
    if shallow_depth_pct > 0.3:
        summary_issues.append("Squat depth often appears shallow.")
    if knee_issue_pct > 0.2:
        summary_issues.append("Knee alignment issues appear in many frames.")

    if not summary_issues:
        summary_issues.append("Overall squat form appears acceptable based on this simple rule set.")

    return {
        "video_path": video_path,
        "total_frames": frame_count,
        "valid_frames": valid_frames,
        "forward_lean_frame_fraction": forward_lean_pct,
        "shallow_depth_frame_fraction": shallow_depth_pct,
        "knee_alignment_issue_frame_fraction": knee_issue_pct,
        "summary_issues": summary_issues,
        "csv_path": csv_output_path if save_csv else None,
    }


def analyze_pushup_video(video_path, draw=False, save_debug_video=False, debug_output_path="results/pushup_debug.mp4", save_csv=False, csv_output_path=None):
    """
    Load a video, run pose estimation frame-by-frame, compute pushup features,
    and apply simple rules. Returns an aggregate summary.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = 0
    valid_frames = 0
    depth_issue_frames = 0
    hip_issue_frames = 0
    frame_features_list = []

    out_writer = None

    os.makedirs("results", exist_ok=True)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            image_height, image_width = frame.shape[:2]
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                features = extract_pushup_features_from_frame(landmarks, image_width, image_height)
                evaluation = evaluate_pushup_frame(features)

                if evaluation["valid"]:
                    valid_frames += 1
                    msgs = evaluation["issues"]
                    if any("depth" in m.lower() or "bent" in m.lower() for m in msgs):
                        depth_issue_frames += 1
                    if any("hip" in m.lower() or "sag" in m.lower() or "pike" in m.lower() for m in msgs):
                        hip_issue_frames += 1

                # Save frame features for CSV export
                if save_csv:
                    frame_data = {
                        "frame_number": frame_count,
                        **features,
                        "valid": evaluation["valid"],
                        "issues": "; ".join(evaluation["issues"]),
                    }
                    frame_features_list.append(frame_data)

                if draw or save_debug_video:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    )

            if save_debug_video and out_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(
                    debug_output_path,
                    fourcc,
                    cap.get(cv2.CAP_PROP_FPS),
                    (image_width, image_height),
                )

            if save_debug_video and out_writer is not None:
                out_writer.write(frame)

        cap.release()
        if out_writer is not None:
            out_writer.release()

    # Save CSV if requested
    if save_csv and frame_features_list:
        if csv_output_path is None:
            csv_output_path = generate_output_filename(video_path, "_pushup_features", "csv")
        save_features_to_csv(frame_features_list, csv_output_path, exercise_type="pushup")

    if valid_frames == 0:
        return {
            "video_path": video_path,
            "total_frames": frame_count,
            "valid_frames": 0,
            "message": "No valid pose frames detected; cannot evaluate pushup form.",
            "csv_path": csv_output_path if save_csv else None,
        }

    depth_issue_pct = depth_issue_frames / valid_frames
    hip_issue_pct = hip_issue_frames / valid_frames

    summary_issues = []
    if depth_issue_pct > 0.3:
        summary_issues.append("Pushup depth often appears insufficient.")
    if hip_issue_pct > 0.2:
        summary_issues.append("Hip alignment issues (sagging or piking) detected in many frames.")

    if not summary_issues:
        summary_issues.append("Overall pushup form appears acceptable based on this simple rule set.")

    return {
        "video_path": video_path,
        "total_frames": frame_count,
        "valid_frames": valid_frames,
        "depth_issue_frame_fraction": depth_issue_pct,
        "hip_issue_frame_fraction": hip_issue_pct,
        "summary_issues": summary_issues,
        "csv_path": csv_output_path if save_csv else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Exercise form analyzer using pose estimation.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video (e.g., data/raw/squat_front.mp4)")
    parser.add_argument("--exercise", type=str, choices=["squat", "pushup"], default="squat", help="Type of exercise to analyze (default: squat)")
    parser.add_argument("--no-debug-video", action="store_true", help="Disable saving debug video.")
    parser.add_argument("--save-csv", action="store_true", help="Save frame-wise features to CSV file.")
    args = parser.parse_args()

    debug = not args.no_debug_video
    
    if args.exercise == "squat":
        debug_output_path = "results/squat_debug.mp4"
        summary = analyze_squat_video(
            video_path=args.video,
            draw=debug,
            save_debug_video=debug,
            debug_output_path=debug_output_path,
            save_csv=args.save_csv,
        )
        print("=== Squat Analysis Summary ===")
    else:  # pushup
        debug_output_path = "results/pushup_debug.mp4"
        summary = analyze_pushup_video(
            video_path=args.video,
            draw=debug,
            save_debug_video=debug,
            debug_output_path=debug_output_path,
            save_csv=args.save_csv,
        )
        print("=== Pushup Analysis Summary ===")
    
    for k, v in summary.items():
        print(f"{k}: {v}")

    if debug and os.path.exists(debug_output_path):
        print(f"\nVideo with landmarks saved to: {debug_output_path}")
    
    if args.save_csv and summary.get("csv_path"):
        print(f"Features saved to: {summary['csv_path']}")


if __name__ == "__main__":
    main()