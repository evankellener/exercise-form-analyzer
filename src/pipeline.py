import argparse
import cv2
import mediapipe as mp
import numpy as np
import os

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
}


def get_landmark_xy(landmarks, landmark_index, image_width, image_height):
    """Convert normalized landmark to pixel coordinates (x, y)."""
    lm = landmarks[landmark_index.value]
    x = int(lm.x * image_width)
    y = int(lm.y * image_height)
    return np.array([x, y], dtype=np.float32)


def angle_between_points(a, b, c):
    """
    Compute angle at point b formed by (a-b-c), in degrees.
    a, b, c are 2D numpy arrays [x, y].
    """
    ba = a - b
    bc = c - b
    ba_norm = np.linalg.norm(ba)
    bc_norm = np.linalg.norm(bc)
    if ba_norm == 0 or bc_norm == 0:
        return None
    ba_unit = ba / ba_norm
    bc_unit = bc / bc_norm
    cos_angle = np.clip(np.dot(ba_unit, bc_unit), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def torso_lean_angle(shoulder, hip):
    """
    Compute torso lean relative to vertical.
    0 degrees = perfectly vertical.
    Larger values = more forward/backward lean.
    """
    v = shoulder - hip
    vertical = np.array([0, -1.0], dtype=np.float32)  # up
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return None
    v_unit = v / v_norm
    cos_angle = np.clip(np.dot(v_unit, vertical), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def extract_squat_features_from_frame(landmarks, image_width, image_height):
    """
    Extract minimal squat-related features from MediaPipe landmarks.
    Currently uses the left side only.
    """
    left_shoulder = get_landmark_xy(landmarks, JOINTS["left_shoulder"], image_width, image_height)
    left_hip = get_landmark_xy(landmarks, JOINTS["left_hip"], image_width, image_height)
    left_knee = get_landmark_xy(landmarks, JOINTS["left_knee"], image_width, image_height)
    left_ankle = get_landmark_xy(landmarks, JOINTS["left_ankle"], image_width, image_height)

    torso_angle = torso_lean_angle(left_shoulder, left_hip)
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


def analyze_squat_video(video_path, draw=False, save_debug_video=False, debug_output_path="results/squat_debug.mp4"):
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

    if valid_frames == 0:
        return {
            "video_path": video_path,
            "total_frames": frame_count,
            "valid_frames": 0,
            "message": "No valid pose frames detected; cannot evaluate squat form."
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
    }


def main():
    parser = argparse.ArgumentParser(description="Squat form analyzer using pose estimation.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video (e.g., data/raw/squat_front.mp4)")
    parser.add_argument("--no-debug-video", action="store_true", help="Disable saving debug video.")
    args = parser.parse_args()

    debug = not args.no_debug_video
    # Derive debug output filename from input video
    input_basename = os.path.splitext(os.path.basename(args.video))[0]
    debug_output_path = os.path.join("results", f"{input_basename}_debug.mp4")

    summary = analyze_squat_video(
        video_path=args.video,
        draw=debug,
        save_debug_video=debug,
        debug_output_path=debug_output_path
    )

    print("=== Squat Analysis Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()