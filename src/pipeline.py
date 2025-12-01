"""
Exercise Form Analyzer Pipeline

This module implements the main pose-estimation and rule-based analysis pipeline
for evaluating exercise form, starting with squats.
"""

import argparse
import csv
import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import mediapipe as mp

from utils import (
    angle_between_points,
    compute_knee_offset,
    get_landmark_xy,
    torso_lean_angle,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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

# Configurable thresholds for squat evaluation
SQUAT_THRESHOLDS = {
    "max_torso_lean": 35.0,           # degrees - maximum acceptable forward lean
    "min_knee_bend_for_depth": 80.0,  # degrees - knee angle above this is shallow
    "max_knee_offset": 0.08,          # normalized - max knee deviation from midpoint
    "forward_lean_threshold": 0.2,    # fraction of frames with lean issues
    "shallow_depth_threshold": 0.3,   # fraction of frames with depth issues
    "knee_alignment_threshold": 0.2,  # fraction of frames with alignment issues
}


def extract_squat_features_from_frame(
    landmarks: list,
    image_width: int,
    image_height: int
) -> Dict[str, Optional[float]]:
    """
    Extract minimal squat-related features from MediaPipe landmarks.
    
    Currently uses the left side only for analysis.
    
    Args:
        landmarks: List of MediaPipe pose landmarks.
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.
        
    Returns:
        Dictionary containing torso angle, knee angle, and knee offset.
    """
    left_shoulder = get_landmark_xy(landmarks, JOINTS["left_shoulder"], image_width, image_height)
    left_hip = get_landmark_xy(landmarks, JOINTS["left_hip"], image_width, image_height)
    left_knee = get_landmark_xy(landmarks, JOINTS["left_knee"], image_width, image_height)
    left_ankle = get_landmark_xy(landmarks, JOINTS["left_ankle"], image_width, image_height)

    torso_angle = torso_lean_angle(left_shoulder, left_hip)
    knee_angle = angle_between_points(left_hip, left_knee, left_ankle)
    knee_offset_norm = compute_knee_offset(
        float(left_hip[0]),
        float(left_knee[0]),
        float(left_ankle[0]),
        image_width
    )

    return {
        "torso_angle_deg": torso_angle,
        "knee_angle_deg": knee_angle,
        "knee_offset_norm": knee_offset_norm,
    }


def evaluate_squat_frame(
    features: Dict[str, Optional[float]],
    thresholds: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Apply simple rules to basic squat features for one frame.
    
    Args:
        features: Dictionary containing extracted squat features.
        thresholds: Optional custom thresholds. Uses SQUAT_THRESHOLDS if None.
        
    Returns:
        Dictionary with 'valid' flag and list of 'issues' detected.
    """
    if thresholds is None:
        thresholds = SQUAT_THRESHOLDS
        
    issues: List[str] = []

    torso_angle = features.get("torso_angle_deg")
    knee_angle = features.get("knee_angle_deg")
    knee_offset = features.get("knee_offset_norm")

    max_torso_lean = thresholds.get("max_torso_lean", 35.0)
    min_knee_bend = thresholds.get("min_knee_bend_for_depth", 80.0)
    max_knee_offset = thresholds.get("max_knee_offset", 0.08)

    if torso_angle is None or knee_angle is None:
        return {
            "valid": False,
            "issues": ["Pose landmarks too unreliable in this frame."]
        }

    if torso_angle > max_torso_lean:
        issues.append("Excessive forward torso lean.")

    if knee_angle > min_knee_bend:
        issues.append("Squat depth appears shallow (knee not flexed enough).")

    if knee_offset is not None and abs(knee_offset) > max_knee_offset:
        issues.append("Possible knee misalignment (valgus/varus).")

    if not issues:
        issues.append("Form looks acceptable for this frame (based on simple rules).")

    return {
        "valid": True,
        "issues": issues
    }


def save_frame_data_to_csv(
    frame_data: List[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Save frame-by-frame analysis data to a CSV file.
    
    Args:
        frame_data: List of dictionaries containing per-frame features and evaluations.
        output_path: Path to the output CSV file.
    """
    if not frame_data:
        logger.warning("No frame data to save to CSV.")
        return
        
    fieldnames = [
        "frame_number",
        "torso_angle_deg",
        "knee_angle_deg",
        "knee_offset_norm",
        "valid",
        "issues"
    ]
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in frame_data:
            # Convert issues list to string for CSV
            row_copy = row.copy()
            if "issues" in row_copy and isinstance(row_copy["issues"], list):
                row_copy["issues"] = "; ".join(row_copy["issues"])
            writer.writerow(row_copy)
    
    logger.info(f"Frame data saved to: {output_path}")


def analyze_squat_video(
    video_path: str,
    draw: bool = False,
    save_debug_video: bool = False,
    debug_output_path: str = "results/squat_debug.mp4",
    save_csv: bool = False,
    csv_output_path: str = "results/frame_analysis.csv"
) -> Dict[str, Any]:
    """
    Load a video, run pose estimation frame-by-frame, compute basic squat features,
    and apply simple rules. Returns an aggregate summary.
    
    Args:
        video_path: Path to the input video file.
        draw: Whether to draw pose landmarks on frames.
        save_debug_video: Whether to save a debug video with landmarks.
        debug_output_path: Path for the debug video output.
        save_csv: Whether to save frame-by-frame data to CSV.
        csv_output_path: Path for the CSV output file.
        
    Returns:
        Dictionary containing analysis summary including frame counts and issue fractions.
        
    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If the video cannot be opened.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    logger.info(f"Starting analysis of video: {video_path}")

    frame_count = 0
    valid_frames = 0
    forward_lean_frames = 0
    shallow_depth_frames = 0
    knee_alignment_issue_frames = 0
    frame_data: List[Dict[str, Any]] = []

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

                # Collect frame data for CSV export
                if save_csv:
                    frame_data.append({
                        "frame_number": frame_count,
                        "torso_angle_deg": features.get("torso_angle_deg"),
                        "knee_angle_deg": features.get("knee_angle_deg"),
                        "knee_offset_norm": features.get("knee_offset_norm"),
                        "valid": evaluation["valid"],
                        "issues": evaluation["issues"]
                    })

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

    logger.info(f"Processed {frame_count} frames, {valid_frames} valid pose detections")

    # Save frame data to CSV if requested
    if save_csv and frame_data:
        save_frame_data_to_csv(frame_data, csv_output_path)

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


def main() -> None:
    """Main entry point for the squat form analyzer CLI."""
    parser = argparse.ArgumentParser(
        description="Squat form analyzer using pose estimation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pipeline.py --video data/raw/squat_front.mp4
  python src/pipeline.py --video data/raw/squat_front.mp4 --save-csv
  python src/pipeline.py --video data/raw/squat_front.mp4 --no-debug-video
        """
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video (e.g., data/raw/squat_front.mp4)"
    )
    parser.add_argument(
        "--no-debug-video",
        action="store_true",
        help="Disable saving debug video with pose landmarks."
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save frame-by-frame analysis data to CSV file."
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="results/frame_analysis.csv",
        help="Path for CSV output file (default: results/frame_analysis.csv)"
    )
    parser.add_argument(
        "--debug-output",
        type=str,
        default="results/squat_debug.mp4",
        help="Path for debug video output (default: results/squat_debug.mp4)"
    )
    args = parser.parse_args()

    debug = not args.no_debug_video

    try:
        summary = analyze_squat_video(
            video_path=args.video,
            draw=debug,
            save_debug_video=debug,
            debug_output_path=args.debug_output,
            save_csv=args.save_csv,
            csv_output_path=args.csv_output
        )

        logger.info("=== Squat Analysis Summary ===")
        for k, v in summary.items():
            logger.info(f"{k}: {v}")

        if debug and os.path.exists(args.debug_output):
            logger.info(f"Video with landmarks saved to: {args.debug_output}")
            
        if args.save_csv and os.path.exists(args.csv_output):
            logger.info(f"Frame analysis data saved to: {args.csv_output}")
            
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        raise SystemExit(1)
    except RuntimeError as e:
        logger.error(f"Error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()