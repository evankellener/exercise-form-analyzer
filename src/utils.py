"""
Shared utility functions for exercise form analysis.
"""
import numpy as np
import csv
import os
from datetime import datetime


def angle_between_points(a, b, c):
    """
    Compute angle at point b formed by (a-b-c), in degrees.
    
    Args:
        a: 2D numpy array [x, y] for first point
        b: 2D numpy array [x, y] for vertex point (where angle is measured)
        c: 2D numpy array [x, y] for third point
    
    Returns:
        Angle in degrees, or None if vectors have zero length
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


def angle_from_vertical(point1, point2):
    """
    Compute angle of line segment relative to vertical axis.
    0 degrees = perfectly vertical (point1 directly above point2).
    
    Args:
        point1: 2D numpy array [x, y] for first point (typically upper, e.g., shoulder)
        point2: 2D numpy array [x, y] for second point (typically lower, e.g., hip)
    
    Returns:
        Angle in degrees, or None if vector has zero length
    """
    v = point1 - point2
    vertical = np.array([0, -1.0], dtype=np.float32)  # up
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return None
    v_unit = v / v_norm
    cos_angle = np.clip(np.dot(v_unit, vertical), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def get_landmark_xy(landmarks, landmark_index, image_width, image_height):
    """
    Convert normalized MediaPipe landmark to pixel coordinates.
    
    Args:
        landmarks: MediaPipe pose landmarks list
        landmark_index: PoseLandmark enum value
        image_width: Width of image in pixels
        image_height: Height of image in pixels
    
    Returns:
        numpy array [x, y] in pixel coordinates
    """
    lm = landmarks[landmark_index.value]
    x = int(lm.x * image_width)
    y = int(lm.y * image_height)
    return np.array([x, y], dtype=np.float32)


def get_landmark_visibility(landmarks, landmark_index):
    """
    Get visibility score for a landmark.
    
    Args:
        landmarks: MediaPipe pose landmarks list
        landmark_index: PoseLandmark enum value
    
    Returns:
        Visibility score (0-1)
    """
    lm = landmarks[landmark_index.value]
    return lm.visibility


def save_features_to_csv(features_list, output_path, exercise_type="squat"):
    """
    Save frame-wise features and evaluations to a CSV file.
    
    Args:
        features_list: List of dicts containing frame features and evaluations
        output_path: Path to output CSV file
        exercise_type: Type of exercise (squat, pushup, etc.)
    """
    if not features_list:
        return
    
    # Handle case where output_path has no directory component
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all unique keys from features
    all_keys = set()
    for f in features_list:
        all_keys.update(f.keys())
    
    fieldnames = ["frame_number", "exercise_type"] + sorted(all_keys - {"frame_number", "exercise_type"})
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for f in features_list:
            row = {"exercise_type": exercise_type}
            row.update(f)
            writer.writerow(row)


def generate_output_filename(input_path, suffix, extension, output_dir="results"):
    """
    Generate an output filename based on input file.
    
    Args:
        input_path: Path to input file
        suffix: Suffix to add (e.g., "_debug", "_features")
        extension: Output file extension (e.g., "mp4", "csv")
        output_dir: Directory for output files
    
    Returns:
        Generated output path
    """
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"{base_name}{suffix}_{timestamp}.{extension}")
