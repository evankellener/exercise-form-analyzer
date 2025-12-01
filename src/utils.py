"""
Utility functions for the Exercise Form Analyzer.

This module contains shared helper functions for geometric calculations,
landmark extraction, and other common operations used across the pipeline.
"""

import numpy as np
import mediapipe as mp
from typing import Optional, Tuple

mp_pose = mp.solutions.pose
POSE_LANDMARKS = mp_pose.PoseLandmark


def get_landmark_xy(
    landmarks: list,
    landmark_index: POSE_LANDMARKS,
    image_width: int,
    image_height: int
) -> np.ndarray:
    """
    Convert normalized MediaPipe landmark to pixel coordinates (x, y).
    
    Args:
        landmarks: List of MediaPipe pose landmarks.
        landmark_index: The PoseLandmark enum value for the desired landmark.
        image_width: Width of the source image in pixels.
        image_height: Height of the source image in pixels.
        
    Returns:
        A numpy array [x, y] with pixel coordinates as float32.
    """
    lm = landmarks[landmark_index.value]
    x = lm.x * image_width
    y = lm.y * image_height
    return np.array([x, y], dtype=np.float32)


def angle_between_points(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray
) -> Optional[float]:
    """
    Compute the angle at point b formed by the vectors (a-b) and (c-b), in degrees.
    
    Args:
        a: 2D numpy array [x, y] for the first point.
        b: 2D numpy array [x, y] for the vertex point.
        c: 2D numpy array [x, y] for the third point.
        
    Returns:
        The angle in degrees, or None if vectors have zero length.
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
    angle_deg = float(np.degrees(angle_rad))
    return angle_deg


def torso_lean_angle(
    shoulder: np.ndarray,
    hip: np.ndarray
) -> Optional[float]:
    """
    Compute the torso lean angle relative to vertical.
    
    0 degrees indicates a perfectly vertical torso.
    Larger values indicate more forward/backward lean.
    
    Args:
        shoulder: 2D numpy array [x, y] for the shoulder position.
        hip: 2D numpy array [x, y] for the hip position.
        
    Returns:
        The lean angle in degrees, or None if the vector has zero length.
    """
    v = shoulder - hip
    vertical = np.array([0, -1.0], dtype=np.float32)  # up direction
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return None
    v_unit = v / v_norm
    cos_angle = np.clip(np.dot(v_unit, vertical), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = float(np.degrees(angle_rad))
    return angle_deg


def compute_knee_offset(
    hip_x: float,
    knee_x: float,
    ankle_x: float,
    image_width: int
) -> float:
    """
    Compute the normalized knee offset from the midpoint between hip and ankle.
    
    This helps detect knee valgus (inward collapse) or varus (outward deviation).
    
    Args:
        hip_x: X coordinate of the hip in pixels.
        knee_x: X coordinate of the knee in pixels.
        ankle_x: X coordinate of the ankle in pixels.
        image_width: Width of the image in pixels for normalization.
        
    Returns:
        Normalized knee offset (positive = knee is to the right of midpoint).
    """
    mid_ha_x = 0.5 * (hip_x + ankle_x)
    knee_offset_norm = (knee_x - mid_ha_x) / image_width
    return knee_offset_norm


def get_landmark_visibility(
    landmarks: list,
    landmark_index: POSE_LANDMARKS
) -> float:
    """
    Get the visibility score for a specific landmark.
    
    Args:
        landmarks: List of MediaPipe pose landmarks.
        landmark_index: The PoseLandmark enum value for the desired landmark.
        
    Returns:
        Visibility score between 0 and 1.
    """
    lm = landmarks[landmark_index.value]
    return float(lm.visibility)
