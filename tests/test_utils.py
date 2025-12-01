"""
Unit tests for the Exercise Form Analyzer utility functions.
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from utils import (
    angle_between_points,
    compute_knee_offset,
    torso_lean_angle,
)


class TestAngleBetweenPoints:
    """Tests for the angle_between_points function."""

    def test_right_angle(self):
        """Test that a 90-degree angle is correctly calculated."""
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 1.0])
        c = np.array([1.0, 1.0])
        angle = angle_between_points(a, b, c)
        assert angle is not None
        assert abs(angle - 90.0) < 0.01

    def test_straight_line(self):
        """Test that a 180-degree angle is correctly calculated."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        c = np.array([2.0, 0.0])
        angle = angle_between_points(a, b, c)
        assert angle is not None
        assert abs(angle - 180.0) < 0.01

    def test_zero_length_vector(self):
        """Test that None is returned for zero-length vectors."""
        a = np.array([1.0, 1.0])
        b = np.array([1.0, 1.0])  # Same as a, creating zero-length vector
        c = np.array([2.0, 2.0])
        angle = angle_between_points(a, b, c)
        assert angle is None

    def test_acute_angle(self):
        """Test an acute angle (less than 90 degrees)."""
        # Create a 45-degree angle
        a = np.array([0.0, -1.0])  # Below b
        b = np.array([0.0, 0.0])   # Vertex at origin
        c = np.array([1.0, -1.0])  # 45 degrees from ba direction
        angle = angle_between_points(a, b, c)
        assert angle is not None
        assert 44.0 < angle < 46.0  # Approximately 45 degrees


class TestTorsoLeanAngle:
    """Tests for the torso_lean_angle function."""

    def test_perfectly_vertical(self):
        """Test that a vertical torso gives 0 degrees."""
        shoulder = np.array([100.0, 100.0])
        hip = np.array([100.0, 200.0])  # Directly below shoulder
        angle = torso_lean_angle(shoulder, hip)
        assert angle is not None
        assert abs(angle) < 0.01

    def test_forward_lean(self):
        """Test that a forward-leaning torso gives a positive angle."""
        shoulder = np.array([150.0, 100.0])  # Forward of hip
        hip = np.array([100.0, 200.0])
        angle = torso_lean_angle(shoulder, hip)
        assert angle is not None
        assert angle > 0.0

    def test_zero_length_vector(self):
        """Test that None is returned when shoulder and hip are at same position."""
        shoulder = np.array([100.0, 100.0])
        hip = np.array([100.0, 100.0])
        angle = torso_lean_angle(shoulder, hip)
        assert angle is None

    def test_45_degree_lean(self):
        """Test a 45-degree forward lean."""
        shoulder = np.array([100.0, 0.0])
        hip = np.array([0.0, 100.0])
        angle = torso_lean_angle(shoulder, hip)
        assert angle is not None
        assert abs(angle - 45.0) < 0.01


class TestComputeKneeOffset:
    """Tests for the compute_knee_offset function."""

    def test_centered_knee(self):
        """Test that a centered knee gives 0 offset."""
        hip_x = 100.0
        knee_x = 100.0
        ankle_x = 100.0
        image_width = 640
        offset = compute_knee_offset(hip_x, knee_x, ankle_x, image_width)
        assert abs(offset) < 0.001

    def test_knee_offset_right(self):
        """Test that a knee to the right gives a positive offset."""
        hip_x = 100.0
        knee_x = 150.0  # Knee is to the right
        ankle_x = 100.0
        image_width = 640
        offset = compute_knee_offset(hip_x, knee_x, ankle_x, image_width)
        assert offset > 0.0

    def test_knee_offset_left(self):
        """Test that a knee to the left gives a negative offset."""
        hip_x = 100.0
        knee_x = 50.0  # Knee is to the left
        ankle_x = 100.0
        image_width = 640
        offset = compute_knee_offset(hip_x, knee_x, ankle_x, image_width)
        assert offset < 0.0

    def test_offset_normalization(self):
        """Test that offset is properly normalized by image width."""
        hip_x = 100.0
        knee_x = 200.0
        ankle_x = 100.0
        image_width = 1000
        offset = compute_knee_offset(hip_x, knee_x, ankle_x, image_width)
        # Midpoint is 100, knee is 200, difference is 100, normalized by 1000 = 0.1
        assert abs(offset - 0.1) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
