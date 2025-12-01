"""
Unit tests for the Exercise Form Analyzer pipeline functions.
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from pipeline import (
    SQUAT_THRESHOLDS,
    evaluate_squat_frame,
)


class TestEvaluateSquatFrame:
    """Tests for the evaluate_squat_frame function."""

    def test_good_form(self):
        """Test that good form is detected correctly."""
        features = {
            "torso_angle_deg": 10.0,  # Within threshold
            "knee_angle_deg": 60.0,   # Deep squat, below threshold
            "knee_offset_norm": 0.02, # Small offset
        }
        result = evaluate_squat_frame(features)
        assert result["valid"] is True
        assert "Form looks acceptable" in result["issues"][0]

    def test_excessive_forward_lean(self):
        """Test that excessive forward lean is detected."""
        features = {
            "torso_angle_deg": 50.0,  # Above threshold (35 degrees)
            "knee_angle_deg": 60.0,   # Good depth
            "knee_offset_norm": 0.02, # Good alignment
        }
        result = evaluate_squat_frame(features)
        assert result["valid"] is True
        assert any("forward torso lean" in issue for issue in result["issues"])

    def test_shallow_squat(self):
        """Test that shallow squat depth is detected."""
        features = {
            "torso_angle_deg": 20.0,  # Good lean
            "knee_angle_deg": 100.0,  # Above threshold (80 degrees = shallow)
            "knee_offset_norm": 0.02, # Good alignment
        }
        result = evaluate_squat_frame(features)
        assert result["valid"] is True
        assert any("shallow" in issue.lower() for issue in result["issues"])

    def test_knee_misalignment(self):
        """Test that knee misalignment is detected."""
        features = {
            "torso_angle_deg": 20.0,  # Good lean
            "knee_angle_deg": 60.0,   # Good depth
            "knee_offset_norm": 0.15, # Above threshold (0.08)
        }
        result = evaluate_squat_frame(features)
        assert result["valid"] is True
        assert any("knee misalignment" in issue for issue in result["issues"])

    def test_multiple_issues(self):
        """Test that multiple issues are detected simultaneously."""
        features = {
            "torso_angle_deg": 50.0,  # Bad lean
            "knee_angle_deg": 100.0,  # Shallow
            "knee_offset_norm": 0.15, # Bad alignment
        }
        result = evaluate_squat_frame(features)
        assert result["valid"] is True
        assert len(result["issues"]) >= 3

    def test_invalid_frame_none_torso(self):
        """Test that frames with None torso angle are marked invalid."""
        features = {
            "torso_angle_deg": None,
            "knee_angle_deg": 60.0,
            "knee_offset_norm": 0.02,
        }
        result = evaluate_squat_frame(features)
        assert result["valid"] is False
        assert any("unreliable" in issue for issue in result["issues"])

    def test_invalid_frame_none_knee(self):
        """Test that frames with None knee angle are marked invalid."""
        features = {
            "torso_angle_deg": 20.0,
            "knee_angle_deg": None,
            "knee_offset_norm": 0.02,
        }
        result = evaluate_squat_frame(features)
        assert result["valid"] is False

    def test_custom_thresholds(self):
        """Test that custom thresholds are applied correctly."""
        features = {
            "torso_angle_deg": 30.0,  # Would be OK with default threshold of 35
            "knee_angle_deg": 60.0,
            "knee_offset_norm": 0.02,
        }
        # Default threshold - should be acceptable
        result_default = evaluate_squat_frame(features)
        assert "Form looks acceptable" in result_default["issues"][0]
        
        # Stricter threshold - should flag the lean
        custom_thresholds = {
            "max_torso_lean": 25.0,  # Stricter
            "min_knee_bend_for_depth": 80.0,
            "max_knee_offset": 0.08,
        }
        result_custom = evaluate_squat_frame(features, thresholds=custom_thresholds)
        assert any("forward torso lean" in issue for issue in result_custom["issues"])


class TestSquatThresholds:
    """Tests for the threshold configuration."""

    def test_thresholds_exist(self):
        """Test that all required thresholds are defined."""
        required_keys = [
            "max_torso_lean",
            "min_knee_bend_for_depth",
            "max_knee_offset",
            "forward_lean_threshold",
            "shallow_depth_threshold",
            "knee_alignment_threshold",
        ]
        for key in required_keys:
            assert key in SQUAT_THRESHOLDS

    def test_thresholds_are_positive(self):
        """Test that all thresholds are positive numbers."""
        for key, value in SQUAT_THRESHOLDS.items():
            assert isinstance(value, (int, float))
            assert value > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
