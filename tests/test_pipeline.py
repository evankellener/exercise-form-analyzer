"""
Unit tests for pipeline module.
"""
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import (
    evaluate_squat_frame,
    evaluate_pushup_frame,
)


def test_evaluate_squat_frame_good_form():
    """Test that good squat form is recognized."""
    features = {
        "torso_angle_deg": 20.0,  # Under 35 threshold
        "knee_angle_deg": 60.0,   # Under 80 threshold (good depth)
        "knee_offset_norm": 0.02,  # Under 0.08 threshold
    }
    
    result = evaluate_squat_frame(features)
    assert result["valid"] is True
    assert any("acceptable" in issue.lower() for issue in result["issues"])


def test_evaluate_squat_frame_forward_lean():
    """Test that excessive forward lean is detected."""
    features = {
        "torso_angle_deg": 45.0,  # Over 35 threshold
        "knee_angle_deg": 60.0,
        "knee_offset_norm": 0.02,
    }
    
    result = evaluate_squat_frame(features)
    assert result["valid"] is True
    assert any("lean" in issue.lower() for issue in result["issues"])


def test_evaluate_squat_frame_shallow_depth():
    """Test that shallow squat depth is detected."""
    features = {
        "torso_angle_deg": 20.0,
        "knee_angle_deg": 120.0,  # Over 80 threshold
        "knee_offset_norm": 0.02,
    }
    
    result = evaluate_squat_frame(features)
    assert result["valid"] is True
    assert any("shallow" in issue.lower() for issue in result["issues"])


def test_evaluate_squat_frame_knee_misalignment():
    """Test that knee misalignment is detected."""
    features = {
        "torso_angle_deg": 20.0,
        "knee_angle_deg": 60.0,
        "knee_offset_norm": 0.15,  # Over 0.08 threshold
    }
    
    result = evaluate_squat_frame(features)
    assert result["valid"] is True
    assert any("knee" in issue.lower() and "alignment" in issue.lower() for issue in result["issues"])


def test_evaluate_squat_frame_invalid_landmarks():
    """Test handling of missing landmarks."""
    features = {
        "torso_angle_deg": None,
        "knee_angle_deg": None,
        "knee_offset_norm": 0.02,
    }
    
    result = evaluate_squat_frame(features)
    assert result["valid"] is False
    assert any("unreliable" in issue.lower() for issue in result["issues"])


def test_evaluate_pushup_frame_good_form():
    """Test that good pushup form is recognized."""
    features = {
        "elbow_angle_deg": 90.0,   # Good depth
        "body_angle_deg": 80.0,
        "hip_deviation_norm": 0.02,  # Under 0.05 threshold
    }
    
    result = evaluate_pushup_frame(features)
    assert result["valid"] is True
    # Good form at bottom of pushup
    assert any("acceptable" in issue.lower() for issue in result["issues"])


def test_evaluate_pushup_frame_arms_extended():
    """Test that extended arms are detected (top of pushup)."""
    features = {
        "elbow_angle_deg": 170.0,  # Arms very extended
        "body_angle_deg": 80.0,
        "hip_deviation_norm": 0.02,
    }
    
    result = evaluate_pushup_frame(features)
    assert result["valid"] is True
    assert any("extended" in issue.lower() for issue in result["issues"])


def test_evaluate_pushup_frame_hip_sag():
    """Test that hip sagging is detected."""
    features = {
        "elbow_angle_deg": 90.0,
        "body_angle_deg": 80.0,
        "hip_deviation_norm": 0.1,  # Hip too low (positive = sagging)
    }
    
    result = evaluate_pushup_frame(features)
    assert result["valid"] is True
    assert any("sag" in issue.lower() for issue in result["issues"])


def test_evaluate_pushup_frame_hip_pike():
    """Test that hip piking is detected."""
    features = {
        "elbow_angle_deg": 90.0,
        "body_angle_deg": 80.0,
        "hip_deviation_norm": -0.1,  # Hip too high (negative = piking)
    }
    
    result = evaluate_pushup_frame(features)
    assert result["valid"] is True
    assert any("pike" in issue.lower() for issue in result["issues"])


def test_evaluate_pushup_frame_invalid_landmarks():
    """Test handling of missing landmarks."""
    features = {
        "elbow_angle_deg": None,
        "body_angle_deg": None,
        "hip_deviation_norm": 0.02,
    }
    
    result = evaluate_pushup_frame(features)
    assert result["valid"] is False
    assert any("unreliable" in issue.lower() for issue in result["issues"])


if __name__ == "__main__":
    # Run all tests
    test_evaluate_squat_frame_good_form()
    print("✓ test_evaluate_squat_frame_good_form passed")
    
    test_evaluate_squat_frame_forward_lean()
    print("✓ test_evaluate_squat_frame_forward_lean passed")
    
    test_evaluate_squat_frame_shallow_depth()
    print("✓ test_evaluate_squat_frame_shallow_depth passed")
    
    test_evaluate_squat_frame_knee_misalignment()
    print("✓ test_evaluate_squat_frame_knee_misalignment passed")
    
    test_evaluate_squat_frame_invalid_landmarks()
    print("✓ test_evaluate_squat_frame_invalid_landmarks passed")
    
    test_evaluate_pushup_frame_good_form()
    print("✓ test_evaluate_pushup_frame_good_form passed")
    
    test_evaluate_pushup_frame_arms_extended()
    print("✓ test_evaluate_pushup_frame_arms_extended passed")
    
    test_evaluate_pushup_frame_hip_sag()
    print("✓ test_evaluate_pushup_frame_hip_sag passed")
    
    test_evaluate_pushup_frame_hip_pike()
    print("✓ test_evaluate_pushup_frame_hip_pike passed")
    
    test_evaluate_pushup_frame_invalid_landmarks()
    print("✓ test_evaluate_pushup_frame_invalid_landmarks passed")
    
    print("\nAll tests passed!")
