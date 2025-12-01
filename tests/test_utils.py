"""
Unit tests for utils module.
"""
import os
import sys
import tempfile
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import (
    angle_between_points,
    angle_from_vertical,
    save_features_to_csv,
)


def test_angle_between_points_right_angle():
    """Test that a 90-degree angle is correctly calculated."""
    a = np.array([0, 0], dtype=np.float32)
    b = np.array([1, 0], dtype=np.float32)
    c = np.array([1, 1], dtype=np.float32)
    
    angle = angle_between_points(a, b, c)
    assert angle is not None
    assert abs(angle - 90.0) < 0.01, f"Expected 90 degrees, got {angle}"


def test_angle_between_points_straight_line():
    """Test that a straight line (180 degrees) is correctly calculated."""
    a = np.array([0, 0], dtype=np.float32)
    b = np.array([1, 0], dtype=np.float32)
    c = np.array([2, 0], dtype=np.float32)
    
    angle = angle_between_points(a, b, c)
    assert angle is not None
    assert abs(angle - 180.0) < 0.01, f"Expected 180 degrees, got {angle}"


def test_angle_between_points_zero_length():
    """Test that zero-length vectors return None."""
    a = np.array([1, 1], dtype=np.float32)
    b = np.array([1, 1], dtype=np.float32)  # Same as a
    c = np.array([2, 2], dtype=np.float32)
    
    angle = angle_between_points(a, b, c)
    assert angle is None


def test_angle_from_vertical_perfectly_vertical():
    """Test that a vertical line returns 0 degrees."""
    point1 = np.array([0, 0], dtype=np.float32)  # Upper point
    point2 = np.array([0, 1], dtype=np.float32)  # Lower point
    
    angle = angle_from_vertical(point1, point2)
    assert angle is not None
    assert abs(angle - 0.0) < 0.01, f"Expected 0 degrees, got {angle}"


def test_angle_from_vertical_horizontal():
    """Test that a horizontal line returns 90 degrees."""
    point1 = np.array([0, 0], dtype=np.float32)
    point2 = np.array([1, 0], dtype=np.float32)
    
    angle = angle_from_vertical(point1, point2)
    assert angle is not None
    assert abs(angle - 90.0) < 0.01, f"Expected 90 degrees, got {angle}"


def test_angle_from_vertical_zero_length():
    """Test that zero-length vectors return None."""
    point1 = np.array([1, 1], dtype=np.float32)
    point2 = np.array([1, 1], dtype=np.float32)
    
    angle = angle_from_vertical(point1, point2)
    assert angle is None


def test_save_features_to_csv():
    """Test that features are saved correctly to CSV."""
    features_list = [
        {"frame_number": 1, "torso_angle_deg": 10.5, "valid": True},
        {"frame_number": 2, "torso_angle_deg": 15.2, "valid": True},
    ]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = f.name
    
    try:
        save_features_to_csv(features_list, temp_path, exercise_type="squat")
        
        # Verify file exists and has content
        assert os.path.exists(temp_path)
        with open(temp_path, "r") as f:
            content = f.read()
            assert "frame_number" in content
            assert "exercise_type" in content
            assert "squat" in content
            assert "10.5" in content
            assert "15.2" in content
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_features_to_csv_empty():
    """Test that empty features list doesn't create file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = f.name
    
    # Remove the file first
    os.unlink(temp_path)
    
    save_features_to_csv([], temp_path, exercise_type="squat")
    
    # File should not exist since features list is empty
    assert not os.path.exists(temp_path)


if __name__ == "__main__":
    # Run all tests
    test_angle_between_points_right_angle()
    print("✓ test_angle_between_points_right_angle passed")
    
    test_angle_between_points_straight_line()
    print("✓ test_angle_between_points_straight_line passed")
    
    test_angle_between_points_zero_length()
    print("✓ test_angle_between_points_zero_length passed")
    
    test_angle_from_vertical_perfectly_vertical()
    print("✓ test_angle_from_vertical_perfectly_vertical passed")
    
    test_angle_from_vertical_horizontal()
    print("✓ test_angle_from_vertical_horizontal passed")
    
    test_angle_from_vertical_zero_length()
    print("✓ test_angle_from_vertical_zero_length passed")
    
    test_save_features_to_csv()
    print("✓ test_save_features_to_csv passed")
    
    test_save_features_to_csv_empty()
    print("✓ test_save_features_to_csv_empty passed")
    
    print("\nAll tests passed!")
