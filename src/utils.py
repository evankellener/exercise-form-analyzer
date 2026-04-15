import os
import sys
import warnings

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MP_VERBOSE"] = "0"

warnings.filterwarnings("ignore")

import numpy as np
import mediapipe as mp

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

# Issue type constants
ISSUE_FORM_ACCEPTABLE = "Form acceptable"
ISSUE_POSE_UNRELIABLE = "Pose landmarks unreliable"


class SuppressStderr:
    """Context manager to suppress stderr during MediaPipe operations."""
    def __enter__(self):
        self._null_fd = os.open(os.devnull, os.O_WRONLY)
        self._original_stderr_fd = os.dup(sys.stderr.fileno())
        os.dup2(self._null_fd, sys.stderr.fileno())
        return self

    def __exit__(self, *args):
        os.dup2(self._original_stderr_fd, sys.stderr.fileno())
        try:
            os.close(self._null_fd)
        except Exception:
            pass
        try:
            os.close(self._original_stderr_fd)
        except Exception:
            pass
        return False


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
