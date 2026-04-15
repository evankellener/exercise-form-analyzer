"""Interactive per-rep labeling tool.

For each clip in data/raw/squats/{good,bad}/:
  1. Extract poses (cached)
  2. Segment into reps
  3. For each rep, play the rep on loop with a HUD
  4. Prompt the user to tag faults

Writes/updates data/labels/rep_labels.csv with one row per rep:
    video, rep_index, start_frame, bottom_frame, end_frame,
    forward_lean, shallow_depth, knee_cave, counted_valid, notes

Usage:
    python -m src.annotate_reps                 # label all unlabeled reps
    python -m src.annotate_reps --video X.mov   # label one clip
    python -m src.annotate_reps --redo          # re-label everything
    python -m src.annotate_reps --dry-run       # just segment + print, don't label
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from .pose import extract_pose_sequence
from .segment import detect_reps, Rep
from .utils import mp_drawing, mp_pose
from .video_io import RotatingCapture


LABELS_CSV = Path("data/labels/rep_labels.csv")
RAW_DIR = Path("data/raw/squats")
FIELDS = [
    "video", "rep_index", "start_frame", "bottom_frame", "end_frame",
    "forward_lean", "shallow_depth", "knee_cave", "counted_valid", "notes",
]


def discover_clips() -> list[Path]:
    clips = []
    for sub in ("good", "bad"):
        d = RAW_DIR / sub
        if d.exists():
            clips.extend(sorted(d.glob("*.mov")))
            clips.extend(sorted(d.glob("*.mp4")))
    return clips


def load_existing() -> dict[tuple[str, int], dict]:
    if not LABELS_CSV.exists():
        return {}
    out = {}
    with LABELS_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["video"], int(row["rep_index"]))
            out[key] = row
    return out


def save_rows(rows: list[dict]) -> None:
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with LABELS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def rep_frames(video_path: Path, rep: Rep) -> list[np.ndarray]:
    cap = RotatingCapture(str(video_path))
    cap.set_frame(rep.start)
    frames = []
    idx = rep.start
    while idx <= rep.end:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        idx += 1
    cap.release()
    return frames


def draw_hud(frame: np.ndarray, text_lines: list[str], landmarks=None) -> np.ndarray:
    out = frame.copy()
    if landmarks is not None:
        mp_drawing.draw_landmarks(out, landmarks, mp_pose.POSE_CONNECTIONS)
    y = 30
    for line in text_lines:
        cv2.rectangle(out, (8, y - 22), (8 + 10 * len(line), y + 6), (0, 0, 0), -1)
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 28
    return out


def play_rep_loop(
    frames: list[np.ndarray],
    window_name: str,
    header: list[str],
    fps: float,
) -> str:
    """Loop the rep until the user presses a label key. Returns the key pressed."""
    delay = max(1, int(1000 / max(fps, 1)))
    keymap = "[1]lean  [2]shallow  [3]kneecave  [4]valid_good  [0]clear  [n]next  [s]skip  [q]quit"
    while True:
        for i, f in enumerate(frames):
            hud = draw_hud(f, header + [f"Frame {i+1}/{len(frames)}", keymap])
            cv2.imshow(window_name, hud)
            k = cv2.waitKey(delay) & 0xFF
            if k != 255:
                return chr(k) if 0 <= k < 128 else ""
    # unreachable


def label_clip(
    video_path: Path,
    existing: dict[tuple[str, int], dict],
    redo: bool,
    dry_run: bool,
) -> list[dict]:
    vp = str(video_path)
    print(f"\n=== {vp} ===")
    pose_data = extract_pose_sequence(video_path)
    reps = detect_reps(pose_data["landmarks"], pose_data["valid"], pose_data["fps"])
    print(f"  Detected {len(reps)} reps (cached={pose_data['from_cache']}).")
    rows = []

    if dry_run:
        for r in reps:
            print(f"  rep {r.index+1}: frames {r.start}-{r.bottom}-{r.end}")
        return rows

    if not reps:
        return rows

    window = f"rep_label :: {video_path.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    for r in reps:
        key = (vp, r.index)
        if key in existing and not redo:
            rows.append(existing[key])
            print(f"  rep {r.index+1}: already labeled, skipping (use --redo to re-label)")
            continue

        frames = rep_frames(video_path, r)
        if not frames:
            continue

        flags = {"forward_lean": 0, "shallow_depth": 0, "knee_cave": 0}
        counted = 1
        header_base = [
            f"{video_path.name}  rep {r.index+1} / {len(reps)}",
        ]
        while True:
            header = header_base + [
                f"lean={flags['forward_lean']}  shallow={flags['shallow_depth']}  kneecave={flags['knee_cave']}  valid={counted}",
            ]
            k = play_rep_loop(frames, window, header, pose_data["fps"])
            if k == "1":
                flags["forward_lean"] ^= 1
            elif k == "2":
                flags["shallow_depth"] ^= 1
            elif k == "3":
                flags["knee_cave"] ^= 1
            elif k == "4":
                counted ^= 1
            elif k == "0":
                flags = {k2: 0 for k2 in flags}
                counted = 1
            elif k == "n":
                break
            elif k == "s":
                flags = None
                break
            elif k == "q":
                cv2.destroyAllWindows()
                return rows

        if flags is None:
            continue

        # Default counted_valid: valid iff no fault flag set (user can override with [4])
        any_fault = any(flags.values())
        inferred_valid = 0 if any_fault else 1
        # If the user toggled [4] to differ from inference, keep their choice.
        final_valid = counted if counted != 1 or not any_fault else 0
        # Simpler rule: counted_valid = 1 only if no faults AND user didn't flip to 0.
        final_valid = 1 if (not any_fault and counted == 1) else 0

        rows.append({
            "video": vp,
            "rep_index": r.index,
            "start_frame": r.start,
            "bottom_frame": r.bottom,
            "end_frame": r.end,
            "forward_lean": flags["forward_lean"],
            "shallow_depth": flags["shallow_depth"],
            "knee_cave": flags["knee_cave"],
            "counted_valid": final_valid,
            "notes": "",
        })
        print(f"  rep {r.index+1}: {flags}  valid={final_valid}")

    cv2.destroyAllWindows()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="Label only this clip (path relative to repo root).")
    ap.add_argument("--redo", action="store_true", help="Re-label reps already in CSV.")
    ap.add_argument("--dry-run", action="store_true", help="Segment and print only; no labeling UI.")
    args = ap.parse_args()

    existing = load_existing()
    all_rows: list[dict] = []
    # Preserve rows from clips we won't touch this run.
    touched_videos: set[str] = set()

    if args.video:
        clips = [Path(args.video)]
    else:
        clips = discover_clips()

    for clip in clips:
        rows = label_clip(clip, existing, redo=args.redo, dry_run=args.dry_run)
        touched_videos.add(str(clip))
        all_rows.extend(rows)

    if args.dry_run:
        return

    # Add back any existing rows from videos we didn't touch.
    for (v, _), row in existing.items():
        if v not in touched_videos:
            all_rows.append(row)

    # Sort for stable diffs.
    all_rows.sort(key=lambda r: (r["video"], int(r["rep_index"])))
    save_rows(all_rows)
    print(f"\nSaved {len(all_rows)} rep labels to {LABELS_CSV}")


if __name__ == "__main__":
    main()
