"""
Interactive annotation CLI: label videos for ground_truth.csv.

Walks data/raw/<exercise>/<label>/, finds videos that aren't yet in
ground_truth.csv, opens each one (system default player), and prompts you to
mark which form issues are present plus the train/test split. Appends a row
per video to ground_truth.csv.

Usage:
    python -m src.annotate
    python -m src.annotate --ground-truth data/ground_truth.csv --data-dir data/raw
    python -m src.annotate --auto-split   # auto-assign 80/20 train/test
"""

import argparse
import csv
import os
import random
import subprocess
import sys

from .extract_features import scan_data_dir
from .squat import ISSUE_FORWARD_LEAN, ISSUE_SHALLOW_DEPTH, ISSUE_KNEE_ALIGNMENT
from .pushup import (
    ISSUE_ASYMMETRIC_ARMS, ISSUE_HIP_SAG, ISSUE_HIP_PIKE, ISSUE_BODY_ALIGNMENT,
)

SQUAT_ISSUES = [ISSUE_FORWARD_LEAN, ISSUE_SHALLOW_DEPTH, ISSUE_KNEE_ALIGNMENT]
PUSHUP_ISSUES = [ISSUE_ASYMMETRIC_ARMS, ISSUE_HIP_SAG, ISSUE_HIP_PIKE, ISSUE_BODY_ALIGNMENT]

FIELDS = ["video", "exercise", "expected_issues", "expected_good", "split", "notes"]


def load_existing(path):
    if not os.path.exists(path):
        return set(), []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {r["video"] for r in rows}, rows


def open_video(path):
    """Open in system default player (non-blocking)."""
    if sys.platform == "darwin":
        subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif sys.platform.startswith("linux"):
        subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]


def prompt_issues(exercise, default_label):
    issues = SQUAT_ISSUES if exercise == "squat" else PUSHUP_ISSUES
    print(f"  Issues ({exercise}):")
    for i, iss in enumerate(issues, 1):
        print(f"    {i}. {iss}")
    print("    0. None (good form)")
    print(f"  Filename label hint: {default_label}")
    raw = input("  Enter issue numbers (comma-separated, blank=use hint): ").strip()

    if not raw:
        if default_label == "good":
            return []
        return None  # ambiguous; prompt again

    if raw == "0":
        return []
    selected = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok.isdigit():
            continue
        idx = int(tok)
        if 1 <= idx <= len(issues):
            selected.append(issues[idx - 1])
    return selected


def prompt_split(auto_split):
    if auto_split is not None:
        return auto_split
    raw = input("  Split [train/test/holdout] (default train): ").strip().lower()
    if raw in ("train", "test", "holdout"):
        return raw
    return "train"


def annotate(unlabeled, gt_path, auto_split_ratio=None):
    rng = random.Random(42)
    new_rows = []
    for i, (path, exercise, hint_label) in enumerate(unlabeled, 1):
        print(f"\n[{i}/{len(unlabeled)}] {path}")
        print(f"  Exercise: {exercise}")
        open_video(path)

        while True:
            issues = prompt_issues(exercise, hint_label)
            if issues is not None:
                break
            print("  Please pick at least one issue or 0 for good form.")

        expected_good = len(issues) == 0
        if auto_split_ratio is not None:
            split = "test" if rng.random() < auto_split_ratio else "train"
        else:
            split = prompt_split(None)
        notes = input("  Notes (optional): ").strip()

        new_rows.append({
            "video": path,
            "exercise": exercise,
            "expected_issues": ";".join(issues),
            "expected_good": "true" if expected_good else "false",
            "split": split,
            "notes": notes,
        })

        # Save incrementally so a Ctrl-C doesn't lose work.
        write_rows(gt_path, new_rows, append=True)
        new_rows = []

    print("\nDone.")


def write_rows(path, rows, append=False):
    if not rows:
        return
    exists = os.path.exists(path)
    mode = "a" if append and exists else "w"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists or mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Interactively annotate videos for ground_truth.csv")
    parser.add_argument("--ground-truth", default="data/ground_truth.csv")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--auto-split", action="store_true",
                        help="Auto-assign 80/20 train/test (skips split prompt)")
    args = parser.parse_args()

    labeled_videos, _ = load_existing(args.ground_truth)
    all_entries = scan_data_dir(args.data_dir)
    unlabeled = [(p, ex, lab) for (p, ex, lab) in all_entries if p not in labeled_videos]

    if not unlabeled:
        print(f"All {len(all_entries)} videos in {args.data_dir} are already labeled.")
        return

    print(f"{len(unlabeled)} videos to label "
          f"({len(all_entries) - len(unlabeled)} already labeled).\n")
    print("For each video the system player will open. Watch, then enter issue numbers.")
    print("Ctrl-C anytime — progress is saved after each video.\n")

    auto_ratio = 0.2 if args.auto_split else None

    try:
        annotate(unlabeled, args.ground_truth, auto_ratio)
    except KeyboardInterrupt:
        print("\nInterrupted; progress saved.")


if __name__ == "__main__":
    main()
