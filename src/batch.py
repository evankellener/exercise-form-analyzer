"""Batch-process all 12 squat clips: annotate + write a rollup CSV.

Usage:
    python -m src.batch                          # use existing model (rf)
    python -m src.batch --model logreg
    python -m src.batch --train                  # re-train on ALL labels first
    python -m src.batch --video one_clip.mov     # only process one clip
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .annotate_video import annotate_video
from .features import build_dataset
from .model import MultiLabelModel, train_and_save, MODEL_DIR
from .pose import extract_pose_sequence


LABELS_CSV = Path("data/labels/rep_labels.csv")
RESULTS_DIR = Path("results")
RAW_DIR = Path("data/raw/squats")


def discover_clips() -> list[Path]:
    clips = []
    for sub in ("good", "bad"):
        d = RAW_DIR / sub
        if d.exists():
            clips.extend(sorted(d.glob("*.mov")))
            clips.extend(sorted(d.glob("*.mp4")))
    return clips


def train_from_all(estimator: str, augment: int, seed: int) -> Path:
    rows = list(csv.DictReader(open(LABELS_CSV)))
    videos = sorted({r["video"] for r in rows})
    poses = {v: extract_pose_sequence(v) for v in videos}
    ds = build_dataset(rows, poses, augment_copies=augment, seed=seed)
    return train_and_save(ds, kind=estimator)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["rf", "logreg"], default="rf")
    ap.add_argument("--train", action="store_true",
                    help="Re-train the model on ALL labeled reps before annotating.")
    ap.add_argument("--augment", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default=None, help="Process only this clip.")
    args = ap.parse_args()

    model_path = MODEL_DIR / f"squat_multilabel_{args.model}.pkl"
    if args.train or not model_path.exists():
        print(f"Training {args.model} on all labels (augment={args.augment})...")
        model_path = train_from_all(args.model, args.augment, args.seed)
        print(f"Saved: {model_path}")

    model = MultiLabelModel.load(model_path)

    clips = [Path(args.video)] if args.video else discover_clips()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "annotated").mkdir(parents=True, exist_ok=True)

    rollup_rows = []
    summaries = []

    for clip in clips:
        out_path = RESULTS_DIR / "annotated" / f"{clip.stem}_annotated.mp4"
        print(f"\n→ {clip.name}")
        info = annotate_video(clip, model, out_path)
        summaries.append(info)
        print(f"  detected={info['n_reps_detected']}  counted={info['n_reps_counted']}  -> {out_path.name}")

        for per in info["per_rep"]:
            rollup_rows.append({
                "video": str(clip),
                "rep": per["rep"],
                "start": per["start"],
                "end": per["end"],
                "counted": per["counted"],
                "faults": ";".join(per["faults"]),
                "proba_lean": round(per["proba"][0], 3),
                "proba_shallow": round(per["proba"][1], 3),
                "proba_cave": round(per["proba"][2], 3),
            })

    rollup_csv = RESULTS_DIR / "squat_reps.csv"
    with rollup_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rollup_rows[0].keys()) if rollup_rows else [])
        w.writeheader()
        w.writerows(rollup_rows)
    print(f"\nWrote rollup: {rollup_csv}  ({len(rollup_rows)} reps across {len(clips)} clips)")

    with (RESULTS_DIR / "batch_summary.json").open("w") as f:
        json.dump(summaries, f, indent=2, default=str)


if __name__ == "__main__":
    main()
