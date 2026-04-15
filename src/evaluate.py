"""Evaluation: leave-one-video-out CV comparing rule-based and ML classifiers.

For each held-out video:
  - Train fold is the 11 other videos. Augmentation is applied ONLY to training.
  - Test is the held-out video's raw (un-augmented) reps.
  - Record per-fault TP/FP/FN/TN and per-video rep-count accuracy.

Outputs:
  results/metrics.json     aggregate + per-fold metrics
  results/report.md        human-readable summary table
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .features import build_dataset, FEATURE_NAMES
from .model import train, LABEL_NAMES
from .pose import extract_pose_sequence
from .rules import classify_batch


LABELS_CSV = Path("data/labels/rep_labels.csv")
RESULTS_DIR = Path("results")


def _pr_f1(y, p):
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-8)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(P, 3), "recall": round(R, 3), "f1": round(F1, 3)}


def lovo_eval(
    rows: list[dict],
    poses: dict[str, dict],
    estimator_kind: str = "rf",
    augment_copies: int = 8,
    seed: int = 0,
) -> dict:
    """Leave-one-video-out evaluation of rules vs. ML classifier."""
    videos = sorted({r["video"] for r in rows})
    per_fold = []
    # Running aggregates for per-fault metrics (raw test reps only).
    y_all = {"rule": [], "ml": [], "truth": []}
    rep_count = {"expected_valid": 0, "rule_valid": 0, "ml_valid": 0,
                 "expected_total": 0, "detected_total": 0}

    for held_out in videos:
        train_rows = [r for r in rows if r["video"] != held_out]
        test_rows = [r for r in rows if r["video"] == held_out]

        ds_train = build_dataset(train_rows, poses,
                                 augment_copies=augment_copies, seed=seed)
        ds_test = build_dataset(test_rows, poses, augment_copies=0)

        if ds_test["X"].shape[0] == 0:
            continue

        model = train(ds_train["X"], ds_train["Y"], kind=estimator_kind)
        ml_pred = model.predict(ds_test["X"])
        rule_pred = classify_batch(ds_test["X"])
        truth = ds_test["Y"]

        y_all["truth"].append(truth)
        y_all["rule"].append(rule_pred)
        y_all["ml"].append(ml_pred)

        # Rep-count metrics: per-video, count reps judged valid.
        exp_valid = int(ds_test["valid"].sum())
        rule_valid = int((1 - rule_pred.max(axis=1)).sum())
        ml_valid = int((1 - ml_pred.max(axis=1)).sum())
        rep_count["expected_valid"] += exp_valid
        rep_count["rule_valid"] += rule_valid
        rep_count["ml_valid"] += ml_valid
        rep_count["expected_total"] += ds_test["X"].shape[0]
        rep_count["detected_total"] += ds_test["X"].shape[0]

        per_fold.append({
            "held_out": held_out,
            "n_test_reps": int(ds_test["X"].shape[0]),
            "expected_valid": exp_valid,
            "rule_valid": rule_valid,
            "ml_valid": ml_valid,
            "truth": truth.tolist(),
            "rule_pred": rule_pred.tolist(),
            "ml_pred": ml_pred.tolist(),
        })

    Y_true = np.concatenate(y_all["truth"], axis=0)
    Y_rule = np.concatenate(y_all["rule"], axis=0)
    Y_ml = np.concatenate(y_all["ml"], axis=0)

    per_fault = {}
    for j, name in enumerate(LABEL_NAMES):
        per_fault[name] = {
            "rule": _pr_f1(Y_true[:, j], Y_rule[:, j]),
            "ml": _pr_f1(Y_true[:, j], Y_ml[:, j]),
        }

    # Overall "counted_valid": a rep counts iff no fault predicted.
    truth_valid = 1 - Y_true.max(axis=1)
    rule_valid = 1 - Y_rule.max(axis=1)
    ml_valid = 1 - Y_ml.max(axis=1)
    valid_metrics = {
        "rule": _pr_f1(truth_valid, rule_valid),
        "ml": _pr_f1(truth_valid, ml_valid),
    }

    return {
        "estimator": estimator_kind,
        "augment_copies": augment_copies,
        "n_videos": len(videos),
        "n_reps_total": int(Y_true.shape[0]),
        "per_fault": per_fault,
        "counted_valid": valid_metrics,
        "rep_count_totals": rep_count,
        "per_fold": per_fold,
    }


def format_report(res: dict) -> str:
    lines = ["# SquatCoach LOVO-CV evaluation\n"]
    lines.append(f"- Estimator: **{res['estimator']}**  |  augment copies: **{res['augment_copies']}**")
    lines.append(f"- Videos (folds): **{res['n_videos']}**  |  Total test reps (raw): **{res['n_reps_total']}**\n")

    lines.append("## Per-fault metrics\n")
    lines.append("| Fault | Model | Precision | Recall | F1 | TP | FP | FN | TN |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, m in res["per_fault"].items():
        for approach in ("rule", "ml"):
            mm = m[approach]
            lines.append(
                f"| {name} | {approach} | {mm['precision']:.2f} | {mm['recall']:.2f} "
                f"| {mm['f1']:.2f} | {mm['tp']} | {mm['fp']} | {mm['fn']} | {mm['tn']} |"
            )
    lines.append("")

    lines.append("## counted_valid (a rep counts iff no fault flagged)\n")
    lines.append("| Model | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|")
    for approach in ("rule", "ml"):
        mm = res["counted_valid"][approach]
        lines.append(
            f"| {approach} | {mm['precision']:.2f} | {mm['recall']:.2f} | {mm['f1']:.2f} |"
        )
    lines.append("")

    lines.append("## Per-fold rep counts\n")
    lines.append("| Video | test reps | expected valid | rule valid | ml valid |")
    lines.append("|---|---|---|---|---|")
    for f in res["per_fold"]:
        lines.append(
            f"| {Path(f['held_out']).name} | {f['n_test_reps']} | "
            f"{f['expected_valid']} | {f['rule_valid']} | {f['ml_valid']} |"
        )
    lines.append("")
    totals = res["rep_count_totals"]
    lines.append(
        f"**Totals**: expected_valid={totals['expected_valid']}, "
        f"rule_valid={totals['rule_valid']}, ml_valid={totals['ml_valid']}"
    )
    return "\n".join(lines)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", choices=["logreg", "rf"], default="rf")
    ap.add_argument("--augment", type=int, default=8, help="augmented copies per training rep")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(LABELS_CSV)))
    videos = sorted({r["video"] for r in rows})
    poses = {v: extract_pose_sequence(v) for v in videos}

    res = lovo_eval(rows, poses, args.estimator, args.augment, args.seed)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    with open(RESULTS_DIR / "report.md", "w") as f:
        f.write(format_report(res))

    print(format_report(res))
    print(f"\nSaved metrics.json and report.md to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
