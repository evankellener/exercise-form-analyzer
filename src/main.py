"""CLI entry point for SquatCoach.

Current subcommands:
    analyze   Run the legacy whole-video rule-based squat analyzer on one clip.
    segment   Extract poses and print detected rep boundaries (quick sanity check).

More subcommands (train, evaluate, batch) land as those modules do.
"""
import argparse
import os

from .squat import analyze_squat_video, print_summary as print_squat_summary
from .pose import extract_pose_sequence
from .segment import detect_reps, summarize as summarize_reps


def cmd_analyze(args):
    debug = not args.no_debug_video
    debug_path = args.debug_path or "results/squat_debug.mp4"
    os.makedirs(os.path.dirname(debug_path) or ".", exist_ok=True)
    res = analyze_squat_video(
        path=args.video,
        draw=debug,
        save_debug=debug,
        debug_path=debug_path,
        show_popup=args.show_popup,
    )
    print_squat_summary(res)
    if debug and os.path.exists(debug_path):
        print(f"\nDebug video saved to: {debug_path}")


def cmd_segment(args):
    pose_data = extract_pose_sequence(args.video, use_cache=not args.no_cache)
    reps = detect_reps(pose_data["landmarks"], pose_data["valid"], pose_data["fps"])
    print(summarize_reps(reps, pose_data["fps"]))


def main():
    parser = argparse.ArgumentParser(description="SquatCoach — squat form analyzer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_an = sub.add_parser("analyze", help="Run legacy whole-video rule-based analyzer.")
    p_an.add_argument("--video", required=True)
    p_an.add_argument("--no-debug-video", action="store_true")
    p_an.add_argument("--debug-path", default=None)
    p_an.add_argument("--show-popup", action="store_true")
    p_an.set_defaults(func=cmd_analyze)

    p_seg = sub.add_parser("segment", help="Extract poses and print detected reps.")
    p_seg.add_argument("--video", required=True)
    p_seg.add_argument("--no-cache", action="store_true")
    p_seg.set_defaults(func=cmd_segment)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
