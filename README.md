# Exercise Form Analyzer Using Pose Estimation

This project implements a simple, rule-based system to analyze exercise form (starting with squats) using pose estimation. It is designed as a capstone-style project for a computer vision course, focusing on:

- Pose estimation with MediaPipe
- Extraction of biomechanical features (joint angles, alignment)
- Rule-based analysis of form quality
- Simple end-to-end pipeline from input video to text feedback

## Project Structure

```text
.
├── data
│   ├── raw/          # Original exercise videos (input)
│   └── processed/    # Any processed keypoints / intermediate data
├── docs/             # Final report, notes
├── models/           # (Reserved) Any saved models or configs
├── notebooks/        # (Optional) Jupyter notebooks for experiments
├── presentation/     # Slides or script for the final presentation
├── results/          # Analysis summaries, debug videos, screenshots
├── src/
│   ├── pipeline.py   # Main pose-estimation + rule-based analysis pipeline
│   └── utils.py      # Shared helper functions for geometric calculations
├── tests/            # Unit tests for the pipeline and utilities
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/exercise-form-analyzer.git
cd exercise-form-analyzer
```

2. **Create and activate a virtual environment (recommended)**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Add your videos**

Place your exercise videos in:

```text
data/raw/
```

For example:

```text
data/raw/squat_front.mp4
data/raw/squat_side.mp4
```

## Usage

Run the basic squat analysis pipeline:

```bash
python src/pipeline.py --video data/raw/squat_front.mp4
```

This will:

- Run MediaPipe Pose on each frame
- Compute simple squat-related features
- Apply heuristic rules (torso lean, squat depth, knee alignment)
- Print an aggregate summary to the terminal
- Optionally save a debug video with pose overlays to `results/squat_debug.mp4`

### Additional Options

```bash
# Save frame-by-frame analysis to CSV
python src/pipeline.py --video data/raw/squat_front.mp4 --save-csv

# Disable debug video output for faster processing
python src/pipeline.py --video data/raw/squat_front.mp4 --no-debug-video

# Customize output paths
python src/pipeline.py --video data/raw/squat_front.mp4 --csv-output results/my_analysis.csv --debug-output results/my_debug.mp4
```

### Configurable Thresholds

You can tweak the evaluation thresholds in `src/pipeline.py` under `SQUAT_THRESHOLDS`:

- `max_torso_lean`: Maximum acceptable forward lean in degrees (default: 35°)
- `min_knee_bend_for_depth`: Knee angle above this is considered shallow (default: 80°)
- `max_knee_offset`: Maximum normalized knee deviation (default: 0.08)

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Extension Ideas

- Add support for **pushups** and **lunges** with their own feature sets and rules.
- ~~Save frame-wise features and system judgments to a CSV in `results/` for easier analysis.~~ ✅ Implemented
- Create a small comparison table: **Human vs System** labels for 3–5 videos.
- Capture screenshots and plots for your final paper and presentation.

## Course Alignment

This repository is structured to support:

1. Pose estimation pipeline
2. Feature extraction
3. Rule-based evaluation logic
4. Complete pipeline from video to summary
5. Evaluation report with small human vs system comparison
6. Short presentation video

You can place your final write-up in `docs/` (e.g., `docs/final_report.md`) and any slides or video scripts in `presentation/`.