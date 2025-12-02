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
│   └── utils.py      # Shared helper functions (if needed)
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
python src/pipeline.py --video data/raw/squats/positive_squat_01.mp4
```

### Analyzing Pushups

Run the pushup analysis pipeline:

```bash
python src/pipeline.py --video data/raw/push-ups/positive_pushup_01.mp4 --exercise pushup
```

### Live Popup Window

Display a real-time popup window with pose landmarks overlaid on the video:

```bash
python src/pipeline.py --video data/raw/squats/positive_squat_01.mp4 --show-popup
```

Press `q` to close the popup window early.

### Command-line Options

- `--video`: Path to the video file (required)
- `--exercise`: Type of exercise to analyze: `squat` or `pushup` (default: `squat`)
- `--no-debug-video`: Skip saving the debug video with pose overlays
- `--show-popup`: Show a real-time popup window with landmarks overlay

This will:

- Run MediaPipe Pose on each frame
- Compute exercise-specific features (squat: torso lean, squat depth, knee alignment; pushup: elbow angles, body alignment, hip position)
- Apply heuristic rules for form evaluation
- Print an aggregate summary to the terminal
- Optionally save a debug video with pose overlays to `results/`
- Optionally display a live popup window with landmarks (using `--show-popup`)

You can tweak thresholds and logic inside `src/pipeline.py` to better match your own labeling.

## Extension Ideas

- Add support for **lunges** with their own feature sets and rules.
- Save frame-wise features and system judgments to a CSV in `results/` for easier analysis.
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