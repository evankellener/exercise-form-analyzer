# Exercise Form Analyzer Using Pose Estimation

This project implements a simple, rule-based system to analyze exercise form (squats and pushups) using pose estimation. It is designed as a capstone-style project for a computer vision course, focusing on:

- Pose estimation with MediaPipe
- Extraction of biomechanical features (joint angles, alignment)
- Rule-based analysis of form quality
- Simple end-to-end pipeline from input video to text feedback
- CSV export for frame-wise analysis

## Supported Exercises

- **Squats**: Analyzes torso lean, squat depth, and knee alignment
- **Pushups**: Analyzes elbow angle (depth), body alignment, and hip position (sag/pike detection)

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
├── results/          # Analysis summaries, debug videos, CSV exports
├── src/
│   ├── pipeline.py   # Main pose-estimation + rule-based analysis pipeline
│   └── utils.py      # Shared helper functions
├── tests/            # Unit tests
│   ├── test_utils.py
│   └── test_pipeline.py
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
data/raw/pushup_side.mp4
```

## Usage

### Analyze Squat Form

```bash
python src/pipeline.py --video data/raw/squat_front.mp4
```

### Analyze Pushup Form

```bash
python src/pipeline.py --video data/raw/pushup_side.mp4 --exercise pushup
```

### Save Frame-wise Features to CSV

```bash
python src/pipeline.py --video data/raw/squat_front.mp4 --save-csv
```

### Disable Debug Video Output

```bash
python src/pipeline.py --video data/raw/squat_front.mp4 --no-debug-video
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--video` | Path to input video (required) |
| `--exercise` | Type of exercise: `squat` (default) or `pushup` |
| `--save-csv` | Save frame-wise features to CSV file |
| `--no-debug-video` | Disable saving debug video with pose overlays |

## Analysis Output

The pipeline will:

- Run MediaPipe Pose on each frame
- Compute exercise-specific features
- Apply heuristic rules for form evaluation
- Print an aggregate summary to the terminal
- Optionally save a debug video with pose overlays to `results/`
- Optionally export frame-wise features to CSV for further analysis

## Running Tests

```bash
python tests/test_utils.py
python tests/test_pipeline.py
```

## Extension Ideas

- Add support for **lunges** with their own feature sets and rules.
- Create a small comparison table: **Human vs System** labels for 3–5 videos.
- Capture screenshots and plots for your final paper and presentation.
- Add visualization of feature trends over time using matplotlib.

## Course Alignment

This repository is structured to support:

1. Pose estimation pipeline
2. Feature extraction
3. Rule-based evaluation logic
4. Complete pipeline from video to summary
5. Evaluation report with small human vs system comparison
6. Short presentation video

You can place your final write-up in `docs/` (e.g., `docs/final_report.md`) and any slides or video scripts in `presentation/`.