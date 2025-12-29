# Continuum Vision POC

**Vision Language Model Pipeline Testing & Accuracy Validation**

Version: 0.2.0 | December 2025 | Continuum Labs

---

## Purpose

A testing framework for evaluating Vision Language Model (VLM) accuracy on counting and scene understanding tasks. While initially focused on childcare compliance monitoring, the goal is to **build robust agentic VLM pipelines** that improve accuracy across diverse scene types.

---

## Current Results

### Static Image Testing (4 images, 5 strategies)

| Strategy | Adult Exact | Child ±2 | API Calls | Verdict |
|----------|-------------|----------|-----------|---------|
| **Basic** | 75% | 100% | 1 | ✅ **USE THIS** |
| Chain-of-Thought | 75% | 100% | 1 | Same accuracy |
| Two-Stage | 75% | 50% | 1 | ❌ Hurts child count |
| Multi-Shot (3x) | 75% | 75% | 3 | ❌ No improvement |
| Verification | 75% | 100% | 2 | ❌ 2x cost, no gain |

### Video Frame Testing (14 frames from stock footage)

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Adult ±1 | 86% | >80% | ✅ Pass |
| Child ±2 | 86% | >90% | ⚠️ Close |
| Easy scenes | 100% | ~100% | ✅ Pass |
| Compliance detection | 100% | 100% | ✅ Pass |
| Processing time | 1.4s/frame | <5s | ✅ Pass |

### Key Findings

**What Works:**
- Basic prompting performs as well as complex approaches
- Model reliably detects "no adults visible" (critical for safety)
- Simple/easy scenes achieve ~100% accuracy
- Consistent JSON output parsing

**What Doesn't Work:**
- Multi-shot consensus (errors are systematic, not random)
- Two-stage enumeration (confuses the model)
- Verification passes (model confirms its own mistakes)

**Known Limitations:**
- 75% ceiling on adult exact accuracy (model limitation)
- Misses adults in busy backgrounds
- Age ambiguity (10-12 year olds sometimes classified as adults)
- Overcounts children in motion-blur frames

---

## Project Structure

```
continuum-vision-poc/
├── README.md                          # This file
├── .env                               # API credentials (not committed)
├── requirements.txt                   # Python dependencies
├── test_comprehensive.py              # Full prompt strategy comparison
│
├── src/
│   ├── __init__.py
│   ├── test_counting.py               # Basic counting test
│   ├── test_with_evidence.py          # Evidence-based counting
│   ├── test_consensus.py              # Multi-inference voting
│   ├── frame_extractor.py             # Extract frames from video files
│   ├── video_pipeline.py              # End-to-end video analysis pipeline
│   ├── analyse_accuracy.py            # Compare VLM output to ground truth
│   └── compliance_monitor.py          # Production monitoring (WIP)
│
├── test_data/
│   ├── frames/                        # Static test images
│   │   ├── 1.jpg                      # 1 adult, 1 child
│   │   ├── 2.jpg                      # 2 adults, 5 children
│   │   ├── 3.jpg                      # 1 adult, 6 children
│   │   └── 4.jpg                      # 0 adults, 3 children
│   ├── videos/                        # Source video files
│   │   └── ChildcareCentreScenes.mp4  # TAFE NSW footage (2:20)
│   ├── extracted/                     # Frames extracted from videos
│   │   └── ChildcareCentreScenes/
│   │       ├── frame_000001.jpg       # Extracted at 10s intervals
│   │       ├── ...
│   │       ├── metadata.json          # Video metadata
│   │       ├── analysis_results.json  # VLM analysis output
│   │       ├── ground_truth.json      # Manual annotations
│   │       ├── accuracy_report.json   # Accuracy metrics
│   │       └── accuracy_report.md     # Human-readable report
│   ├── ground_truth.csv               # Static image ground truth
│   └── scenarios.md                   # Test scenario descriptions
│
├── docs/
│   ├── MODEL_SPECIFICATION.md         # Llama 3.2 11B Vision specs
│   ├── TESTING_LOG.md                 # Chronological testing notes
│   ├── TEST_RESULTS_SUMMARY.md        # Strategy comparison results
│   ├── accuracy_report.md             # Latest accuracy analysis
│   └── failure_modes.md               # Documented limitations
│
├── discovery/                         # API exploration scripts
│   ├── 01_test_text_only.py
│   ├── 02_test_vision_input.py
│   └── 03_test_formats.py
│
├── prompts/
│   └── v1_baseline.txt                # Current best prompt
│
├── config/
│   ├── ground_truth.json              # Ground truth annotations
│   └── settings.py                    # Configuration
│
└── evaluation/
    ├── run_evaluation.py              # Batch evaluation runner
    └── calculate_metrics.py           # Metrics calculation
```

---

## Quick Start

### Installation

```bash
cd ~/continuum-workspace/continuum-vision-poc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or: source ~/venvs/vision-poc/bin/activate

# Install dependencies
pip install openai python-dotenv pillow

# Configure API
cat > .env << 'EOF'
RESETDATA_BASE_URL=https://models.au-syd.resetdata.ai/v1
RESETDATA_API_KEY=your_key_here
RESETDATA_VISION_MODEL=meta/llama-3.2-11b-vision-instruct:shared
EOF
```

### Run Tests

```bash
# 1. Test prompt strategies on static images
python test_comprehensive.py

# 2. Extract frames from video (every 10 seconds)
python src/frame_extractor.py test_data/videos/YourVideo.mp4 --interval 10

# 3. Run VLM analysis on extracted frames
python src/video_pipeline.py test_data/extracted/YourVideo/

# 4. Compare to ground truth (after manual annotation)
python src/analyse_accuracy.py test_data/extracted/YourVideo/
```

---

## Core Scripts

### `src/frame_extractor.py`

Extract frames from video files at fixed intervals using FFmpeg.

```bash
# Extract frame every 10 seconds
python src/frame_extractor.py video.mp4 --interval 10

# Extract every 5 seconds as PNG
python src/frame_extractor.py video.mp4 --interval 5 --format png

# Custom output directory
python src/frame_extractor.py video.mp4 --output ./my_frames/
```

**Output:**
- `frame_000001.jpg`, `frame_000002.jpg`, ...
- `metadata.json` with video info and timestamps

### `src/video_pipeline.py`

End-to-end pipeline: extract frames → VLM analysis → structured results.

```bash
# Analyse video (extracts frames automatically)
python src/video_pipeline.py video.mp4 --interval 10

# Analyse pre-extracted frames
python src/video_pipeline.py --frames-dir test_data/extracted/MyVideo/
```

**Output:**
- `analysis_results.json` with counts per frame
- Summary statistics printed to console

### `src/analyse_accuracy.py`

Compare VLM predictions against ground truth annotations.

```bash
# Run accuracy analysis (requires ground_truth.json in directory)
python src/analyse_accuracy.py test_data/extracted/ChildcareCentreScenes/
```

**Output:**
- `accuracy_report.json` (machine-readable)
- `accuracy_report.md` (human-readable with tables)
- Breakdown by difficulty level and scene type

### `test_comprehensive.py`

Compare multiple prompt strategies on static test images.

```bash
python test_comprehensive.py
```

Tests: Basic, Chain-of-Thought, Two-Stage, Multi-Shot, Verification

---

## Ground Truth Format

### For Video Frames (`ground_truth.json`)

```json
{
  "video_name": "ChildcareCentreScenes",
  "annotator": "human",
  "annotation_date": "2025-12-29",
  "frames": {
    "frame_000001.jpg": {
      "timestamp": "00:00",
      "adult_count": 4,
      "child_count": 7,
      "adult_count_range": [3, 5],
      "child_count_range": [6, 8],
      "difficulty": "hard",
      "scene_type": "indoor_classroom",
      "notes": "Title overlay partially obscures view",
      "compliance_scenario": null
    },
    "frame_000009.jpg": {
      "timestamp": "01:20",
      "adult_count": 0,
      "child_count": 1,
      "difficulty": "easy",
      "scene_type": "outdoor_amphitheatre",
      "compliance_scenario": "potential_supervision_gap"
    }
  }
}
```

### For Static Images (`ground_truth.csv`)

```csv
filename,adult_count,child_count,difficulty,notes
1.jpg,1,1,easy,Clear indoor scene
2.jpg,2,5,medium,Group activity
3.jpg,1,6,medium,Reading circle
4.jpg,0,3,easy,Children only
```

---

## Model Details

**Current Model:** Llama 3.2 11B Vision (`meta/llama-3.2-11b-vision-instruct`)

| Spec | Value |
|------|-------|
| Parameters | 10.6B |
| Context Length | 128K tokens |
| Max Image Resolution | 1120×1120 |
| VQAv2 Benchmark | 75.2% |
| DocVQA Benchmark | 88.4% |

**Best Prompt (Basic):**
```
Count the adults and children visible in this childcare image.
Respond with JSON only: {"adult_count": X, "child_count": Y}
```

See [docs/MODEL_SPECIFICATION.md](docs/MODEL_SPECIFICATION.md) for full technical details.

---

## Accuracy by Scene Difficulty

From video frame testing (14 frames):

| Difficulty | Frames | Adult ±1 | Child ±2 |
|------------|--------|----------|----------|
| Easy | 5 | 100% | 100% |
| Medium | 6 | 83% | 83% |
| Hard | 3 | 67% | 67% |

**Insight:** Model accuracy is highly correlated with scene complexity. Easy scenes (≤5 people, clear visibility) achieve near-perfect accuracy.

---

## Future Directions

### Agentic Pipeline Strategies to Test

| Strategy | Concept | Hypothesis |
|----------|---------|------------|
| Region-based | Divide image into quadrants, count each | Reduces complexity per inference |
| Zoom-and-count | Crop busy areas, re-analyse at detail | Focuses attention on problem regions |
| Confidence routing | Simple → basic; complex → multi-step | Adaptive compute allocation |
| Self-correction | Count, then "check your work" prompt | May catch obvious errors |
| Scene-first | Describe scene → extract counts | Forces reasoning before counting |
| Multi-model ensemble | Compare outputs across models | Different models, different biases |

### Diverse Scene Types Needed

| Scene Type | Purpose |
|------------|---------|
| Crowded retail | High density, occlusion |
| Empty rooms | Baseline validation |
| Outdoor spaces | Variable lighting, motion |
| Office/meetings | Adults only |
| Public transit | Extreme counts |
| Sports/events | Mass gatherings |

---

## Development Notes

### Environment

```bash
# Activate virtual environment
source ~/venvs/vision-poc/bin/activate

# Check API connectivity
python -c "from openai import OpenAI; print('OK')"
```

### Adding New Test Videos

1. Place video in `test_data/videos/`
2. Extract frames: `python src/frame_extractor.py test_data/videos/new.mp4`
3. Manually annotate: Create `ground_truth.json` in extracted directory
4. Run analysis: `python src/analyse_accuracy.py test_data/extracted/new/`

### API Rate Limiting

Default 0.5s delay between API calls. Adjust with `--delay` parameter:

```bash
python src/video_pipeline.py video.mp4 --delay 1.0  # 1 second between calls
```

---

## License

Proprietary - Continuum Labs Pty Ltd

---

## Document History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-12-29 | Initial POC documentation |
| 0.2.0 | 2025-12-29 | Added video pipeline, accuracy analysis, testing results |