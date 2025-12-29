# Continuum Vision POC

Pre-investment validation of Vision Language Model capability for childcare compliance monitoring.

## Objective

Prove VLM counting accuracy before committing to Thor hardware investment (~$12K-15K per centre).

### Success Criteria

| Metric | Target |
|--------|--------|
| Adult count accuracy (within ±1) | >90% |
| Child count accuracy (within ±2) | >80% |
| JSON parse rate | >95% |

## Quick Start

```bash
# 1. Setup environment
conda create -n vision-poc python=3.11 -y
conda activate vision-poc
pip install -r requirements.txt

# 2. Configure API
cp .env.example .env
# Edit .env with your ResetData API key

# 3. Run discovery tests
python discovery/01_test_text_only.py
python discovery/02_test_vision_input.py
# If 02 fails: python discovery/03_test_formats.py

# 4. Add test images
# See test_data/scenarios.md for guidance

# 5. Run evaluation
python evaluation/run_evaluation.py
python evaluation/calculate_metrics.py
```

## Project Structure

```
continuum-vision-poc/
├── config/
│   └── settings.py           # Configuration management
├── discovery/                 # Phase 1: API format testing
│   ├── 01_test_text_only.py  # Verify connectivity
│   ├── 02_test_vision_input.py # Test image input
│   ├── 03_test_formats.py    # Alternative formats
│   └── results/              # Test outputs
├── test_data/
│   ├── frames/               # Test images (gitignored)
│   ├── ground_truth.csv      # Manual counts
│   └── scenarios.md          # Test coverage guide
├── prompts/
│   └── v1_baseline.txt       # Compliance prompt
├── evaluation/
│   ├── run_evaluation.py     # Process all frames
│   ├── calculate_metrics.py  # Compare to ground truth
│   └── results/              # Evaluation outputs
├── docs/
│   ├── api_discovery.md      # API behaviour log
│   ├── accuracy_report.md    # Results template
│   └── failure_modes.md      # Honest limitations
├── proxy/                    # Optional proxy (if needed)
│   └── src/
├── k8s/                      # K8s deployment (optional)
├── demo/                     # Investor demo materials
└── requirements.txt
```

## 7-Day Timeline

| Day | Phase | Tasks |
|-----|-------|-------|
| 1 | Discovery | API connectivity tests, format validation |
| 1-3 | Test Data | Source stock footage, extract frames |
| 3 | Ground Truth | Manual count all frames |
| 4-5 | Prompts | Iterate on compliance prompt |
| 5-6 | Evaluation | Run full evaluation, calculate metrics |
| 6-7 | Demo | Prepare investor materials |

## Key Files

| File | Purpose |
|------|---------|
| `prompts/v1_baseline.txt` | Compliance analysis prompt |
| `test_data/ground_truth.csv` | Human-verified counts |
| `docs/failure_modes.md` | Honest limitations |
| `docs/accuracy_report.md` | Results summary |

## Go/No-Go Decision

| Outcome | Action |
|---------|--------|
| All targets met | ✅ Proceed to Thor + pilot |
| Below targets | ⚠️ Test 90B model or improve prompts |
| Fundamental gap | ❌ Reassess before hardware |

## Model

Using `meta/llama-3.2-11b-vision-instruct:shared` via ResetData API.

Same model that will run on Thor hardware, validating capability without hardware investment.
