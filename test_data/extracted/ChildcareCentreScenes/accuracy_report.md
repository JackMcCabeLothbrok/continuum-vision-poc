# VLM Accuracy Analysis Report

**Date:** 2025-12-29T05:19:29.250235

**Model:** `meta/llama-3.2-11b-vision-instruct:shared`

**Frames Directory:** `test_data/extracted/ChildcareCentreScenes`

## Overall Accuracy

| Metric | Adults | Children |
|--------|--------|----------|
| Exact Match | 8/14 (57.1%) | 11/14 (78.6%) |
| Within Range | 10/14 (71.4%) | 12/14 (85.7%) |
| Within ±1/±2 | 12/14 (85.7%) | 12/14 (85.7%) |

## Accuracy by Difficulty

| Difficulty | Frames | Adult ±1 | Child ±2 |
|------------|--------|----------|----------|
| Easy | 5 | 100% | 100% |
| Medium | 6 | 83% | 83% |
| Hard | 3 | 67% | 67% |

## Accuracy by Scene Type

| Scene | Frames | Adult ±1 | Child ±2 |
|-------|--------|----------|----------|
| indoor_classroom | 3 | 67% | 67% |
| outdoor_amphitheatre | 1 | 100% | 100% |
| outdoor_playground | 2 | 100% | 100% |
| outdoor_sandpit | 1 | 100% | 100% |
| indoor_craft | 6 | 83% | 83% |
| outdoor_fence | 1 | 100% | 100% |

## Frame-by-Frame Results

| Frame | Time | GT Adults | Pred Adults | GT Children | Pred Children | Adult | Child |
|-------|------|-----------|-------------|-------------|---------------|-------|-------|
| frame_000001.jpg | 00:00 | 4 | 3 | 7 | 7 | ✅ | ✅ |
| frame_000002.jpg | 00:10 | 5 | 3 | 6 | 6 | ❌ | ✅ |
| frame_000003.jpg | 00:20 | 3 | 3 | 5 | 8 | ✅ | ❌ |
| frame_000004.jpg | 00:30 | 3 | 3 | 3 | 3 | ✅ | ✅ |
| frame_000005.jpg | 00:40 | 6 | 4 | 6 | 6 | ❌ | ✅ |
| frame_000006.jpg | 00:50 | 4 | 3 | 5 | 8 | ✅ | ❌ |
| frame_000007.jpg | 01:00 | 4 | 4 | 6 | 7 | ✅ | ✅ |
| frame_000008.jpg | 01:10 | 3 | 3 | 6 | 6 | ✅ | ✅ |
| frame_000009.jpg | 01:20 | 0 | 0 | 1 | 1 | ✅ | ✅ |
| frame_000010.jpg | 01:30 | 0 | 0 | 1 | 1 | ✅ | ✅ |
| frame_000011.jpg | 01:40 | 1 | 0 | 0 | 0 | ✅ | ✅ |
| frame_000012.jpg | 01:50 | 0 | 0 | 0 | 0 | ✅ | ✅ |
| frame_000013.jpg | 02:00 | 2 | 1 | 1 | 1 | ✅ | ✅ |
| frame_000014.jpg | 02:10 | 0 | 0 | 1 | 1 | ✅ | ✅ |

## Compliance Scenario Detection

| Frame | Scenario | Detected |
|-------|----------|----------|
| frame_000009.jpg | potential_supervision_gap | ✅ Yes |

## Failure Analysis

### Frames with Adult Errors (outside ±1)

- **frame_000002.jpg** (indoor_classroom, medium): GT=5, Pred=3 — undercounted by 2
  - Notes: Clear shot. 3 TAFE students in uniform + 2 other adults. 6 toddlers at tables.

- **frame_000005.jpg** (indoor_craft, hard): GT=6, Pred=4 — undercounted by 2
  - Notes: Wide shot with many people. Multiple activity stations. Adults in background easily missed.

### Frames with Child Errors (outside ±2)

- **frame_000003.jpg** (indoor_classroom, hard): GT=5, Pred=8 — overcounted by 3
  - Notes: Motion blur. Carla in foreground. Background activity areas partially visible.

- **frame_000006.jpg** (indoor_craft, medium): GT=5, Pred=8 — overcounted by 3
  - Notes: Craft activity. Carla + 3-4 other adults. Children at table with playdough.

---
*Generated: 2025-12-29T05:19:29.250235*