# Test Scenarios for VLM Evaluation

## Required Coverage

We need test images covering these scenarios to properly evaluate the model's capability for childcare compliance monitoring.

### Scenario Types

| Scenario | Description | Priority | Min Frames |
|----------|-------------|----------|------------|
| `indoor_play` | Children playing indoors, typical daytime activity | High | 10 |
| `outdoor` | Playground, outdoor play areas | High | 10 |
| `sleep_room` | Rest/nap time, low lighting | High | 5 |
| `meal_time` | Eating at tables | Medium | 5 |
| `transitions` | People entering/leaving rooms | Medium | 5 |
| `high_occupancy` | 10+ children in frame | High | 5 |

### Lighting Conditions

| Condition | Description | Target Frames |
|-----------|-------------|---------------|
| `natural` | Daylight through windows | 40% |
| `artificial` | Indoor fluorescent/LED only | 30% |
| `mixed` | Combination of natural + artificial | 20% |
| `dim` | Low light (sleep rooms) | 10% |

### Occlusion Levels

| Level | Description | % Frame Affected |
|-------|-------------|------------------|
| `low` | <20% people partially obscured | Most frames |
| `medium` | 20-50% partial obstruction | Some frames |
| `high` | >50% obstruction | Few frames (edge cases) |

## File Naming Convention

```
{scenario}_{lighting}_{occlusion}_{sequence}.jpg
```

Examples:
- `indoor_play_natural_low_001.jpg`
- `outdoor_natural_medium_003.jpg`
- `sleep_room_dim_low_002.jpg`

## Ground Truth Recording

For each frame, record in `ground_truth.csv`:

1. `frame_id` - Unique identifier
2. `filename` - Exact filename
3. `adult_count` - Number of adults (verified by human)
4. `child_count` - Number of children (verified by human)
5. `scenario` - Scenario type from list above
6. `lighting` - Lighting condition
7. `occlusion_level` - How much obstruction
8. `notes` - Any relevant details

## Sourcing Options

### Option 1: Stock Footage (Recommended for POC)
- Shutterstock, Getty, Adobe Stock
- Search: "childcare", "daycare", "preschool classroom"
- Budget: $150-300 for 5-10 clips
- Extract frames at 1 fps

### Option 2: Partner Centre
- Film staff only (no children)
- Document consent process
- Best for realistic environment

### Option 3: Staged Footage
- Adult volunteers only
- Good for testing edge cases
- Full control over scenarios

## Target: 50+ Frames Total

| Scenario | Target |
|----------|--------|
| indoor_play | 15 |
| outdoor | 15 |
| sleep_room | 5 |
| meal_time | 5 |
| transitions | 5 |
| high_occupancy | 5 |
| **Total** | **50** |
