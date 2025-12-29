# VLM Failure Modes Documentation

## Purpose

Document when and why the VLM fails to accurately analyze childcare scenes. This honest assessment is crucial for:

1. Setting accurate expectations with investors
2. Identifying prompt improvements
3. Understanding product limitations
4. Guiding future model/fine-tuning decisions

---

## Counting Failures

### Over-Counting Patterns

**When does the model count more people than present?**

| Scenario | Frequency | Example | Possible Cause |
|----------|-----------|---------|----------------|
| | | | |

### Under-Counting Patterns

**When does the model count fewer people than present?**

| Scenario | Frequency | Example | Possible Cause |
|----------|-----------|---------|----------------|
| | | | |

---

## JSON Parse Failures

### Common Patterns

| Pattern | Frequency | Example |
|---------|-----------|---------|
| Markdown code blocks | | Model wraps JSON in \`\`\`json...\`\`\` |
| Preamble text | | "Here is my analysis:" before JSON |
| Trailing explanation | | JSON followed by additional commentary |
| Invalid JSON syntax | | Missing quotes, trailing commas |

### Mitigation Strategies

1. **Current prompt adjustments:**
   - 

2. **Post-processing:**
   - Strip markdown code blocks
   - Extract JSON from mixed content

---

## Scenario-Specific Failures

### Sleep Rooms (Low Light)

**Issues observed:**
- 

**Root cause:**
- 

**Mitigation:**
- 

### High Occupancy (10+ children)

**Issues observed:**
- 

**Root cause:**
- 

**Mitigation:**
- 

### Partial Occlusion

**Issues observed:**
- 

**Root cause:**
- 

**Mitigation:**
- 

### Transitions (People Moving)

**Issues observed:**
- 

**Root cause:**
- 

**Mitigation:**
- 

---

## Known Limitations (By Design)

These are fundamental limitations that cannot be fixed with prompting:

### Cannot Detect
1. **Grooming behaviour** - Relational/verbal, unfolds over months
2. **Emotional states** - Cannot reliably assess distress, anxiety
3. **Interaction quality** - Can see interaction occurs, not assess quality
4. **Audio cues** - No audio processing in current architecture

### Cannot Guarantee
1. **Precise counts** - Use ranges (±1 adults, ±2 children)
2. **100% coverage** - Camera blind spots exist
3. **Real-time alerts** - Processing latency 1-3 seconds

---

## Investor Communication

### How to Present Limitations

**Frame as "Honest About What We Can't Do":**

> "We're transparent about limitations because overclaiming would backfire in a sensitive sector like childcare. Our system is one layer in comprehensive protection—it doesn't replace proper staffing, training, or cultural change."

### Specific Talking Points

1. **Counting accuracy:** "We provide estimates, not precise counts. All flags prompt human review."

2. **Grooming detection:** "Our cameras see physical environments, not relationships. Grooming is detected through reporting culture and professional vigilance, not AI."

3. **Not surveillance:** "We document positive practice, not surveil educators. The output is compliance reports, not watched footage."

---

## Improvement Roadmap

### Short-term (Prompt Engineering)
- [ ] Improve JSON formatting instructions
- [ ] Add confidence calibration guidance
- [ ] Test structured output formats

### Medium-term (Model Selection)
- [ ] Evaluate 90B model for accuracy improvement
- [ ] Test Qwen2.5-VL as alternative
- [ ] Compare latency vs accuracy tradeoffs

### Long-term (Fine-tuning)
- [ ] Collect operational data with consent
- [ ] Develop childcare-specific fine-tuning dataset
- [ ] Evaluate domain-specific model improvements
