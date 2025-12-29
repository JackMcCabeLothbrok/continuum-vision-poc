# API Discovery Log

## ResetData Vision API Behaviour

Document findings from the discovery phase tests.

### Test 1: Text-Only Connectivity

**Date:** [Fill in]

**Result:** ☐ Success / ☐ Failed

**Observations:**
- Latency: ___s
- Response format: Standard OpenAI / Other
- Notes:

---

### Test 2: Vision Input (OpenAI Format)

**Date:** [Fill in]

**Result:** ☐ Success / ☐ Failed

**Format Tested:**
```json
{
  "model": "meta/llama-3.2-11b-vision-instruct:shared",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "..."},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]
  }]
}
```

**Observations:**
- Latency: ___s
- Image processed correctly: ☐ Yes / ☐ No
- Notes:

---

### Test 3: Alternative Formats (if needed)

**Formats Tested:**

| Format | Result | Notes |
|--------|--------|-------|
| OpenAI Standard | | |
| With input_type | | |
| Separate images field | | |
| Images in message | | |
| Base64 without data URI | | |

**Working Format:**

**Proxy Required:** ☐ Yes / ☐ No

---

## Key Findings

### What Works
- 

### What Doesn't Work
- 

### Recommended Approach
- 

---

## Comparison to Embedding API

The embedding proxy required `input_type` parameter translation.

**Vision API Requirements:**
- Similar translation needed: ☐ Yes / ☐ No
- Details:
