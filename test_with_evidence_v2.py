import os, base64, json, re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    base_url=os.getenv('RESETDATA_BASE_URL'),
    api_key=os.getenv('RESETDATA_API_KEY')
)

GROUND_TRUTH = {
    "daycare_scene.jpg": {"adults": 2, "children": 5},
    "childcare_02.jpg": {"adults": 1, "children": 1},
    "group_01.jpg": {"adults": 1, "children": 6},
    "group_02.jpg": {"adults": 0, "children": 4},
    "group_03.jpg": {"adults": 0, "children": 3},
    "test_real.jpg": {"adults": 0, "children": 1},
}

prompt = """Analyze this childcare image. Count ONLY people you can actually see.

CRITICAL RULES:
- Count only what is VISIBLE in the image
- Do NOT assume adults are present just because children are supervised
- If you only see hands, count based on hand size (small hands = children)
- If no adult face or body is visible, adult_count = 0

For each person, note their location. Respond in JSON:
{
  "adults": [
    {"location": "where in image", "description": "what they're wearing/doing"}
  ],
  "children": [
    {"location": "where in image", "description": "brief description"}
  ],
  "adult_count": <total from adults array>,
  "child_count": <total from children array>,
  "supervision_present": true/false,
  "activity": "what's happening"
}

If no adults visible, use: "adults": [], "adult_count": 0, "supervision_present": false

IMPORTANT: Start your response with { and end with }. No other text."""

def extract_json(text):
    """Extract JSON from response that may have preamble text"""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except:
        pass
    
    # Look for JSON object in text
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    
    # Try to extract counts from text even if JSON fails
    adult_match = re.search(r'"adult_count":\s*(\d+)', text)
    child_match = re.search(r'"child_count":\s*(\d+)', text)
    no_adults = "no visible adults" in text.lower() or '"adults": []' in text
    
    if adult_match or child_match or no_adults:
        return {
            "adult_count": 0 if no_adults else (int(adult_match.group(1)) if adult_match else None),
            "child_count": int(child_match.group(1)) if child_match else None,
            "adults": [],
            "parsed_from_text": True
        }
    
    return None

results = []
frames_dir = Path('test_data/frames')

for img_path in sorted(frames_dir.glob('*.jpg')):
    filename = img_path.name
    print(f"\n{'='*70}")
    print(f"Image: {filename}")
    
    with open(img_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    response = client.chat.completions.create(
        model=os.getenv('RESETDATA_VISION_MODEL'),
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }],
        max_tokens=800,
        temperature=0.1
    )
    
    result = response.choices[0].message.content
    data = extract_json(result)
    
    if data:
        vlm_adults = data.get('adult_count', 0)
        vlm_children = data.get('child_count', 0)
        
        gt = GROUND_TRUTH.get(filename, {})
        gt_adults = gt.get('adults', '?')
        gt_children = gt.get('children', '?')
        
        adult_diff = vlm_adults - gt_adults if gt_adults != '?' else '?'
        child_diff = vlm_children - gt_children if gt_children != '?' else '?'
        
        adult_ok = "✅" if adult_diff == 0 else f"❌ ({'+' if adult_diff > 0 else ''}{adult_diff})"
        child_ok = "✅" if abs(child_diff) <= 1 else f"❌ ({'+' if child_diff > 0 else ''}{child_diff})"
        
        print(f"VLM:    Adults={vlm_adults}, Children={vlm_children}")
        print(f"TRUTH:  Adults={gt_adults}, Children={gt_children}")
        print(f"RESULT: Adults {adult_ok} | Children {child_ok}")
        
        if data.get('parsed_from_text'):
            print(f"(Parsed from text, not clean JSON)")
        
        results.append({
            "image": filename,
            "vlm_adults": vlm_adults,
            "vlm_children": vlm_children,
            "gt_adults": gt_adults,
            "gt_children": gt_children,
            "adult_accurate": adult_diff == 0,
            "child_within_1": abs(child_diff) <= 1 if child_diff != '?' else None,
            "child_within_2": abs(child_diff) <= 2 if child_diff != '?' else None
        })
    else:
        print(f"FAILED to parse")
        print(f"Raw: {result[:300]}...")

# Summary
print(f"\n{'='*70}")
print("FINAL ACCURACY SUMMARY")
print(f"{'='*70}")
valid = [r for r in results if r['adult_accurate'] is not None]
if valid:
    adult_acc = sum(1 for r in valid if r['adult_accurate']) / len(valid) * 100
    child_1 = sum(1 for r in valid if r['child_within_1']) / len(valid) * 100
    child_2 = sum(1 for r in valid if r['child_within_2']) / len(valid) * 100
    
    print(f"Images tested:              {len(valid)}")
    print(f"Adult accuracy (exact):     {adult_acc:.1f}%  (target: >90%)")
    print(f"Child accuracy (within ±1): {child_1:.1f}%")
    print(f"Child accuracy (within ±2): {child_2:.1f}%  (target: >80%)")
    print(f"\nJSON parse rate:            {len(valid)}/{len(list(frames_dir.glob('*.jpg')))*100//len(list(frames_dir.glob('*.jpg')))}%")
