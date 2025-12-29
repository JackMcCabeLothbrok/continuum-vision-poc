import os, base64, json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    base_url=os.getenv('RESETDATA_BASE_URL'),
    api_key=os.getenv('RESETDATA_API_KEY')
)

# Ground truth from manual inspection
GROUND_TRUTH = {
    "daycare_scene.jpg": {"adults": 2, "children": 5},  # The good daycare image
    "childcare_02.jpg": {"adults": 1, "children": 1},
    "group_01.jpg": {"adults": 1, "children": 6},
    "group_02.jpg": {"adults": 0, "children": 4},  # Hands-only, estimate
    "group_03.jpg": {"adults": 0, "children": 3},  # NO adult in this image
    "test_real.jpg": {"adults": 0, "children": 1},
}

prompt = """You are analyzing a childcare facility image for compliance monitoring.

Count the people visible and respond in JSON format:
{
  "adult_count": <number of adults/staff visible>,
  "child_count": <number of children visible>,
  "adult_confidence": "high" | "medium" | "low",
  "child_confidence": "high" | "medium" | "low",
  "activity": "<what children are doing>",
  "supervision_ratio": "<e.g. 1:5 if 1 adult and 5 children>"
}

Counting guidelines:
- Adults = anyone who appears 18+ or is clearly a staff member/teacher
- Children = anyone who appears under school age (roughly 0-5 years)
- Only count people you can actually see (face or significant body portion)
- Do NOT assume adults are present if you cannot see them
- If only hands visible, count based on hand size (small = child)

Respond with valid JSON only, no other text."""

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
        max_tokens=400,
        temperature=0.1
    )
    
    result = response.choices[0].message.content
    
    try:
        data = json.loads(result.strip())
        vlm_adults = data.get('adult_count', 0)
        vlm_children = data.get('child_count', 0)
        
        # Compare to ground truth
        gt = GROUND_TRUTH.get(filename, {})
        gt_adults = gt.get('adults', '?')
        gt_children = gt.get('children', '?')
        
        adult_diff = vlm_adults - gt_adults if gt_adults != '?' else '?'
        child_diff = vlm_children - gt_children if gt_children != '?' else '?'
        
        adult_ok = "✅" if adult_diff == 0 else f"❌ ({'+' if adult_diff > 0 else ''}{adult_diff})"
        child_ok = "✅" if child_diff == 0 else f"⚠️ ({'+' if child_diff > 0 else ''}{child_diff})"
        
        print(f"{'='*70}")
        print(f"VLM:    Adults={vlm_adults}, Children={vlm_children}")
        print(f"TRUTH:  Adults={gt_adults}, Children={gt_children}")
        print(f"RESULT: Adults {adult_ok} | Children {child_ok}")
        print(f"Activity: {data.get('activity')}")
        print(f"Ratio: {data.get('supervision_ratio')}")
        
        results.append({
            "image": filename,
            "vlm_adults": vlm_adults,
            "vlm_children": vlm_children,
            "gt_adults": gt_adults,
            "gt_children": gt_children,
            "adult_accurate": adult_diff == 0,
            "child_accurate": abs(child_diff) <= 1 if child_diff != '?' else None
        })
        
    except Exception as e:
        print(f"Parse failed: {e}")
        print(f"Raw: {result}")

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
valid = [r for r in results if r['adult_accurate'] is not None]
adult_acc = sum(1 for r in valid if r['adult_accurate']) / len(valid) * 100
child_acc = sum(1 for r in valid if r['child_accurate']) / len(valid) * 100
print(f"Adult accuracy (exact):     {adult_acc:.1f}%")
print(f"Child accuracy (within ±1): {child_acc:.1f}%")
