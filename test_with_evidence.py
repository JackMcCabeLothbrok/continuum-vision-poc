import os, base64, json
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

# KEY CHANGE: Require location evidence for each person
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

JSON only, no other text."""

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
        max_tokens=600,
        temperature=0.1
    )
    
    result = response.choices[0].message.content
    
    try:
        data = json.loads(result.strip())
        vlm_adults = data.get('adult_count', 0)
        vlm_children = data.get('child_count', 0)
        adults_list = data.get('adults', [])
        
        gt = GROUND_TRUTH.get(filename, {})
        gt_adults = gt.get('adults', '?')
        gt_children = gt.get('children', '?')
        
        adult_diff = vlm_adults - gt_adults if gt_adults != '?' else '?'
        child_diff = vlm_children - gt_children if gt_children != '?' else '?'
        
        adult_ok = "✅" if adult_diff == 0 else f"❌ ({'+' if adult_diff > 0 else ''}{adult_diff})"
        child_ok = "✅" if abs(child_diff) <= 1 else f"❌ ({'+' if child_diff > 0 else ''}{child_diff})"
        
        print(f"{'='*70}")
        print(f"VLM:    Adults={vlm_adults}, Children={vlm_children}")
        print(f"TRUTH:  Adults={gt_adults}, Children={gt_children}")
        print(f"RESULT: Adults {adult_ok} | Children {child_ok}")
        
        # Show the evidence for adults
        if adults_list:
            print(f"\nAdult evidence:")
            for i, adult in enumerate(adults_list, 1):
                loc = adult.get('location', 'unknown')
                desc = adult.get('description', 'unknown')
                print(f"  {i}. Location: {loc} | {desc}")
        else:
            print(f"\nNo adults detected (empty list)")
            
        print(f"\nSupervision present: {data.get('supervision_present')}")
        print(f"Activity: {data.get('activity')}")
        
    except Exception as e:
        print(f"Parse failed: {e}")
        print(f"Raw: {result[:500]}")

print(f"\n{'='*70}")
print("KEY: Does requiring evidence reduce adult hallucination?")
print("Watch for group_02.jpg and group_03.jpg - should show 0 adults")
print(f"{'='*70}")
