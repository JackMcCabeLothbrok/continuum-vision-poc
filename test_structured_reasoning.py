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
}

# Strategy 1: Binary questions (like "which figure is different")
PROMPT_BINARY = """Look at this childcare image.

Answer these YES/NO questions:
1. Is there at least one ADULT visible in this image? (Look for: taller person, mature features, staff clothing)
2. Are there CHILDREN visible? 
3. Is active supervision occurring? (Adult watching/interacting with children)

Then estimate counts:
- Adults: Give a range (e.g., "1-2" or "0")
- Children: Give a range (e.g., "3-5")

Format:
{
  "adult_visible": true/false,
  "children_visible": true/false,
  "supervision_active": true/false,
  "adult_estimate": "X-Y",
  "child_estimate": "X-Y"
}

JSON only."""

# Strategy 2: Entity extraction style
PROMPT_ENTITY_STYLE = """You are analyzing a childcare facility image.

Task: Extract information about each PERSON visible in the image.

For each person you can see, create an entry:
{
  "person_id": 1,
  "position": "where in frame",
  "age_category": "adult" or "child",
  "visual_evidence": "why you classified them this way"
}

Then summarize:
{
  "people": [...entries above...],
  "total_adults": X,
  "total_children": Y
}

Only include people you can actually SEE (face or body visible).
Respond with JSON only."""

# Strategy 3: Math-style step-by-step reasoning
PROMPT_MATH_STYLE = """Look at this image like a visual puzzle.

Step 1: Identify each PERSON in the image (list them by position: left, center, right, background)
Step 2: For each person, determine: ADULT (appears 18+, larger body) or CHILD (small, young features)
Step 3: Count your lists

Example format:
- Person 1: center-left, ADULT (wearing blue, standing tall)
- Person 2: center, CHILD (small, sitting)
- Person 3: right, CHILD (small features)

After listing all people, provide final count in JSON:
{"adult_count": X, "child_count": Y}

IMPORTANT: If you cannot see a person clearly (only hands, or blocked), note it but don't count.
Begin your analysis:"""

def extract_json(text):
    try:
        return json.loads(text.strip())
    except:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None

def test_prompt(prompt_name, prompt_text):
    print(f"\n{'='*70}")
    print(f"STRATEGY: {prompt_name}")
    print(f"{'='*70}")
    
    frames_dir = Path('test_data/frames')
    
    for img_path in sorted(frames_dir.glob('*.jpg'))[:5]:
        filename = img_path.name
        if filename == 'test_real.jpg':
            continue
            
        gt = GROUND_TRUTH.get(filename, {})
        
        with open(img_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode()
        
        response = client.chat.completions.create(
            model=os.getenv('RESETDATA_VISION_MODEL'),
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }],
            max_tokens=800,
            temperature=0.0
        )
        
        result = response.choices[0].message.content
        
        print(f"\n--- {filename} (Truth: {gt.get('adults')} adults, {gt.get('children')} children) ---")
        
        # For math-style, show the reasoning
        if "Step 1" in prompt_text:
            print(f"Reasoning: {result[:500]}...")
            
        data = extract_json(result)
        if data:
            if 'adult_visible' in data:
                # Binary style
                gt_has_adult = gt.get('adults', 0) > 0
                pred_has_adult = data.get('adult_visible', False)
                adult_ok = "✅" if gt_has_adult == pred_has_adult else "❌"
                print(f"Adult visible: {pred_has_adult} (truth: {gt_has_adult}) {adult_ok}")
                print(f"Estimates: Adults={data.get('adult_estimate')}, Children={data.get('child_estimate')}")
            elif 'total_adults' in data:
                # Entity style
                vlm = data.get('total_adults', 0)
                ok = "✅" if vlm == gt.get('adults') else "❌"
                print(f"Adults: {vlm} (truth: {gt.get('adults')}) {ok}")
                if 'people' in data:
                    print(f"People entries: {len(data.get('people', []))}")
            else:
                # Standard style
                vlm = data.get('adult_count', 0)
                ok = "✅" if vlm == gt.get('adults') else "❌"
                print(f"Adults: {vlm} (truth: {gt.get('adults')}) {ok}")
        else:
            print(f"Parse failed. Raw: {result[:200]}...")

print("Testing 3 strategies inspired by AWS success patterns...\n")

test_prompt("Binary Questions", PROMPT_BINARY)
test_prompt("Entity Extraction", PROMPT_ENTITY_STYLE)
test_prompt("Math-Style Reasoning", PROMPT_MATH_STYLE)
