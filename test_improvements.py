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

# Strategy 1: Chain-of-thought with explicit counting
PROMPT_COT = """Look at this childcare image carefully. 

Step 1: Scan left to right across the image
Step 2: For each ADULT (18+ years, staff/teacher), note their position
Step 3: For each CHILD (under 5 years), note their position
Step 4: Count your lists

CRITICAL: Only count people whose face OR significant body is visible. 
Do NOT assume adults exist if you cannot see them.
Hands alone do not count as a person unless attached to a visible body.

Respond in JSON:
{
  "adult_positions": ["position1", "position2"],
  "child_positions": ["position1", "position2"],
  "adult_count": <length of adult_positions>,
  "child_count": <length of child_positions>
}

JSON only."""

# Strategy 2: Conservative bias
PROMPT_CONSERVATIVE = """Analyze this childcare image for compliance monitoring.

YOUR BIAS: When uncertain, undercount rather than overcount.
- If you're not sure if someone is an adult or older child, count as child
- If you only see partial body, don't count
- If you cannot clearly identify a person, don't count

Count visible people:
{
  "adult_count": <number - err on lower side>,
  "child_count": <number - err on lower side>,
  "confidence": "high" | "medium" | "low"
}

JSON only."""

# Strategy 3: Range-based (our actual product approach)
PROMPT_RANGES = """Analyze this childcare image for compliance monitoring.

Instead of exact counts, provide ranges to account for uncertainty:

{
  "adult_range": {"min": X, "max": Y},
  "child_range": {"min": X, "max": Y},
  "supervision_assessment": "adequate" | "potentially_inadequate" | "no_supervision_visible",
  "activity": "brief description"
}

Guidelines:
- min = people you're highly confident about
- max = including uncertain/partial figures
- If no adults visible at all, use {"min": 0, "max": 0}

JSON only."""

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
    print(f"TESTING: {prompt_name}")
    print(f"{'='*70}")
    
    results = []
    frames_dir = Path('test_data/frames')
    
    for img_path in sorted(frames_dir.glob('*.jpg'))[:5]:  # Skip test_real
        filename = img_path.name
        if filename == 'test_real.jpg':
            continue
            
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
            max_tokens=600,
            temperature=0.0  # Deterministic
        )
        
        data = extract_json(response.choices[0].message.content)
        gt = GROUND_TRUTH.get(filename, {})
        
        if data:
            # Handle different response formats
            if 'adult_range' in data:
                vlm_adults_min = data['adult_range']['min']
                vlm_adults_max = data['adult_range']['max']
                gt_adults = gt.get('adults', 0)
                adult_ok = vlm_adults_min <= gt_adults <= vlm_adults_max
                print(f"{filename}: Adults {vlm_adults_min}-{vlm_adults_max} (truth: {gt_adults}) {'✅' if adult_ok else '❌'}")
            else:
                vlm_adults = data.get('adult_count', 0)
                gt_adults = gt.get('adults', 0)
                adult_ok = vlm_adults == gt_adults
                print(f"{filename}: Adults {vlm_adults} (truth: {gt_adults}) {'✅' if adult_ok else '❌'}")
                results.append(adult_ok)
        else:
            print(f"{filename}: Parse failed")
    
    if results:
        acc = sum(results) / len(results) * 100
        print(f"\nAdult accuracy: {acc:.1f}%")

# Test all strategies
test_prompt("Chain-of-Thought", PROMPT_COT)
test_prompt("Conservative Bias", PROMPT_CONSERVATIVE)
test_prompt("Range-Based", PROMPT_RANGES)
