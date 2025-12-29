"""
Test multi-inference consensus for improved counting accuracy.
Hypothesis: 3 runs with majority vote > 1 run accuracy
"""
import os, base64, json, re
from pathlib import Path
from collections import Counter
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

PROMPT = """Analyze this childcare image for compliance monitoring.

Count the people visible:
- ADULTS: People who appear 18+ years old (staff, teachers, parents)
- CHILDREN: Young children (under 5 years old)

CRITICAL RULES:
1. Only count people whose face OR significant body is clearly visible
2. Do NOT assume adults exist if you cannot see them
3. Hands alone do not count as a person

Respond with JSON only:
{"adult_count": X, "child_count": Y}"""

def extract_json(text):
    try:
        return json.loads(text.strip())
    except:
        match = re.search(r'\{[^}]+\}', text)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return None

def analyze_image(img_b64, temperature=0.3):
    """Single inference"""
    response = client.chat.completions.create(
        model=os.getenv('RESETDATA_VISION_MODEL'),
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }],
        max_tokens=100,
        temperature=temperature  # Some variance for diversity
    )
    return extract_json(response.choices[0].message.content)

def consensus_vote(results, method='median'):
    """Aggregate multiple results"""
    adult_counts = [r['adult_count'] for r in results if r]
    child_counts = [r['child_count'] for r in results if r]
    
    if not adult_counts:
        return None
    
    if method == 'median':
        adult_counts.sort()
        child_counts.sort()
        mid = len(adult_counts) // 2
        return {
            'adult_count': adult_counts[mid],
            'child_count': child_counts[mid]
        }
    elif method == 'mode':
        return {
            'adult_count': Counter(adult_counts).most_common(1)[0][0],
            'child_count': Counter(child_counts).most_common(1)[0][0]
        }
    elif method == 'min':  # Conservative
        return {
            'adult_count': min(adult_counts),
            'child_count': min(child_counts)
        }

print("="*70)
print("TESTING: Single-shot vs 3-shot Consensus")
print("="*70)

frames_dir = Path('test_data/frames')
single_correct = 0
consensus_correct = 0
total = 0

for img_path in sorted(frames_dir.glob('*.jpg')):
    filename = img_path.name
    if filename not in GROUND_TRUTH:
        continue
    
    gt = GROUND_TRUTH[filename]
    total += 1
    
    with open(img_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    print(f"\n--- {filename} (Truth: {gt['adults']}A, {gt['children']}C) ---")
    
    # Single shot (temperature=0 for deterministic)
    single = analyze_image(img_b64, temperature=0.0)
    
    # 3-shot with slight temperature for variance
    run1 = analyze_image(img_b64, temperature=0.3)
    run2 = analyze_image(img_b64, temperature=0.3)
    run3 = analyze_image(img_b64, temperature=0.3)
    
    runs = [run1, run2, run3]
    consensus = consensus_vote(runs, method='median')
    
    # Show individual runs
    print(f"  Single (t=0):  {single}")
    print(f"  Run 1 (t=0.3): {run1}")
    print(f"  Run 2 (t=0.3): {run2}")
    print(f"  Run 3 (t=0.3): {run3}")
    print(f"  Consensus:     {consensus}")
    
    # Check accuracy
    if single and single['adult_count'] == gt['adults']:
        single_correct += 1
        print(f"  Single adult: ✅")
    else:
        print(f"  Single adult: ❌")
    
    if consensus and consensus['adult_count'] == gt['adults']:
        consensus_correct += 1
        print(f"  Consensus adult: ✅")
    else:
        print(f"  Consensus adult: ❌")

print(f"\n{'='*70}")
print(f"RESULTS:")
print(f"  Single-shot adult accuracy: {single_correct}/{total} = {single_correct/total*100:.1f}%")
print(f"  3-shot consensus accuracy:  {consensus_correct}/{total} = {consensus_correct/total*100:.1f}%")
print(f"{'='*70}")
