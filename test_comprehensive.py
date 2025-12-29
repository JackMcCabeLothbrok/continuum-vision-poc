#!/usr/bin/env python3
"""
test_comprehensive.py - Comprehensive VLM Counting Test

Tests multiple strategies to improve counting accuracy:
1. Basic counting (baseline)
2. Chain-of-thought reasoning (step-by-step)
3. Two-stage identification (find people, then classify)
4. Multi-shot consensus (3 runs, median vote)
5. Self-verification (count, then verify)

Usage: python test_comprehensive.py

Ground Truth:
  1.jpg: 1 adult, 1 child
  2.jpg: 2 adults, 5 children  
  3.jpg: 1 adult, 6 children
  4.jpg: 0 adults, 3 children (hallucination test)
"""

import os
import base64
import json
import re
import time
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize client
client = OpenAI(
    base_url=os.getenv('RESETDATA_BASE_URL'),
    api_key=os.getenv('RESETDATA_API_KEY')
)
MODEL = os.getenv('RESETDATA_VISION_MODEL')

# Ground truth for new images
GROUND_TRUTH = {
    "1.jpg": {"adults": 1, "children": 1, "notes": "Adult in blue, child in white"},
    "2.jpg": {"adults": 2, "children": 5, "notes": "Two women, 5 toddlers"},
    "3.jpg": {"adults": 1, "children": 6, "notes": "Woman reading to 6 children"},
    "4.jpg": {"adults": 0, "children": 3, "notes": "Children only - hallucination test"},
}

# ============================================================================
# PROMPTS - Different strategies
# ============================================================================

PROMPT_BASIC = """Count the adults and children visible in this childcare image.
Respond with JSON only: {"adult_count": X, "child_count": Y}"""

PROMPT_CHAIN_OF_THOUGHT = """You are analyzing a childcare facility image for compliance monitoring.

TASK: Count adults (18+ years) and children (under 6 years) visible in this image.

Think step by step:

STEP 1 - SCAN THE IMAGE
Look at the entire image from left to right, top to bottom.
List every person you can see, noting their approximate position.

STEP 2 - CLASSIFY EACH PERSON
For each person identified, determine if they are:
- ADULT: Full-grown person, staff member, teacher, or parent (18+)
- CHILD: Young child, toddler, preschool-aged (under 6)
- UNCERTAIN: Cannot determine (e.g., only partial view, teen)

STEP 3 - COUNT ONLY VISIBLE PEOPLE
Only count people whose face OR significant body portion is clearly visible.
Do NOT count:
- People you assume are there but cannot see
- Hands only
- Reflections

STEP 4 - FINAL COUNT
Sum up your classifications.

Respond with your reasoning, then provide the final JSON:
{"adult_count": X, "child_count": Y, "confidence": "high/medium/low"}"""

PROMPT_TWO_STAGE_IDENTIFY = """STAGE 1: Identify all people visible in this childcare image.

For EACH person you can see, describe:
- Position in image (left/center/right, foreground/background)
- What they are wearing
- What they are doing
- Approximate age category: ADULT (18+), CHILD (under 6), or UNCERTAIN

List each person as a numbered entry. Only list people you can actually see.

Format your response as:
Person 1: [position], [clothing], [action], [age category]
Person 2: [position], [clothing], [action], [age category]
...

Then provide totals:
ADULTS: [count]
CHILDREN: [count]"""

PROMPT_TWO_STAGE_COUNT = """Based on the people you identified, provide the final count.

CRITICAL RULES:
- If you classified someone as UNCERTAIN, decide: do they appear more adult-like or child-like?
- Only count people you explicitly identified - do not add any
- If you listed 0 adults, the adult count must be 0

Respond with JSON only:
{"adult_count": X, "child_count": Y}"""

PROMPT_VERIFICATION = """You previously counted {adult_count} adults and {child_count} children in this image.

Please VERIFY this count by re-examining the image carefully.

Check:
1. Did you count any person twice?
2. Did you miss anyone visible?
3. Did you count someone who isn't actually visible (hallucination)?
4. For each adult counted, can you describe where they are?

If your original count was correct, confirm it.
If you find an error, provide the corrected count.

Respond with JSON:
{{"adult_count": X, "child_count": Y, "verification": "confirmed" or "corrected", "notes": "..."}}"""


# ============================================================================
# API HELPER FUNCTIONS
# ============================================================================

def encode_image(image_path: Path) -> str:
    """Encode image to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def call_vlm(img_b64: str, prompt: str, temperature: float = 0.1, max_tokens: int = 800) -> str:
    """Call the VLM API and return response text."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content


def extract_json(text: str) -> Optional[Dict]:
    """Extract JSON from response text."""
    # Try direct parse
    try:
        return json.loads(text.strip())
    except:
        pass
    
    # Try to find JSON in text
    patterns = [
        r'\{[^{}]*"adult_count"[^{}]*\}',
        r'\{[^{}]*"adults"[^{}]*\}',
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group()
                return json.loads(json_str)
            except:
                continue
    
    # Try to extract numbers manually
    adult_match = re.search(r'adult.*?(\d+)|(\d+)\s*adult', text.lower())
    child_match = re.search(r'child.*?(\d+)|(\d+)\s*child', text.lower())
    
    if adult_match or child_match:
        adult = int(adult_match.group(1) or adult_match.group(2)) if adult_match else 0
        child = int(child_match.group(1) or child_match.group(2)) if child_match else 0
        return {"adult_count": adult, "child_count": child}
    
    return None


# ============================================================================
# TESTING STRATEGIES
# ============================================================================

def test_basic(img_b64: str) -> Dict:
    """Strategy 1: Basic single-shot counting."""
    response = call_vlm(img_b64, PROMPT_BASIC, temperature=0.0)
    result = extract_json(response)
    return result or {"adult_count": -1, "child_count": -1, "error": "parse_failed"}


def test_chain_of_thought(img_b64: str) -> Tuple[Dict, str]:
    """Strategy 2: Chain-of-thought reasoning."""
    response = call_vlm(img_b64, PROMPT_CHAIN_OF_THOUGHT, temperature=0.1, max_tokens=1200)
    result = extract_json(response)
    return result or {"adult_count": -1, "child_count": -1, "error": "parse_failed"}, response


def test_two_stage(img_b64: str) -> Tuple[Dict, str]:
    """Strategy 3: Two-stage identify then count."""
    # Stage 1: Identify all people
    stage1_response = call_vlm(img_b64, PROMPT_TWO_STAGE_IDENTIFY, temperature=0.1, max_tokens=1000)
    
    # Stage 2: Provide final count (using conversation context would be ideal, but we'll parse)
    # Extract counts from stage 1 response
    result = extract_json(stage1_response)
    
    if not result:
        # Try to extract from text format
        adults_match = re.search(r'ADULTS:\s*(\d+)', stage1_response, re.IGNORECASE)
        children_match = re.search(r'CHILDREN:\s*(\d+)', stage1_response, re.IGNORECASE)
        
        if adults_match and children_match:
            result = {
                "adult_count": int(adults_match.group(1)),
                "child_count": int(children_match.group(1))
            }
    
    return result or {"adult_count": -1, "child_count": -1, "error": "parse_failed"}, stage1_response


def test_multi_shot(img_b64: str, num_shots: int = 3) -> Tuple[Dict, List[Dict]]:
    """Strategy 4: Multi-shot consensus voting."""
    results = []
    
    for i in range(num_shots):
        response = call_vlm(img_b64, PROMPT_CHAIN_OF_THOUGHT, temperature=0.3, max_tokens=1200)
        result = extract_json(response)
        if result and result.get('adult_count', -1) >= 0:
            results.append(result)
        time.sleep(0.5)  # Rate limiting
    
    if not results:
        return {"adult_count": -1, "child_count": -1, "error": "no_valid_results"}, []
    
    # Median voting
    adult_counts = sorted([r['adult_count'] for r in results])
    child_counts = sorted([r['child_count'] for r in results])
    
    mid = len(adult_counts) // 2
    consensus = {
        "adult_count": adult_counts[mid],
        "child_count": child_counts[mid],
        "num_shots": len(results),
        "all_adults": adult_counts,
        "all_children": child_counts
    }
    
    return consensus, results


def test_with_verification(img_b64: str) -> Tuple[Dict, str, str]:
    """Strategy 5: Count then verify."""
    # Initial count with CoT
    initial_response = call_vlm(img_b64, PROMPT_CHAIN_OF_THOUGHT, temperature=0.1, max_tokens=1200)
    initial_result = extract_json(initial_response)
    
    if not initial_result or initial_result.get('adult_count', -1) < 0:
        return {"adult_count": -1, "child_count": -1, "error": "initial_failed"}, initial_response, ""
    
    # Verification pass
    verify_prompt = PROMPT_VERIFICATION.format(
        adult_count=initial_result['adult_count'],
        child_count=initial_result['child_count']
    )
    
    verify_response = call_vlm(img_b64, verify_prompt, temperature=0.0, max_tokens=600)
    verify_result = extract_json(verify_response)
    
    final_result = verify_result if verify_result else initial_result
    
    return final_result, initial_response, verify_response


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def evaluate_result(result: Dict, ground_truth: Dict) -> Dict:
    """Evaluate a result against ground truth."""
    if result.get('error'):
        return {"adult_correct": False, "child_within_1": False, "child_within_2": False, "error": True}
    
    vlm_a = result.get('adult_count', -1)
    vlm_c = result.get('child_count', -1)
    gt_a = ground_truth['adults']
    gt_c = ground_truth['children']
    
    return {
        "vlm_adults": vlm_a,
        "vlm_children": vlm_c,
        "gt_adults": gt_a,
        "gt_children": gt_c,
        "adult_correct": vlm_a == gt_a,
        "adult_diff": vlm_a - gt_a,
        "child_within_1": abs(vlm_c - gt_c) <= 1,
        "child_within_2": abs(vlm_c - gt_c) <= 2,
        "child_diff": vlm_c - gt_c,
        "error": False
    }


def run_all_tests():
    """Run all test strategies on all images."""
    
    print("=" * 80)
    print("COMPREHENSIVE VLM COUNTING TEST")
    print("Testing 5 strategies across 4 images")
    print("=" * 80)
    
    frames_dir = Path('test_data/frames')
    
    if not frames_dir.exists():
        print(f"ERROR: {frames_dir} not found")
        return
    
    # Results storage
    all_results = {
        "basic": [],
        "cot": [],
        "two_stage": [],
        "multi_shot": [],
        "verified": []
    }
    
    for img_path in sorted(frames_dir.glob('*.jpg')):
        filename = img_path.name
        
        if filename not in GROUND_TRUTH:
            continue
        
        gt = GROUND_TRUTH[filename]
        img_b64 = encode_image(img_path)
        
        print(f"\n{'='*80}")
        print(f"IMAGE: {filename}")
        print(f"GROUND TRUTH: {gt['adults']} adults, {gt['children']} children")
        print(f"Notes: {gt['notes']}")
        print("="*80)
        
        # Strategy 1: Basic
        print("\n[1] BASIC COUNTING...")
        basic_result = test_basic(img_b64)
        basic_eval = evaluate_result(basic_result, gt)
        print(f"    Result: {basic_result.get('adult_count')}A, {basic_result.get('child_count')}C")
        print(f"    Adult: {'✅' if basic_eval['adult_correct'] else '❌'} | Child (±2): {'✅' if basic_eval['child_within_2'] else '❌'}")
        all_results["basic"].append(basic_eval)
        
        # Strategy 2: Chain-of-thought
        print("\n[2] CHAIN-OF-THOUGHT...")
        cot_result, cot_response = test_chain_of_thought(img_b64)
        cot_eval = evaluate_result(cot_result, gt)
        print(f"    Result: {cot_result.get('adult_count')}A, {cot_result.get('child_count')}C")
        print(f"    Adult: {'✅' if cot_eval['adult_correct'] else '❌'} | Child (±2): {'✅' if cot_eval['child_within_2'] else '❌'}")
        all_results["cot"].append(cot_eval)
        
        # Strategy 3: Two-stage
        print("\n[3] TWO-STAGE IDENTIFICATION...")
        two_stage_result, two_stage_response = test_two_stage(img_b64)
        two_stage_eval = evaluate_result(two_stage_result, gt)
        print(f"    Result: {two_stage_result.get('adult_count')}A, {two_stage_result.get('child_count')}C")
        print(f"    Adult: {'✅' if two_stage_eval['adult_correct'] else '❌'} | Child (±2): {'✅' if two_stage_eval['child_within_2'] else '❌'}")
        all_results["two_stage"].append(two_stage_eval)
        
        # Strategy 4: Multi-shot consensus
        print("\n[4] MULTI-SHOT CONSENSUS (3 runs)...")
        multi_result, multi_runs = test_multi_shot(img_b64, num_shots=3)
        multi_eval = evaluate_result(multi_result, gt)
        print(f"    Individual runs: Adults={multi_result.get('all_adults')}, Children={multi_result.get('all_children')}")
        print(f"    Consensus: {multi_result.get('adult_count')}A, {multi_result.get('child_count')}C")
        print(f"    Adult: {'✅' if multi_eval['adult_correct'] else '❌'} | Child (±2): {'✅' if multi_eval['child_within_2'] else '❌'}")
        all_results["multi_shot"].append(multi_eval)
        
        # Strategy 5: With verification
        print("\n[5] COUNT + VERIFICATION...")
        verify_result, _, _ = test_with_verification(img_b64)
        verify_eval = evaluate_result(verify_result, gt)
        print(f"    Result: {verify_result.get('adult_count')}A, {verify_result.get('child_count')}C")
        if verify_result.get('verification'):
            print(f"    Verification: {verify_result.get('verification')}")
        print(f"    Adult: {'✅' if verify_eval['adult_correct'] else '❌'} | Child (±2): {'✅' if verify_eval['child_within_2'] else '❌'}")
        all_results["verified"].append(verify_eval)
        
        # Small delay between images
        time.sleep(1)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("FINAL SUMMARY - ACCURACY BY STRATEGY")
    print("="*80)
    
    strategies = [
        ("Basic", "basic"),
        ("Chain-of-Thought", "cot"),
        ("Two-Stage", "two_stage"),
        ("Multi-Shot (3x)", "multi_shot"),
        ("With Verification", "verified")
    ]
    
    print(f"\n{'Strategy':<20} {'Adult Exact':>12} {'Child ±1':>12} {'Child ±2':>12}")
    print("-"*60)
    
    best_adult = 0
    best_strategy = ""
    
    for name, key in strategies:
        results = all_results[key]
        total = len([r for r in results if not r.get('error')])
        
        if total == 0:
            print(f"{name:<20} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
            continue
        
        adult_acc = sum(1 for r in results if r.get('adult_correct')) / total * 100
        child_1 = sum(1 for r in results if r.get('child_within_1')) / total * 100
        child_2 = sum(1 for r in results if r.get('child_within_2')) / total * 100
        
        print(f"{name:<20} {adult_acc:>11.1f}% {child_1:>11.1f}% {child_2:>11.1f}%")
        
        if adult_acc > best_adult:
            best_adult = adult_acc
            best_strategy = name
    
    print("-"*60)
    print(f"\n🏆 Best Strategy for Adult Accuracy: {best_strategy} ({best_adult:.1f}%)")
    
    # Hallucination check (image 4.jpg)
    print("\n" + "="*80)
    print("HALLUCINATION TEST (4.jpg - should be 0 adults)")
    print("="*80)
    
    for name, key in strategies:
        results = all_results[key]
        # Find result for 4.jpg (last one)
        if len(results) >= 4:
            r = results[3]
            status = "✅ PASS" if r.get('vlm_adults') == 0 else f"❌ FAIL (said {r.get('vlm_adults')} adults)"
            print(f"  {name:<20}: {status}")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    run_all_tests()