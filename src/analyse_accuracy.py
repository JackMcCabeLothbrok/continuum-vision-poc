#!/usr/bin/env python3
"""
analyse_accuracy.py - VLM Accuracy Testing Framework

A modular framework for testing different VLM prompting strategies
and agentic pipelines to improve counting accuracy.

Strategies Available:
    - basic: Simple direct prompt (baseline)
    - detailed: More specific instructions about edge cases
    - scene_first: Describe scene, then extract counts
    - region_based: Divide image into quadrants, count each
    - confidence_aware: Assess complexity first, route accordingly
    - all: Run all strategies and compare

Usage:
    # Run with default (detailed) strategy
    python src/analyse_accuracy.py test_data/extracted/ChildcareCentreScenes/
    
    # Run specific strategy
    python src/analyse_accuracy.py frames/ --strategy scene_first
    
    # Compare all strategies
    python src/analyse_accuracy.py frames/ --strategy all

Author: Continuum Labs
Version: 2.0.0
"""

import os
import sys
import json
import base64
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

client = OpenAI(
    base_url=os.getenv('RESETDATA_BASE_URL'),
    api_key=os.getenv('RESETDATA_API_KEY')
)
MODEL = os.getenv('RESETDATA_VISION_MODEL')


# =============================================================================
# PROMPTS - Carefully engineered for accuracy
# =============================================================================

PROMPTS = {
    # Original baseline - simple and direct
    "basic": """Count the adults and children visible in this image.
Respond with JSON only: {"adult_count": X, "child_count": Y}""",

    # Detailed prompt with edge case handling
    "detailed": """Analyze this image and count the people visible.

DEFINITIONS:
- ADULT: Anyone who appears 18 years or older (includes teenagers who look adult-sized)
- CHILD: Anyone who appears under 18 years old

COUNTING RULES:
1. Count ONLY people you can clearly see (full body, partial body, or face visible)
2. Do NOT count: reflections, photos on walls, people on screens, mannequins
3. If someone's back is to the camera but they're clearly a person, count them
4. If you can only see hands/feet, do NOT count them as a separate person
5. When uncertain about age, use body size relative to surroundings as a guide

COUNT CAREFULLY. Take your time. Double-check before responding.

Respond with JSON only: {"adult_count": X, "child_count": Y}""",

    # Scene description first, then extraction
    "scene_first_describe": """Describe this image in detail. Focus on:
1. The setting/environment
2. Each person visible - their approximate age (child/adult) and what they're doing
3. How many total people you can see

Be specific about each individual person.""",

    "scene_first_extract": """Based on your description, provide the final count.
Remember:
- ADULT = appears 18+ years old
- CHILD = appears under 18 years old

Respond with JSON only: {"adult_count": X, "child_count": Y}""",

    # Complexity assessment
    "complexity_assess": """Look at this image and assess its complexity for counting people.

Rate the following (1-5 scale):
- How many people are visible? (1=0-2, 2=3-5, 3=6-10, 4=11-20, 5=20+)
- How much occlusion/overlap? (1=none, 5=severe)
- How clear is the image? (1=very clear, 5=blurry/dark)

Respond with JSON: {"people_estimate": X, "occlusion": X, "clarity": X, "total_complexity": X}""",

    # Region-based counting
    "region_count": """Focus ONLY on the {region} of this image.

Count the adults (18+) and children (under 18) visible in ONLY this region.
Do not count anyone outside this region.

Respond with JSON only: {{"adult_count": X, "child_count": Y}}""",

    # Verification prompt
    "verify": """You previously counted {adult_count} adults and {child_count} children.

Look at the image again carefully. Check:
1. Did you miss anyone partially visible or in the background?
2. Did you double-count anyone?
3. Did you correctly classify ages?

If your count was correct, respond: {{"adult_count": {adult_count}, "child_count": {child_count}, "confident": true}}
If you need to revise, respond: {{"adult_count": X, "child_count": Y, "confident": false, "revision_reason": "..."}}"""
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FrameResult:
    """Result for a single frame analysis."""
    frame_name: str
    timestamp: str
    scene_type: str
    difficulty: str
    strategy: str
    
    # Ground truth
    gt_adults: int
    gt_children: int
    gt_adults_range: Tuple[int, int]
    gt_children_range: Tuple[int, int]
    
    # Prediction
    pred_adults: int
    pred_children: int
    
    # Accuracy
    adult_exact: bool
    adult_within_1: bool
    adult_within_range: bool
    child_exact: bool
    child_within_2: bool
    child_within_range: bool
    
    # Metadata
    api_calls: int
    processing_time: float
    raw_responses: List[str] = field(default_factory=list)
    notes: str = ""
    compliance_scenario: Optional[str] = None


@dataclass
class StrategyResult:
    """Aggregated results for a strategy."""
    strategy_name: str
    total_frames: int
    
    # Accuracy metrics
    adult_exact_pct: float
    adult_within_1_pct: float
    child_exact_pct: float
    child_within_2_pct: float
    
    # By difficulty
    easy_adult_1: float
    easy_child_2: float
    medium_adult_1: float
    medium_child_2: float
    hard_adult_1: float
    hard_child_2: float
    
    # Cost
    avg_api_calls: float
    avg_time: float
    
    # Compliance
    compliance_detection_rate: float


# =============================================================================
# VLM INTERFACE
# =============================================================================

def encode_image(image_path: Path) -> str:
    """Encode image to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def call_vlm(
    prompt: str, 
    image_path: Optional[Path] = None,
    image_b64: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 500
) -> Dict:
    """
    Make a VLM API call with optional image.
    
    Returns dict with response text, timing, and success status.
    """
    if image_path and not image_b64:
        image_b64 = encode_image(image_path)
    
    start_time = time.time()
    
    try:
        content = [{"type": "text", "text": prompt}]
        if image_b64:
            content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            })
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return {
            "text": response.choices[0].message.content,
            "time": time.time() - start_time,
            "success": True
        }
        
    except Exception as e:
        return {
            "text": "",
            "time": time.time() - start_time,
            "success": False,
            "error": str(e)
        }


def parse_json_response(text: str) -> Optional[Dict]:
    """Extract JSON from response text."""
    import re
    
    # Try to find JSON object
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Try whole text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    return None


def extract_counts(response: Dict) -> Tuple[int, int]:
    """Extract adult and child counts from parsed response."""
    adult = response.get("adult_count", response.get("adults", -1))
    child = response.get("child_count", response.get("children", -1))
    return max(0, int(adult)) if adult != -1 else -1, max(0, int(child)) if child != -1 else -1


# =============================================================================
# COUNTING STRATEGIES
# =============================================================================

class CountingStrategy(ABC):
    """Base class for counting strategies."""
    
    name: str = "base"
    description: str = ""
    
    @abstractmethod
    def count(self, image_path: Path) -> Tuple[int, int, int, List[str], float]:
        """
        Count adults and children in image.
        
        Returns: (adult_count, child_count, api_calls, raw_responses, total_time)
        """
        pass


class BasicStrategy(CountingStrategy):
    """Simple direct prompt - baseline."""
    
    name = "basic"
    description = "Simple direct prompt (baseline)"
    
    def count(self, image_path: Path) -> Tuple[int, int, int, List[str], float]:
        result = call_vlm(PROMPTS["basic"], image_path=image_path)
        
        if not result["success"]:
            return -1, -1, 1, [result.get("error", "")], result["time"]
        
        parsed = parse_json_response(result["text"])
        if not parsed:
            return -1, -1, 1, [result["text"]], result["time"]
        
        adults, children = extract_counts(parsed)
        return adults, children, 1, [result["text"]], result["time"]


class DetailedStrategy(CountingStrategy):
    """Detailed prompt with edge case instructions."""
    
    name = "detailed"
    description = "Detailed prompt with counting rules and edge cases"
    
    def count(self, image_path: Path) -> Tuple[int, int, int, List[str], float]:
        result = call_vlm(PROMPTS["detailed"], image_path=image_path)
        
        if not result["success"]:
            return -1, -1, 1, [result.get("error", "")], result["time"]
        
        parsed = parse_json_response(result["text"])
        if not parsed:
            return -1, -1, 1, [result["text"]], result["time"]
        
        adults, children = extract_counts(parsed)
        return adults, children, 1, [result["text"]], result["time"]


class SceneFirstStrategy(CountingStrategy):
    """Describe scene first, then extract counts."""
    
    name = "scene_first"
    description = "Two-step: describe scene, then extract counts"
    
    def count(self, image_path: Path) -> Tuple[int, int, int, List[str], float]:
        img_b64 = encode_image(image_path)
        responses = []
        total_time = 0
        
        # Step 1: Describe scene
        result1 = call_vlm(
            PROMPTS["scene_first_describe"], 
            image_b64=img_b64,
            max_tokens=800
        )
        total_time += result1["time"]
        responses.append(result1["text"])
        
        if not result1["success"]:
            return -1, -1, 1, responses, total_time
        
        # Step 2: Extract counts (with description context)
        extract_prompt = f"""Based on your previous description:
"{result1['text'][:1000]}"

Now provide the final count.
- ADULT = appears 18+ years old  
- CHILD = appears under 18 years old

Respond with JSON only: {{"adult_count": X, "child_count": Y}}"""
        
        result2 = call_vlm(extract_prompt, image_b64=img_b64)
        total_time += result2["time"]
        responses.append(result2["text"])
        
        if not result2["success"]:
            return -1, -1, 2, responses, total_time
        
        parsed = parse_json_response(result2["text"])
        if not parsed:
            return -1, -1, 2, responses, total_time
        
        adults, children = extract_counts(parsed)
        return adults, children, 2, responses, total_time


class RegionBasedStrategy(CountingStrategy):
    """Divide image into regions and count each."""
    
    name = "region_based"
    description = "Divide into 4 quadrants, count each, aggregate"
    
    def count(self, image_path: Path) -> Tuple[int, int, int, List[str], float]:
        img_b64 = encode_image(image_path)
        responses = []
        total_time = 0
        
        regions = ["top-left quadrant", "top-right quadrant", 
                   "bottom-left quadrant", "bottom-right quadrant"]
        
        region_counts = []
        
        for region in regions:
            prompt = PROMPTS["region_count"].format(region=region)
            result = call_vlm(prompt, image_b64=img_b64)
            total_time += result["time"]
            responses.append(f"[{region}] {result['text']}")
            
            if result["success"]:
                parsed = parse_json_response(result["text"])
                if parsed:
                    adults, children = extract_counts(parsed)
                    region_counts.append((adults, children))
        
        if not region_counts:
            return -1, -1, len(regions), responses, total_time
        
        # Aggregate (with overlap correction - reduce by ~20% for quadrant overlap)
        total_adults = sum(r[0] for r in region_counts if r[0] >= 0)
        total_children = sum(r[1] for r in region_counts if r[1] >= 0)
        
        # Apply overlap correction (people on boundaries get double-counted)
        overlap_factor = 0.75
        final_adults = max(0, round(total_adults * overlap_factor))
        final_children = max(0, round(total_children * overlap_factor))
        
        return final_adults, final_children, len(regions), responses, total_time


class ConfidenceAwareStrategy(CountingStrategy):
    """Assess complexity first, then route to appropriate method."""
    
    name = "confidence_aware"
    description = "Assess scene complexity, route to simple or detailed analysis"
    
    def count(self, image_path: Path) -> Tuple[int, int, int, List[str], float]:
        img_b64 = encode_image(image_path)
        responses = []
        total_time = 0
        
        # Step 1: Quick complexity assessment
        result1 = call_vlm(PROMPTS["complexity_assess"], image_b64=img_b64)
        total_time += result1["time"]
        responses.append(f"[complexity] {result1['text']}")
        
        complexity = 5  # Default to complex if assessment fails
        if result1["success"]:
            parsed = parse_json_response(result1["text"])
            if parsed:
                complexity = parsed.get("total_complexity", 5)
        
        # Step 2: Route based on complexity
        if complexity <= 2:
            # Simple scene - use basic prompt
            result2 = call_vlm(PROMPTS["basic"], image_b64=img_b64)
        else:
            # Complex scene - use detailed prompt
            result2 = call_vlm(PROMPTS["detailed"], image_b64=img_b64)
        
        total_time += result2["time"]
        responses.append(f"[count] {result2['text']}")
        
        if not result2["success"]:
            return -1, -1, 2, responses, total_time
        
        parsed = parse_json_response(result2["text"])
        if not parsed:
            return -1, -1, 2, responses, total_time
        
        adults, children = extract_counts(parsed)
        return adults, children, 2, responses, total_time


class VerifiedStrategy(CountingStrategy):
    """Count then verify with self-check."""
    
    name = "verified"
    description = "Initial count + verification pass"
    
    def count(self, image_path: Path) -> Tuple[int, int, int, List[str], float]:
        img_b64 = encode_image(image_path)
        responses = []
        total_time = 0
        
        # Step 1: Initial count with detailed prompt
        result1 = call_vlm(PROMPTS["detailed"], image_b64=img_b64)
        total_time += result1["time"]
        responses.append(f"[initial] {result1['text']}")
        
        if not result1["success"]:
            return -1, -1, 1, responses, total_time
        
        parsed1 = parse_json_response(result1["text"])
        if not parsed1:
            return -1, -1, 1, responses, total_time
        
        initial_adults, initial_children = extract_counts(parsed1)
        
        # Step 2: Verification
        verify_prompt = PROMPTS["verify"].format(
            adult_count=initial_adults,
            child_count=initial_children
        )
        
        result2 = call_vlm(verify_prompt, image_b64=img_b64)
        total_time += result2["time"]
        responses.append(f"[verify] {result2['text']}")
        
        if not result2["success"]:
            return initial_adults, initial_children, 2, responses, total_time
        
        parsed2 = parse_json_response(result2["text"])
        if not parsed2:
            return initial_adults, initial_children, 2, responses, total_time
        
        # Use verified count
        final_adults, final_children = extract_counts(parsed2)
        return final_adults, final_children, 2, responses, total_time


# Strategy registry
STRATEGIES: Dict[str, CountingStrategy] = {
    "basic": BasicStrategy(),
    "detailed": DetailedStrategy(),
    "scene_first": SceneFirstStrategy(),
    "region_based": RegionBasedStrategy(),
    "confidence_aware": ConfidenceAwareStrategy(),
    "verified": VerifiedStrategy(),
}


# =============================================================================
# ACCURACY EVALUATION
# =============================================================================

def evaluate_prediction(
    pred: int, 
    gt: int, 
    gt_range: Tuple[int, int]
) -> Dict[str, bool]:
    """Evaluate prediction against ground truth."""
    return {
        "exact": pred == gt,
        "within_1": abs(pred - gt) <= 1,
        "within_2": abs(pred - gt) <= 2,
        "within_range": gt_range[0] <= pred <= gt_range[1],
        "error": pred - gt
    }


def run_analysis(
    frames_dir: Path,
    strategy_name: str = "detailed",
    ground_truth_path: Optional[Path] = None,
    delay: float = 0.5
) -> Dict:
    """
    Run analysis with specified strategy.
    
    Returns comprehensive report with results and metrics.
    """
    
    # Load ground truth
    gt_path = ground_truth_path or frames_dir / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_path}")
    
    with open(gt_path) as f:
        gt_data = json.load(f)
    
    gt_frames = gt_data.get("frames", {})
    
    # Get strategy
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGIES.keys())}")
    
    strategy = STRATEGIES[strategy_name]
    
    print("=" * 70)
    print(f"VLM ACCURACY ANALYSIS")
    print("=" * 70)
    print(f"Strategy: {strategy.name} - {strategy.description}")
    print(f"Model: {MODEL}")
    print(f"Frames: {len(gt_frames)}")
    print("=" * 70)
    
    results: List[FrameResult] = []
    
    for i, (frame_name, gt) in enumerate(gt_frames.items()):
        frame_path = frames_dir / frame_name
        
        if not frame_path.exists():
            print(f"⚠️  Frame not found: {frame_name}")
            continue
        
        print(f"\n[{i+1}/{len(gt_frames)}] {frame_name}")
        print(f"    Scene: {gt.get('scene_type', gt.get('scene', 'unknown'))} | Difficulty: {gt['difficulty']}")
        print(f"    Ground Truth: {gt['adults']}A, {gt['children']}C", end="")
        
        # Run strategy
        pred_adults, pred_children, api_calls, raw_responses, proc_time = strategy.count(frame_path)
        
        if pred_adults == -1 or pred_children == -1:
            print(f" | ❌ Analysis failed")
            continue
        
        # Evaluate
        adult_range = tuple(gt.get('adults_range', [gt['adults'], gt['adults']]))
        child_range = tuple(gt.get('children_range', [gt['children'], gt['children']]))
        
        adult_eval = evaluate_prediction(pred_adults, gt['adults'], adult_range)
        child_eval = evaluate_prediction(pred_children, gt['children'], child_range)
        
        # Create result
        result = FrameResult(
            frame_name=frame_name,
            timestamp=gt.get('timestamp', ''),
            scene_type=gt.get('scene_type', gt.get('scene', '')),
            difficulty=gt['difficulty'],
            strategy=strategy_name,
            gt_adults=gt['adults'],
            gt_children=gt['children'],
            gt_adults_range=adult_range,
            gt_children_range=child_range,
            pred_adults=pred_adults,
            pred_children=pred_children,
            adult_exact=adult_eval['exact'],
            adult_within_1=adult_eval['within_1'],
            adult_within_range=adult_eval['within_range'],
            child_exact=child_eval['exact'],
            child_within_2=child_eval['within_2'],
            child_within_range=child_eval['within_range'],
            api_calls=api_calls,
            processing_time=proc_time,
            raw_responses=raw_responses,
            notes=gt.get('notes', ''),
            compliance_scenario=gt.get('compliance_scenario', gt.get('compliance_flag'))
        )
        
        results.append(result)
        
        # Print result
        adult_icon = "✅" if adult_eval['within_1'] else "❌"
        child_icon = "✅" if child_eval['within_2'] else "❌"
        print(f" | VLM: {pred_adults}A, {pred_children}C | {adult_icon}{child_icon} ({api_calls} calls, {proc_time:.2f}s)")
        
        time.sleep(delay)
    
    # Calculate summary
    summary = calculate_summary(results, strategy_name)
    print_summary(summary)
    
    return {
        "analysis_time": datetime.now().isoformat(),
        "strategy": strategy_name,
        "strategy_description": strategy.description,
        "model": MODEL,
        "frames_dir": str(frames_dir),
        "summary": summary,
        "results": [asdict(r) for r in results]
    }


def calculate_summary(results: List[FrameResult], strategy_name: str) -> Dict:
    """Calculate summary statistics."""
    n = len(results)
    if n == 0:
        return {"error": "No results"}
    
    # Overall accuracy
    adult_exact = sum(1 for r in results if r.adult_exact) / n * 100
    adult_within_1 = sum(1 for r in results if r.adult_within_1) / n * 100
    child_exact = sum(1 for r in results if r.child_exact) / n * 100
    child_within_2 = sum(1 for r in results if r.child_within_2) / n * 100
    
    # By difficulty
    by_difficulty = {}
    for diff in ['easy', 'medium', 'hard']:
        diff_results = [r for r in results if r.difficulty == diff]
        if diff_results:
            dn = len(diff_results)
            by_difficulty[diff] = {
                "count": dn,
                "adult_exact": sum(1 for r in diff_results if r.adult_exact) / dn * 100,
                "adult_within_1": sum(1 for r in diff_results if r.adult_within_1) / dn * 100,
                "child_exact": sum(1 for r in diff_results if r.child_exact) / dn * 100,
                "child_within_2": sum(1 for r in diff_results if r.child_within_2) / dn * 100,
            }
    
    # By scene type
    by_scene = {}
    scenes = set(r.scene_type for r in results if r.scene_type)
    for scene in scenes:
        scene_results = [r for r in results if r.scene_type == scene]
        sn = len(scene_results)
        by_scene[scene] = {
            "count": sn,
            "adult_within_1": sum(1 for r in scene_results if r.adult_within_1) / sn * 100,
            "child_within_2": sum(1 for r in scene_results if r.child_within_2) / sn * 100,
        }
    
    # Compliance detection
    compliance_results = [r for r in results if r.compliance_scenario]
    compliance_detected = 0
    compliance_details = []
    for r in compliance_results:
        if r.compliance_scenario == "potential_supervision_gap":
            detected = r.pred_adults == 0 and r.pred_children > 0
        else:
            detected = None
        compliance_details.append({
            "frame": r.frame_name,
            "scenario": r.compliance_scenario,
            "detected": detected
        })
        if detected:
            compliance_detected += 1
    
    compliance_rate = (compliance_detected / len(compliance_results) * 100) if compliance_results else 100.0
    
    # Cost metrics
    avg_api_calls = sum(r.api_calls for r in results) / n
    avg_time = sum(r.processing_time for r in results) / n
    total_time = sum(r.processing_time for r in results)
    
    return {
        "strategy": strategy_name,
        "total_frames": n,
        "overall": {
            "adult_exact": f"{adult_exact:.1f}%",
            "adult_within_1": f"{adult_within_1:.1f}%",
            "child_exact": f"{child_exact:.1f}%",
            "child_within_2": f"{child_within_2:.1f}%",
        },
        "by_difficulty": by_difficulty,
        "by_scene": by_scene,
        "compliance": {
            "detection_rate": f"{compliance_rate:.1f}%",
            "details": compliance_details
        },
        "cost": {
            "avg_api_calls": f"{avg_api_calls:.1f}",
            "avg_time_seconds": f"{avg_time:.2f}",
            "total_time_seconds": f"{total_time:.2f}"
        }
    }


def print_summary(summary: Dict):
    """Print formatted summary."""
    print("\n" + "=" * 70)
    print(f"RESULTS: {summary['strategy'].upper()}")
    print("=" * 70)
    
    overall = summary['overall']
    print(f"\n📊 OVERALL ACCURACY ({summary['total_frames']} frames)")
    print(f"   Adults:   Exact {overall['adult_exact']:>6} | Within ±1 {overall['adult_within_1']:>6}")
    print(f"   Children: Exact {overall['child_exact']:>6} | Within ±2 {overall['child_within_2']:>6}")
    
    print(f"\n📈 BY DIFFICULTY")
    for diff, stats in summary.get('by_difficulty', {}).items():
        print(f"   {diff.upper():6} ({stats['count']}): Adult ±1: {stats['adult_within_1']:>5.0f}% | Child ±2: {stats['child_within_2']:>5.0f}%")
    
    print(f"\n🏠 BY SCENE TYPE")
    for scene, stats in summary.get('by_scene', {}).items():
        print(f"   {scene:25} ({stats['count']}): Adult ±1: {stats['adult_within_1']:>5.0f}% | Child ±2: {stats['child_within_2']:>5.0f}%")
    
    compliance = summary.get('compliance', {})
    if compliance.get('details'):
        print(f"\n⚠️  COMPLIANCE DETECTION: {compliance['detection_rate']}")
        for c in compliance['details']:
            status = "✅ DETECTED" if c['detected'] else "❌ MISSED" if c['detected'] is False else "⚪ N/A"
            print(f"   {c['frame']}: {c['scenario']} — {status}")
    
    cost = summary['cost']
    print(f"\n💰 COST: {cost['avg_api_calls']} API calls/frame | {cost['avg_time_seconds']}s avg")
    print("=" * 70)


def compare_strategies(
    frames_dir: Path,
    strategies: List[str],
    ground_truth_path: Optional[Path] = None,
    delay: float = 0.5
) -> Dict:
    """Run and compare multiple strategies."""
    
    all_results = {}
    
    for strategy_name in strategies:
        print(f"\n{'#' * 70}")
        print(f"# TESTING STRATEGY: {strategy_name.upper()}")
        print(f"{'#' * 70}")
        
        result = run_analysis(
            frames_dir=frames_dir,
            strategy_name=strategy_name,
            ground_truth_path=ground_truth_path,
            delay=delay
        )
        all_results[strategy_name] = result
        
        # Brief pause between strategies
        time.sleep(1)
    
    # Print comparison
    print("\n" + "=" * 70)
    print("STRATEGY COMPARISON")
    print("=" * 70)
    print(f"\n{'Strategy':<20} {'Adult ±1':>10} {'Child ±2':>10} {'API Calls':>10} {'Time':>10}")
    print("-" * 70)
    
    for name, result in all_results.items():
        summary = result['summary']
        adult = summary['overall']['adult_within_1']
        child = summary['overall']['child_within_2']
        calls = summary['cost']['avg_api_calls']
        time_s = summary['cost']['avg_time_seconds']
        print(f"{name:<20} {adult:>10} {child:>10} {calls:>10} {time_s:>10}")
    
    print("=" * 70)
    
    return all_results


def save_reports(report: Dict, output_dir: Path):
    """Save JSON and Markdown reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    strategy = report['strategy']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON report
    json_path = output_dir / f"accuracy_{strategy}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📊 JSON: {json_path}")
    
    # Markdown report
    md_path = output_dir / f"accuracy_{strategy}_{timestamp}.md"
    generate_markdown_report(report, md_path)
    print(f"📄 Markdown: {md_path}")


def generate_markdown_report(report: Dict, output_path: Path):
    """Generate markdown report."""
    summary = report['summary']
    
    md = [
        f"# VLM Accuracy Report: {report['strategy'].upper()}",
        f"\n**Strategy:** {report['strategy_description']}",
        f"\n**Model:** `{report['model']}`",
        f"\n**Date:** {report['analysis_time']}",
        f"\n**Frames:** {summary['total_frames']}",
        
        "\n## Overall Accuracy",
        "\n| Metric | Adults | Children |",
        "|--------|--------|----------|",
        f"| Exact Match | {summary['overall']['adult_exact']} | {summary['overall']['child_exact']} |",
        f"| Within Tolerance | {summary['overall']['adult_within_1']} | {summary['overall']['child_within_2']} |",
        
        "\n## By Difficulty",
        "\n| Difficulty | Frames | Adult ±1 | Child ±2 |",
        "|------------|--------|----------|----------|",
    ]
    
    for diff, stats in summary.get('by_difficulty', {}).items():
        md.append(f"| {diff.capitalize()} | {stats['count']} | {stats['adult_within_1']:.0f}% | {stats['child_within_2']:.0f}% |")
    
    md.extend([
        "\n## By Scene Type",
        "\n| Scene | Frames | Adult ±1 | Child ±2 |",
        "|-------|--------|----------|----------|",
    ])
    
    for scene, stats in summary.get('by_scene', {}).items():
        md.append(f"| {scene} | {stats['count']} | {stats['adult_within_1']:.0f}% | {stats['child_within_2']:.0f}% |")
    
    md.extend([
        "\n## Frame Results",
        "\n| Frame | GT | Pred | Adult | Child |",
        "|-------|-----|------|-------|-------|",
    ])
    
    for r in report['results']:
        adult_icon = "✅" if r['adult_within_1'] else "❌"
        child_icon = "✅" if r['child_within_2'] else "❌"
        md.append(f"| {r['frame_name']} | {r['gt_adults']}A,{r['gt_children']}C | {r['pred_adults']}A,{r['pred_children']}C | {adult_icon} | {child_icon} |")
    
    md.extend([
        "\n## Cost",
        f"\n- Average API calls: {summary['cost']['avg_api_calls']}",
        f"- Average time: {summary['cost']['avg_time_seconds']}s",
        f"- Total time: {summary['cost']['total_time_seconds']}s",
        f"\n---\n*Generated: {report['analysis_time']}*"
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(md))


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='VLM Accuracy Testing Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategies:
    basic           Simple direct prompt (baseline)
    detailed        Detailed prompt with counting rules
    scene_first     Two-step: describe then extract
    region_based    Divide into quadrants, count each
    confidence_aware  Assess complexity, route accordingly
    verified        Initial count + verification pass
    all             Compare all strategies

Examples:
    python src/analyse_accuracy.py frames/ --strategy detailed
    python src/analyse_accuracy.py frames/ --strategy all
        """
    )
    
    parser.add_argument('frames_dir', type=Path, help='Directory with frames and ground_truth.json')
    parser.add_argument('--strategy', '-s', default='detailed', 
                        choices=list(STRATEGIES.keys()) + ['all'],
                        help='Counting strategy to use (default: detailed)')
    parser.add_argument('--ground-truth', '-g', type=Path, help='Path to ground truth JSON')
    parser.add_argument('--output', '-o', type=Path, help='Output directory')
    parser.add_argument('--delay', '-d', type=float, default=0.5, help='Delay between API calls')
    
    args = parser.parse_args()
    
    if not args.frames_dir.exists():
        print(f"❌ Directory not found: {args.frames_dir}")
        sys.exit(1)
    
    output_dir = args.output or args.frames_dir
    
    try:
        if args.strategy == 'all':
            # Compare all strategies
            all_results = compare_strategies(
                frames_dir=args.frames_dir,
                strategies=list(STRATEGIES.keys()),
                ground_truth_path=args.ground_truth,
                delay=args.delay
            )
            
            # Save comparison
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_path = output_dir / f"strategy_comparison_{timestamp}.json"
            with open(comparison_path, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"\n📊 Comparison saved: {comparison_path}")
            
        else:
            # Single strategy
            report = run_analysis(
                frames_dir=args.frames_dir,
                strategy_name=args.strategy,
                ground_truth_path=args.ground_truth,
                delay=args.delay
            )
            save_reports(report, output_dir)
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()