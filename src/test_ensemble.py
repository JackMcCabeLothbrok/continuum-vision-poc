#!/usr/bin/env python3
"""
test_ensemble.py - Ensemble Verification Experiment

Tests whether analyzing the SAME frame multiple times and aggregating
results improves counting accuracy over a single pass.

Hypothesis: VLM outputs have variance. Multiple passes + voting/median
can reduce errors and improve reliability.

Usage:
    python src/test_ensemble.py test_data/extracted/ChildcareCentreScenes/
    python src/test_ensemble.py frames_dir/ --passes 3 --strategy median
    python src/test_ensemble.py frames_dir/ --passes 5 --strategy vote

Author: Continuum Labs
Version: 1.0.0
Date: December 2025
"""

import os
import sys
import json
import base64
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter

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
# PROMPTS
# =============================================================================

COUNTING_PROMPT = """Analyze this image and count the people visible.

DEFINITIONS:
- ADULT: Anyone who appears 18 years or older
- CHILD: Anyone who appears under 18 years old

COUNTING RULES:
1. Count ONLY people you can clearly see (full body, partial body, or face visible)
2. Do NOT count: reflections, photos on walls, people on screens, mannequins
3. If someone's back is to the camera but they're clearly a person, count them
4. Partial people at frame edges: count if you can identify them as a person

COUNT CAREFULLY. Double-check before responding.

Respond with JSON only: {"adult_count": X, "child_count": Y}"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EnsembleResult:
    """Result for ensemble analysis of a single frame."""
    frame_name: str
    
    # Ground truth
    gt_adults: int
    gt_children: int
    
    # Per-pass predictions
    pass_results: List[Dict[str, int]]
    
    # Aggregated predictions (by strategy)
    strategies: Dict[str, Dict[str, int]]
    
    # Selected strategy result
    pred_adults: int
    pred_children: int
    strategy_used: str
    
    # Accuracy
    adult_exact: bool
    adult_within_1: bool
    child_exact: bool
    child_within_2: bool
    
    # Variance info
    adult_variance: int
    child_variance: int
    adult_agreement: float  # % of passes that agree with final
    child_agreement: float
    
    # Metadata
    processing_time: float
    num_passes: int


@dataclass 
class SinglePassResult:
    """Result for single-pass baseline."""
    frame_name: str
    gt_adults: int
    gt_children: int
    pred_adults: int
    pred_children: int
    adult_exact: bool
    adult_within_1: bool
    child_exact: bool
    child_within_2: bool
    processing_time: float


# =============================================================================
# VLM INTERFACE
# =============================================================================

def encode_image(image_path: Path) -> str:
    """Encode image to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def call_vlm(
    image_path: Path,
    temperature: float = 0.0,
    max_tokens: int = 150
) -> Dict:
    """Call VLM with a single image."""
    img_b64 = encode_image(image_path)
    
    content = [
        {"type": "text", "text": COUNTING_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
    ]
    
    start_time = time.time()
    
    try:
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
    start = text.find('{')
    if start == -1:
        return None
    
    depth = 0
    end = start
    for i, char in enumerate(text[start:], start):
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


# =============================================================================
# ENSEMBLE ANALYSIS
# =============================================================================

def analyze_frame_ensemble(
    image_path: Path,
    num_passes: int = 3,
    delay: float = 0.3,
    temperature: float = 0.0
) -> Dict:
    """
    Analyze the SAME frame multiple times and aggregate results.
    
    Args:
        image_path: Path to the frame image
        num_passes: Number of times to analyze the same image
        delay: Delay between API calls
        temperature: VLM temperature (0.0 for deterministic, >0 for variation)
    
    Returns:
        Dict with per-pass results and aggregated strategies
    """
    pass_results = []
    total_time = 0
    
    for i in range(num_passes):
        result = call_vlm(image_path, temperature=temperature)
        total_time += result.get("time", 0)
        
        if result["success"]:
            parsed = parse_json_response(result["text"])
            if parsed:
                pass_results.append({
                    "pass": i + 1,
                    "adults": parsed.get("adult_count", -1),
                    "children": parsed.get("child_count", -1),
                    "raw": result["text"]
                })
            else:
                pass_results.append({
                    "pass": i + 1,
                    "adults": -1,
                    "children": -1,
                    "parse_error": True,
                    "raw": result["text"]
                })
        else:
            pass_results.append({
                "pass": i + 1,
                "adults": -1,
                "children": -1,
                "error": result.get("error")
            })
        
        if i < num_passes - 1:
            time.sleep(delay)
    
    # Filter valid results
    valid_results = [r for r in pass_results if r["adults"] >= 0]
    
    if not valid_results:
        return {
            "success": False,
            "error": "All passes failed",
            "time": total_time,
            "pass_results": pass_results
        }
    
    adult_counts = [r["adults"] for r in valid_results]
    child_counts = [r["children"] for r in valid_results]
    
    # Calculate different aggregation strategies
    strategies = {}
    
    # MAX - highest count seen
    strategies["max"] = {
        "adults": max(adult_counts),
        "children": max(child_counts)
    }
    
    # MIN - lowest count seen
    strategies["min"] = {
        "adults": min(adult_counts),
        "children": min(child_counts)
    }
    
    # MEDIAN - middle value
    sorted_adults = sorted(adult_counts)
    sorted_children = sorted(child_counts)
    strategies["median"] = {
        "adults": sorted_adults[len(sorted_adults) // 2],
        "children": sorted_children[len(sorted_children) // 2]
    }
    
    # AVERAGE (rounded)
    strategies["average"] = {
        "adults": round(sum(adult_counts) / len(adult_counts)),
        "children": round(sum(child_counts) / len(child_counts))
    }
    
    # MODE (most common) - voting
    adult_counter = Counter(adult_counts)
    child_counter = Counter(child_counts)
    strategies["vote"] = {
        "adults": adult_counter.most_common(1)[0][0],
        "children": child_counter.most_common(1)[0][0]
    }
    
    # FIRST - just use first result (baseline comparison)
    strategies["first"] = {
        "adults": valid_results[0]["adults"],
        "children": valid_results[0]["children"]
    }
    
    # Calculate variance
    adult_variance = max(adult_counts) - min(adult_counts)
    child_variance = max(child_counts) - min(child_counts)
    
    return {
        "success": True,
        "time": total_time,
        "pass_results": pass_results,
        "valid_passes": len(valid_results),
        "strategies": strategies,
        "adult_counts": adult_counts,
        "child_counts": child_counts,
        "adult_variance": adult_variance,
        "child_variance": child_variance
    }


def run_ensemble_experiment(
    frames_dir: Path,
    frame_names: List[str],
    gt_data: Dict,
    num_passes: int = 3,
    strategy: str = "vote",
    delay: float = 0.3,
    temperature: float = 0.0
) -> List[EnsembleResult]:
    """Run ensemble analysis on all frames."""
    results = []
    
    for i, frame_name in enumerate(frame_names):
        frame_path = frames_dir / frame_name
        if not frame_path.exists():
            print(f"    ⚠️  Frame not found: {frame_name}")
            continue
        
        # Get ground truth
        gt = gt_data.get(frame_name, {})
        gt_adults = gt.get('adults', gt.get('adult_count', 0))
        gt_children = gt.get('children', gt.get('child_count', 0))
        
        print(f"\n[{i+1}/{len(frame_names)}] {frame_name} (GT: {gt_adults}A, {gt_children}C)")
        
        # Run ensemble analysis
        result = analyze_frame_ensemble(
            frame_path,
            num_passes=num_passes,
            delay=delay,
            temperature=temperature
        )
        
        if not result["success"]:
            print(f"    ❌ Failed: {result.get('error')}")
            continue
        
        # Get prediction using selected strategy
        strategies = result["strategies"]
        if strategy not in strategies:
            strategy = "vote"
        
        pred_adults = strategies[strategy]["adults"]
        pred_children = strategies[strategy]["children"]
        
        # Calculate accuracy
        adult_exact = pred_adults == gt_adults
        adult_within_1 = abs(pred_adults - gt_adults) <= 1
        child_exact = pred_children == gt_children
        child_within_2 = abs(pred_children - gt_children) <= 2
        
        # Calculate agreement (what % of passes agree with final result)
        adult_counts = result["adult_counts"]
        child_counts = result["child_counts"]
        adult_agreement = adult_counts.count(pred_adults) / len(adult_counts) * 100
        child_agreement = child_counts.count(pred_children) / len(child_counts) * 100
        
        # Print per-pass results
        pass_str = ", ".join([f"{r['adults']}A/{r['children']}C" for r in result["pass_results"] if r.get("adults", -1) >= 0])
        print(f"    Passes: [{pass_str}]")
        
        adult_icon = "✅" if adult_within_1 else "❌"
        child_icon = "✅" if child_within_2 else "❌"
        print(f"    {strategy.upper()}: {pred_adults}A, {pred_children}C | {adult_icon}{child_icon} ({result['time']:.2f}s)")
        print(f"    Variance: adult={result['adult_variance']}, child={result['child_variance']} | Agreement: {adult_agreement:.0f}%A, {child_agreement:.0f}%C")
        
        ensemble_result = EnsembleResult(
            frame_name=frame_name,
            gt_adults=gt_adults,
            gt_children=gt_children,
            pass_results=result["pass_results"],
            strategies=strategies,
            pred_adults=pred_adults,
            pred_children=pred_children,
            strategy_used=strategy,
            adult_exact=adult_exact,
            adult_within_1=adult_within_1,
            child_exact=child_exact,
            child_within_2=child_within_2,
            adult_variance=result["adult_variance"],
            child_variance=result["child_variance"],
            adult_agreement=adult_agreement,
            child_agreement=child_agreement,
            processing_time=result["time"],
            num_passes=num_passes
        )
        
        results.append(ensemble_result)
    
    return results


def run_single_pass_baseline(
    frames_dir: Path,
    frame_names: List[str],
    gt_data: Dict,
    delay: float = 0.3
) -> List[SinglePassResult]:
    """Run single-pass baseline for comparison."""
    results = []
    
    for frame_name in frame_names:
        frame_path = frames_dir / frame_name
        if not frame_path.exists():
            continue
        
        gt = gt_data.get(frame_name, {})
        gt_adults = gt.get('adults', gt.get('adult_count', 0))
        gt_children = gt.get('children', gt.get('child_count', 0))
        
        result = call_vlm(frame_path)
        
        if result["success"]:
            parsed = parse_json_response(result["text"])
            if parsed:
                pred_adults = parsed.get("adult_count", -1)
                pred_children = parsed.get("child_count", -1)
                
                adult_exact = pred_adults == gt_adults
                adult_within_1 = abs(pred_adults - gt_adults) <= 1
                child_exact = pred_children == gt_children
                child_within_2 = abs(pred_children - gt_children) <= 2
                
                results.append(SinglePassResult(
                    frame_name=frame_name,
                    gt_adults=gt_adults,
                    gt_children=gt_children,
                    pred_adults=pred_adults,
                    pred_children=pred_children,
                    adult_exact=adult_exact,
                    adult_within_1=adult_within_1,
                    child_exact=child_exact,
                    child_within_2=child_within_2,
                    processing_time=result["time"]
                ))
        
        time.sleep(delay)
    
    return results


# =============================================================================
# REPORTING
# =============================================================================

def calculate_metrics(results: List[EnsembleResult]) -> Dict:
    """Calculate aggregate metrics for ensemble results."""
    n = len(results)
    if n == 0:
        return {"error": "No results"}
    
    return {
        "total_frames": n,
        "adult_exact_pct": sum(1 for r in results if r.adult_exact) / n * 100,
        "adult_within_1_pct": sum(1 for r in results if r.adult_within_1) / n * 100,
        "child_exact_pct": sum(1 for r in results if r.child_exact) / n * 100,
        "child_within_2_pct": sum(1 for r in results if r.child_within_2) / n * 100,
        "avg_time_seconds": sum(r.processing_time for r in results) / n,
        "avg_adult_variance": sum(r.adult_variance for r in results) / n,
        "avg_child_variance": sum(r.child_variance for r in results) / n,
        "avg_adult_agreement": sum(r.adult_agreement for r in results) / n,
        "avg_child_agreement": sum(r.child_agreement for r in results) / n,
        "zero_variance_pct": sum(1 for r in results if r.adult_variance == 0 and r.child_variance == 0) / n * 100
    }


def calculate_baseline_metrics(results: List[SinglePassResult]) -> Dict:
    """Calculate metrics for baseline results."""
    n = len(results)
    if n == 0:
        return {"error": "No results"}
    
    return {
        "total_frames": n,
        "adult_exact_pct": sum(1 for r in results if r.adult_exact) / n * 100,
        "adult_within_1_pct": sum(1 for r in results if r.adult_within_1) / n * 100,
        "child_exact_pct": sum(1 for r in results if r.child_exact) / n * 100,
        "child_within_2_pct": sum(1 for r in results if r.child_within_2) / n * 100,
        "avg_time_seconds": sum(r.processing_time for r in results) / n
    }


def compare_strategies(results: List[EnsembleResult], gt_data: Dict) -> Dict:
    """Compare accuracy of different aggregation strategies."""
    strategies = ["first", "max", "min", "median", "average", "vote"]
    comparison = {}
    
    for strat in strategies:
        adult_within_1 = 0
        child_within_2 = 0
        
        for r in results:
            if strat in r.strategies:
                pred_a = r.strategies[strat]["adults"]
                pred_c = r.strategies[strat]["children"]
                
                if abs(pred_a - r.gt_adults) <= 1:
                    adult_within_1 += 1
                if abs(pred_c - r.gt_children) <= 2:
                    child_within_2 += 1
        
        n = len(results)
        comparison[strat] = {
            "adult_within_1_pct": adult_within_1 / n * 100 if n > 0 else 0,
            "child_within_2_pct": child_within_2 / n * 100 if n > 0 else 0
        }
    
    return comparison


def print_results(
    ensemble_results: List[EnsembleResult],
    baseline_results: Optional[List[SinglePassResult]] = None
):
    """Print formatted results."""
    
    print("\n" + "=" * 70)
    print("ENSEMBLE VERIFICATION RESULTS")
    print("=" * 70)
    
    metrics = calculate_metrics(ensemble_results)
    
    print(f"\n📊 ENSEMBLE ACCURACY ({metrics['total_frames']} frames, {ensemble_results[0].num_passes} passes each)")
    print(f"   Strategy: {ensemble_results[0].strategy_used.upper()}")
    print(f"   Adults:   Exact {metrics['adult_exact_pct']:>5.1f}% | Within ±1 {metrics['adult_within_1_pct']:>5.1f}%")
    print(f"   Children: Exact {metrics['child_exact_pct']:>5.1f}% | Within ±2 {metrics['child_within_2_pct']:>5.1f}%")
    print(f"   Avg Time: {metrics['avg_time_seconds']:.2f}s per frame ({ensemble_results[0].num_passes} passes)")
    
    print(f"\n📈 VARIANCE ANALYSIS")
    print(f"   Avg adult variance:  {metrics['avg_adult_variance']:.2f} (range across passes)")
    print(f"   Avg child variance:  {metrics['avg_child_variance']:.2f}")
    print(f"   Zero variance:       {metrics['zero_variance_pct']:.1f}% of frames (all passes agreed)")
    print(f"   Avg adult agreement: {metrics['avg_adult_agreement']:.1f}%")
    print(f"   Avg child agreement: {metrics['avg_child_agreement']:.1f}%")
    
    # Compare all strategies
    print(f"\n📊 STRATEGY COMPARISON")
    print("-" * 50)
    strategy_comparison = compare_strategies(ensemble_results, {})
    print(f"   {'Strategy':<10} {'Adult ±1':>12} {'Child ±2':>12}")
    print(f"   {'-'*10} {'-'*12} {'-'*12}")
    for strat, scores in strategy_comparison.items():
        print(f"   {strat:<10} {scores['adult_within_1_pct']:>11.1f}% {scores['child_within_2_pct']:>11.1f}%")
    
    # Baseline comparison
    if baseline_results:
        baseline_metrics = calculate_baseline_metrics(baseline_results)
        
        print(f"\n📊 SINGLE-PASS BASELINE ({baseline_metrics['total_frames']} frames)")
        print(f"   Adults:   Exact {baseline_metrics['adult_exact_pct']:>5.1f}% | Within ±1 {baseline_metrics['adult_within_1_pct']:>5.1f}%")
        print(f"   Children: Exact {baseline_metrics['child_exact_pct']:>5.1f}% | Within ±2 {baseline_metrics['child_within_2_pct']:>5.1f}%")
        print(f"   Avg Time: {baseline_metrics['avg_time_seconds']:.2f}s per frame")
        
        print(f"\n🔄 COMPARISON (Ensemble vs Single-pass)")
        adult_delta = metrics['adult_within_1_pct'] - baseline_metrics['adult_within_1_pct']
        child_delta = metrics['child_within_2_pct'] - baseline_metrics['child_within_2_pct']
        time_factor = metrics['avg_time_seconds'] / baseline_metrics['avg_time_seconds']
        
        print(f"   Adult ±1:  {adult_delta:>+5.1f}% {'✅ BETTER' if adult_delta > 0 else '❌ WORSE' if adult_delta < 0 else '= SAME'}")
        print(f"   Child ±2:  {child_delta:>+5.1f}% {'✅ BETTER' if child_delta > 0 else '❌ WORSE' if child_delta < 0 else '= SAME'}")
        print(f"   Time:      {time_factor:.1f}x slower")
    
    print("\n" + "=" * 70)


def save_report(
    ensemble_results: List[EnsembleResult],
    baseline_results: Optional[List[SinglePassResult]],
    output_dir: Path,
    strategy: str,
    num_passes: int
):
    """Save JSON report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = {
        "experiment": "ensemble_verification",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "num_passes": num_passes,
        "strategy": strategy,
        "ensemble_metrics": calculate_metrics(ensemble_results),
        "strategy_comparison": compare_strategies(ensemble_results, {}),
        "ensemble_results": [asdict(r) for r in ensemble_results],
    }
    
    if baseline_results:
        report["baseline_metrics"] = calculate_baseline_metrics(baseline_results)
        report["baseline_results"] = [asdict(r) for r in baseline_results]
    
    output_path = output_dir / f"ensemble_experiment_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📊 Report saved: {output_path}")
    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Ensemble Verification Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tests whether analyzing the SAME frame multiple times improves accuracy.

Strategies:
  first   - Use first pass only (baseline equivalent)
  max     - Take maximum count across passes
  min     - Take minimum count across passes  
  median  - Take median count across passes
  average - Take average (rounded) across passes
  vote    - Take most common count (majority vote)

Example:
    python src/test_ensemble.py test_data/extracted/ChildcareCentreScenes/
    python src/test_ensemble.py frames/ --passes 3 --strategy vote
    python src/test_ensemble.py frames/ --passes 5 --strategy median
        """
    )
    
    parser.add_argument('frames_dir', type=Path, help='Directory with frames and ground_truth.json')
    parser.add_argument('--passes', '-p', type=int, default=3, help='Number of passes per frame (default: 3)')
    parser.add_argument('--strategy', '-s', type=str, default='vote',
                        choices=['first', 'max', 'min', 'median', 'average', 'vote'],
                        help='Aggregation strategy (default: vote)')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip single-pass baseline test')
    parser.add_argument('--delay', '-d', type=float, default=0.3, help='Delay between API calls')
    parser.add_argument('--temperature', '-t', type=float, default=0.0, 
                        help='VLM temperature (0.0=deterministic, >0 adds variation)')
    
    args = parser.parse_args()
    
    if not args.frames_dir.exists():
        print(f"❌ Directory not found: {args.frames_dir}")
        sys.exit(1)
    
    # Load ground truth
    gt_path = args.frames_dir / "ground_truth.json"
    if not gt_path.exists():
        print(f"❌ Ground truth not found: {gt_path}")
        sys.exit(1)
    
    with open(gt_path) as f:
        gt_data = json.load(f)
    
    gt_frames = gt_data.get("frames", {})
    
    # Get sorted frame names
    frame_names = sorted([
        f for f in gt_frames.keys()
        if (args.frames_dir / f).exists()
    ])
    
    print("=" * 70)
    print("ENSEMBLE VERIFICATION EXPERIMENT")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Frames: {len(frame_names)}")
    print(f"Passes per frame: {args.passes}")
    print(f"Strategy: {args.strategy.upper()}")
    print(f"Temperature: {args.temperature}")
    print("=" * 70)
    
    # Run ensemble analysis
    print(f"\n{'#' * 70}")
    print(f"# PHASE 1: ENSEMBLE ANALYSIS ({args.passes} passes per frame)")
    print(f"{'#' * 70}")
    
    ensemble_results = run_ensemble_experiment(
        args.frames_dir,
        frame_names,
        gt_frames,
        num_passes=args.passes,
        strategy=args.strategy,
        delay=args.delay,
        temperature=args.temperature
    )
    
    # Run baseline (optional)
    baseline_results = None
    if not args.skip_baseline:
        print(f"\n{'#' * 70}")
        print("# PHASE 2: SINGLE-PASS BASELINE")
        print(f"{'#' * 70}")
        
        print(f"\nRunning single-pass on {len(frame_names)} frames...")
        baseline_results = run_single_pass_baseline(
            args.frames_dir,
            frame_names,
            gt_frames,
            delay=args.delay
        )
        
        for r in baseline_results:
            adult_icon = "✅" if r.adult_within_1 else "❌"
            child_icon = "✅" if r.child_within_2 else "❌"
            print(f"    {r.frame_name}: GT {r.gt_adults}A,{r.gt_children}C | Pred {r.pred_adults}A,{r.pred_children}C | {adult_icon}{child_icon}")
    
    # Print results
    print_results(ensemble_results, baseline_results)
    
    # Save report
    save_report(ensemble_results, baseline_results, args.frames_dir, args.strategy, args.passes)


if __name__ == '__main__':
    main()