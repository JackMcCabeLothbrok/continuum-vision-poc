#!/usr/bin/env python3
"""
analyse_accuracy.py - Run VLM analysis and compare against ground truth

Uses the Basic prompting strategy (winner from testing) and generates
a detailed accuracy report.

Usage:
    python src/analyse_accuracy.py test_data/extracted/ChildcareCentreScenes/

Output:
    - Console summary
    - accuracy_report.json (detailed results)
    - accuracy_report.md (human-readable report)
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

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize client
client = OpenAI(
    base_url=os.getenv('RESETDATA_BASE_URL'),
    api_key=os.getenv('RESETDATA_API_KEY')
)
MODEL = os.getenv('RESETDATA_VISION_MODEL')

# Best performing prompt from testing (Basic strategy)
COUNTING_PROMPT = """Count the adults and children visible in this childcare image.
Respond with JSON only: {"adult_count": X, "child_count": Y}"""


@dataclass
class FrameResult:
    """Result for a single frame analysis."""
    frame_name: str
    timestamp: str
    scene: str
    difficulty: str
    
    # Ground truth
    gt_adults: int
    gt_adults_range: Tuple[int, int]
    gt_children: int
    gt_children_range: Tuple[int, int]
    
    # VLM prediction
    pred_adults: int
    pred_children: int
    
    # Accuracy metrics
    adult_exact_match: bool
    adult_within_range: bool
    adult_within_1: bool
    child_exact_match: bool
    child_within_range: bool
    child_within_2: bool
    
    # Metadata
    processing_time: float
    raw_response: str
    notes: str = ""
    compliance_flag: str = ""


def encode_image(image_path: Path) -> str:
    """Encode image to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def analyse_frame(image_path: Path) -> Dict:
    """Analyse a single frame with the VLM using Basic prompt."""
    img_b64 = encode_image(image_path)
    
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": COUNTING_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }],
            max_tokens=100,
            temperature=0.0
        )
        
        elapsed = time.time() - start_time
        response_text = response.choices[0].message.content
        
        # Parse JSON
        try:
            import re
            match = re.search(r'\{[^{}]*\}', response_text)
            if match:
                result = json.loads(match.group())
            else:
                result = json.loads(response_text.strip())
            
            return {
                "adult_count": result.get("adult_count", -1),
                "child_count": result.get("child_count", -1),
                "raw_response": response_text,
                "processing_time": elapsed,
                "success": True
            }
        except json.JSONDecodeError:
            return {
                "adult_count": -1,
                "child_count": -1,
                "raw_response": response_text,
                "processing_time": elapsed,
                "success": False,
                "error": "JSON parse failed"
            }
            
    except Exception as e:
        return {
            "adult_count": -1,
            "child_count": -1,
            "raw_response": "",
            "processing_time": time.time() - start_time,
            "success": False,
            "error": str(e)
        }


def evaluate_accuracy(
    pred: int, 
    gt: int, 
    gt_range: Tuple[int, int]
) -> Dict[str, bool]:
    """Evaluate prediction accuracy against ground truth."""
    return {
        "exact_match": pred == gt,
        "within_range": gt_range[0] <= pred <= gt_range[1],
        "within_1": abs(pred - gt) <= 1,
        "within_2": abs(pred - gt) <= 2,
        "error": pred - gt
    }


def run_accuracy_analysis(
    frames_dir: Path,
    ground_truth_path: Optional[Path] = None,
    delay_between_frames: float = 0.5
) -> Dict:
    """
    Run VLM analysis on all frames and compare to ground truth.
    """
    
    # Load ground truth
    if ground_truth_path is None:
        ground_truth_path = frames_dir / 'ground_truth.json'
    
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth not found: {ground_truth_path}")
    
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)
    
    gt_frames = ground_truth['frames']
    
    print("=" * 70)
    print("VLM ACCURACY ANALYSIS")
    print("=" * 70)
    print(f"Model: {MODEL}")
    print(f"Prompt: Basic (winner from testing)")
    print(f"Frames directory: {frames_dir}")
    print(f"Ground truth: {ground_truth_path}")
    print(f"Total frames: {len(gt_frames)}")
    print("=" * 70)
    
    results: List[FrameResult] = []
    
    for frame_name, gt_data in gt_frames.items():
        frame_path = frames_dir / frame_name
        
        if not frame_path.exists():
            print(f"⚠️  Frame not found: {frame_name}")
            continue
        
        print(f"\n[{len(results)+1}/{len(gt_frames)}] {frame_name} ({gt_data['timestamp']})")
        print(f"    Scene: {gt_data['scene']} | Difficulty: {gt_data['difficulty']}")
        print(f"    Ground Truth: {gt_data['adults']}A, {gt_data['children']}C", end="")
        
        # Run VLM
        vlm_result = analyse_frame(frame_path)
        
        if not vlm_result['success']:
            print(f" | ❌ VLM Error: {vlm_result.get('error', 'unknown')}")
            continue
        
        pred_adults = vlm_result['adult_count']
        pred_children = vlm_result['child_count']
        
        # Evaluate accuracy
        adult_eval = evaluate_accuracy(
            pred_adults, 
            gt_data['adults'], 
            tuple(gt_data['adults_range'])
        )
        child_eval = evaluate_accuracy(
            pred_children, 
            gt_data['children'], 
            tuple(gt_data['children_range'])
        )
        
        # Create result
        result = FrameResult(
            frame_name=frame_name,
            timestamp=gt_data['timestamp'],
            scene=gt_data['scene'],
            difficulty=gt_data['difficulty'],
            gt_adults=gt_data['adults'],
            gt_adults_range=tuple(gt_data['adults_range']),
            gt_children=gt_data['children'],
            gt_children_range=tuple(gt_data['children_range']),
            pred_adults=pred_adults,
            pred_children=pred_children,
            adult_exact_match=adult_eval['exact_match'],
            adult_within_range=adult_eval['within_range'],
            adult_within_1=adult_eval['within_1'],
            child_exact_match=child_eval['exact_match'],
            child_within_range=child_eval['within_range'],
            child_within_2=child_eval['within_2'],
            processing_time=vlm_result['processing_time'],
            raw_response=vlm_result['raw_response'],
            notes=gt_data.get('notes', ''),
            compliance_flag=gt_data.get('compliance_flag', '')
        )
        
        results.append(result)
        
        # Print result
        adult_status = "✅" if adult_eval['within_1'] else "❌"
        child_status = "✅" if child_eval['within_2'] else "❌"
        print(f" | VLM: {pred_adults}A, {pred_children}C | {adult_status}{child_status} ({vlm_result['processing_time']:.2f}s)")
        
        # Rate limiting
        time.sleep(delay_between_frames)
    
    # Calculate summary statistics
    summary = calculate_summary(results)
    
    # Build full report
    report = {
        "analysis_time": datetime.now().isoformat(),
        "model": MODEL,
        "prompt": COUNTING_PROMPT,
        "frames_directory": str(frames_dir),
        "ground_truth_source": str(ground_truth_path),
        "summary": summary,
        "frame_results": [asdict(r) for r in results]
    }
    
    # Print summary
    print_summary(summary)
    
    return report


def calculate_summary(results: List[FrameResult]) -> Dict:
    """Calculate summary statistics from results."""
    n = len(results)
    if n == 0:
        return {"error": "No results"}
    
    # Adult metrics
    adult_exact = sum(1 for r in results if r.adult_exact_match)
    adult_range = sum(1 for r in results if r.adult_within_range)
    adult_within_1 = sum(1 for r in results if r.adult_within_1)
    
    # Child metrics
    child_exact = sum(1 for r in results if r.child_exact_match)
    child_range = sum(1 for r in results if r.child_within_range)
    child_within_2 = sum(1 for r in results if r.child_within_2)
    
    # By difficulty
    by_difficulty = {}
    for diff in ['easy', 'medium', 'hard']:
        diff_results = [r for r in results if r.difficulty == diff]
        if diff_results:
            by_difficulty[diff] = {
                "count": len(diff_results),
                "adult_exact": sum(1 for r in diff_results if r.adult_exact_match) / len(diff_results) * 100,
                "adult_within_1": sum(1 for r in diff_results if r.adult_within_1) / len(diff_results) * 100,
                "child_exact": sum(1 for r in diff_results if r.child_exact_match) / len(diff_results) * 100,
                "child_within_2": sum(1 for r in diff_results if r.child_within_2) / len(diff_results) * 100
            }
    
    # By scene type
    by_scene = {}
    scene_types = set(r.scene for r in results)
    for scene in scene_types:
        scene_results = [r for r in results if r.scene == scene]
        by_scene[scene] = {
            "count": len(scene_results),
            "adult_within_1": sum(1 for r in scene_results if r.adult_within_1) / len(scene_results) * 100,
            "child_within_2": sum(1 for r in scene_results if r.child_within_2) / len(scene_results) * 100
        }
    
    # Compliance scenario detection
    compliance_frames = [r for r in results if r.compliance_flag]
    compliance_detection = []
    for r in compliance_frames:
        if r.compliance_flag == "potential_supervision_gap":
            detected = r.pred_adults == 0 and r.pred_children > 0
        else:
            detected = None
        compliance_detection.append({
            "frame": r.frame_name,
            "flag": r.compliance_flag,
            "detected": detected
        })
    
    # Processing time
    avg_time = sum(r.processing_time for r in results) / n
    
    return {
        "total_frames": n,
        "overall": {
            "adult_exact_match": f"{adult_exact}/{n} ({adult_exact/n*100:.1f}%)",
            "adult_within_range": f"{adult_range}/{n} ({adult_range/n*100:.1f}%)",
            "adult_within_1": f"{adult_within_1}/{n} ({adult_within_1/n*100:.1f}%)",
            "child_exact_match": f"{child_exact}/{n} ({child_exact/n*100:.1f}%)",
            "child_within_range": f"{child_range}/{n} ({child_range/n*100:.1f}%)",
            "child_within_2": f"{child_within_2}/{n} ({child_within_2/n*100:.1f}%)"
        },
        "by_difficulty": by_difficulty,
        "by_scene": by_scene,
        "compliance_detection": compliance_detection,
        "processing": {
            "avg_time_per_frame": f"{avg_time:.2f}s",
            "total_time": f"{sum(r.processing_time for r in results):.2f}s"
        }
    }


def print_summary(summary: Dict):
    """Print formatted summary to console."""
    print("\n" + "=" * 70)
    print("ACCURACY SUMMARY")
    print("=" * 70)
    
    overall = summary['overall']
    print(f"\n📊 OVERALL ACCURACY ({summary['total_frames']} frames)")
    print(f"   Adults:")
    print(f"      Exact match:  {overall['adult_exact_match']}")
    print(f"      Within ±1:    {overall['adult_within_1']}")
    print(f"   Children:")
    print(f"      Exact match:  {overall['child_exact_match']}")
    print(f"      Within ±2:    {overall['child_within_2']}")
    
    print(f"\n📈 BY DIFFICULTY")
    for diff, stats in summary.get('by_difficulty', {}).items():
        print(f"   {diff.upper()} ({stats['count']} frames):")
        print(f"      Adult ±1: {stats['adult_within_1']:.0f}% | Child ±2: {stats['child_within_2']:.0f}%")
    
    print(f"\n🏠 BY SCENE TYPE")
    for scene, stats in summary.get('by_scene', {}).items():
        print(f"   {scene} ({stats['count']}): Adult ±1: {stats['adult_within_1']:.0f}% | Child ±2: {stats['child_within_2']:.0f}%")
    
    compliance = summary.get('compliance_detection', [])
    if compliance:
        print(f"\n⚠️  COMPLIANCE SCENARIOS")
        for c in compliance:
            status = "✅ DETECTED" if c['detected'] else "❌ MISSED"
            print(f"   {c['frame']}: {c['flag']} — {status}")
    
    print(f"\n⏱️  PROCESSING: {summary['processing']['avg_time_per_frame']} avg per frame")
    print("=" * 70)


def generate_markdown_report(report: Dict, output_path: Path):
    """Generate a human-readable markdown report."""
    
    md = []
    md.append("# VLM Accuracy Analysis Report")
    md.append(f"\n**Date:** {report['analysis_time']}")
    md.append(f"\n**Model:** `{report['model']}`")
    md.append(f"\n**Frames Directory:** `{report['frames_directory']}`")
    
    summary = report['summary']
    overall = summary['overall']
    
    md.append("\n## Overall Accuracy")
    md.append("\n| Metric | Adults | Children |")
    md.append("|--------|--------|----------|")
    md.append(f"| Exact Match | {overall['adult_exact_match']} | {overall['child_exact_match']} |")
    md.append(f"| Within Range | {overall['adult_within_range']} | {overall['child_within_range']} |")
    md.append(f"| Within ±1/±2 | {overall['adult_within_1']} | {overall['child_within_2']} |")
    
    md.append("\n## Accuracy by Difficulty")
    md.append("\n| Difficulty | Frames | Adult ±1 | Child ±2 |")
    md.append("|------------|--------|----------|----------|")
    for diff, stats in summary.get('by_difficulty', {}).items():
        md.append(f"| {diff.capitalize()} | {stats['count']} | {stats['adult_within_1']:.0f}% | {stats['child_within_2']:.0f}% |")
    
    md.append("\n## Accuracy by Scene Type")
    md.append("\n| Scene | Frames | Adult ±1 | Child ±2 |")
    md.append("|-------|--------|----------|----------|")
    for scene, stats in summary.get('by_scene', {}).items():
        md.append(f"| {scene} | {stats['count']} | {stats['adult_within_1']:.0f}% | {stats['child_within_2']:.0f}% |")
    
    md.append("\n## Frame-by-Frame Results")
    md.append("\n| Frame | Time | GT Adults | Pred Adults | GT Children | Pred Children | Adult | Child |")
    md.append("|-------|------|-----------|-------------|-------------|---------------|-------|-------|")
    
    for r in report['frame_results']:
        adult_status = "✅" if r['adult_within_1'] else "❌"
        child_status = "✅" if r['child_within_2'] else "❌"
        md.append(f"| {r['frame_name']} | {r['timestamp']} | {r['gt_adults']} | {r['pred_adults']} | {r['gt_children']} | {r['pred_children']} | {adult_status} | {child_status} |")
    
    compliance = summary.get('compliance_detection', [])
    if compliance:
        md.append("\n## Compliance Scenario Detection")
        md.append("\n| Frame | Scenario | Detected |")
        md.append("|-------|----------|----------|")
        for c in compliance:
            status = "✅ Yes" if c['detected'] else "❌ No"
            md.append(f"| {c['frame']} | {c['flag']} | {status} |")
    
    md.append("\n## Failure Analysis")
    md.append("\n### Frames with Adult Errors (outside ±1)")
    failures = [r for r in report['frame_results'] if not r['adult_within_1']]
    if failures:
        for r in failures:
            error = r['pred_adults'] - r['gt_adults']
            direction = "overcounted" if error > 0 else "undercounted"
            md.append(f"\n- **{r['frame_name']}** ({r['scene']}, {r['difficulty']}): GT={r['gt_adults']}, Pred={r['pred_adults']} — {direction} by {abs(error)}")
            md.append(f"  - Notes: {r['notes']}")
    else:
        md.append("\nNo adult counting failures outside ±1 tolerance.")
    
    md.append("\n### Frames with Child Errors (outside ±2)")
    failures = [r for r in report['frame_results'] if not r['child_within_2']]
    if failures:
        for r in failures:
            error = r['pred_children'] - r['gt_children']
            direction = "overcounted" if error > 0 else "undercounted"
            md.append(f"\n- **{r['frame_name']}** ({r['scene']}, {r['difficulty']}): GT={r['gt_children']}, Pred={r['pred_children']} — {direction} by {abs(error)}")
            md.append(f"  - Notes: {r['notes']}")
    else:
        md.append("\nNo child counting failures outside ±2 tolerance.")
    
    md.append(f"\n---\n*Generated: {report['analysis_time']}*")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(md))
    
    print(f"\n📄 Markdown report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run VLM accuracy analysis against ground truth'
    )
    parser.add_argument(
        'frames_dir',
        type=Path,
        help='Directory containing extracted frames and ground_truth.json'
    )
    parser.add_argument(
        '--ground-truth', '-g',
        type=Path,
        help='Path to ground truth JSON (default: frames_dir/ground_truth.json)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output directory for reports (default: frames_dir/)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API calls in seconds (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Validate
    if not args.frames_dir.exists():
        print(f"❌ Frames directory not found: {args.frames_dir}")
        sys.exit(1)
    
    # Set output directory
    output_dir = args.output or args.frames_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    try:
        report = run_accuracy_analysis(
            frames_dir=args.frames_dir,
            ground_truth_path=args.ground_truth,
            delay_between_frames=args.delay
        )
        
        # Save JSON report
        json_path = output_dir / 'accuracy_report.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📊 JSON report saved to: {json_path}")
        
        # Generate markdown report
        md_path = output_dir / 'accuracy_report.md'
        generate_markdown_report(report, md_path)
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
