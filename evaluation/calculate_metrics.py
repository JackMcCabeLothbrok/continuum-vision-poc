#!/usr/bin/env python3
"""
calculate_metrics.py
====================
Compare VLM results against ground truth and calculate accuracy metrics.

Usage:
    python evaluation/calculate_metrics.py
    python evaluation/calculate_metrics.py --results evaluation/results/evaluation_20241229.json
"""

import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def load_ground_truth() -> pd.DataFrame:
    """Load ground truth CSV."""
    gt_file = project_root / 'test_data' / 'ground_truth.csv'
    return pd.read_csv(gt_file)


def load_latest_results() -> dict:
    """Load the most recent evaluation results."""
    results_dir = project_root / 'evaluation' / 'results'
    result_files = sorted(results_dir.glob('evaluation_*.json'), reverse=True)
    
    if not result_files:
        return None
    
    with open(result_files[0]) as f:
        return json.load(f)


def calculate_metrics(ground_truth: pd.DataFrame, results: dict) -> dict:
    """Calculate accuracy metrics comparing VLM output to ground truth."""
    
    metrics = {
        "adult_count": {"errors": [], "within_1": 0, "total": 0},
        "child_count": {"errors": [], "within_2": 0, "total": 0},
        "json_parse": {"success": 0, "total": 0},
        "by_scenario": {},
        "by_lighting": {}
    }
    
    # Create lookup from ground truth
    gt_lookup = {row['filename']: row for _, row in ground_truth.iterrows()}
    
    for result in results.get('results', []):
        frame_name = result.get('frame')
        
        metrics["json_parse"]["total"] += 1
        
        if not result.get('parsed'):
            continue
        
        metrics["json_parse"]["success"] += 1
        
        # Check if we have ground truth for this frame
        if frame_name not in gt_lookup:
            continue
        
        gt = gt_lookup[frame_name]
        pred = result.get('data', {})
        
        # Adult count accuracy
        if 'staff_count' in pred:
            pred_adults = pred['staff_count']
            gt_adults = gt['adult_count']
            error = abs(pred_adults - gt_adults)
            
            metrics["adult_count"]["errors"].append(error)
            metrics["adult_count"]["total"] += 1
            if error <= 1:
                metrics["adult_count"]["within_1"] += 1
        
        # Child count accuracy
        if 'child_count' in pred:
            pred_children = pred['child_count']
            gt_children = gt['child_count']
            error = abs(pred_children - gt_children)
            
            metrics["child_count"]["errors"].append(error)
            metrics["child_count"]["total"] += 1
            if error <= 2:
                metrics["child_count"]["within_2"] += 1
        
        # Track by scenario
        scenario = gt.get('scenario', 'unknown')
        if scenario not in metrics["by_scenario"]:
            metrics["by_scenario"][scenario] = {"adult_errors": [], "child_errors": []}
        if 'staff_count' in pred:
            metrics["by_scenario"][scenario]["adult_errors"].append(
                abs(pred['staff_count'] - gt['adult_count'])
            )
        if 'child_count' in pred:
            metrics["by_scenario"][scenario]["child_errors"].append(
                abs(pred['child_count'] - gt['child_count'])
            )
        
        # Track by lighting
        lighting = gt.get('lighting', 'unknown')
        if lighting not in metrics["by_lighting"]:
            metrics["by_lighting"][lighting] = {"adult_errors": [], "child_errors": []}
        if 'staff_count' in pred:
            metrics["by_lighting"][lighting]["adult_errors"].append(
                abs(pred['staff_count'] - gt['adult_count'])
            )
        if 'child_count' in pred:
            metrics["by_lighting"][lighting]["child_errors"].append(
                abs(pred['child_count'] - gt['child_count'])
            )
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, help='Specific results file to analyze')
    args = parser.parse_args()
    
    # Load data
    console.print("\n[bold]Loading data...[/bold]\n")
    
    ground_truth = load_ground_truth()
    console.print(f"Ground truth: {len(ground_truth)} frames")
    
    if args.results:
        with open(args.results) as f:
            results = json.load(f)
    else:
        results = load_latest_results()
    
    if not results:
        console.print("[red]No evaluation results found. Run run_evaluation.py first.[/red]")
        sys.exit(1)
    
    console.print(f"Results file: {len(results.get('results', []))} frames analyzed")
    
    # Calculate metrics
    metrics = calculate_metrics(ground_truth, results)
    
    # Display results
    console.print("\n" + "="*70)
    console.print("[bold]ACCURACY METRICS[/bold]")
    console.print("="*70 + "\n")
    
    # Success criteria table
    table = Table(title="Success Criteria Assessment")
    table.add_column("Metric", style="cyan")
    table.add_column("Target", style="yellow")
    table.add_column("Actual", style="green")
    table.add_column("Status")
    
    # JSON parse rate
    parse_rate = metrics["json_parse"]["success"] / metrics["json_parse"]["total"] * 100 if metrics["json_parse"]["total"] > 0 else 0
    table.add_row(
        "JSON Parse Rate",
        ">95%",
        f"{parse_rate:.1f}%",
        "✅ PASS" if parse_rate >= 95 else "❌ FAIL"
    )
    
    # Adult accuracy
    adult_rate = metrics["adult_count"]["within_1"] / metrics["adult_count"]["total"] * 100 if metrics["adult_count"]["total"] > 0 else 0
    table.add_row(
        "Adult Count (within ±1)",
        ">90%",
        f"{adult_rate:.1f}%",
        "✅ PASS" if adult_rate >= 90 else "❌ FAIL"
    )
    
    # Child accuracy
    child_rate = metrics["child_count"]["within_2"] / metrics["child_count"]["total"] * 100 if metrics["child_count"]["total"] > 0 else 0
    table.add_row(
        "Child Count (within ±2)",
        ">80%",
        f"{child_rate:.1f}%",
        "✅ PASS" if child_rate >= 80 else "❌ FAIL"
    )
    
    console.print(table)
    
    # MAE table
    if metrics["adult_count"]["errors"]:
        console.print("\n")
        mae_table = Table(title="Mean Absolute Error")
        mae_table.add_column("Count Type")
        mae_table.add_column("MAE")
        mae_table.add_column("Samples")
        
        adult_mae = sum(metrics["adult_count"]["errors"]) / len(metrics["adult_count"]["errors"])
        mae_table.add_row("Adults", f"{adult_mae:.2f}", str(len(metrics["adult_count"]["errors"])))
        
        child_mae = sum(metrics["child_count"]["errors"]) / len(metrics["child_count"]["errors"])
        mae_table.add_row("Children", f"{child_mae:.2f}", str(len(metrics["child_count"]["errors"])))
        
        console.print(mae_table)
    
    # Performance by scenario
    if metrics["by_scenario"]:
        console.print("\n")
        scenario_table = Table(title="Performance by Scenario")
        scenario_table.add_column("Scenario")
        scenario_table.add_column("Adult MAE")
        scenario_table.add_column("Child MAE")
        scenario_table.add_column("Samples")
        
        for scenario, data in metrics["by_scenario"].items():
            adult_mae = sum(data["adult_errors"]) / len(data["adult_errors"]) if data["adult_errors"] else "-"
            child_mae = sum(data["child_errors"]) / len(data["child_errors"]) if data["child_errors"] else "-"
            samples = max(len(data["adult_errors"]), len(data["child_errors"]))
            
            if isinstance(adult_mae, float):
                adult_mae = f"{adult_mae:.2f}"
            if isinstance(child_mae, float):
                child_mae = f"{child_mae:.2f}"
            
            scenario_table.add_row(scenario, adult_mae, child_mae, str(samples))
        
        console.print(scenario_table)
    
    # Overall verdict
    console.print("\n")
    all_pass = parse_rate >= 95 and adult_rate >= 90 and child_rate >= 80
    
    if all_pass:
        console.print(Panel.fit(
            "[bold green]✅ ALL CRITERIA MET[/bold green]\n\n"
            "VLM demonstrates sufficient accuracy for childcare compliance monitoring.\n"
            "Recommend proceeding to Thor hardware pilot.",
            title="VERDICT"
        ))
    else:
        console.print(Panel.fit(
            "[bold yellow]⚠️ SOME CRITERIA NOT MET[/bold yellow]\n\n"
            "Consider:\n"
            "• Improving prompts\n"
            "• Testing 90B model variant\n"
            "• Investigating failure cases in docs/failure_modes.md",
            title="VERDICT"
        ))
    
    # Save metrics
    results_dir = project_root / 'evaluation' / 'results'
    metrics_file = results_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(metrics_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "pass_all_criteria": all_pass,
            "json_parse_rate": parse_rate,
            "adult_within_1_rate": adult_rate,
            "child_within_2_rate": child_rate,
            "adult_mae": sum(metrics["adult_count"]["errors"]) / len(metrics["adult_count"]["errors"]) if metrics["adult_count"]["errors"] else None,
            "child_mae": sum(metrics["child_count"]["errors"]) / len(metrics["child_count"]["errors"]) if metrics["child_count"]["errors"] else None,
        }, f, indent=2)
    
    console.print(f"\n[dim]Metrics saved to: {metrics_file}[/dim]\n")


if __name__ == "__main__":
    main()
