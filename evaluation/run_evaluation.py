#!/usr/bin/env python3
"""
run_evaluation.py
=================
Process all test frames through VLM and collect results.

Usage:
    python evaluation/run_evaluation.py
    python evaluation/run_evaluation.py --frames-dir ./test_data/frames
    python evaluation/run_evaluation.py --limit 10  # Process only first 10
"""

import os
import sys
import json
import base64
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from PIL import Image
from io import BytesIO

# Load environment
load_dotenv(project_root / '.env')

console = Console()


def load_prompt():
    """Load the compliance prompt template."""
    prompt_file = project_root / 'prompts' / 'v1_baseline.txt'
    with open(prompt_file) as f:
        return f.read()


def encode_image(image_path: Path) -> str:
    """Load and encode image to base64."""
    with Image.open(image_path) as img:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large (optional, saves bandwidth)
        max_size = 1024
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


def analyze_frame(client, model: str, prompt: str, image_base64: str) -> dict:
    """Send frame to VLM and get analysis."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        response_text = completion.choices[0].message.content
        
        # Try to parse as JSON
        try:
            # Handle potential markdown code blocks
            clean_response = response_text.strip()
            if clean_response.startswith('```'):
                clean_response = clean_response.split('```')[1]
                if clean_response.startswith('json'):
                    clean_response = clean_response[4:]
            clean_response = clean_response.strip()
            
            parsed = json.loads(clean_response)
            return {
                "success": True,
                "parsed": True,
                "data": parsed,
                "raw_response": response_text
            }
        except json.JSONDecodeError:
            return {
                "success": True,
                "parsed": False,
                "data": None,
                "raw_response": response_text,
                "parse_error": "Failed to parse JSON"
            }
            
    except Exception as e:
        return {
            "success": False,
            "parsed": False,
            "error": f"{type(e).__name__}: {str(e)}"
        }


def main():
    parser = argparse.ArgumentParser(description='Run VLM evaluation on test frames')
    parser.add_argument('--frames-dir', type=Path, default=project_root / 'test_data' / 'frames')
    parser.add_argument('--limit', type=int, help='Limit number of frames to process')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    
    # Configuration
    base_url = os.getenv('RESETDATA_BASE_URL')
    api_key = os.getenv('RESETDATA_API_KEY')
    model = os.getenv('RESETDATA_VISION_MODEL')
    
    if not api_key:
        console.print("[red]ERROR: RESETDATA_API_KEY not set[/red]")
        sys.exit(1)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png'}
    frames = [f for f in args.frames_dir.iterdir() 
              if f.suffix.lower() in image_extensions]
    
    if not frames:
        console.print(f"[yellow]No images found in {args.frames_dir}[/yellow]")
        console.print("Add test images first. See test_data/scenarios.md for guidance.")
        sys.exit(1)
    
    frames = sorted(frames)
    if args.limit:
        frames = frames[:args.limit]
    
    console.print(f"\n[bold]Processing {len(frames)} frames...[/bold]\n")
    
    # Load prompt
    prompt = load_prompt()
    
    # Create client
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    # Process frames
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Analyzing frames...", total=len(frames))
        
        for frame_path in frames:
            progress.update(task, description=f"Processing {frame_path.name}...")
            
            start_time = datetime.now()
            
            # Encode image
            try:
                image_base64 = encode_image(frame_path)
            except Exception as e:
                results.append({
                    "frame": frame_path.name,
                    "success": False,
                    "error": f"Failed to load image: {e}"
                })
                progress.advance(task)
                continue
            
            # Analyze
            result = analyze_frame(client, model, prompt, image_base64)
            result["frame"] = frame_path.name
            result["latency_seconds"] = (datetime.now() - start_time).total_seconds()
            results.append(result)
            
            progress.advance(task)
    
    # Summary
    console.print("\n" + "="*60)
    console.print("[bold]EVALUATION SUMMARY[/bold]")
    console.print("="*60 + "\n")
    
    successful = [r for r in results if r.get('success')]
    parsed = [r for r in results if r.get('parsed')]
    
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total frames", str(len(results)))
    table.add_row("Successful API calls", f"{len(successful)} ({100*len(successful)/len(results):.1f}%)")
    table.add_row("Valid JSON responses", f"{len(parsed)} ({100*len(parsed)/len(results):.1f}%)")
    
    if successful:
        avg_latency = sum(r['latency_seconds'] for r in successful) / len(successful)
        table.add_row("Average latency", f"{avg_latency:.2f}s")
    
    console.print(table)
    
    # Save results
    results_dir = project_root / 'evaluation' / 'results'
    results_dir.mkdir(exist_ok=True)
    
    output_file = args.output or results_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "frames_processed": len(results),
            "success_rate": len(successful) / len(results) if results else 0,
            "parse_rate": len(parsed) / len(results) if results else 0,
            "results": results
        }, f, indent=2)
    
    console.print(f"\n[dim]Results saved to: {output_file}[/dim]")
    console.print("\n[bold]Next:[/bold] Run calculate_metrics.py to compare against ground truth\n")


if __name__ == "__main__":
    main()
