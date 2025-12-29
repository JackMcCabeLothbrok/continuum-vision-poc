#!/usr/bin/env python3
"""
video_pipeline.py - End-to-end video analysis pipeline

Extracts frames from video and runs VLM analysis on each frame.

Usage:
    python src/video_pipeline.py test_data/videos/ChildcareCentreScenes.mp4

Or with pre-extracted frames:
    python src/video_pipeline.py --frames-dir test_data/extracted/ChildcareCentreScenes/
"""

import os
import sys
import json
import base64
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

# Initialize client
client = OpenAI(
    base_url=os.getenv('RESETDATA_BASE_URL'),
    api_key=os.getenv('RESETDATA_API_KEY')
)
MODEL = os.getenv('RESETDATA_VISION_MODEL')

# Best performing prompt from testing
COUNTING_PROMPT = """Count the adults and children visible in this childcare image.
Respond with JSON only: {"adult_count": X, "child_count": Y}"""


def encode_image(image_path: Path) -> str:
    """Encode image to base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def analyse_frame(image_path: Path, prompt: str = COUNTING_PROMPT) -> Dict:
    """
    Analyse a single frame with the VLM.
    
    Returns dict with counts and metadata.
    """
    img_b64 = encode_image(image_path)
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }],
            max_tokens=100,
            temperature=0.0
        )
        
        response_text = response.choices[0].message.content
        
        # Parse JSON from response
        try:
            # Try direct parse
            result = json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from text
            import re
            match = re.search(r'\{[^{}]*\}', response_text)
            if match:
                result = json.loads(match.group())
            else:
                result = {"error": "parse_failed", "raw": response_text}
        
        return {
            "adult_count": result.get("adult_count", -1),
            "child_count": result.get("child_count", -1),
            "raw_response": response_text,
            "success": True
        }
        
    except Exception as e:
        return {
            "adult_count": -1,
            "child_count": -1,
            "error": str(e),
            "success": False
        }


def extract_frames_from_video(video_path: Path, interval: float = 10.0) -> Path:
    """Extract frames from video, returns output directory."""
    from frame_extractor import extract_frames
    
    video_name = video_path.stem
    output_dir = Path('test_data/extracted') / video_name
    
    extract_frames(
        video_path=video_path,
        output_dir=output_dir,
        interval_seconds=interval
    )
    
    return output_dir


def run_pipeline(
    frames_dir: Path,
    output_path: Optional[Path] = None,
    delay_between_frames: float = 0.5
) -> Dict:
    """
    Run VLM analysis on all frames in directory.
    
    Args:
        frames_dir: Directory containing extracted frames
        output_path: Path to save results JSON
        delay_between_frames: Seconds between API calls (rate limiting)
    
    Returns:
        Dictionary with all results and summary
    """
    
    # Load metadata if available
    metadata_path = frames_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            extraction_metadata = json.load(f)
    else:
        extraction_metadata = None
    
    # Find all frame images
    frame_files = sorted(frames_dir.glob('frame_*.jpg')) + sorted(frames_dir.glob('frame_*.png'))
    
    if not frame_files:
        # Also try numbered files
        frame_files = sorted(frames_dir.glob('*.jpg')) + sorted(frames_dir.glob('*.png'))
    
    if not frame_files:
        raise RuntimeError(f"No frame images found in {frames_dir}")
    
    print(f"\n🎬 Running VLM Pipeline")
    print(f"   Frames directory: {frames_dir}")
    print(f"   Total frames: {len(frame_files)}")
    print(f"   Model: {MODEL}")
    print("=" * 60)
    
    results = []
    start_time = time.time()
    
    for i, frame_path in enumerate(frame_files):
        # Get timestamp from metadata or calculate from index
        if extraction_metadata:
            frame_meta = next(
                (f for f in extraction_metadata.get('frames', []) if f['filename'] == frame_path.name),
                None
            )
            timestamp = frame_meta['timestamp_formatted'] if frame_meta else f"frame_{i}"
        else:
            timestamp = f"frame_{i:04d}"
        
        print(f"\n[{i+1}/{len(frame_files)}] {frame_path.name} ({timestamp})")
        
        # Analyse frame
        frame_start = time.time()
        analysis = analyse_frame(frame_path)
        frame_time = time.time() - frame_start
        
        # Add metadata
        analysis['frame_path'] = str(frame_path)
        analysis['frame_index'] = i
        analysis['timestamp'] = timestamp
        analysis['processing_time_seconds'] = round(frame_time, 2)
        
        results.append(analysis)
        
        # Print result
        if analysis['success']:
            print(f"   Adults: {analysis['adult_count']}, Children: {analysis['child_count']} ({frame_time:.2f}s)")
        else:
            print(f"   ❌ Error: {analysis.get('error', 'unknown')}")
        
        # Rate limiting
        if i < len(frame_files) - 1:
            time.sleep(delay_between_frames)
    
    total_time = time.time() - start_time
    
    # Calculate summary statistics
    successful = [r for r in results if r['success']]
    
    if successful:
        adult_counts = [r['adult_count'] for r in successful]
        child_counts = [r['child_count'] for r in successful]
        
        summary = {
            'total_frames': len(frame_files),
            'successful_analyses': len(successful),
            'failed_analyses': len(results) - len(successful),
            'adult_count_range': [min(adult_counts), max(adult_counts)],
            'adult_count_avg': round(sum(adult_counts) / len(adult_counts), 1),
            'child_count_range': [min(child_counts), max(child_counts)],
            'child_count_avg': round(sum(child_counts) / len(child_counts), 1),
            'total_processing_time_seconds': round(total_time, 2),
            'avg_time_per_frame_seconds': round(total_time / len(frame_files), 2)
        }
    else:
        summary = {
            'total_frames': len(frame_files),
            'successful_analyses': 0,
            'failed_analyses': len(results),
            'error': 'All analyses failed'
        }
    
    # Compile full results
    pipeline_results = {
        'pipeline_run_time': datetime.now().isoformat(),
        'frames_directory': str(frames_dir),
        'model': MODEL,
        'prompt': COUNTING_PROMPT,
        'extraction_metadata': extraction_metadata,
        'summary': summary,
        'frame_results': results
    }
    
    # Save results
    if output_path is None:
        output_path = frames_dir / 'analysis_results.json'
    
    with open(output_path, 'w') as f:
        json.dump(pipeline_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 PIPELINE SUMMARY")
    print("=" * 60)
    print(f"   Frames analysed: {summary['total_frames']}")
    print(f"   Successful: {summary.get('successful_analyses', 0)}")
    print(f"   Failed: {summary.get('failed_analyses', 0)}")
    if 'adult_count_range' in summary:
        print(f"   Adult count range: {summary['adult_count_range'][0]}-{summary['adult_count_range'][1]} (avg: {summary['adult_count_avg']})")
        print(f"   Child count range: {summary['child_count_range'][0]}-{summary['child_count_range'][1]} (avg: {summary['child_count_avg']})")
    print(f"   Total time: {summary.get('total_processing_time_seconds', 0):.1f}s")
    print(f"   Avg per frame: {summary.get('avg_time_per_frame_seconds', 0):.2f}s")
    print(f"\n📄 Results saved to: {output_path}")
    
    return pipeline_results


def main():
    parser = argparse.ArgumentParser(
        description='Run VLM analysis pipeline on video or extracted frames'
    )
    
    parser.add_argument(
        'input',
        type=Path,
        nargs='?',
        help='Video file (.mp4) or frames directory'
    )
    parser.add_argument(
        '--frames-dir',
        type=Path,
        help='Use pre-extracted frames directory'
    )
    parser.add_argument(
        '--interval', '-i',
        type=float,
        default=10.0,
        help='Frame extraction interval in seconds (default: 10)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        help='Output path for results JSON'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API calls in seconds (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Determine input type
    if args.frames_dir:
        frames_dir = args.frames_dir
    elif args.input:
        if args.input.is_dir():
            frames_dir = args.input
        elif args.input.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            print(f"📹 Extracting frames from: {args.input}")
            frames_dir = extract_frames_from_video(args.input, args.interval)
        else:
            print(f"❌ Unknown input type: {args.input}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
    
    # Validate frames directory
    if not frames_dir.exists():
        print(f"❌ Frames directory not found: {frames_dir}")
        sys.exit(1)
    
    # Run pipeline
    try:
        results = run_pipeline(
            frames_dir=frames_dir,
            output_path=args.output,
            delay_between_frames=args.delay
        )
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
