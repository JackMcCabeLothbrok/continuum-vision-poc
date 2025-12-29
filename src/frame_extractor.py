#!/usr/bin/env python3
"""
frame_extractor.py - Extract frames from video at fixed intervals

Usage:
    python src/frame_extractor.py test_data/videos/ChildcareCentreScenes.mp4 --interval 10

This creates frames in test_data/extracted/<video_name>/
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def get_video_info(video_path: Path) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        str(video_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    
    data = json.loads(result.stdout)
    
    # Extract relevant info
    video_stream = next(
        (s for s in data.get('streams', []) if s['codec_type'] == 'video'),
        None
    )
    
    if not video_stream:
        raise RuntimeError("No video stream found")
    
    format_info = data.get('format', {})
    
    return {
        'duration': float(format_info.get('duration', 0)),
        'width': int(video_stream.get('width', 0)),
        'height': int(video_stream.get('height', 0)),
        'fps': eval(video_stream.get('r_frame_rate', '30/1')),  # e.g., "30/1"
        'codec': video_stream.get('codec_name', 'unknown'),
        'filename': format_info.get('filename', ''),
        'size_bytes': int(format_info.get('size', 0)),
    }


def extract_frames(
    video_path: Path,
    output_dir: Path,
    interval_seconds: float = 10.0,
    output_format: str = 'jpg',
    quality: int = 85
) -> dict:
    """
    Extract frames from video at fixed intervals.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        interval_seconds: Time between frames (default: 10 seconds)
        output_format: Output image format (jpg, png)
        quality: JPEG quality (1-100)
    
    Returns:
        Dictionary with extraction metadata
    """
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video info
    print(f"📹 Analysing video: {video_path.name}")
    video_info = get_video_info(video_path)
    
    duration = video_info['duration']
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   Resolution: {video_info['width']}x{video_info['height']}")
    print(f"   FPS: {video_info['fps']:.2f}")
    
    # Calculate expected frames
    expected_frames = int(duration / interval_seconds) + 1
    print(f"   Expected frames: {expected_frames} (every {interval_seconds}s)")
    
    # Output pattern
    output_pattern = output_dir / f"frame_%06d.{output_format}"
    
    # FFmpeg command
    # -vf "fps=1/10" means 1 frame every 10 seconds
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', f'fps=1/{interval_seconds}',
        '-q:v', str(int((100 - quality) / 100 * 31)),  # Convert quality to FFmpeg scale
        '-y',  # Overwrite output files
        str(output_pattern)
    ]
    
    print(f"\n⚙️  Extracting frames...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FFmpeg stderr: {result.stderr}")
        raise RuntimeError(f"FFmpeg failed with code {result.returncode}")
    
    # Count extracted frames
    extracted_files = sorted(output_dir.glob(f"frame_*.{output_format}"))
    num_frames = len(extracted_files)
    
    print(f"✅ Extracted {num_frames} frames to {output_dir}/")
    
    # Create metadata file
    metadata = {
        'source_video': str(video_path),
        'extraction_time': datetime.now().isoformat(),
        'interval_seconds': interval_seconds,
        'video_info': video_info,
        'num_frames': num_frames,
        'frames': [
            {
                'filename': f.name,
                'timestamp_seconds': i * interval_seconds,
                'timestamp_formatted': f"{int(i * interval_seconds // 60):02d}:{int(i * interval_seconds % 60):02d}"
            }
            for i, f in enumerate(extracted_files)
        ]
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Metadata saved to {metadata_path}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from video at fixed intervals'
    )
    parser.add_argument(
        'video',
        type=Path,
        help='Path to input video file'
    )
    parser.add_argument(
        '--interval', '-i',
        type=float,
        default=10.0,
        help='Seconds between frames (default: 10)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output directory (default: test_data/extracted/<video_name>/)'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['jpg', 'png'],
        default='jpg',
        help='Output format (default: jpg)'
    )
    parser.add_argument(
        '--quality', '-q',
        type=int,
        default=85,
        help='JPEG quality 1-100 (default: 85)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.video.exists():
        print(f"❌ Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Set default output directory
    if args.output is None:
        video_name = args.video.stem  # Filename without extension
        args.output = Path('test_data/extracted') / video_name
    
    # Extract frames
    try:
        metadata = extract_frames(
            video_path=args.video,
            output_dir=args.output,
            interval_seconds=args.interval,
            output_format=args.format,
            quality=args.quality
        )
        
        print(f"\n📊 Summary:")
        print(f"   Video duration: {metadata['video_info']['duration']:.1f}s")
        print(f"   Frames extracted: {metadata['num_frames']}")
        print(f"   Output directory: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
