#!/usr/bin/env python3
"""
03_test_formats.py
==================
Purpose: Try alternative formats if standard OpenAI format fails

What this tests:
- NVIDIA NIM-specific parameters
- Different image encoding approaches
- Direct httpx requests (bypassing OpenAI client)

Run: python discovery/03_test_formats.py
"""

import os
import sys
import json
import base64
from datetime import datetime
from pathlib import Path
from io import BytesIO

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import httpx
from rich.console import Console
from rich.panel import Panel
from PIL import Image, ImageDraw

# Load environment
load_dotenv(project_root / '.env')

console = Console()


def create_test_image():
    """Create a simple test image."""
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 100, 150], fill='blue')
    draw.rectangle([150, 50, 200, 150], fill='blue')
    draw.rectangle([250, 80, 280, 130], fill='red')
    draw.rectangle([300, 80, 330, 130], fill='red')
    draw.rectangle([320, 85, 345, 125], fill='red')
    draw.text((10, 10), "Test: 2 blue, 3 red rectangles", fill='black')
    
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def test_format(name, request_body, headers, base_url):
    """Test a specific request format."""
    console.print(f"\n[yellow]Testing format: {name}...[/yellow]")
    
    start_time = datetime.now()
    
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=request_body
            )
            response.raise_for_status()
            
        elapsed = (datetime.now() - start_time).total_seconds()
        data = response.json()
        
        response_text = data.get('choices', [{}])[0].get('message', {}).get('content', 'No content')
        
        console.print(f"[green]✓ SUCCESS with format: {name}[/green]")
        console.print(f"  Response: {response_text[:200]}...")
        console.print(f"  Latency: {elapsed:.2f}s")
        
        return {
            "format": name,
            "success": True,
            "latency_seconds": elapsed,
            "response": response_text,
            "status_code": response.status_code
        }
        
    except httpx.HTTPStatusError as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        console.print(f"[red]✗ FAILED: {name}[/red]")
        console.print(f"  Status: {e.response.status_code}")
        console.print(f"  Error: {e.response.text[:300]}...")
        
        return {
            "format": name,
            "success": False,
            "latency_seconds": elapsed,
            "status_code": e.response.status_code,
            "error": e.response.text
        }
        
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        console.print(f"[red]✗ FAILED: {name}[/red]")
        console.print(f"  Error: {type(e).__name__}: {str(e)[:200]}")
        
        return {
            "format": name,
            "success": False,
            "latency_seconds": elapsed,
            "error": f"{type(e).__name__}: {str(e)}"
        }


def main():
    """Try various API formats to find what works."""
    
    base_url = os.getenv('RESETDATA_BASE_URL', 'https://models.au-syd.resetdata.ai/v1')
    api_key = os.getenv('RESETDATA_API_KEY')
    model = os.getenv('RESETDATA_VISION_MODEL', 'meta/llama-3.2-11b-vision-instruct:shared')
    
    if not api_key:
        console.print("[red]ERROR: RESETDATA_API_KEY not set[/red]")
        return
    
    console.print(Panel.fit(
        f"[bold]Test 3: Alternative Format Discovery[/bold]\n\n"
        f"Base URL: {base_url}\n"
        f"Model: {model}\n\n"
        f"Testing multiple request formats to find what works...",
        title="Configuration"
    ))
    
    # Create test image
    img_base64 = create_test_image()
    prompt = "How many rectangles are in this image? Count blue and red separately."
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    results = []
    
    # Format 1: Standard OpenAI (reference - likely already failed)
    format1 = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }
        ],
        "max_tokens": 200
    }
    results.append(test_format("OpenAI Standard", format1, headers, base_url))
    
    # Format 2: With input_type (like embeddings required)
    format2 = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }
        ],
        "max_tokens": 200,
        "input_type": "query"
    }
    results.append(test_format("With input_type=query", format2, headers, base_url))
    
    # Format 3: Image as separate field
    format3 = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "images": [img_base64],
        "max_tokens": 200
    }
    results.append(test_format("Separate images field", format3, headers, base_url))
    
    # Format 4: Content as string with image in different structure
    format4 = {
        "model": model,
        "messages": [
            {
                "role": "user", 
                "content": prompt,
                "images": [f"data:image/jpeg;base64,{img_base64}"]
            }
        ],
        "max_tokens": 200
    }
    results.append(test_format("Images in message object", format4, headers, base_url))
    
    # Format 5: NVIDIA NIM style (image as base64 without data URI prefix)
    format5 = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_base64}}
                ]
            }
        ],
        "max_tokens": 200
    }
    results.append(test_format("Base64 without data URI", format5, headers, base_url))
    
    # Summary
    console.print("\n" + "="*60)
    console.print("[bold]SUMMARY[/bold]")
    console.print("="*60)
    
    working_formats = [r for r in results if r['success']]
    
    if working_formats:
        console.print(f"\n[green]✓ {len(working_formats)} format(s) work![/green]\n")
        for fmt in working_formats:
            console.print(f"  • {fmt['format']} ({fmt['latency_seconds']:.2f}s)")
        
        console.print("\n[bold]Recommendation:[/bold]")
        best = min(working_formats, key=lambda x: x['latency_seconds'])
        console.print(f"  Use format: [green]{best['format']}[/green]")
        
        if best['format'] == "OpenAI Standard":
            console.print("  [green]No proxy needed![/green]")
        else:
            console.print("  [yellow]Proxy may be needed to translate format[/yellow]")
    else:
        console.print("\n[red]✗ No formats worked![/red]")
        console.print("\nPossible issues:")
        console.print("  • API key expired (30-minute keys)")
        console.print("  • Model not available")
        console.print("  • Vision not supported on this model variant")
        console.print("\nCheck the error messages above for details.")
    
    # Save all results
    results_dir = project_root / 'discovery' / 'results'
    results_dir.mkdir(exist_ok=True)
    
    result_file = results_dir / f"03_formats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "formats_tested": len(results),
            "formats_working": len(working_formats),
            "results": results
        }, f, indent=2)
    
    console.print(f"\n[dim]Full results saved to: {result_file}[/dim]\n")
    
    return len(working_formats) > 0


if __name__ == "__main__":
    console.print("\n[bold blue]═══ Continuum Vision POC - Format Discovery ═══[/bold blue]\n")
    
    success = main()
    sys.exit(0 if success else 1)
