#!/usr/bin/env python3
"""
02_test_vision_input.py
=======================
Purpose: Test if vision model accepts images in standard OpenAI format

What this tests:
- Base64 image input
- Multi-part content (text + image)
- Response to visual queries

Run: python discovery/02_test_vision_input.py
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
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from PIL import Image, ImageDraw

# Load environment
load_dotenv(project_root / '.env')

console = Console()


def create_test_image():
    """Create a simple test image with colored rectangles."""
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some colored rectangles (simulating people/objects)
    draw.rectangle([50, 50, 100, 150], fill='blue')    # "Adult" 1
    draw.rectangle([150, 50, 200, 150], fill='blue')   # "Adult" 2
    draw.rectangle([250, 80, 280, 130], fill='red')    # "Child" 1
    draw.rectangle([300, 80, 330, 130], fill='red')    # "Child" 2
    draw.rectangle([320, 85, 345, 125], fill='red')    # "Child" 3
    
    # Add some text
    draw.text((10, 10), "Test Image: 2 blue rectangles, 3 red rectangles", fill='black')
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return img_base64, img


def test_vision_input():
    """Test vision input with standard OpenAI format."""
    
    # Configuration
    base_url = os.getenv('RESETDATA_BASE_URL', 'https://models.au-syd.resetdata.ai/v1')
    api_key = os.getenv('RESETDATA_API_KEY')
    model = os.getenv('RESETDATA_VISION_MODEL', 'meta/llama-3.2-11b-vision-instruct:shared')
    
    if not api_key:
        console.print("[red]ERROR: RESETDATA_API_KEY not set in .env file[/red]")
        return False, None
    
    console.print(Panel.fit(
        f"[bold]Test 2: Vision Input (OpenAI Format)[/bold]\n\n"
        f"Base URL: {base_url}\n"
        f"Model: {model}",
        title="Configuration"
    ))
    
    # Create test image
    console.print("\n[yellow]Creating test image...[/yellow]")
    img_base64, img = create_test_image()
    
    # Save test image for reference
    results_dir = project_root / 'discovery' / 'results'
    results_dir.mkdir(exist_ok=True)
    test_img_path = results_dir / 'test_image.jpg'
    img.save(test_img_path)
    console.print(f"[dim]Test image saved to: {test_img_path}[/dim]")
    
    # Create client
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    # Test request with image - Standard OpenAI format
    console.print("\n[yellow]Sending vision request (OpenAI format)...[/yellow]")
    
    prompt = "Describe what you see in this image. Count the blue rectangles and red rectangles separately."
    
    start_time = datetime.now()
    
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
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=300,
            stream=False
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Extract response
        response_text = completion.choices[0].message.content
        
        console.print(Panel.fit(
            f"[green]SUCCESS - OpenAI format works![/green]\n\n"
            f"Response:\n{response_text}\n\n"
            f"Latency: {elapsed:.2f}s\n"
            f"Model: {completion.model}",
            title="Result"
        ))
        
        # Save result
        result = {
            "test": "02_vision_input",
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "format": "openai_standard",
            "latency_seconds": elapsed,
            "model": completion.model,
            "prompt": prompt,
            "response": response_text,
            "finish_reason": completion.choices[0].finish_reason,
            "proxy_required": False
        }
        
        result_file = results_dir / f"02_vision_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        console.print(f"\n[dim]Result saved to: {result_file}[/dim]")
        return True, result
        
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        error_str = str(e)
        
        console.print(Panel.fit(
            f"[red]FAILED with OpenAI format[/red]\n\n"
            f"Error: {type(e).__name__}\n"
            f"Message: {error_str[:500]}...\n\n"
            f"Elapsed: {elapsed:.2f}s\n\n"
            f"[yellow]→ This may require NIM format translation (like embeddings)[/yellow]\n"
            f"[yellow]→ Run 03_test_formats.py to try alternative formats[/yellow]",
            title="Result"
        ))
        
        # Save error result
        result = {
            "test": "02_vision_input",
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "format": "openai_standard",
            "latency_seconds": elapsed,
            "error_type": type(e).__name__,
            "error_message": error_str,
            "proxy_required": "unknown - try 03_test_formats.py"
        }
        
        result_file = results_dir / f"02_vision_input_ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return False, result


if __name__ == "__main__":
    console.print("\n[bold blue]═══ Continuum Vision POC - Discovery Phase ═══[/bold blue]\n")
    
    success, result = test_vision_input()
    
    if success:
        console.print("\n[green]✓ Vision input works with standard OpenAI format![/green]")
        console.print("[green]  No proxy needed - can call ResetData directly.[/green]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print("  1. Add test images to test_data/frames/")
        console.print("  2. Run evaluation/run_evaluation.py")
    else:
        console.print("\n[yellow]⚠ OpenAI format failed. Try alternative formats:[/yellow]")
        console.print("  python discovery/03_test_formats.py\n")
    
    sys.exit(0 if success else 1)
