#!/usr/bin/env python3
"""
01_test_text_only.py
====================
Purpose: Verify basic API connectivity with text-only request

What this tests:
- API key authentication
- Model availability  
- Response format

Run: python discovery/01_test_text_only.py
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

# Load environment
load_dotenv(project_root / '.env')

console = Console()

def test_text_only():
    """Test basic text connectivity to ResetData Vision API."""
    
    # Configuration
    base_url = os.getenv('RESETDATA_BASE_URL', 'https://models.au-syd.resetdata.ai/v1')
    api_key = os.getenv('RESETDATA_API_KEY')
    model = os.getenv('RESETDATA_VISION_MODEL', 'meta/llama-3.2-11b-vision-instruct:shared')
    
    if not api_key:
        console.print("[red]ERROR: RESETDATA_API_KEY not set in .env file[/red]")
        console.print("Copy .env.example to .env and add your API key")
        return False
    
    console.print(Panel.fit(
        f"[bold]Test 1: Text-Only Connectivity[/bold]\n\n"
        f"Base URL: {base_url}\n"
        f"Model: {model}\n"
        f"API Key: {api_key[:20]}...{api_key[-10:]}",
        title="Configuration"
    ))
    
    # Create client
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    # Test request
    console.print("\n[yellow]Sending text-only request...[/yellow]")
    
    start_time = datetime.now()
    
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Respond concisely."},
                {"role": "user", "content": "Count from 1 to 5 and confirm you are operational."},
            ],
            temperature=0.2,
            max_tokens=100,
            stream=False
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Extract response
        response_text = completion.choices[0].message.content
        
        console.print(Panel.fit(
            f"[green]SUCCESS[/green]\n\n"
            f"Response: {response_text}\n\n"
            f"Latency: {elapsed:.2f}s\n"
            f"Model: {completion.model}\n"
            f"Finish reason: {completion.choices[0].finish_reason}",
            title="Result"
        ))
        
        # Save result
        result = {
            "test": "01_text_only",
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "latency_seconds": elapsed,
            "model": completion.model,
            "response": response_text,
            "finish_reason": completion.choices[0].finish_reason
        }
        
        results_dir = project_root / 'discovery' / 'results'
        results_dir.mkdir(exist_ok=True)
        
        result_file = results_dir / f"01_text_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        console.print(f"\n[dim]Result saved to: {result_file}[/dim]")
        return True
        
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        
        console.print(Panel.fit(
            f"[red]FAILED[/red]\n\n"
            f"Error: {type(e).__name__}\n"
            f"Message: {str(e)}\n\n"
            f"Elapsed: {elapsed:.2f}s",
            title="Result"
        ))
        
        # Save error result
        result = {
            "test": "01_text_only",
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "latency_seconds": elapsed,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }
        
        results_dir = project_root / 'discovery' / 'results'
        results_dir.mkdir(exist_ok=True)
        
        result_file = results_dir / f"01_text_only_ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return False


if __name__ == "__main__":
    console.print("\n[bold blue]═══ Continuum Vision POC - Discovery Phase ═══[/bold blue]\n")
    
    success = test_text_only()
    
    if success:
        console.print("\n[green]✓ Text connectivity verified. Proceed to 02_test_vision_input.py[/green]\n")
    else:
        console.print("\n[red]✗ Text connectivity failed. Check API key and try again.[/red]\n")
    
    sys.exit(0 if success else 1)
