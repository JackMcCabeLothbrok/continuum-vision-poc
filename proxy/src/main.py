#!/usr/bin/env python3
"""
Vision API Proxy
================
Only implement if ResetData requires non-standard format translation.

Based on discovery tests:
- If 02_test_vision_input.py succeeds: This proxy is NOT needed
- If it fails and 03_test_formats.py finds working format: Implement translation here

See embedding proxy for reference implementation pattern.
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import httpx

app = FastAPI(title="Vision API Proxy", version="0.1.0")


class ImageContent(BaseModel):
    type: str = "image_url"
    image_url: dict


class TextContent(BaseModel):
    type: str = "text"
    text: str


class Message(BaseModel):
    role: str
    content: Union[str, List[Union[TextContent, ImageContent]]]


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.2


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    Translate request if needed and forward to ResetData.
    
    Implement translation logic here based on 03_test_formats.py findings.
    """
    
    base_url = os.getenv('RESETDATA_BASE_URL')
    api_key = os.getenv('RESETDATA_API_KEY')
    
    if not base_url or not api_key:
        raise HTTPException(500, "Missing configuration")
    
    # TODO: Add format translation if needed based on discovery results
    # For now, pass through unchanged
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=request.model_dump()
        )
        
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.text)
        
        return response.json()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
