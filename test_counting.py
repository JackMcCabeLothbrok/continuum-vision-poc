import os, base64, json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    base_url=os.getenv('RESETDATA_BASE_URL'),
    api_key=os.getenv('RESETDATA_API_KEY')
)

prompt = """You are analyzing a childcare facility image for compliance monitoring.

Count the people visible and respond in JSON format:
{
  "adult_count": <number of adults/staff visible>,
  "child_count": <number of children visible>,
  "adult_confidence": "high" | "medium" | "low",
  "child_confidence": "high" | "medium" | "low",
  "activity": "<what children are doing>",
  "notes": "<any supervision concerns or observations>"
}

Counting guidelines:
- Adults = anyone who appears 18+ or is clearly a staff member/teacher
- Children = anyone who appears under school age (roughly 0-5 years)
- If partially visible, still count but lower confidence
- If uncertain between adult/child, note in observations

Respond with valid JSON only, no other text."""

frames_dir = Path('test_data/frames')
for img_path in sorted(frames_dir.glob('*.jpg')):
    print(f"\n{'='*60}")
    print(f"Image: {img_path.name}")
    print('='*60)
    
    with open(img_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    
    response = client.chat.completions.create(
        model=os.getenv('RESETDATA_VISION_MODEL'),
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }],
        max_tokens=400,
        temperature=0.1  # Lower for more consistent counting
    )
    
    result = response.choices[0].message.content
    
    # Try to parse JSON
    try:
        clean = result.strip()
        if clean.startswith('```'):
            clean = clean.split('```')[1].replace('json', '').strip()
        data = json.loads(clean)
        print(f"Adults: {data.get('adult_count')} ({data.get('adult_confidence')} confidence)")
        print(f"Children: {data.get('child_count')} ({data.get('child_confidence')} confidence)")
        print(f"Activity: {data.get('activity')}")
        print(f"Notes: {data.get('notes')}")
    except Exception as e:
        print(f"Parse failed: {e}")
        print(f"Raw: {result}")
