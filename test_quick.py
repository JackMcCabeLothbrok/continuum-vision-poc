import os, base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
    base_url=os.getenv('RESETDATA_BASE_URL'),
    api_key=os.getenv('RESETDATA_API_KEY')
)

# Load real image
with open('test_data/frames/test_real.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

print(f"Image size: {len(img_b64)} bytes base64")

response = client.chat.completions.create(
    model=os.getenv('RESETDATA_VISION_MODEL'),
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "How many people are in this image? Describe what you see."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]
    }],
    max_tokens=500,
    temperature=0.2
)

print(f"Finish reason: {response.choices[0].finish_reason}")
print(f"Response: {response.choices[0].message.content}")
