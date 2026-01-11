
import os
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("SECOND_OPENAI_API_KEY")

async def test_key():
    print(f"Testing Key: {API_KEY[:10]}...")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            print(f"Status: {resp.status}")
            data = await resp.json()
            print(f"Response: {data}")
            if resp.status == 429:
                print("\nRate Limit Headers:")
                pw = resp.headers
                for k in ['x-ratelimit-limit-tokens', 'x-ratelimit-remaining-tokens', 'x-ratelimit-reset-tokens']:
                    print(f"  {k}: {pw.get(k)}")

if __name__ == "__main__":
    if not API_KEY:
        print("SECOND_OPENAI_API_KEY not found")
    else:
        asyncio.run(test_key())
