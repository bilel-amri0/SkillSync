#!/usr/bin/env python3
"""
Simple API test with manual env loading
"""
import asyncio
import aiohttp
import os
from pathlib import Path
async def test_jsearch_manual():
    """Test JSearch API with manual env loading"""
    print("🔍 Manual JSearch API Test")
    print("=" * 30)
    
    # Manual loading of .env
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Environment loaded manually")
    else:
        print("❌ .env file not found")
        return
    
    # Get API key
    api_key = os.getenv('JSEARCH_RAPIDAPI_KEY')
    if not api_key:
        print("❌ JSearch API key not found")
        return
    
    print(f"🔑 API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test API call
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        'X-RapidAPI-Key': api_key,
        'X-RapidAPI-Host': 'jsearch.p.rapidapi.com'
    }
    params = {
        'query': 'Python developer',
        'page': '1',
        'num_pages': '1'
    }
    
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            print("📡 Making API request...")
            async with session.get(url, headers=headers, params=params) as response:
                print(f"📊 Response status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    job_count = len(data.get('data', []))
                    print(f"✅ SUCCESS! Found {job_count} jobs")
                    
                    if job_count > 0:
                        sample_job = data['data'][0]
                        print(f"📋 Sample job: '{sample_job.get('job_title', 'N/A')}' at {sample_job.get('employer_name', 'N/A')}")
                        return True
                else:
                    error_text = await response.text()
                    print(f"❌ API Error: {error_text[:200]}")
                    return False
                    
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False
async def main():
    success = await test_jsearch_manual()
    if success:
        print("\n🎉 API test successful! Your configuration works!")
        print("🚀 Next: python main_simple_for_frontend.py")
    else:
        print("\n⚠️ API test failed. Check your configuration.")
if __name__ == "__main__":
    asyncio.run(main())