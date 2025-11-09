#!/usr/bin/env python3
"""
🔧 Test spécifique de la correction JSearch
"""
import asyncio
import aiohttp
import os
from dotenv import load_dotenv
async def test_jsearch_direct():
    """Test direct JSearch API avec gestion des None"""
    load_dotenv()
    
    url = "https://jsearch.p.rapidapi.com/search"
    
    # Test avec différents paramètres
    test_cases = [
        {"query": "python developer", "location": "remote"},
        {"query": "python", "location": ""},
        {"query": "javascript", "location": None},
    ]
    
    headers = {
        'X-RapidAPI-Key': os.getenv('JSEARCH_RAPIDAPI_KEY'),
        'X-RapidAPI-Host': 'jsearch.p.rapidapi.com'
    }
    
    print("🔍 Test JSearch API avec correction NoneType...")
    
    async with aiohttp.ClientSession() as session:
        for i, test_case in enumerate(test_cases, 1):
            query = test_case["query"]
            location = test_case["location"]
            
            # Safe string handling like in our fix
            safe_query = query or ""
            safe_location = location or ""
            search_query = f"{safe_query} {safe_location}".strip()
            
            params = {
                "query": search_query,
                "page": "1",
                "num_pages": "1"
            }
            
            print(f"\n   Test {i}: query='{query}', location='{location}'")
            print(f"   Search string: '{search_query}'")
            
            try:
                async with session.get(url, headers=headers, params=params) as response:
                    print(f"   Status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        jobs_count = len(data.get('data', []))
                        print(f"   Jobs found: {jobs_count}")
                        
                        # Test safe data extraction
                        if jobs_count > 0:
                            job = data['data'][0]
                            
                            # Test the problematic fields
                            job_city = job.get('job_city') or ''
                            job_country = job.get('job_country') or ''
                            location_str = f"{job_city}, {job_country}".strip(', ')
                            
                            description = job.get('job_description') or ''
                            description_preview = (description[:100] + '...') if description else ''
                            
                            print(f"   ✅ Location extraction: '{location_str}'")
                            print(f"   ✅ Description extraction: {len(description_preview)} chars")
                    else:
                        text = await response.text()
                        print(f"   Error: {text[:200]}")
            except Exception as e:
                print(f"   Exception: {e}")
async def test_service_jsearch():
    """Test via notre service"""
    print("\n🔍 Test via service multi-API...")
    
    from services.multi_job_api_service import get_job_service
    
    service = get_job_service()
    
    try:
        async with aiohttp.ClientSession() as session:
            jobs = await service._search_jsearch(session, "python", "remote")
            print(f"   ✅ Service JSearch: {len(jobs)} jobs")
            if jobs:
                job = jobs[0]
                print(f"   📋 Exemple: '{job.title}' chez '{job.company}'")
                print(f"   📍 Location: '{job.location}'")
    except Exception as e:
        print(f"   ❌ Service JSearch erreur: {e}")
async def main():
    await test_jsearch_direct()
    await test_service_jsearch()
if __name__ == "__main__":
    asyncio.run(main())