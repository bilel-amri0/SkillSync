#!/usr/bin/env python3
"""
🔧 Test des corrections LinkedIn et FindWork
"""

import asyncio
import os
from dotenv import load_dotenv
from services.multi_job_api_service import get_job_service

async def test_fixes():
    """Test spécifique pour LinkedIn et FindWork"""
    print("🔧 Test des corrections API...")
    print("=" * 50)
    
    # Load environment
    load_dotenv()
    
    # Initialize service
    service = get_job_service()
    
    # Test LinkedIn spécifiquement
    print("\n🔍 Test LinkedIn (correction du parsing)...")
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            linkedin_jobs = await service._search_linkedin(session, "python developer", "remote")
            print(f"   ✅ LinkedIn: {len(linkedin_jobs)} jobs trouvés")
            if linkedin_jobs:
                job = linkedin_jobs[0]
                print(f"   📋 Exemple: '{job.title}' chez '{job.company}'")
    except Exception as e:
        print(f"   ❌ LinkedIn erreur: {e}")
    
    # Test FindWork spécifiquement
    print("\n🔍 Test FindWork (correction des paramètres)...")
    try:
        async with aiohttp.ClientSession() as session:
            findwork_jobs = await service._search_findwork(session, "python developer", "")
            print(f"   ✅ FindWork: {len(findwork_jobs)} jobs trouvés")
            if findwork_jobs:
                job = findwork_jobs[0]
                print(f"   📋 Exemple: '{job.title}' chez '{job.company}'")
    except Exception as e:
        print(f"   ❌ FindWork erreur: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Test terminé !")

if __name__ == "__main__":
    asyncio.run(test_fixes())
