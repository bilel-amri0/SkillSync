#!/usr/bin/env python3
"""
Test script for job APIs to verify they're working correctly
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.services.multi_job_api_service import get_job_service

async def test_apis():
    """Test all configured job APIs"""
    service = get_job_service()
    
    print("🔍 Testing Job APIs...")
    print("=" * 50)
    
    # Test with a common tech query
    test_query = "Python"
    test_location = "fr"
    
    print(f"🎯 Searching for: '{test_query}' in '{test_location}'")
    print()
    
    # Get API status first
    status = await service.get_api_status()
    print("📊 API Status:")
    for api_name, api_status in status.items():
        status_icon = "✅" if api_status['enabled'] else "❌"
        print(f"  {status_icon} {api_name}: {api_status['status']} (priority: {api_status['priority']})")
    print()
    
    # Run the search
    print("🚀 Starting job search...")
    jobs = await service.search_jobs(test_query, test_location)
    
    print(f"🎉 Total jobs found: {len(jobs)}")
    print()
    
    if jobs:
        print("📋 Sample jobs:")
        for i, job in enumerate(jobs[:5]):  # Show first 5 jobs
            print(f"  {i+1}. {job.title} at {job.company}")
            print(f"     📍 {job.location} | 🏢 {job.source}")
            if job.salary:
                print(f"     💰 {job.salary}")
            print(f"     🔗 {job.url}")
            print()
    else:
        print("❌ No jobs found! Check your API configuration.")
        print()
        print("🔧 Troubleshooting tips:")
        print("  1. Check that at least one API is enabled")
        print("  2. Verify API keys are set correctly in environment variables")
        print("  3. Check internet connection")
        print("  4. Try a different search query")

if __name__ == "__main__":
    try:
        asyncio.run(test_apis())
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n🚨 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
