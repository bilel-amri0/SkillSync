#!/usr/bin/env python3
"""
SkillSync API Testing Script
Tests all configured job APIs individually and provides detailed status report
"""

import asyncio
import aiohttp
import os
import sys
from datetime import datetime
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent))

from services.multi_job_api_service import get_job_service

class APITester:
    """Test suite for all job APIs"""
    
    def __init__(self):
        self.job_service = get_job_service()
        self.test_query = "software developer"
        self.test_location = "New York"
        
    async def run_all_tests(self):
        """Run comprehensive tests on all APIs"""
        print("\n🚀 SkillSync Multi-API Test Suite")
        print("=" * 50)
        print(f"Test Query: '{self.test_query}'")
        print(f"Test Location: '{self.test_location}'")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print("=" * 50)
        
        # Get API status
        api_status = await self.job_service.get_api_status()
        
        # Test each API individually
        results = {}
        total_enabled = 0
        total_working = 0
        
        for api_name, config in api_status.items():
            if config['enabled']:
                total_enabled += 1
                print(f"\n🔍 Testing {api_name.upper()}...")
                result = await self._test_single_api(api_name)
                results[api_name] = result
                
                if result['success']:
                    total_working += 1
                    print(f"   ✅ SUCCESS: {result['job_count']} jobs found")
                    if result['sample_job']:
                        sample = result['sample_job']
                        print(f"   💼 Sample: '{sample['title']}' at {sample['company']}")
                else:
                    print(f"   ❌ FAILED: {result['error']}")
            else:
                print(f"\n⏸️  {api_name.upper()}: DISABLED (missing API keys)")
                results[api_name] = {'success': False, 'error': 'Not configured', 'job_count': 0}
        
        # Test full multi-API search
        print(f"\n🔍 Testing FULL MULTI-API SEARCH...")
        full_search_result = await self._test_full_search()
        
        # Generate final report
        self._generate_report(results, total_enabled, total_working, full_search_result)
        
    async def _test_single_api(self, api_name: str) -> dict:
        """Test a single API"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                jobs = await self.job_service._search_single_api(
                    session, api_name, self.test_query, self.test_location, []
                )
                
                return {
                    'success': True,
                    'job_count': len(jobs),
                    'sample_job': {
                        'title': jobs[0].title,
                        'company': jobs[0].company,
                        'location': jobs[0].location
                    } if jobs else None,
                    'error': None
                }
                
        except Exception as e:
            return {
                'success': False,
                'job_count': 0,
                'sample_job': None,
                'error': str(e)
            }
    
    async def _test_full_search(self) -> dict:
        """Test the full multi-API search"""
        try:
            jobs = await self.job_service.search_jobs(
                self.test_query, self.test_location, ["Python", "JavaScript"]
            )
            
            # Count jobs by source
            source_counts = {}
            for job in jobs:
                source_counts[job.source] = source_counts.get(job.source, 0) + 1
            
            return {
                'success': True,
                'total_jobs': len(jobs),
                'source_breakdown': source_counts,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'total_jobs': 0,
                'source_breakdown': {},
                'error': str(e)
            }
    
    def _generate_report(self, results: dict, total_enabled: int, total_working: int, full_search: dict):
        """Generate comprehensive test report"""
        print("\n" + "=" * 50)
        print("📈 FINAL TEST REPORT")
        print("=" * 50)
        
        # API Status Summary
        print(f"📊 API Status Summary:")
        print(f"   Total APIs Configured: {total_enabled}")
        print(f"   Working APIs: {total_working}")
        print(f"   Success Rate: {(total_working/total_enabled*100):.1f}%" if total_enabled > 0 else "   Success Rate: 0%")
        
        # Individual API Results
        print(f"\n🔍 Individual API Results:")
        for api_name, result in results.items():
            status_icon = "✅" if result['success'] else "❌"
            job_count = result['job_count']
            print(f"   {status_icon} {api_name.upper():<12} | Jobs: {job_count:<3} | Status: {'OK' if result['success'] else 'FAILED'}")
            if not result['success'] and result['error'] != 'Not configured':
                print(f"      Error: {result['error']}")
        
        # Full Search Results
        print(f"\n🌐 Full Multi-API Search:")
        if full_search['success']:
            print(f"   ✅ SUCCESS: {full_search['total_jobs']} total jobs found")
            print(f"   📉 Source Breakdown:")
            for source, count in full_search['source_breakdown'].items():
                print(f"      • {source}: {count} jobs")
        else:
            print(f"   ❌ FAILED: {full_search['error']}")
        
        # Configuration Check
        print(f"\n⚙️ Configuration Status:")
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            print(f"   ✅ .env file found at: {env_file}")
        else:
            print(f"   ❌ .env file not found!")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if total_working == 0:
            print("   ⚠️  No APIs are working. Check your API keys in .env file.")
            print("   🔑 Run: python quick_setup_apis.py to reconfigure")
        elif total_working < total_enabled:
            print(f"   ⚠️  {total_enabled - total_working} APIs are not working. Check their configuration.")
        else:
            print("   🎉 All configured APIs are working perfectly!")
            print("   🚀 Your SkillSync system is ready for production!")
        
        print("\n" + "=" * 50)
        print("🏁 Test completed successfully!")
        print("=" * 50)

async def main():
    """Main test runner"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    tester = APITester()
    await tester.run_all_tests()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)