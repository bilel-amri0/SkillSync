"""
SkillSync E2E Test Script
Tests the complete CV upload and portfolio generation flow
"""

import requests
import json
import time
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8001"
FRONTEND_URL = "http://localhost:5173"

# Test CV content
TEST_CV_CONTENT = """
John Doe
Senior Full Stack Developer
john.doe@email.com | +1-555-0123 | San Francisco, CA

PROFESSIONAL SUMMARY
Experienced Full Stack Developer with 8+ years building scalable web applications.
Expert in React, Node.js, Python, and cloud technologies.

TECHNICAL SKILLS
Frontend: React, TypeScript, Next.js, Vue.js, HTML5, CSS3, Tailwind CSS
Backend: Node.js, Python, FastAPI, Express, Django, RESTful APIs
Database: PostgreSQL, MongoDB, Redis, MySQL
Cloud & DevOps: AWS, Docker, Kubernetes, CI/CD, GitHub Actions
Tools: Git, Jest, Pytest, Webpack, Vite

WORK EXPERIENCE

Senior Full Stack Developer | TechCorp Inc. | 2020-Present
- Led development of enterprise SaaS platform serving 500K+ users
- Architected microservices infrastructure using Node.js and Docker
- Implemented real-time features with WebSockets and Redis
- Reduced API response time by 60% through optimization
- Mentored team of 5 junior developers

Full Stack Developer | StartupXYZ | 2018-2020
- Built e-commerce platform from scratch using MERN stack
- Integrated payment systems (Stripe, PayPal) and third-party APIs
- Implemented automated testing achieving 85% code coverage
- Designed and deployed AWS infrastructure with Terraform
- Collaborated with UX team to improve conversion rates by 40%

Software Engineer | Digital Agency | 2015-2018
- Developed custom WordPress plugins and React applications
- Created RESTful APIs for mobile and web clients
- Implemented responsive designs for 50+ client websites
- Optimized database queries reducing load times by 50%

EDUCATION
Bachelor of Science in Computer Science
University of California, Berkeley | 2011-2015
GPA: 3.8/4.0

CERTIFICATIONS
- AWS Certified Solutions Architect - Associate
- MongoDB Certified Developer
- Google Cloud Professional Cloud Architect

PROJECTS
- Open Source Contributor to React and Node.js projects
- Built developer tools used by 10K+ developers monthly
- Published NPM packages with 50K+ downloads
"""

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_health_check():
    """Test 1: Health check endpoint"""
    print_section("TEST 1: Health Check")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        print(f"✅ Backend is running")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Backend health check failed: {e}")
        return False

def test_analytics_endpoint():
    """Test 2: Analytics dashboard endpoint"""
    print_section("TEST 2: Analytics Dashboard")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/analytics/dashboard", timeout=5)
        print(f"✅ Analytics endpoint working")
        print(f"   Status: {response.status_code}")
        data = response.json()
        print(f"   CVs Analyzed: {data['data']['overview']['total_cvs']}")
        print(f"   Jobs Analyzed: {data['data']['overview']['jobs_analyzed']}")
        return True
    except Exception as e:
        print(f"❌ Analytics endpoint failed: {e}")
        return False

def test_cv_upload():
    """Test 3: CV upload and analysis"""
    print_section("TEST 3: CV Upload & Analysis")
    try:
        # Create temporary CV file
        cv_file_path = Path("test_cv_upload.txt")
        cv_file_path.write_text(TEST_CV_CONTENT, encoding='utf-8')
        
        print("📄 Uploading CV...")
        with open(cv_file_path, 'rb') as f:
            files = {'file': ('test_cv.txt', f, 'text/plain')}
            data = {'extract_skills': 'true'}
            response = requests.post(
                f"{API_BASE_URL}/api/v1/upload-cv",
                files=files,
                data=data,
                timeout=30
            )
        
        # Clean up
        cv_file_path.unlink()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ CV analysis successful")
            print(f"   Analysis ID: {result['analysis_id']}")
            print(f"   Name: {result.get('personal_info', {}).get('name', 'N/A')}")
            print(f"   Skills Found: {len(result.get('skills', []))}")
            print(f"   Job Titles: {len(result.get('job_titles', []))}")
            print(f"   Education: {len(result.get('education', []))}")
            print(f"   Confidence: {result.get('confidence_score', 0):.2f}")
            
            # Print first 5 skills
            print(f"\n   Top Skills Detected:")
            for skill in result.get('skills', [])[:5]:
                print(f"     • {skill}")
            
            return result
        else:
            print(f"❌ CV upload failed")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ CV upload test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_job_search(cv_data):
    """Test 4: Job matching with CV skills"""
    print_section("TEST 4: Job Matching")
    try:
        if not cv_data:
            print("⚠️  Skipping (no CV data)")
            return False
        
        # Extract skills from CV
        skills = cv_data.get('skills', [])[:5]
        
        print(f"🔍 Searching jobs for skills: {', '.join(skills[:3])}...")
        payload = {
            "query": "Full Stack Developer",
            "location": "Remote",
            "skills": skills,
            "max_results": 10
        }
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/jobs/search",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            total_jobs = result.get('total_results', 0)
            jobs = result.get('jobs', [])
            
            print(f"✅ Job search successful")
            print(f"   Total Results: {total_jobs}")
            print(f"   Jobs Retrieved: {len(jobs)}")
            
            if jobs:
                print(f"\n   Sample Job Matches:")
                for job in jobs[:3]:
                    print(f"     • {job.get('title', 'N/A')} at {job.get('company', 'N/A')}")
                    print(f"       Location: {job.get('location', 'N/A')}")
            
            return True
        else:
            print(f"⚠️  Job search returned status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Job search test failed: {e}")
        return False

def test_frontend_access():
    """Test 5: Frontend accessibility"""
    print_section("TEST 5: Frontend Access")
    try:
        response = requests.get(FRONTEND_URL, timeout=5)
        if response.status_code == 200:
            print(f"✅ Frontend is accessible")
            print(f"   URL: {FRONTEND_URL}")
            print(f"   Status: {response.status_code}")
            return True
        else:
            print(f"⚠️  Frontend returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Frontend access failed: {e}")
        return False

def run_all_tests():
    """Run all tests sequentially"""
    print("\n" + "="*60)
    print("  SKILLSYNC E2E TEST SUITE")
    print("  Testing complete application flow")
    print("="*60)
    
    results = {}
    cv_data = None
    
    # Test 1: Health Check
    results['health'] = test_health_check()
    time.sleep(1)
    
    # Test 2: Analytics
    results['analytics'] = test_analytics_endpoint()
    time.sleep(1)
    
    # Test 3: CV Upload
    cv_data = test_cv_upload()
    results['cv_upload'] = cv_data is not None
    time.sleep(1)
    
    # Test 4: Job Search
    results['job_search'] = test_job_search(cv_data)
    time.sleep(1)
    
    # Test 5: Frontend
    results['frontend'] = test_frontend_access()
    
    # Summary
    print_section("TEST SUMMARY")
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"Total Tests: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"\nSuccess Rate: {(passed/total)*100:.1f}%\n")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name.replace('_', ' ').title()}")
    
    print("\n" + "="*60)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Application is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    run_all_tests()
