"""
Complete Frontend-Backend Integration Test
Tests the full flow: Upload CV  Extract Text  Advanced ML Analysis
"""
import requests
import json

BASE_URL = "http://localhost:8001"

def test_text_extraction():
    """Test /api/v1/extract-text endpoint"""
    print("=" * 70)
    print("TEST 1: Text Extraction from TXT file")
    print("=" * 70)
    
    # Create sample CV text
    cv_text = """John Doe
Senior Software Engineer
john.doe@example.com | +1-555-0123
GitHub: github.com/johndoe | LinkedIn: linkedin.com/in/johndoe

SUMMARY
Experienced software engineer with 8+ years in full-stack development, specializing in
cloud-native applications, microservices architecture, and machine learning systems.

WORK EXPERIENCE
Senior Software Engineer | Tech Corp | 2021 - Present
- Led team of 5 engineers in developing cloud-based microservices
- Implemented CI/CD pipelines reducing deployment time by 60%
- Technologies: Python, React, AWS, Docker, Kubernetes

Software Engineer | StartupXYZ | 2018 - 2021
- Developed RESTful APIs serving 100K+ daily users
- Built real-time dashboard using React and WebSockets
- Technologies: Django, PostgreSQL, Redis

EDUCATION
Master of Science in Computer Science | MIT | 2016
AWS Certified Solutions Architect (2022)

PROJECTS
- OpenAI Chat Application: Built chatbot using GPT-4 API, served 10K users
- E-commerce Platform: Full-stack marketplace with payment integration
"""
    
    # Create a file-like object
    files = {'file': ('test_cv.txt', cv_text.encode(), 'text/plain')}
    
    try:
        response = requests.post(f"{BASE_URL}/api/v1/extract-text", files=files)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"    Text extracted: {data['length']} characters")
            print(f"   Preview: {data['cv_text'][:200]}...")
            return data['cv_text']
        else:
            print(f"    Failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"    Exception: {e}")
        return None


def test_advanced_ml_analysis(cv_text=None):
    """Test /api/v1/analyze-cv-advanced endpoint"""
    print("\n" + "=" * 70)
    print("TEST 2: Advanced ML Analysis")
    print("=" * 70)
    
    if not cv_text:
        cv_text = """John Doe
Senior Software Engineer
john.doe@example.com
Python, React, AWS, Docker, Kubernetes
"""
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/analyze-cv-advanced",
            json={"cv_content": cv_text}
        )
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"    Analysis completed!")
            print(f"    Skills: {len(data['skills'])} detected")
            print(f"      {', '.join(data['skills'][:10])}")
            
            if data.get('seniority_level'):
                print(f"    Seniority: {data['seniority_level']}")
            
            if data.get('industries'):
                print(f"    Industries: {len(data['industries'])}")
                for industry, conf in data['industries'][:3]:
                    print(f"       {industry}: {conf*100:.1f}%")
            
            if data.get('projects'):
                print(f"    Projects: {len(data['projects'])} detected")
                for proj in data['projects'][:2]:
                    print(f"       {proj.get('name', 'Unnamed')}")
            
            if data.get('portfolio_links'):
                links = data['portfolio_links']
                if links.get('github'):
                    print(f"    GitHub: {links['github']}")
                if links.get('linkedin'):
                    print(f"    LinkedIn: {links['linkedin']}")
            
            print(f"     Processing: {data.get('processing_time_ms', 0)}ms")
            print(f"    Confidence: {data['confidence_score']*100:.1f}%")
            
            return data
        else:
            print(f"    Failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"    Exception: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_health():
    """Test backend health"""
    print("=" * 70)
    print("TEST 0: Backend Health Check")
    print("=" * 70)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"    Backend healthy: {data['service']}")
            print(f"   Features: {', '.join([k for k, v in data.get('features', {}).items() if v])}")
            return True
        else:
            print(f"    Backend unhealthy")
            return False
            
    except Exception as e:
        print(f"    Backend not responding: {e}")
        return False


def main():
    print("\n" + "" * 35)
    print("   COMPLETE FRONTEND-BACKEND INTEGRATION TEST")
    print("" * 35 + "\n")
    
    # Test 0: Health check
    if not test_health():
        print("\n Backend is not running. Start it with:")
        print("   cd backend && python -m uvicorn main_simple_for_frontend:app --reload --port 8001")
        return
    
    # Test 1: Text extraction
    cv_text = test_text_extraction()
    
    # Test 2: Advanced ML analysis
    if cv_text:
        result = test_advanced_ml_analysis(cv_text)
    else:
        print("\n  Skipping ML analysis (text extraction failed)")
        result = test_advanced_ml_analysis()  # Try with minimal CV
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if result:
        print(" All tests passed!")
        print("\n Next step: Test in frontend")
        print("   1. Open http://localhost:5173")
        print("   2. Go to CV Analysis page")
        print("   3. Toggle 'Advanced ML' ON")
        print("   4. Upload a CV file (PDF or TXT)")
        print("   5. Verify all ML features display:")
        print("       Seniority level card")
        print("       Industry classification")
        print("       Detected projects")
        print("       Portfolio links")
    else:
        print(" Some tests failed")
        print("\n Troubleshooting:")
        print("   1. Check backend logs for errors")
        print("   2. Verify ML models are loaded")
        print("   3. Check if production_cv_parser_final.py has issues")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
