"""
Test script for Phase 1.5 API Layer
Tests CV analysis and portfolio generation endpoints.
"""

import requests
import json
from pathlib import Path

# API Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"


def test_health_check():
    """Test root endpoint."""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    assert response.json()["status"] == "SkillSync API is online"
    print("✅ Health check passed")


def test_cv_analysis_health():
    """Test CV analysis health endpoint."""
    print("\n" + "="*60)
    print("TEST 2: CV Analysis Health")
    print("="*60)
    
    response = requests.get(f"{API_BASE}/analyze/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200
    print("✅ CV Analysis health check passed")


def test_cv_upload_and_analyze():
    """Test CV upload and analysis."""
    print("\n" + "="*60)
    print("TEST 3: CV Upload and Analysis")
    print("="*60)
    
    # Create a sample CV file
    sample_cv_content = """
John Doe
Software Engineer
john.doe@email.com | +1-234-567-8900

PROFESSIONAL SUMMARY
Experienced software engineer with 5 years of experience in full-stack development.
Passionate about building scalable applications and solving complex problems.

SKILLS
- Programming Languages: Python, JavaScript, TypeScript, Java
- Web Frameworks: React, FastAPI, Django, Node.js
- Databases: PostgreSQL, MongoDB, Redis
- Cloud Platforms: AWS, Azure
- DevOps Tools: Docker, Kubernetes, CI/CD, Git

EXPERIENCE
Senior Software Engineer | Tech Corp | 2021 - Present
- Led development of microservices architecture using FastAPI and Docker
- Implemented CI/CD pipelines reducing deployment time by 60%
- Mentored junior developers and conducted code reviews

Software Engineer | StartupCo | 2019 - 2021
- Developed full-stack applications using React and Django
- Integrated third-party APIs and payment gateways
- Improved application performance by 40%

EDUCATION
Bachelor of Science in Computer Science
University of Technology | 2019
GPA: 3.8/4.0

CERTIFICATIONS
- AWS Certified Solutions Architect
- Docker Certified Associate
"""
    
    # Save to temporary file
    temp_file = Path("temp_cv.txt")
    temp_file.write_text(sample_cv_content)
    
    try:
        # Upload CV
        with open(temp_file, 'rb') as f:
            files = {'file': ('sample_cv.txt', f, 'text/plain')}
            response = requests.post(
                f"{API_BASE}/analyze",
                files=files,
                params={'extract_skills': True}
            )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ CV Analysis Successful!")
            print(f"Personal Info: {json.dumps(data['data']['personal_info'], indent=2)}")
            print(f"Skills Found: {len(data['data']['skills'])}")
            print(f"First 3 Skills: {data['data']['skills'][:3]}")
            print(f"Experience Entries: {len(data['data']['experience'])}")
            print(f"Education Entries: {len(data['data']['education'])}")
            
            # Return cv_data for portfolio test
            return data['data']
        else:
            print(f"❌ Error: {response.text}")
            return None
            
    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()


def test_portfolio_templates():
    """Test portfolio templates endpoint."""
    print("\n" + "="*60)
    print("TEST 4: List Portfolio Templates")
    print("="*60)
    
    response = requests.get(f"{API_BASE}/templates")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Available Templates: {list(data['templates'].keys())}")
        print("✅ Templates endpoint passed")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False


def test_portfolio_color_schemes():
    """Test portfolio color schemes endpoint."""
    print("\n" + "="*60)
    print("TEST 5: List Color Schemes")
    print("="*60)
    
    response = requests.get(f"{API_BASE}/color-schemes")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Available Color Schemes: {data['color_schemes']}")
        print("✅ Color schemes endpoint passed")
        return True
    else:
        print(f"❌ Error: {response.text}")
        return False


def test_portfolio_generation(cv_data):
    """Test portfolio generation."""
    print("\n" + "="*60)
    print("TEST 6: Portfolio Generation")
    print("="*60)
    
    if not cv_data:
        print("⚠️ Skipping - no CV data available")
        return
    
    # Prepare portfolio request
    portfolio_request = {
        "cv_data": cv_data,
        "template_id": "modern",
        "color_scheme": "blue",
        "include_photo": True,
        "include_projects": True,
        "include_contact_form": True,
        "dark_mode": False
    }
    
    print("Sending portfolio generation request...")
    response = requests.post(
        f"{API_BASE}/generate-portfolio",
        json=portfolio_request
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        # Save ZIP file
        output_file = Path("generated_portfolio.zip")
        output_file.write_bytes(response.content)
        
        print(f"✅ Portfolio Generated!")
        print(f"File Size: {len(response.content)} bytes ({len(response.content)/1024:.1f} KB)")
        print(f"Saved to: {output_file.absolute()}")
        print(f"Headers: {dict(response.headers)}")
        
        # Check ZIP is valid
        import zipfile
        if zipfile.is_zipfile(output_file):
            with zipfile.ZipFile(output_file, 'r') as zf:
                print(f"ZIP Contents: {zf.namelist()}")
                print("✅ Valid ZIP file")
        else:
            print("❌ Invalid ZIP file")
    else:
        print(f"❌ Error: {response.text}")


def main():
    """Run all tests."""
    print("\n" + "🚀"*30)
    print("PHASE 1.5 API LAYER TESTING")
    print("🚀"*30)
    
    try:
        # Test 1: Health check
        test_health_check()
        
        # Test 2: CV analysis health
        test_cv_analysis_health()
        
        # Test 3: CV upload and analysis
        cv_data = test_cv_upload_and_analyze()
        
        # Test 4: Portfolio templates
        test_portfolio_templates()
        
        # Test 5: Color schemes
        test_portfolio_color_schemes()
        
        # Test 6: Portfolio generation
        if cv_data:
            test_portfolio_generation(cv_data)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n⚠️ NOTE: Make sure the API server is running on http://localhost:8000")
    print("Run: python backend/main.py\n")
    
    input("Press Enter to start tests...")
    main()
