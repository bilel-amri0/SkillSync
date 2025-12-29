"""
Test Both CV Analysis Endpoints
Tests standard and advanced ML endpoints side-by-side
"""

import requests
import json

# Test CV
cv_text = """
John Doe
Senior Software Engineer
john.doe@example.com | +1-555-0123
GitHub: github.com/johndoe | LinkedIn: linkedin.com/in/johndoe

Experienced software engineer with 8+ years in full-stack development, specializing in 
cloud-native applications, microservices architecture, and machine learning systems.

WORK EXPERIENCE

Senior Software Engineer | Tech Corp | 2021 - Present
- Led development of cloud-based platform serving 1M+ users
- Increased system performance by 45% through optimization
- Managed team of 5 junior engineers
- Implemented CI/CD pipeline reducing deployment time by 60%
- Technologies: Python, React, AWS, Docker, Kubernetes

Software Engineer | StartupXYZ | 2018 - 2021
- Developed RESTful APIs handling 100K+ daily requests
- Built machine learning models with 92% accuracy
- Reduced infrastructure costs by $50K annually
- Technologies: Python, Django, PostgreSQL, Redis

EDUCATION
Master of Science in Computer Science | MIT | 2016
AWS Certified Solutions Architect (2022)

PROJECTS
- OpenAI Chat Application: Built chatbot using GPT-4 API, served 10K users
"""

print("=" * 70)
print(" TESTING BOTH CV ANALYSIS ENDPOINTS")
print("=" * 70)

base_url = "http://localhost:8001"

# Test 1: Standard endpoint
print("\n1 Testing STANDARD endpoint (/api/v1/analyze-cv)...")
try:
    response1 = requests.post(
        f"{base_url}/api/v1/analyze-cv",
        json={"cv_content": cv_text},
        timeout=30
    )
    
    if response1.status_code == 200:
        result1 = response1.json()
        print(f"    Status: {response1.status_code}")
        print(f"    Skills found: {len(result1.get('skills', []))}")
        print(f"    Experience: {result1.get('experience_years', 0)} years")
        print(f"    Processing time: {result1.get('processing_time_ms', 0)}ms")
        print(f"    Confidence: {result1.get('confidence_score', 0):.2%}")
        print(f"    Top skills: {', '.join(result1.get('skills', [])[:5])}")
    else:
        print(f"    Failed: {response1.status_code}")
        print(f"   {response1.text}")
except Exception as e:
    print(f"    Error: {e}")

# Test 2: Advanced ML endpoint
print("\n2 Testing ADVANCED ML endpoint (/api/v1/analyze-cv-advanced)...")
try:
    response2 = requests.post(
        f"{base_url}/api/v1/analyze-cv-advanced",
        json={"cv_content": cv_text},
        timeout=60
    )
    
    if response2.status_code == 200:
        result2 = response2.json()
        print(f"    Status: {response2.status_code}")
        print(f"    Skills found: {len(result2.get('skills', []))}")
        print(f"    Seniority: {result2.get('seniority_level', 'Unknown')}")
        print(f"    Industries: {len(result2.get('industries', []))}")
        print(f"    Projects: {len(result2.get('projects', []))}")
        print(f"    Portfolio links: GitHub={result2.get('portfolio_links', {}).get('github', 'N/A')}")
        print(f"    Processing time: {result2.get('processing_time_ms', 0)}ms")
        print(f"    Confidence: {result2.get('confidence_score', 0):.2%}")
        print(f"    Top skills: {', '.join(result2.get('skills', [])[:5])}")
        industries_str = ', '.join([f"{i[0]} ({i[1]:.2f})" for i in result2.get('industries', [])[:3]])
        print(f"    Industries: {industries_str}")
    else:
        print(f"    Failed: {response2.status_code}")
        print(f"   {response2.text}")
except Exception as e:
    print(f"    Error: {e}")

print("\n" + "=" * 70)
print(" TESTING COMPLETE")
print("=" * 70)
print("\n Comparison:")
if 'result1' in locals() and 'result2' in locals():
    print(f"   Standard: {len(result1.get('skills', []))} skills, "
          f"{result1.get('processing_time_ms', 0)}ms, "
          f"{result1.get('confidence_score', 0):.2%} confidence")
    print(f"   Advanced: {len(result2.get('skills', []))} skills, "
          f"{result2.get('processing_time_ms', 0)}ms, "
          f"{result2.get('confidence_score', 0):.2%} confidence")
    print(f"   Advanced ML benefits:")
    print(f"      + {len(result2.get('industries', []))} industries classified")
    print(f"      + {len(result2.get('projects', []))} projects extracted")
    print(f"      + ML-predicted seniority: {result2.get('seniority_level', 'N/A')}")
    print(f"      + Portfolio links extracted")
else:
    print("   Could not compare - one or both tests failed")

print("\n Next step: Use the advanced endpoint in your frontend!")
print("   URL: POST http://localhost:8001/api/v1/analyze-cv-advanced")
