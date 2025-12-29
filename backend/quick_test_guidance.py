"""
Quick Career Guidance Test (No user input required)
"""
import requests
import json

BASE_URL = "http://localhost:8001"

SAMPLE_CV = """
Bilel Amri
bilela329@gmail.com | github.com/Bilel-Amri | linkedin.com/in/amri-bilel-53092b283

Student engineer at Tek-Up, passionate about AI and machine learning.

SKILLS
Python, C, C++, JavaScript, HTML, CSS, React, FastAPI, Django, Bootstrap
Machine Learning, Deep Learning, NLP, Computer Vision, TensorFlow, PyTorch
Docker, Git, GitHub, Linux, MATLAB, Problem Solving

PROJECTS
- AI Chatbot: NLP-powered chatbot with TensorFlow and FastAPI
- Sports Analytics: React dashboard with ML predictions

LANGUAGES: French, English, Arabic
"""

print(" Testing Career Guidance Engine\n")

try:
    response = requests.post(
        f"{BASE_URL}/api/v1/career-guidance",
        json={"cv_content": SAMPLE_CV},
        timeout=60
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        print("\n SUCCESS!\n")
        print(f" Jobs Recommended: {len(result['recommended_jobs'])}")
        for job in result['recommended_jobs'][:3]:
            print(f"    {job['title']}: {job['match_score']}")
        
        print(f"\n Certifications: {len(result['recommended_certifications'])}")
        for cert in result['recommended_certifications'][:3]:
            print(f"    {cert['name']} ({cert['priority']} priority)")
        
        print(f"\n  Roadmap Phases: {len(result['learning_roadmap']['phases'])}")
        print(f"   Timeline: {result['learning_roadmap']['timeline']}")
        
        print(f"\n XAI Insights Available: ")
        print(f"   Top Match: {result['xai_insights']['job_matching_logic']['top_match']['job']}")
        
        # Save
        with open('career_guidance_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n Full result saved to: career_guidance_result.json")
        
    else:
        print(f"\n Error: {response.text}")
        
except Exception as e:
    print(f"\n Failed: {e}")
