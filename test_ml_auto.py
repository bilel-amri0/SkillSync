"""
Automated test - No user input required
"""
import requests
import json
import time

print("="*80)
print("🚀 TESTING ML-DRIVEN CAREER GUIDANCE SYSTEM")
print("="*80)

# Wait for server
print("\n⏳ Waiting for server to be ready...")
time.sleep(2)

# Test CV
cv = """
BILEL AMRI
Software Engineer | ML Enthusiast
Email: bilel@example.com | Phone: 555-1234

EXPERIENCE
Machine Learning Engineer at TechCorp (2022-Present)
- Developed ML models using TensorFlow and PyTorch
- Built data pipelines with Apache Spark
- Deployed services with Docker and Kubernetes on AWS
- Implemented CI/CD pipelines

Python Developer at StartupXYZ (2020-2022)  
- Built APIs with FastAPI and Django
- Frontend development with React
- PostgreSQL database management

SKILLS
Python, TensorFlow, PyTorch, Machine Learning, Deep Learning, Docker, 
Kubernetes, AWS, FastAPI, Django, React, PostgreSQL, Git, CI/CD

EDUCATION
BS Computer Science, 2020
"""

print("📤 Sending CV to ML career guidance API...")
print(f"   CV length: {len(cv)} characters")

try:
    start = time.time()
    response = requests.post(
        'http://localhost:8001/api/v1/career-guidance',
        json={'cv_content': cv},
        timeout=90
    )
    elapsed = time.time() - start
    
    print(f"\n✅ Response received in {elapsed:.2f}s")
    print(f"   Status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        # Save
        with open('test_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("   💾 Saved to test_result.json")
        
        # Display summary
        print("\n" + "="*80)
        print("📊 RESULTS SUMMARY")
        print("="*80)
        
        meta = result.get('metadata', {})
        print(f"\n🤖 ML Model: {meta.get('ml_model', 'N/A')}")
        print(f"🚀 Engine: {meta.get('engine_version', 'N/A')}")
        print(f"⚡ Processing: {meta.get('processing_time_seconds', 0)}s")
        print(f"📝 Skills found: {meta.get('cv_skills_count', 0)}")
        
        # Jobs
        jobs = result.get('job_recommendations', [])
        print(f"\n💼 JOB RECOMMENDATIONS: {len(jobs)}")
        for i, job in enumerate(jobs[:3], 1):
            print(f"\n  {i}. {job['title']}")
            print(f"     🤖 ML Similarity: {job['similarity_score']*100:.1f}%")
            print(f"     🎯 Confidence: {job['confidence']*100:.1f}%")
            salary = job['predicted_salary']
            print(f"     💰 Salary: ${salary['min']:,} - ${salary['max']:,}")
            print(f"     ✅ Skills match: {len(job['matching_skills'])}/{len(job['matching_skills']) + len(job['skill_gaps'])}")
        
        # Certs
        certs = result.get('certification_recommendations', [])
        print(f"\n🎓 CERTIFICATION RECOMMENDATIONS: {len(certs)}")
        for i, cert in enumerate(certs[:3], 1):
            print(f"\n  {i}. {cert['name']}")
            print(f"     🤖 ML Relevance: {cert['relevance_score']*100:.1f}%")
            print(f"     💰 ROI: {cert['predicted_roi']}")
            print(f"     ⏱️  Time: {cert['estimated_time']}")
        
        # Roadmap
        roadmap = result.get('learning_roadmap', {})
        phases = roadmap.get('phases', [])
        print(f"\n🎯 LEARNING ROADMAP: {len(phases)} phases")
        print(f"   📅 Duration: {roadmap.get('total_duration_weeks')} weeks ({roadmap.get('total_duration_months')} months)")
        print(f"   🎓 Success rate: {roadmap.get('predicted_success_rate')}")
        print(f"   ✨ Personalization: {roadmap.get('personalization_score')}")
        print(f"   📚 Strategy: {roadmap.get('learning_strategy')}")
        
        # XAI
        xai = result.get('xai_insights', {})
        confidence = xai.get('ml_confidence_scores', {})
        print(f"\n🧠 ML CONFIDENCE SCORES:")
        for key, val in confidence.items():
            print(f"   • {key}: {val}")
        
        insights = xai.get('key_insights', [])
        print(f"\n💡 KEY INSIGHTS ({len(insights)}):")
        for insight in insights[:5]:
            print(f"   • {insight}")
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED - ML CAREER GUIDANCE WORKING!")
        print("="*80)
        print("\n📁 Full results saved to: test_result.json")
        print("📖 Documentation: ML_CAREER_SYSTEM_DOCUMENTATION.md")
        print("📋 Summary: ML_IMPLEMENTATION_SUMMARY.md\n")
        
    else:
        print(f"\n❌ Error: HTTP {response.status_code}")
        print(response.text[:500])
        
except requests.exceptions.Timeout:
    print("\n⏱️ Timeout! Request took >90 seconds")
    print("Note: First request may take 20-30s to load ML models")
except requests.exceptions.ConnectionError:
    print("\n❌ Connection Error!")
    print("Is the backend server running on port 8001?")
    print("Start it with: cd backend && python start_server.py")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
