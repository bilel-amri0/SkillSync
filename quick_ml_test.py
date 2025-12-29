"""
Quick test of ML Career Guidance System
Simple version without rich formatting
"""
import requests
import json
from datetime import datetime

# Sample CV
SAMPLE_CV = """
John Doe - Software Engineer
Email: john@example.com | Phone: 555-1234

EXPERIENCE
Senior Python Developer at TechCorp (2021-Present)
- Built ML models with TensorFlow and PyTorch
- Deployed applications using Docker and Kubernetes
- Worked with AWS, PostgreSQL, and Redis

SKILLS
Python, JavaScript, TensorFlow, PyTorch, Docker, Kubernetes, AWS, React, SQL

EDUCATION
BS Computer Science, 2020
"""

def test():
    print("=" * 80)
    print("🚀 ML CAREER GUIDANCE - QUICK TEST")
    print("=" * 80)
    
    url = "http://localhost:8001/api/v1/career-guidance"
    
    print("\n📤 Sending request to API...")
    start = datetime.now()
    
    try:
        response = requests.post(
            url,
            json={"cv_content": SAMPLE_CV},
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        elapsed = (datetime.now() - start).total_seconds()
        
        if response.status_code == 200:
            print(f"✅ Success! ({elapsed:.2f}s)")
            
            result = response.json()
            
            # Save
            with open('quick_ml_result.json', 'w') as f:
                json.dump(result, f, indent=2)
            print("💾 Saved to quick_ml_result.json")
            
            # Summary
            print("\n" + "=" * 80)
            print("📊 RESULTS SUMMARY")
            print("=" * 80)
            
            metadata = result.get('metadata', {})
            print(f"\n🤖 ML Model: {metadata.get('ml_model', 'N/A')}")
            print(f"⚡ Processing Time: {metadata.get('processing_time_seconds', 0)}s")
            print(f"📝 CV Skills Found: {metadata.get('cv_skills_count', 0)}")
            
            # Jobs
            jobs = result.get('job_recommendations', [])
            print(f"\n💼 JOB RECOMMENDATIONS: {len(jobs)}")
            for i, job in enumerate(jobs[:3], 1):
                print(f"\n  #{i} {job['title']}")
                print(f"      ML Similarity: {job['similarity_score']*100:.1f}%")
                print(f"      ML Confidence: {job['confidence']*100:.1f}%")
                salary = job['predicted_salary']
                print(f"      Predicted Salary: ${salary['min']:,} - ${salary['max']:,}")
                print(f"      Matching Skills: {', '.join(job['matching_skills'][:4])}")
            
            # Certs
            certs = result.get('certification_recommendations', [])
            print(f"\n🎓 CERTIFICATION RECOMMENDATIONS: {len(certs)}")
            for i, cert in enumerate(certs[:3], 1):
                print(f"\n  #{i} {cert['name']}")
                print(f"      ML Relevance: {cert['relevance_score']*100:.1f}%")
                print(f"      Predicted ROI: {cert['predicted_roi']}")
                print(f"      Time: {cert['estimated_time']}")
            
            # Roadmap
            roadmap = result.get('learning_roadmap', {})
            print(f"\n🎯 LEARNING ROADMAP:")
            print(f"   Total Duration: {roadmap.get('total_duration_weeks')} weeks")
            print(f"   ML Success Prediction: {roadmap.get('predicted_success_rate')}")
            print(f"   Personalization: {roadmap.get('personalization_score')}")
            print(f"   Strategy: {roadmap.get('learning_strategy')}")
            print(f"   Phases: {len(roadmap.get('phases', []))}")
            
            # XAI
            xai = result.get('xai_insights', {})
            print(f"\n🧠 EXPLAINABLE AI:")
            confidence = xai.get('ml_confidence_scores', {})
            for key, value in confidence.items():
                print(f"   • {key}: {value}")
            
            key_insights = xai.get('key_insights', [])
            print(f"\n💡 KEY INSIGHTS:")
            for insight in key_insights[:5]:
                print(f"   • {insight}")
            
            print("\n" + "=" * 80)
            print("✅ ML CAREER GUIDANCE TEST COMPLETE")
            print("=" * 80)
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("⏱️ Request timed out (>60s)")
        print("First request may take 20-30s for model loading")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("\n🤖 ML-Driven Career Guidance System - Quick Test")
    print("\nThis tests the 100% ML-powered career guidance:")
    print("✅ Semantic job matching")
    print("✅ ML salary predictions")
    print("✅ Intelligent cert ranking")
    print("✅ Optimized learning paths")
    print("✅ Complete explainability (XAI)\n")
    print("Make sure backend server is running on port 8001!\n")
    
    input("Press Enter to start test...")
    test()
