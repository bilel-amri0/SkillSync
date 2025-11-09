#!/usr/bin/env python3
"""
🧪 TEST BACKEND ML HYBRIDE
"""

from ml_backend_hybrid import get_ml_backend

def test_hybrid_backend():
    print("🧪 Test Backend ML Hybride")
    print("="*50)
    
    # Initialisation
    backend = get_ml_backend()
    
    # Status système
    status = backend.get_system_status()
    print("📊 Status système:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Test scoring compétences
    print("\n🎯 Test scoring compétences:")
    job_skills = ["Python", "Machine Learning", "Data Science"]
    user_skills = ["Python", "Data Analysis", "SQL"]
    
    score = backend.score_skill_match(job_skills, user_skills)
    print(f"   Score correspondance: {score:.3f}")
    
    # Test analyse sentiment
    print("\n😊 Test analyse sentiment:")
    job_desc = "Great opportunity to work with cutting-edge technology in a dynamic team!"
    sentiment = backend.analyze_job_sentiment(job_desc)
    print(f"   Sentiment: {sentiment}")
    
    # Test recommandations
    print("\n🎯 Test recommandations:")
    user_profile = {
        "skills": ["Python", "Data Science", "Machine Learning"]
    }
    
    jobs_data = [
        {
            "id": 1,
            "title": "Data Scientist",
            "required_skills": ["Python", "Machine Learning", "Statistics"],
            "description": "Exciting role in AI development"
        },
        {
            "id": 2, 
            "title": "Web Developer",
            "required_skills": ["JavaScript", "React", "Node.js"],
            "description": "Build amazing web applications"
        }
    ]
    
    recommendations = backend.get_recommendations(user_profile, jobs_data)
    
    for rec in recommendations:
        print(f"   📋 {rec['title']}: Score {rec['score']:.3f}")
    
    print("\n🎉 Tests terminés avec succès!")

if __name__ == "__main__":
    test_hybrid_backend()