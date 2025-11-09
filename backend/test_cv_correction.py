#!/usr/bin/env python3
"""
Test rapide de la correction CV
"""

import requests
import json

BASE_URL = "http://127.0.0.1:8001"

def test_cv_correction():
    """Test de la correction CV"""
    
    print("🧪 Test Correction CV")
    print("=" * 30)
    
    # Vérifier serveur
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Serveur non accessible")
            return False
        print("✅ Serveur accessible")
    except:
        print("❌ Serveur non accessible")
        return False
    
    # Test analyse CV
    cv_content = """
    Jane Smith
    Full Stack Developer
    
    Technical Skills:
    - Python, Django, Flask
    - JavaScript, React, Vue.js
    - PostgreSQL, MongoDB
    - Docker, AWS, Git
    
    Experience:
    - 4 years as Software Engineer
    - 2 years as Frontend Developer
    
    Education:
    - Master in Computer Science
    """
    
    print("\n📝 Test POST /api/v1/analyze-cv...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/analyze-cv",
            json={"cv_content": cv_content},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS: Status 200")
            print(f"   📊 Skills: {len(data['skills'])} trouvées")
            print(f"   🎯 Score: {data['confidence_score']}")
            print(f"   💼 Titres: {len(data['job_titles'])} trouvés")
            print(f"   📅 Timestamp: {data['timestamp']}")
            
            # Afficher quelques compétences trouvées
            if data['skills']:
                print(f"   🔧 Exemples skills: {data['skills'][:3]}")
            
            return True
        else:
            print(f"   ❌ ERROR: Status {response.status_code}")
            print(f"   📄 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
        return False

if __name__ == "__main__":
    success = test_cv_correction()
    
    if success:
        print("\n🎉 Correction CV validée !")
    else:
        print("\n❌ Correction CV échouée")
