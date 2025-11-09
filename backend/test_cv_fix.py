#!/usr/bin/env python3
"""
Test rapide pour vérifier que l'erreur CV est corrigée
"""

import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8001"

def test_cv_endpoints():
    """Test CV upload and text analysis endpoints"""
    print("🧪 Test rapide de correction CV...")
    print("=" * 50)
    
    # Test 1: CV Text Analysis
    print("\n📝 Test analyse de texte CV...")
    cv_text = """
    John Doe
    Software Developer
    
    Skills: Python, JavaScript, React, Node.js, SQL, Docker
    Experience: 3 years as Full Stack Developer
    Education: Computer Science Degree
    """
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/analyze-cv",
            json={"cv_content": cv_text},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS: Status {response.status_code}")
            print(f"   📊 Skills trouvées: {len(data['skills'])}")
            print(f"   🎯 Score: {data['confidence_score']}")
        else:
            print(f"   ❌ ERROR: Status {response.status_code}")
            print(f"   📄 Response: {response.text}")
            
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
    
    # Test 2: CV File Upload
    print("\n📄 Test upload de fichier CV...")
    
    # Create a simple text file
    test_cv_content = cv_text.encode('utf-8')
    
    try:
        files = {'file': ('test_cv.txt', test_cv_content, 'text/plain')}
        response = requests.post(
            f"{BASE_URL}/api/v1/upload-cv",
            files=files,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS: Status {response.status_code}")
            print(f"   📊 Skills trouvées: {len(data['skills'])}")
            print(f"   🎯 Score: {data['confidence_score']}")
        else:
            print(f"   ❌ ERROR: Status {response.status_code}")
            print(f"   📄 Response: {response.text}")
            
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")

if __name__ == "__main__":
    print("🔍 Vérification de l'accès au serveur...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Serveur accessible")
            test_cv_endpoints()
        else:
            print("❌ Serveur non accessible")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur connexion serveur: {e}")
        print("💡 Assurez-vous que le serveur fonctionne: python START_SERVER.py")
        sys.exit(1)
