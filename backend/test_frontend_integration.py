#!/usr/bin/env python3
"""
🔧 Test d'intégration frontend - Backend SkillSync
"""

import requests
import json
import time

def test_post_job_search():
    """Test POST /api/v1/jobs/search (utilisé par le frontend)"""
    url = "http://127.0.0.1:8001/api/v1/jobs/search"
    
    # Test comme le fait le frontend
    payload = {
        "query": "Python Developer",
        "location": "remote",
        "skills": ["Python", "Django", "React"],
        "max_results": 10
    }
    
    print("🔍 Test POST /api/v1/jobs/search...")
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS: {data['total_count']} jobs found")
            print(f"   📊 Sources: {', '.join(data['sources_used'])}")
            print(f"   ⏱️ Time: {data['search_time_ms']}ms")
            return True
        else:
            print(f"   ❌ ERROR: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
        return False

def test_cv_upload():
    """Test POST /api/v1/upload-cv"""
    url = "http://127.0.0.1:8001/api/v1/upload-cv"
    
    # Créer un CV de test
    cv_content = """
John Doe
Senior Python Developer

EXPERIENCE:
- 5 years Python development
- 3 years Django and Flask
- 2 years React and JavaScript
- Machine Learning with TensorFlow
- AWS and Docker experience

SKILLS:
Python, Django, Flask, React, JavaScript, SQL, PostgreSQL, 
Git, Docker, Kubernetes, AWS, Machine Learning, TensorFlow

EDUCATION:
Master's in Computer Science
    """
    
    print("\n📄 Test POST /api/v1/upload-cv...")
    try:
        # Test avec un fichier texte
        files = {
            'file': ('cv.txt', cv_content, 'text/plain')
        }
        
        response = requests.post(url, files=files, timeout=30)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS: CV analyzed")
            print(f"   🛠️ Skills found: {len(data['skills'])} - {', '.join(data['skills'][:5])}...")
            print(f"   📊 Experience: {data['experience_years']} years")
            print(f"   💼 Job titles: {data['job_titles']}")
            return True
        else:
            print(f"   ❌ ERROR: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
        return False

def test_cv_analyze_text():
    """Test POST /api/v1/analyze-cv"""
    url = "http://127.0.0.1:8001/api/v1/analyze-cv"
    
    cv_text = """
Jane Smith - Full Stack Developer

TECHNICAL SKILLS:
- Frontend: React, Vue.js, Angular, HTML, CSS, JavaScript
- Backend: Node.js, Python, Django, Flask
- Database: MongoDB, PostgreSQL, MySQL
- Cloud: AWS, Azure, Docker, Kubernetes
- Tools: Git, Jenkins, CI/CD

EXPERIENCE:
Senior Full Stack Developer (3 years)
Software Engineer (2 years)
Junior Developer (1 year)
    """
    
    payload = {
        "cv_content": cv_text,
        "format": "text"
    }
    
    print("\n📝 Test POST /api/v1/analyze-cv...")
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS: CV text analyzed")
            print(f"   🛠️ Skills: {', '.join(data['skills'])}")
            print(f"   📊 Experience: {data['experience_years']} years")
            print(f"   📋 Summary: {data['summary']}")
            return True
        else:
            print(f"   ❌ ERROR: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
        return False

def test_api_status():
    """Test GET /api/v1/jobs/status"""
    url = "http://127.0.0.1:8001/api/v1/jobs/status"
    
    print("\n📊 Test GET /api/v1/jobs/status...")
    try:
        response = requests.get(url, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ SUCCESS: {data['total_enabled']} APIs enabled")
            print(f"   🔧 System status: {data['system_status']}")
            return True
        else:
            print(f"   ❌ ERROR: {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ EXCEPTION: {e}")
        return False

def main():
    print("🧪 Test d'intégration Frontend-Backend SkillSync")
    print("=" * 60)
    
    # Test de connexion de base
    try:
        response = requests.get("http://127.0.0.1:8001/health", timeout=5)
        if response.status_code != 200:
            print("❌ Serveur non accessible ! Démarrez avec: python START_SERVER.py")
            return
        print("✅ Serveur accessible")
    except:
        print("❌ Serveur non accessible ! Démarrez avec: python START_SERVER.py")
        return
    
    # Tests des endpoints
    tests = [
        ("Status API", test_api_status),
        ("Job Search POST", test_post_job_search),
        ("CV Upload", test_cv_upload),
        ("CV Text Analysis", test_cv_analyze_text)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name}...")
        result = test_func()
        results.append((test_name, result))
        time.sleep(1)  # Pause entre les tests
    
    # Résumé des résultats
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DES TESTS:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Résultat: {passed}/{len(results)} tests réussis")
    
    if passed == len(results):
        print("\n🎉 TOUS LES TESTS PASSÉS !")
        print("🚀 Votre backend est 100% compatible avec le frontend !")
    else:
        print("\n⚠️ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")

if __name__ == "__main__":
    main()
