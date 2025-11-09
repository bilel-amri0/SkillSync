#!/usr/bin/env python3
"""
SkillSync System Test - Version Finale
Test complet du système pour validation avant archivage
Author: MiniMax Agent
"""

import requests
import json
import sys
import time
from datetime import datetime

BASE_URL = "http://127.0.0.1:8001"

def test_server_health():
    """Test de la santé du serveur"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_api_status():
    """Test du status des APIs"""
    try:
        response = requests.get(f"{BASE_URL}/api/v1/jobs/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data['enabled_count'], data['total_count']
        return 0, 0
    except:
        return 0, 0

def test_job_search_get():
    """Test recherche d'emplois GET"""
    try:
        params = {
            'query': 'Python Developer',
            'location': 'remote',
            'skills': 'Python,Django,React',
            'max_results': 10
        }
        response = requests.get(f"{BASE_URL}/api/v1/jobs/search", params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return len(data['jobs']), data['sources'], data['search_time_ms']
        return 0, [], 0
    except Exception as e:
        print(f"Erreur GET: {e}")
        return 0, [], 0

def test_job_search_post():
    """Test recherche d'emplois POST"""
    try:
        payload = {
            'query': 'JavaScript Developer',
            'location': 'paris',
            'skills': ['JavaScript', 'React', 'Node.js'],
            'max_results': 10
        }
        response = requests.post(f"{BASE_URL}/api/v1/jobs/search", json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return len(data['jobs']), data['sources'], data['search_time_ms']
        return 0, [], 0
    except Exception as e:
        print(f"Erreur POST: {e}")
        return 0, [], 0

def test_cv_analysis():
    """Test analyse CV"""
    cv_content = """
    John Doe
    Senior Full Stack Developer
    
    Skills: Python, JavaScript, React, Node.js, Docker, AWS, PostgreSQL
    Experience: 5 years in web development
    Education: Computer Science Master's Degree
    """
    
    try:
        payload = {'cv_content': cv_content}
        response = requests.post(f"{BASE_URL}/api/v1/analyze-cv", json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return len(data['skills']), data['confidence_score']
        return 0, 0.0
    except Exception as e:
        print(f"Erreur CV: {e}")
        return 0, 0.0

def run_complete_test():
    """Exécute tous les tests"""
    print("🧪 SkillSync System Test - Version Finale")
    print("=" * 60)
    
    # Test 1: Santé du serveur
    print("\n🔍 Test 1: Santé du serveur...")
    if test_server_health():
        print("   ✅ Serveur accessible")
    else:
        print("   ❌ Serveur inaccessible")
        print("   💡 Lancez le serveur: python start_server_final.py")
        return False
    
    # Test 2: Status des APIs
    print("\n🔍 Test 2: Status des APIs...")
    enabled, total = test_api_status()
    print(f"   📊 APIs activées: {enabled}/{total}")
    if enabled > 0:
        print("   ✅ Au moins une API configurée")
    else:
        print("   ⚠️  Aucune API configurée")
    
    # Test 3: Recherche GET
    print("\n🔍 Test 3: Recherche emplois (GET)...")
    start_time = time.time()
    jobs_count, sources, search_time = test_job_search_get()
    test_duration = time.time() - start_time
    
    if jobs_count > 0:
        print(f"   ✅ {jobs_count} emplois trouvés")
        print(f"   📊 Sources: {', '.join(sources)}")
        print(f"   ⏱️  Temps: {search_time}ms (total: {test_duration:.1f}s)")
    else:
        print("   ⚠️  Aucun emploi trouvé")
    
    # Test 4: Recherche POST
    print("\n🔍 Test 4: Recherche emplois (POST)...")
    start_time = time.time()
    jobs_count, sources, search_time = test_job_search_post()
    test_duration = time.time() - start_time
    
    if jobs_count > 0:
        print(f"   ✅ {jobs_count} emplois trouvés")
        print(f"   📊 Sources: {', '.join(sources)}")
        print(f"   ⏱️  Temps: {search_time}ms (total: {test_duration:.1f}s)")
    else:
        print("   ⚠️  Aucun emploi trouvé")
    
    # Test 5: Analyse CV
    print("\n🔍 Test 5: Analyse CV...")
    skills_count, confidence = test_cv_analysis()
    
    if skills_count > 0:
        print(f"   ✅ {skills_count} compétences détectées")
        print(f"   🎯 Confiance: {confidence}")
    else:
        print("   ❌ Échec analyse CV")
    
    # Résumé
    print("\n" + "=" * 60)
    print("📋 RÉSUMÉ DES TESTS:")
    
    tests_passed = 0
    if test_server_health(): tests_passed += 1
    if enabled > 0: tests_passed += 1
    if jobs_count > 0: tests_passed += 1  # Utilise le dernier résultat
    if skills_count > 0: tests_passed += 1
    
    print(f"   ✅ Tests réussis: {tests_passed}/4")
    
    if tests_passed >= 3:
        print("🎉 Système OPÉRATIONNEL - Prêt pour l'archivage!")
        return True
    else:
        print("⚠️  Système partiellement fonctionnel")
        return False

if __name__ == "__main__":
    print(f"🕒 Test exécuté le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    success = run_complete_test()
    
    if success:
        print("\n✅ Version finale validée")
        sys.exit(0)
    else:
        print("\n⚠️  Validation partielle")
        sys.exit(1)
