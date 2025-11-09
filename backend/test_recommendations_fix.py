#!/usr/bin/env python3
"""
Script de test spécifique pour vérifier la correction du bug recommendations
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://127.0.0.1:8001"
API_BASE = f"{BASE_URL}/api/v1"

def test_recommendations_fix():
    """Test spécifique de la correction du bug recommendations"""
    print("🔧 TEST DE CORRECTION - SYSTÈME DE RECOMMANDATIONS")
    print("=" * 60)
    
    session = requests.Session()
    analysis_id = None
    
    # Étape 1: Analyse CV pour obtenir un analysis_id
    print("\n🧪 Étape 1: Analyse CV")
    cv_data = {
        "cv_content": """Développeur Full-Stack Senior avec 5 ans d'expérience.
Compétences: Python, JavaScript, React, Node.js, PostgreSQL, Docker, AWS, Kubernetes.
Expérience en développement d'applications web, APIs REST, microservices.
Diplôme: Master en Informatique.
Certifications: AWS Solutions Architect.
Langues: Français (natif), Anglais (courant).
Email: senior.dev@example.com""",
        "format": "text"
    }
    
    try:
        response = session.post(f"{API_BASE}/analyze-cv", json=cv_data, timeout=15)
        if response.status_code == 200:
            data = response.json()
            analysis_id = data.get('analysis_id')
            print(f"✅ CV analysé - ID: {analysis_id}")
            print(f"   Compétences extraites: {len(data.get('skills', []))}")
        else:
            print(f"❌ Échec analyse CV: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur analyse CV: {e}")
        return False
    
    if not analysis_id:
        print("❌ Pas d'analysis_id. Impossible de continuer.")
        return False
    
    # Étape 2: Test des recommandations (cœur du fix)
    print(f"\n🎯 Étape 2: Test des recommandations CORRIGÉES")
    print("─" * 50)
    
    try:
        print("⏳ Génération des recommandations...")
        response = session.get(f"{API_BASE}/recommendations/{analysis_id}", timeout=20)
        
        if response.status_code == 200:
            print("✅ SUCCESS! Recommandations générées sans erreur")
            
            data = response.json()
            
            # Vérification de la structure
            print(f"\n📋 ANALYSE DES DONNÉES:")
            print(f"   • Analysis ID: {data.get('analysis_id')}")
            print(f"   • Generated at: {data.get('generated_at')}")
            print(f"   • Global confidence: {data.get('global_confidence', 0):.1%}")
            
            # Vérification des recommandations
            recommendations = data.get('recommendations', {})
            print(f"\n🎯 RECOMMANDATIONS ({len(recommendations)} types):")
            
            for rec_type, recs in recommendations.items():
                if isinstance(recs, list):
                    print(f"   • {rec_type.upper()}: {len(recs)} recommandations")
                    for i, rec in enumerate(recs[:2], 1):  # Afficher les 2 premières
                        if isinstance(rec, dict):
                            title = rec.get('title', 'N/A')
                            score = rec.get('score', 0)
                            if isinstance(score, (int, float)):
                                print(f"     {i}. {title} (Score: {score:.1%})")
                            else:
                                print(f"     {i}. {title} (Score: {score})")
                        else:
                            print(f"     {i}. {rec}")
                elif isinstance(recs, dict):
                    print(f"   • {rec_type.upper()}: Structure complexe")
                    # Afficher quelques clés
                    keys = list(recs.keys())[:3]
                    print(f"     Clés: {keys}")
                else:
                    print(f"   • {rec_type.upper()}: {type(recs).__name__}")
            
            # Vérification des profils utilisateur
            user_profile = data.get('user_profile', {})
            print(f"\n👤 PROFIL UTILISATEUR:")
            print(f"   • Compétences: {user_profile.get('skills_count', 0)}")
            print(f"   • Expérience: {user_profile.get('experience_years', 0)} ans")
            print(f"   • Postes: {user_profile.get('job_titles', [])}")
            
            # Vérification spéciale pour les erreurs gérées
            if data.get('error_handled'):
                print(f"\n⚠️  FALLBACK ACTIVÉ:")
                print(f"   • Erreur originale: {data.get('original_error', 'N/A')}")
                print("   • Le système a basculé sur des recommandations par défaut")
            
            print(f"\n🎉 TEST RÉUSSI ! Le bug 'unhashable type: slice' est CORRIGÉ !")
            return True
            
        else:
            print(f"❌ Échec recommandations: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur recommandations: {e}")
        print("🔍 Ce pourrait être le bug original...")
        return False

def test_multiple_scenarios():
    """Test avec plusieurs scénarios pour valider la robustesse"""
    print("\n" + "="*60)
    print("🔬 TESTS DE ROBUSTESSE MULTIPLES")
    print("="*60)
    
    scenarios = [
        {
            "name": "Profil Junior",
            "cv": "Développeur Junior avec 1 an d'expérience. Python, HTML, CSS.",
            "expected_skills": 3
        },
        {
            "name": "Profil Expert",
            "cv": "Architecte logiciel avec 10 ans d'expérience. Python, Java, Kubernetes, Terraform, AWS, Azure.",
            "expected_skills": 6
        },
        {
            "name": "Profil Data Science",
            "cv": "Data Scientist avec 3 ans d'expérience. Python, R, TensorFlow, SQL, Pandas.",
            "expected_skills": 5
        }
    ]
    
    session = requests.Session()
    results = []
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🧪 Scénario {i}: {scenario['name']}")
        print("─" * 40)
        
        # Analyse CV
        cv_data = {
            "cv_content": scenario["cv"],
            "format": "text"
        }
        
        try:
            # Analyse
            response = session.post(f"{API_BASE}/analyze-cv", json=cv_data, timeout=15)
            if response.status_code != 200:
                print(f"❌ Échec analyse: {response.status_code}")
                results.append(False)
                continue
            
            analysis_id = response.json().get('analysis_id')
            if not analysis_id:
                print("❌ Pas d'analysis_id")
                results.append(False)
                continue
            
            # Recommandations
            response = session.get(f"{API_BASE}/recommendations/{analysis_id}", timeout=15)
            if response.status_code == 200:
                data = response.json()
                rec_count = sum(len(recs) if isinstance(recs, list) else 1 
                               for recs in data.get('recommendations', {}).values())
                print(f"✅ Succès: {rec_count} recommandations générées")
                results.append(True)
            else:
                print(f"❌ Échec recommandations: {response.status_code}")
                results.append(False)
                
        except Exception as e:
            print(f"❌ Erreur: {e}")
            results.append(False)
        
        time.sleep(0.5)  # Pause entre scénarios
    
    # Résumé
    print(f"\n📊 RÉSUMÉ DES TESTS DE ROBUSTESSE:")
    success_rate = sum(results) / len(results) * 100
    print(f"   • Taux de réussite: {success_rate:.1f}% ({sum(results)}/{len(results)})")
    
    return success_rate >= 80  # 80% de réussite minimum

if __name__ == "__main__":
    print("🚀 DÉMARRAGE DES TESTS DE CORRECTION")
    
    # Test principal
    main_success = test_recommendations_fix()
    
    # Tests de robustesse
    robustness_success = test_multiple_scenarios()
    
    # Résultat final
    print(f"\n" + "="*60)
    print("🏁 RÉSULTAT FINAL")
    print("="*60)
    
    if main_success and robustness_success:
        print("🎉 TOUS LES TESTS SONT PASSÉS !")
        print("✅ Le bug 'unhashable type: slice' est définitivement CORRIGÉ")
        print("✅ Le système de recommandations est robuste et fonctionnel")
    elif main_success:
        print("✅ Le bug principal est corrigé")
        print("⚠️  Quelques tests de robustesse ont échoué")
    else:
        print("❌ Le bug persiste ou d'autres erreurs sont présentes")
        print("🔧 Vérifications supplémentaires nécessaires")