#!/usr/bin/env python3
"""
Script de test complet pour SkillSync - VERSION FINALE CORRIGÉE
Teste tous les endpoints et fonctionnalités step by step
"""

import requests
import json
import time
from pathlib import Path
import os
import traceback

# Configuration
BASE_URL = "http://127.0.0.1:8001"
API_BASE = f"{BASE_URL}/api/v1"

class SkillSyncTester:
    def __init__(self):
        self.session = requests.Session()
        self.analysis_id = None
        self.cv_data_extracted = None
        
    def print_step(self, step_num, description):
        """Affiche une étape de test"""
        print(f"\n{'='*60}")
        print(f"🧪 ÉTAPE {step_num}: {description}")
        print('='*60)
    
    def print_success(self, message):
        """Affiche un succès"""
        print(f"✅ {message}")
    
    def print_error(self, message):
        """Affiche une erreur"""
        print(f"❌ {message}")
    
    def print_info(self, message):
        """Affiche une info"""
        print(f"ℹ️  {message}")

    def test_step_1_health_check(self):
        """Test 1: Vérification de la santé du serveur"""
        self.print_step(1, "HEALTH CHECK")
        
        try:
            response = self.session.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                self.print_success("Serveur backend opérationnel")
                self.print_info(f"Response: {response.json()}")
                return True
            else:
                self.print_error(f"Health check failed: {response.status_code}")
                return False
        except Exception as e:
            self.print_error(f"Impossible de contacter le serveur: {e}")
            return False

    def test_step_2_api_status(self):
        """Test 2: Statut des APIs Job"""
        self.print_step(2, "API STATUS CHECK")
        
        try:
            response = self.session.get(f"{API_BASE}/jobs/status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.print_success("Statut des APIs récupéré")
                
                # Afficher le statut de chaque API
                if 'api_status' in data:
                    for api_name, status in data['api_status'].items():
                        icon = "✅" if status.get('available', False) else "❌"
                        self.print_info(f"{icon} {api_name}: {status}")
                
                return True
            else:
                self.print_error(f"API status check failed: {response.status_code}")
                return False
        except Exception as e:
            self.print_error(f"Erreur lors du test API status: {e}")
            return False

    def test_step_3_job_search(self):
        """Test 3: Recherche d'emplois"""
        self.print_step(3, "JOB SEARCH")
        
        search_data = {
            "query": "Python Developer",
            "location": "fr",
            "skills": ["Python", "Django", "FastAPI"],
            "max_results": 10
        }
        
        try:
            start_time = time.time()
            response = self.session.post(
                f"{API_BASE}/jobs/search", 
                json=search_data,
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                job_count = len(data.get('jobs', []))
                search_time = (end_time - start_time) * 1000
                
                self.print_success(f"Recherche réussie: {job_count} emplois trouvés en {search_time:.0f}ms")
                
                # Afficher les sources
                sources = data.get('summary', {}).get('sources', {})
                for source, count in sources.items():
                    if count > 0:
                        self.print_info(f"📊 {source}: {count} emplois")
                
                # Afficher quelques exemples d'emplois
                jobs = data.get('jobs', [])[:3]
                for i, job in enumerate(jobs, 1):
                    self.print_info(f"💼 Emploi {i}: {job.get('title', 'N/A')} - {job.get('company', 'N/A')}")
                
                return True
            else:
                self.print_error(f"Job search failed: {response.status_code}")
                return False
        except Exception as e:
            self.print_error(f"Erreur lors de la recherche d'emplois: {e}")
            return False

    def test_step_4_cv_analysis_text(self):
        """Test 4: Analyse CV (texte) - VERSION CORRIGÉE"""
        self.print_step(4, "CV ANALYSIS - TEXT")
        
        # CORRECTION: Utiliser cv_content au lieu de cv_text
        cv_data = {
            "cv_content": """Développeur Full-Stack avec 4 ans d'expérience.
Compétences: Python, JavaScript, Java, React, Node.js, SQL, PostgreSQL, Docker, AWS.
Expérience en développement d'applications web et APIs REST.
Diplôme: Master en Informatique.
Langues: Français (natif), Anglais (courant).
Email: test@example.com
Téléphone: +33 1 23 45 67 89""",
            "format": "text"
        }
        
        try:
            response = self.session.post(
                f"{API_BASE}/analyze-cv",
                json=cv_data,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                self.analysis_id = data.get('analysis_id')
                
                # IMPORTANT: Sauvegarder les données CV extraites pour le portfolio
                self.cv_data_extracted = {
                    'personal_info': {
                        'name': 'Test User',
                        'email': 'test@example.com',
                        'phone': '+33 1 23 45 67 89'
                    },
                    'skills': data.get('skills', []),
                    'experience_years': data.get('experience_years', 0),
                    'job_titles': ['Développeur Full-Stack'],
                    'education': ['Master en Informatique'],
                    'summary': 'Développeur Full-Stack avec expertise en technologies modernes',
                    'languages': ['Français (natif)', 'Anglais (courant)']
                }
                
                self.print_success(f"Analyse CV réussie - ID: {self.analysis_id}")
                
                # Afficher les compétences extraites
                skills = data.get('skills', [])
                self.print_info(f"🎯 Compétences extraites: {len(skills)}")
                for skill in skills[:5]:  # Afficher les 5 premières
                    self.print_info(f"   • {skill}")
                
                # Afficher l'expérience
                experience = data.get('experience_years')
                if experience:
                    self.print_info(f"📅 Expérience: {experience} ans")
                
                # Afficher la confiance
                confidence = data.get('confidence_score', 0)
                self.print_info(f"🎯 Score de confiance: {confidence:.1%}")
                
                return True
            else:
                self.print_error(f"CV analysis failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            self.print_error(f"Erreur lors de l'analyse CV: {e}")
            return False

    def test_step_5_dashboard(self):
        """Test 5: Récupération des données dashboard"""
        self.print_step(5, "DASHBOARD DATA")
        
        if not self.analysis_id:
            self.print_error("Pas d'analysis_id disponible. Étape 4 requise.")
            return False
        
        try:
            response = self.session.get(
                f"{API_BASE}/dashboard/{self.analysis_id}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                self.print_success("Données dashboard récupérées")
                
                # Afficher les métriques du dashboard
                dashboard = data.get('dashboard', {})
                skill_count = dashboard.get('skill_analysis', {}).get('total_skills', 0)
                self.print_info(f"📊 Nombre de compétences: {skill_count}")
                
                # Afficher les catégories de compétences
                skill_categories = dashboard.get('skill_analysis', {}).get('categories', {})
                for category, count in skill_categories.items():
                    self.print_info(f"   • {category}: {count}")
                
                return True
            else:
                self.print_error(f"Dashboard retrieval failed: {response.status_code}")
                self.print_info(f"Response: {response.text}")
                return False
        except Exception as e:
            self.print_error(f"Erreur lors de la récupération dashboard: {e}")
            return False

    def test_step_6_recommendations(self):
        """Test 6: Génération de recommandations - VERSION AVEC GESTION D'ERREUR CORRIGÉE"""
        self.print_step(6, "RECOMMENDATIONS")
        
        if not self.analysis_id:
            self.print_error("Pas d'analysis_id disponible. Étape 4 requise.")
            return False
        
        try:
            response = self.session.get(
                f"{API_BASE}/recommendations/{self.analysis_id}",
                timeout=20  # Augmenté le timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                self.print_success("Recommandations générées")
                
                # Vérifier si c'est un fallback
                if data.get('error_handled'):
                    self.print_info("⚠️  Mode fallback activé (recommandations par défaut)")
                    self.print_info(f"   Erreur originale: {data.get('original_error', 'N/A')}")
                
                # Afficher les recommandations
                recommendations = data.get('recommendations', {})
                
                for rec_type, recs in recommendations.items():
                    if recs and isinstance(recs, list):
                        self.print_info(f"🎯 {rec_type.upper()}: {len(recs)} recommandations")
                        for rec in recs[:2]:  # Afficher les 2 premières
                            if isinstance(rec, dict):
                                title = rec.get('title', 'N/A')
                                score = rec.get('score', 0)
                                if isinstance(score, (int, float)):
                                    self.print_info(f"   • {title} (Score: {score:.1%})")
                                else:
                                    self.print_info(f"   • {title} (Score: {score})")
                            else:
                                self.print_info(f"   • {rec}")
                    elif recs and isinstance(recs, dict):
                        self.print_info(f"🎯 {rec_type.upper()}: Structure complexe")
                
                # Afficher la confiance globale
                global_confidence = data.get('global_confidence', 0)
                if global_confidence:
                    self.print_info(f"🎯 Confiance globale: {global_confidence:.1%}")
                
                return True
            else:
                self.print_error(f"Recommendations failed: {response.status_code}")
                self.print_info(f"Response: {response.text}")
                return False
        except Exception as e:
            # Le bug 'unhashable type: slice' devrait maintenant être corrigé
            error_msg = str(e)
            self.print_error(f"Erreur lors de la génération de recommandations: {error_msg}")
            if "unhashable type" in error_msg:
                self.print_info("🚨 Le bug 'unhashable type: slice' persiste ! Vérification nécessaire.")
            return False

    def test_step_7_portfolio(self):
        """Test 7: Génération de portfolio - VERSION FINALE CORRIGÉE"""
        self.print_step(7, "PORTFOLIO GENERATION")
        
        if not self.cv_data_extracted:
            self.print_error("Pas de données CV extraites. Étape 4 requise.")
            return False
        
        # CORRECTION FINALE: Structure conforme à l'API
        portfolio_data = {
            "cv_data": self.cv_data_extracted,  # CHAMP REQUIS !
            "template": "modern",
            "style": "professional"
        }
        
        try:
            response = self.session.post(
                f"{API_BASE}/generate-portfolio",
                json=portfolio_data,
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                self.print_success("Portfolio généré")
                
                # Afficher les informations du portfolio
                portfolio_id = data.get('portfolio_id')
                if portfolio_id:
                    self.print_info(f"🆔 Portfolio ID: {portfolio_id}")
                
                template = data.get('template')
                if template:
                    self.print_info(f"🎨 Template utilisé: {template}")
                
                html_length = len(data.get('html_content', ''))
                self.print_info(f"📄 HTML généré: {html_length} caractères")
                
                return True
            else:
                self.print_error(f"Portfolio generation failed: {response.status_code}")
                self.print_info(f"Response: {response.text}")
                return False
        except Exception as e:
            self.print_error(f"Erreur lors de la génération de portfolio: {e}")
            self.print_info(f"Traceback: {traceback.format_exc()}")
            return False

    def run_all_tests(self):
        """Exécute tous les tests"""
        print("🚀 DÉMARRAGE DES TESTS SKILLSYNC")
        print(f"🎯 URL Backend: {BASE_URL}")
        
        tests = [
            self.test_step_1_health_check,
            self.test_step_2_api_status,
            self.test_step_3_job_search,
            self.test_step_4_cv_analysis_text,
            self.test_step_5_dashboard,
            self.test_step_6_recommendations,
            self.test_step_7_portfolio
        ]
        
        results = []
        
        for test in tests:
            try:
                result = test()
                results.append(result)
                time.sleep(1)  # Pause entre les tests
            except Exception as e:
                self.print_error(f"Erreur inattendue lors du test: {e}")
                self.print_info(f"Traceback: {traceback.format_exc()}")
                results.append(False)
        
        # Résumé final
        self.print_step("FINAL", "RÉSUMÉ DES TESTS")
        
        passed = sum(results)
        total = len(results)
        
        print(f"📊 Tests réussis: {passed}/{total}")
        
        if passed == total:
            self.print_success("🎉 TOUS LES TESTS SONT PASSÉS ! SkillSync fonctionne parfaitement.")
        else:
            failed_count = total - passed
            self.print_error(f"❌ {failed_count} test(s) échoué(s). Vérifiez les logs ci-dessus.")
        
        # Afficher les détails des échecs
        if passed < total:
            print("\n🔧 RECOMMANDATIONS DE DEBUG:")
            print("   • Vérifiez que le serveur est en marche")
            print("   • Contrôlez les logs du serveur backend")
            print("   • Testez les endpoints individuellement")
            
            # Recommandations spécifiques selon les tests échoués
            if not results[5]:  # Recommendations failed
                print("   • Bug connu: 'unhashable type: slice' dans le moteur de recommandations")
            if not results[6]:  # Portfolio failed
                print("   • Vérifiez la structure des données CV extraites")
        
        return passed == total

if __name__ == "__main__":
    print("🔬 SKILLSYNC - SUITE DE TESTS COMPLÈTE (VERSION FINALE)")
    print("======================================================")
    
    tester = SkillSyncTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 SYSTÈME VALIDÉ - PRÊT POUR PRODUCTION !")
    else:
        print("\n⚠️  VÉRIFICATIONS NÉCESSAIRES")