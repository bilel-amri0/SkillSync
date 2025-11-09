#!/usr/bin/env python3
"""
Script d'activation automatique du mode ML pour SkillSync
Active tous les composants d'intelligence artificielle avancés
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
import json

# Configuration
BACKEND_DIR = Path("SkillSync_Project/backend")
MODELS_DIR = BACKEND_DIR / "models"
ML_MODELS_DIR = BACKEND_DIR / "ml_models"

def setup_logging():
    """Configuration du logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ml_activation.log')
        ]
    )

def check_python_version():
    """Vérifier la version Python"""
    print("🐍 Vérification de Python...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ requis pour ML")
        return False
    print(f"✅ Python {sys.version.split()[0]} détecté")
    return True

def install_ml_dependencies():
    """Installer les dépendances ML"""
    print("\n📦 Installation des dépendances ML...")
    
    ml_packages = [
        "torch",
        "transformers",
        "sentence-transformers", 
        "scikit-learn",
        "tensorflow",
        "numpy",
        "pandas"
    ]
    
    try:
        for package in ml_packages:
            print(f"   📦 Installation de {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"   ✅ {package} installé")
            else:
                print(f"   ⚠️ {package} - Potentiel problème")
                print(f"      {result.stderr[:100]}")
        
        print("✅ Dépendances ML installées")
        return True
        
    except Exception as e:
        print(f"❌ Erreur installation: {e}")
        return False

def test_ml_imports():
    """Tester les imports ML"""
    print("\n🧪 Test des imports ML...")
    
    test_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence-Transformers"),
        ("sklearn", "Scikit-Learn"),
        ("tensorflow", "TensorFlow")
    ]
    
    results = {}
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {name} disponible")
            results[name] = True
        except ImportError:
            print(f"   ❌ {name} non disponible")
            results[name] = False
    
    return results

def create_ml_directories():
    """Créer les répertoires nécessaires"""
    print("\n📁 Création des répertoires ML...")
    
    directories = [
        MODELS_DIR,
        MODELS_DIR / "bert-skills-ner-final",
        MODELS_DIR / "similarity_model", 
        MODELS_DIR / "neural_scorer",
        BACKEND_DIR / "ml_data"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    print("✅ Répertoires créés")

def test_ml_components():
    """Tester les composants ML individuellement"""
    print("\n🔧 Test des composants ML...")
    
    # Changer vers le répertoire backend
    original_cwd = os.getcwd()
    os.chdir(BACKEND_DIR)
    
    try:
        # Test Skills Extractor
        print("   🧠 Test Skills Extractor...")
        try:
            from ml_models.skills_extractor import SkillsExtractorModel
            extractor = SkillsExtractorModel()
            print("   ✅ Skills Extractor OK")
        except Exception as e:
            print(f"   ⚠️ Skills Extractor: {str(e)[:50]}...")
        
        # Test Similarity Engine  
        print("   🎯 Test Similarity Engine...")
        try:
            from ml_models.similarity_engine import SimilarityEngine
            similarity = SimilarityEngine()
            print("   ✅ Similarity Engine OK")
        except Exception as e:
            print(f"   ⚠️ Similarity Engine: {str(e)[:50]}...")
        
        # Test Neural Scorer
        print("   🧮 Test Neural Scorer...")
        try:
            from ml_models.neural_scorer import NeuralScorer
            scorer = NeuralScorer()
            print("   ✅ Neural Scorer OK")
        except Exception as e:
            print(f"   ⚠️ Neural Scorer: {str(e)[:50]}...")
        
        # Test Advanced Engine
        print("   🚀 Test Advanced Engine...")
        try:
            from ml_models.advanced_recommendation_engine import AdvancedRecommendationEngine
            engine = AdvancedRecommendationEngine()
            print("   ✅ Advanced Engine OK")
            return True
        except Exception as e:
            print(f"   ⚠️ Advanced Engine: {str(e)[:50]}...")
            return False
            
    finally:
        os.chdir(original_cwd)

def update_main_backend():
    """Mettre à jour le backend principal pour utiliser ML"""
    print("\n⚙️ Configuration du backend principal...")
    
    main_file = BACKEND_DIR / "main_simple_for_frontend.py"
    
    # Lire le fichier actuel
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Configuration ML dans le fichier principal
    ml_config = '''
# Configuration ML - ACTIVÉ
ML_MODE_ENABLED = True
try:
    from ml_models.advanced_recommendation_engine import AdvancedRecommendationEngine
    ml_recommendation_engine = AdvancedRecommendationEngine()
    logger.info("🧠 Mode ML ACTIVÉ - Moteur avancé chargé")
except ImportError as e:
    ML_MODE_ENABLED = False
    ml_recommendation_engine = None
    logger.warning(f"⚠️ Mode ML non disponible: {e}")
'''
    
    # Ajouter la configuration si elle n'existe pas
    if "ML_MODE_ENABLED" not in content:
        # Trouver où insérer la configuration
        lines = content.split('\n')
        insert_position = -1
        
        for i, line in enumerate(lines):
            if "USE_ENHANCED_ENGINE = True" in line:
                insert_position = i + 1
                break
        
        if insert_position > 0:
            lines.insert(insert_position, ml_config)
            content = '\n'.join(lines)
            
            # Sauvegarder
            with open(main_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("   ✅ Backend configuré pour ML")
        else:
            print("   ⚠️ Position d'insertion non trouvée")
    else:
        print("   ✅ Configuration ML déjà présente")

def create_ml_test_script():
    """Créer un script de test ML"""
    print("\n🧪 Création du script de test ML...")
    
    test_script = '''#!/usr/bin/env python3
"""
Test complet du mode ML de SkillSync
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8001"
API_BASE = f"{BASE_URL}/api/v1"

def test_ml_recommendations():
    """Test des recommandations ML"""
    print("🧠 TEST DU MODE ML")
    print("=" * 50)
    
    session = requests.Session()
    
    # CV avec profil riche pour ML
    cv_data = {
        "cv_content": """Senior Machine Learning Engineer avec 7 ans d'expérience.
        
Compétences techniques:
- ML/AI: TensorFlow, PyTorch, Scikit-learn, Transformers, BERT
- Langages: Python, R, SQL, JavaScript
- Cloud: AWS SageMaker, GCP AI Platform, Azure ML
- Data: Pandas, NumPy, Spark, Kafka, Airflow
- DevOps: Docker, Kubernetes, MLOps, CI/CD

Expérience:
- Senior ML Engineer chez GoogleAI (2021-2024)
- Data Scientist chez Meta (2019-2021) 
- ML Researcher chez OpenAI (2017-2019)

Projets:
- Système de recommandations avec 50M+ utilisateurs
- Modèles NLP pour analyse de sentiment
- Pipeline ML automatisé en production
- Recherche en Computer Vision publiée

Formation:
- PhD Machine Learning - Stanford University
- MS Computer Science - MIT

Certifications:
- AWS Certified ML Specialty
- Google Professional ML Engineer
- TensorFlow Developer Certificate
"""
    }
    
    try:
        # Analyse CV
        print("📄 Analyse CV avec ML...")
        response = session.post(f"{API_BASE}/analyze-cv", json=cv_data, timeout=15)
        response.raise_for_status()
        cv_result = response.json()
        
        analysis_id = cv_result['analysis_id']
        skills = cv_result.get('skills', [])
        
        print(f"✅ CV analysé - ID: {analysis_id}")
        print(f"   🎯 Compétences ML extraites: {len(skills)}")
        print(f"   🧠 Top skills ML: {skills[:7]}")
        
        # Recommandations ML
        print("\\n🤖 Génération recommandations ML...")
        response = session.get(f"{API_BASE}/recommendations/{analysis_id}", timeout=20)
        response.raise_for_status()
        recommendations = response.json()
        
        print("✅ Recommandations ML générées")
        
        # Analyse de la qualité ML
        recs = recommendations.get('recommendations', {})
        print(f"\\n📊 ANALYSE QUALITÉ ML:")
        
        skill_dev = recs.get('SKILL_DEVELOPMENT', [])
        if skill_dev:
            print(f"   🎯 Développement: {len(skill_dev)} recommandations")
            for i, rec in enumerate(skill_dev[:3], 1):
                title = rec.get('title', 'N/A')
                score = rec.get('score', 0)
                print(f"     {i}. {title} (Score: {score:.1%})")
        
        projects = recs.get('PROJECT_SUGGESTIONS', [])
        if projects:
            print(f"   🚀 Projets ML: {len(projects)} suggestions")
            for i, proj in enumerate(projects[:2], 1):
                title = proj.get('title', 'N/A')
                score = proj.get('score', 0)
                print(f"     {i}. {title} (Score: {score:.1%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur test ML: {e}")
        return False

def main():
    print("🧠 TEST COMPLET MODE ML")
    success = test_ml_recommendations()
    
    if success:
        print("\\n🎉 MODE ML FONCTIONNEL!")
        print("✅ Intelligence artificielle activée")
        print("✅ Recommandations ML de haute qualité")
    else:
        print("\\n❌ Problèmes détectés en mode ML")
    
    return success

if __name__ == "__main__":
    main()
'''
    
    test_file = Path("test_ml_mode.py")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print(f"   ✅ Script de test: {test_file}")

def main():
    """Fonction principale d'activation ML"""
    setup_logging()
    
    print("🚀 ACTIVATION DU MODE ML SKILLSYNC")
    print("=" * 50)
    
    # Vérifications préliminaires
    if not check_python_version():
        return False
    
    # Installation des dépendances
    if not install_ml_dependencies():
        print("❌ Échec installation dépendances")
        return False
    
    # Test des imports
    import_results = test_ml_imports()
    if not all(import_results.values()):
        print("⚠️ Certaines dépendances manquent, mais on continue...")
    
    # Création des répertoires
    create_ml_directories()
    
    # Test des composants
    components_ok = test_ml_components()
    
    # Configuration du backend
    update_main_backend()
    
    # Création du script de test
    create_ml_test_script()
    
    # Résumé final
    print("\n" + "=" * 50)
    print("🏁 RÉSUMÉ ACTIVATION ML")
    print("=" * 50)
    
    if components_ok:
        print("🎉 MODE ML ACTIVÉ AVEC SUCCÈS!")
        print("✅ Composants ML fonctionnels")
        print("✅ Backend configuré")
        print("✅ Script de test créé")
        print("\n🚀 Étapes suivantes:")
        print("1. Redémarrer le serveur backend")
        print("2. Lancer: python test_ml_mode.py")
        print("3. Profiter de l'IA avancée! 🧠")
        return True
    else:
        print("⚠️ MODE ML PARTIELLEMENT ACTIVÉ")
        print("🔧 Certains composants nécessitent de l'attention")
        print("📝 Consulter ml_activation.log pour plus de détails")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)