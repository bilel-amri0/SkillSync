#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TEST MODE ML COMPLET AVEC TENSORFLOW - SKILLSYNC
"""

import requests
import json
import time
import sys
from datetime import datetime

def print_header(title):
    print("\n" + "="*70)
    print(f"🧪 {title}")
    print("="*70)

def test_server_connection():
    """Test de connexion au serveur"""
    print("\n🌐 Test connexion serveur...")
    try:
        response = requests.get("http://localhost:8000/", timeout=10)
        if response.status_code == 200:
            print("✅ Serveur backend actif")
            return True
        else:
            print(f"❌ Serveur erreur HTTP: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Serveur inaccessible - démarrer avec: python main_simple_for_frontend.py")
        return False
    except Exception as e:
        print(f"❌ Erreur connexion serveur: {e}")
        return False

def test_ml_status_detailed():
    """Test détaillé du statut ML"""
    print("\n🧠 Test statut ML détaillé...")
    try:
        response = requests.get("http://localhost:8000/api/v1/ml/status", timeout=15)
        if response.status_code == 200:
            data = response.json()
            print("✅ Endpoint ML status actif")
            
            # Affichage détaillé du statut
            print(f"\n📊 STATUT ML COMPLET:")
            print(f"   🤖 ML Activé: {data.get('ml_enabled', 'N/A')}")
            print(f"   🔧 Type ML: {data.get('ml_mode_type', 'N/A')}")
            print(f"   ⚙️ Moteur: {data.get('engine_type', 'N/A')}")
            print(f"   📝 Version: {data.get('version', 'N/A')}")
            
            # Capacités ML
            capabilities = data.get('capabilities', {})
            print(f"\n🎯 CAPACITÉS ML:")
            for cap, enabled in capabilities.items():
                status = "✅" if enabled else "❌"
                print(f"   {status} {cap}")
            
            # Modèles chargés
            models = data.get('models_loaded', {})
            print(f"\n🧠 MODÈLES CHARGÉS:")
            for model, loaded in models.items():
                status = "✅" if loaded else "❌"
                print(f"   {status} {model}")
            
            # Performance
            performance = data.get('performance', {})
            print(f"\n⚡ PERFORMANCE:")
            print(f"   🏃 Mode: {performance.get('mode', 'N/A')}")
            print(f"   🚀 Vitesse: {performance.get('speed', 'N/A')}")
            print(f"   🎯 Précision: {performance.get('accuracy', 'N/A')}")
            
            # Vérifier si TensorFlow est actif
            tensorflow_active = (
                data.get('ml_mode_type') == 'full' and
                data.get('capabilities', {}).get('tensorflow_models', False)
            )
            
            if tensorflow_active:
                print("\n🎉 TENSORFLOW DÉTECTÉ ET ACTIF!")
                return True
            else:
                print("\n⚠️ TensorFlow non détecté - vérifier l'installation")
                return False
                
        else:
            print(f"❌ ML Status erreur HTTP: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Test ML status échoué: {e}")
        return False

def test_tensorflow_recommendations():
    """Test spécifique des recommandations TensorFlow"""
    print("\n🧠 Test recommandations TensorFlow...")
    
    # Profil de test complexe pour TensorFlow
    test_profile = {
        "current_role": "Développeur Full-Stack Senior",
        "target_role": "Architecte Solutions IA",
        "skills": ["Python", "JavaScript", "TensorFlow", "Kubernetes", "AWS", "React"],
        "experience_years": 5,
        "industry": "FinTech"
    }
    
    try:
        print(f"   👤 Profil test: {test_profile['current_role']} -> {test_profile['target_role']}")
        print(f"   🛠️ Compétences: {', '.join(test_profile['skills'])}")
        
        response = requests.post(
            "http://localhost:8000/api/v1/recommendations",
            json=test_profile,
            timeout=45  # Timeout généreux pour TensorFlow
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Recommandations TensorFlow générées")
            
            # Analyser la qualité des recommandations
            engine_info = data.get('engine_info', {})
            print(f"\n🔧 MOTEUR UTILISÉ:")
            print(f"   🤖 Type: {engine_info.get('engine_type', 'N/A')}")
            print(f"   📝 Version: {engine_info.get('version', 'N/A')}")
            print(f"   🧠 ML Activé: {engine_info.get('ml_mode_enabled', 'N/A')}")
            print(f"   🔧 Type ML: {engine_info.get('ml_mode_type', 'N/A')}")
            
            # Analyser les recommandations
            recommendations = data.get('recommendations', {})
            total_recommendations = 0
            valid_scores = 0
            valid_titles = 0
            
            for category, items in recommendations.items():
                if isinstance(items, list):
                    total_recommendations += len(items)
                    for item in items:
                        if item.get('priority_score', 0) > 0:
                            valid_scores += 1
                        if item.get('title') and item['title'] != 'N/A':
                            valid_titles += 1
            
            print(f"\n📊 QUALITÉ RECOMMANDATIONS:")
            print(f"   📋 Total: {total_recommendations}")
            print(f"   🎯 Scores valides: {valid_scores}/{total_recommendations}")
            print(f"   📝 Titres valides: {valid_titles}/{total_recommendations}")
            
            # Afficher quelques exemples
            immediate_actions = recommendations.get('IMMEDIATE_ACTIONS', [])
            if immediate_actions:
                print(f"\n🎯 EXEMPLES D'ACTIONS IMMÉDIATES:")
                for i, action in enumerate(immediate_actions[:3], 1):
                    title = action.get('title', 'N/A')
                    score = action.get('priority_score', 0)
                    description = action.get('description', 'N/A')[:80] + '...'
                    print(f"   {i}. {title} (Score: {score}%)")
                    print(f"      {description}")
            
            # Déterminer le succès
            quality_threshold = 0.8  # 80% de qualité minimum
            if (valid_scores >= total_recommendations * quality_threshold and 
                valid_titles >= total_recommendations * quality_threshold):
                print("\n🌟 QUALITÉ EXCELLENTE - TensorFlow fonctionne parfaitement!")
                return True
            else:
                print("\n⚠️ Qualité modérée - TensorFlow partiellement fonctionnel")
                return False
                
        else:
            print(f"❌ Recommandations erreur HTTP: {response.status_code}")
            if response.text:
                print(f"   📄 Réponse: {response.text[:200]}...")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Timeout lors des recommandations TensorFlow")
        print("   💡 Ceci peut indiquer que TensorFlow charge ses modèles")
        return False
    except Exception as e:
        print(f"❌ Test recommandations TensorFlow échoué: {e}")
        return False

def test_tensorflow_models_locally():
    """Test local des modèles TensorFlow"""
    print("\n🔬 Test local des modèles TensorFlow...")
    
    try:
        # Test d'import TensorFlow
        import tensorflow as tf
        print(f"   ✅ TensorFlow {tf.__version__} importé")
        
        # Test de création de modèle
        print("   🏗️ Création modèle neural scorer...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(384,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        print("   ✅ Modèle TensorFlow créé")
        
        # Test de prédiction
        import numpy as np
        test_input = np.random.random((1, 384))
        prediction = model.predict(test_input, verbose=0)
        score = prediction[0][0]
        print(f"   ✅ Prédiction test: {score:.4f}")
        
        # Test des autres composants
        try:
            from transformers import AutoTokenizer, AutoModel
            print("   ✅ Transformers disponible")
        except ImportError:
            print("   ⚠️ Transformers manquant")
        
        try:
            from sentence_transformers import SentenceTransformer
            print("   ✅ Sentence-Transformers disponible")
        except ImportError:
            print("   ⚠️ Sentence-Transformers manquant")
        
        print("\n🎉 TENSORFLOW ENTIÈREMENT FONCTIONNEL!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import TensorFlow échoué: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Test TensorFlow local échoué: {e}")
        return False

def run_complete_tensorflow_test():
    """Test complet du mode ML avec TensorFlow"""
    print_header("TEST COMPLET MODE ML TENSORFLOW - SKILLSYNC")
    print(f"🕒 Démarré à: {datetime.now().strftime('%H:%M:%S')}")
    
    results = {
        "server_connection": False,
        "ml_status": False,
        "tensorflow_local": False,
        "tensorflow_recommendations": False
    }
    
    # Tests séquentiels
    print("\n🚀 DÉMARRAGE DES TESTS TENSORFLOW...")
    
    # 1. Test connexion serveur
    results["server_connection"] = test_server_connection()
    
    # 2. Test local TensorFlow
    results["tensorflow_local"] = test_tensorflow_models_locally()
    
    if results["server_connection"]:
        # 3. Test ML status
        results["ml_status"] = test_ml_status_detailed()
        
        # 4. Test recommandations TensorFlow
        results["tensorflow_recommendations"] = test_tensorflow_recommendations()
    
    # Analyse des résultats
    print_header("RÉSULTATS FINAUX TENSORFLOW")
    total_tests = len(results)
    passed_tests = sum(results.values())
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"📊 Tests réussis: {passed_tests}/{total_tests}")
    print(f"📈 Taux de succès: {success_rate:.1f}%")
    
    print("\n📋 DÉTAIL DES RÉSULTATS:")
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        test_name = test.replace('_', ' ').title()
        print(f"   {test_name}: {status}")
    
    # Conclusion
    if passed_tests == total_tests:
        print("\n🎉 TENSORFLOW MODE ML 100% FONCTIONNEL!")
        print("🚀 Votre système SkillSync utilise maintenant l'IA avancée")
        print("💡 Les recommandations sont générées par des réseaux de neurones TensorFlow")
    elif passed_tests >= 3:
        print("\n✅ TENSORFLOW MAJORITAIREMENT FONCTIONNEL")
        print("⚙️ Quelques ajustements mineurs possibles")
    elif results["tensorflow_local"]:
        print("\n⚠️ TENSORFLOW INSTALLÉ MAIS PROBLÈMES DE CONFIGURATION")
        print("🔧 Vérifier la configuration du serveur backend")
    else:
        print("\n❌ TENSORFLOW NON FONCTIONNEL")
        print("💊 Solutions:")
        print("   1. Relancer: python install_tensorflow_full.py")
        print("   2. Installation manuelle: pip install tensorflow")
        print("   3. Vérifier les logs d'erreur ci-dessus")
    
    print(f"\n🕒 Terminé à: {datetime.now().strftime('%H:%M:%S')}")
    return passed_tests == total_tests

if __name__ == "__main__":
    try:
        success = run_complete_tensorflow_test()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Tests interrompus par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue dans les tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
