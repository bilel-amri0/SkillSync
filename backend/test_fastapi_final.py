#!/usr/bin/env python3
"""
🧪 Test FastAPI après réparation typing_extensions
================================================
"""
def test_imports():
    print("🧪 TEST DES IMPORTS CRITIQUES")
    print("=" * 50)
    
    # Test 1: typing_extensions
    try:
        import typing_extensions
        print("✅ typing_extensions importé")
        
        # Test des fonctions critiques
        try:
            from typing_extensions import TypeAliasType
            print("✅ TypeAliasType disponible")
        except ImportError as e:
            print(f"❌ TypeAliasType: {e}")
            
        try:
            from typing_extensions import Sentinel
            print("✅ Sentinel disponible")
        except ImportError as e:
            print(f"❌ Sentinel: {e}")
            
    except ImportError as e:
        print(f"❌ typing_extensions: {e}")
        return False
    
    # Test 2: pydantic_core
    try:
        import pydantic_core
        print("✅ pydantic_core importé")
    except ImportError as e:
        print(f"❌ pydantic_core: {e}")
        return False
    
    # Test 3: pydantic
    try:
        import pydantic
        print(f"✅ pydantic importé (v{pydantic.__version__})")
    except ImportError as e:
        print(f"❌ pydantic: {e}")
        return False
    
    # Test 4: FastAPI
    try:
        from fastapi import FastAPI
        print("✅ FastAPI importé avec succès!")
        
        # Test création d'une app simple
        app = FastAPI(title="Test")
        print("✅ Instance FastAPI créée")
        
        return True
        
    except ImportError as e:
        print(f"❌ FastAPI: {e}")
        return False
def test_server_start():
    """Test si le serveur peut démarrer"""
    print("\n🚀 TEST DÉMARRAGE SERVEUR")
    print("=" * 30)
    
    try:
        # Import du serveur principal
        import main_simple_for_frontend
        print("✅ main_simple_for_frontend importé sans erreur!")
        print("🎉 LE SERVEUR PEUT DÉMARRER!")
        return True
        
    except ImportError as e:
        print(f"❌ Erreur import serveur: {e}")
        return False
    except Exception as e:
        print(f"⚠️ Autre erreur: {e}")
        return False
if __name__ == "__main__":
    success1 = test_imports()
    
    if success1:
        success2 = test_server_start()
        
        if success2:
            print("\n🎉 RÉPARATION COMPLÈTEMENT RÉUSSIE!")
            print("\n📋 PROCHAINES ÉTAPES:")
            print("   1. Démarrer le serveur: python main_simple_for_frontend.py")
            print("   2. Ouvrir http://localhost:8000")
            print("   3. Tester l'API ML : http://localhost:8000/api/v1/ml/status")
        else:
            print("\n⚠️ FastAPI fonctionne mais problème avec le serveur principal")
    else:
        print("\n❌ Réparation incomplète - autres actions nécessaires")