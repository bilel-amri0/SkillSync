#!/usr/bin/env python3
"""
Test final corrigé - noms d'import corrects
"""

def test_quick():
    modules = [
        ("scikit-learn", "sklearn"),
        ("python-dotenv", "dotenv"),
        ("fastapi", "fastapi"),
        ("openai", "openai"),
        ("spacy model", "spacy.load('en_core_web_sm')")
    ]
    
    print("🔍 VÉRIFICATION RAPIDE DES MODULES CRITIQUES:")
    print("-" * 50)
    
    all_ok = True
    
    try:
        import sklearn
        print(f"✅ scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("❌ scikit-learn: NON INSTALLÉ")
        all_ok = False
    
    try:
        import dotenv
        print("✅ python-dotenv: OK")
    except ImportError:
        print("❌ python-dotenv: NON INSTALLÉ")
        all_ok = False
    
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print(f"✅ SpaCy + modèle: {nlp.meta['version']}")
    except Exception as e:
        print(f"❌ SpaCy modèle: {e}")
        all_ok = False
    
    try:
        import fastapi, openai, langchain
        print(f"✅ FastAPI: {fastapi.__version__}")
        print(f"✅ OpenAI: {openai.__version__}")
        print(f"✅ LangChain: {langchain.__version__}")
    except Exception as e:
        print(f"❌ Modules principaux: {e}")
        all_ok = False
    
    print("-" * 50)
    if all_ok:
        print("🎉 SUCCÈS ! Tous les modules critiques sont OK!")
        print("🚀 VOTRE APPLICATION EST PRÊTE À DÉMARRER !")
        print("\n💡 Commande pour démarrer:")
        print("   python main_simple_for_frontend.py")
    else:
        print("⚠️ Certains modules nécessitent une installation.")
    
    return all_ok

if __name__ == "__main__":
    test_quick()