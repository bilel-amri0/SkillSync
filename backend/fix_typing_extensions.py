#!/usr/bin/env python3
"""
🔧 Script de réparation pour typing_extensions
====================================================
Problème : ImportError: cannot import name 'TypeAliasType' from 'typing_extensions'
Solution : Mise à jour de typing_extensions vers une version compatible
"""

import subprocess
import sys

def run_command(cmd, description):
    """Exécute une commande avec gestion d'erreur"""
    print(f"\n🔄 {description}...")
    print(f"   Commande: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        
        if result.stdout:
            print(f"✅ Sortie:\n{result.stdout}")
        if result.stderr and result.returncode == 0:
            print(f"⚠️ Avertissements:\n{result.stderr}")
        elif result.stderr and result.returncode != 0:
            print(f"❌ Erreurs:\n{result.stderr}")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout après 120 secondes")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    print("🚀 RÉPARATION TYPING_EXTENSIONS")
    print("=" * 50)
    
    # Étape 1: Mettre à jour typing_extensions
    success1 = run_command(
        "pip install --upgrade typing_extensions>=4.8.0",
        "Mise à jour typing_extensions"
    )
    
    # Étape 2: Mettre à jour pydantic si nécessaire
    success2 = run_command(
        "pip install --upgrade pydantic>=2.0.0",
        "Mise à jour pydantic"
    )
    
    # Étape 3: Test d'import rapide
    print("\n🧪 Test d'import...")
    try:
        import typing_extensions
        from typing_extensions import TypeAliasType
        print("✅ typing_extensions.TypeAliasType importé avec succès")
        
        import pydantic
        print(f"✅ Pydantic version: {pydantic.__version__}")
        
        from fastapi import FastAPI
        print("✅ FastAPI importé avec succès")
        
        print("\n🎉 RÉPARATION TERMINÉE AVEC SUCCÈS!")
        print("\n📋 Prochaines étapes:")
        print("   1. Redémarrer le serveur: python main_simple_for_frontend.py")
        print("   2. Tester l'API sur http://localhost:8000")
        
    except ImportError as e:
        print(f"❌ Erreur d'import persistante: {e}")
        print("\n🔍 Diagnostic supplémentaire requis...")
        
        # Afficher les versions installées
        run_command("pip show typing_extensions", "Version typing_extensions")
        run_command("pip show pydantic", "Version pydantic")
        run_command("pip show fastapi", "Version fastapi")

if __name__ == "__main__":
    main()