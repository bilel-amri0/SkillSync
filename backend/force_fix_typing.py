#!/usr/bin/env python3
"""
🔥 Réparation RADICALE typing_extensions
================================================
Problème : typing_extensions 4.15.0 trop ancienne
Solution : Force installation version récente
"""
import subprocess
import sys
import time
def run_command(cmd, description):
    """Exécute une commande avec gestion d'erreur"""
    print(f"\n🔄 {description}...")
    print(f"   Commande: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
        
        if result.stdout:
            # Limiter l'affichage pour éviter spam
            stdout_lines = result.stdout.split('\n')
            if len(stdout_lines) > 10:
                print(f"✅ Sortie:\n" + '\n'.join(stdout_lines[:5]))
                print("   [... lignes supprimées ...]")
                print('\n'.join(stdout_lines[-3:]))
            else:
                print(f"✅ Sortie:\n{result.stdout}")
                
        if result.stderr and result.returncode != 0:
            print(f"❌ Erreurs:\n{result.stderr}")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout après 180 secondes")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False
def main():
    print("🔥 RÉPARATION RADICALE TYPING_EXTENSIONS")
    print("=" * 50)
    
    # Étape 1: Désinstaller typing_extensions complètement
    print("\n🗑️ Étape 1: Désinstallation complète...")
    run_command(
        "pip uninstall typing_extensions -y",
        "Désinstallation typing_extensions"
    )
    
    # Attendre un peu
    time.sleep(2)
    
    # Étape 2: Nettoyer le cache pip
    print("\n🧹 Étape 2: Nettoyage cache...")
    run_command(
        "pip cache purge",
        "Nettoyage cache pip"
    )
    
    # Étape 3: Réinstaller avec version spécifique
    print("\n📦 Étape 3: Réinstallation forcée...")
    success = run_command(
        "pip install typing_extensions==4.12.0 --force-reinstall --no-cache-dir",
        "Installation typing_extensions 4.12.0"
    )
    
    if not success:
        print("\n🔄 Tentative avec version 4.8.0...")
        success = run_command(
            "pip install typing_extensions==4.8.0 --force-reinstall --no-cache-dir",
            "Installation typing_extensions 4.8.0"
        )
    
    # Étape 4: Test immédiat
    print("\n🧪 Étape 4: Test des imports...")
    try:
        import typing_extensions
        print(f"✅ typing_extensions version: {typing_extensions.__version__}")
        
        # Test des fonctions critiques
        try:
            from typing_extensions import TypeAliasType
            print("✅ TypeAliasType disponible")
        except ImportError:
            print("❌ TypeAliasType manquant")
            
        try:
            from typing_extensions import Sentinel
            print("✅ Sentinel disponible")
        except ImportError:
            print("❌ Sentinel manquant")
            
        # Test FastAPI
        try:
            from fastapi import FastAPI
            print("✅ FastAPI importé avec succès!")
            
            print("\n🎉 RÉPARATION TERMINÉE AVEC SUCCÈS!")
            print("\n📋 Prochaines étapes:")
            print("   1. Redémarrer le serveur: python main_simple_for_frontend.py")
            print("   2. Tester l'API sur http://localhost:8000")
            
            return True
            
        except ImportError as e:
            print(f"❌ FastAPI toujours en erreur: {e}")
            return False
            
    except ImportError as e:
        print(f"❌ typing_extensions toujours en erreur: {e}")
        
        # Solution alternative : downgrade pydantic
        print("\n🔄 Solution alternative: downgrade pydantic...")
        run_command(
            "pip install pydantic==1.10.8 --force-reinstall",
            "Downgrade pydantic vers 1.10.8"
        )
        
        # Test final
        try:
            from fastapi import FastAPI
            print("✅ FastAPI fonctionne avec pydantic 1.10.8!")
            return True
        except ImportError as e:
            print(f"❌ Échec final: {e}")
            return False
if __name__ == "__main__":
    main()