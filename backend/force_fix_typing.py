#!/usr/bin/env python3
"""
 Rparation RADICALE typing_extensions
================================================
Problme : typing_extensions 4.15.0 trop ancienne
Solution : Force installation version rcente
"""
import subprocess
import sys
import time
def run_command(cmd, description):
    """Excute une commande avec gestion d'erreur"""
    print(f"\n {description}...")
    print(f"   Commande: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=180)
        
        if result.stdout:
            # Limiter l'affichage pour viter spam
            stdout_lines = result.stdout.split('\n')
            if len(stdout_lines) > 10:
                print(f" Sortie:\n" + '\n'.join(stdout_lines[:5]))
                print("   [... lignes supprimes ...]")
                print('\n'.join(stdout_lines[-3:]))
            else:
                print(f" Sortie:\n{result.stdout}")
                
        if result.stderr and result.returncode != 0:
            print(f" Erreurs:\n{result.stderr}")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f" Timeout aprs 180 secondes")
        return False
    except Exception as e:
        print(f" Erreur: {e}")
        return False
def main():
    print(" RPARATION RADICALE TYPING_EXTENSIONS")
    print("=" * 50)
    
    # tape 1: Dsinstaller typing_extensions compltement
    print("\n tape 1: Dsinstallation complte...")
    run_command(
        "pip uninstall typing_extensions -y",
        "Dsinstallation typing_extensions"
    )
    
    # Attendre un peu
    time.sleep(2)
    
    # tape 2: Nettoyer le cache pip
    print("\n tape 2: Nettoyage cache...")
    run_command(
        "pip cache purge",
        "Nettoyage cache pip"
    )
    
    # tape 3: Rinstaller avec version spcifique
    print("\n tape 3: Rinstallation force...")
    success = run_command(
        "pip install typing_extensions==4.12.0 --force-reinstall --no-cache-dir",
        "Installation typing_extensions 4.12.0"
    )
    
    if not success:
        print("\n Tentative avec version 4.8.0...")
        success = run_command(
            "pip install typing_extensions==4.8.0 --force-reinstall --no-cache-dir",
            "Installation typing_extensions 4.8.0"
        )
    
    # tape 4: Test immdiat
    print("\n tape 4: Test des imports...")
    try:
        import typing_extensions
        print(f" typing_extensions version: {typing_extensions.__version__}")
        
        # Test des fonctions critiques
        try:
            from typing_extensions import TypeAliasType
            print(" TypeAliasType disponible")
        except ImportError:
            print(" TypeAliasType manquant")
            
        try:
            from typing_extensions import Sentinel
            print(" Sentinel disponible")
        except ImportError:
            print(" Sentinel manquant")
            
        # Test FastAPI
        try:
            from fastapi import FastAPI
            print(" FastAPI import avec succs!")
            
            print("\n RPARATION TERMINE AVEC SUCCS!")
            print("\n Prochaines tapes:")
            print("   1. Redmarrer le serveur: python main_simple_for_frontend.py")
            print("   2. Tester l'API sur http://localhost:8000")
            
            return True
            
        except ImportError as e:
            print(f" FastAPI toujours en erreur: {e}")
            return False
            
    except ImportError as e:
        print(f" typing_extensions toujours en erreur: {e}")
        
        # Solution alternative : downgrade pydantic
        print("\n Solution alternative: downgrade pydantic...")
        run_command(
            "pip install pydantic==1.10.8 --force-reinstall",
            "Downgrade pydantic vers 1.10.8"
        )
        
        # Test final
        try:
            from fastapi import FastAPI
            print(" FastAPI fonctionne avec pydantic 1.10.8!")
            return True
        except ImportError as e:
            print(f" chec final: {e}")
            return False
if __name__ == "__main__":
    main()