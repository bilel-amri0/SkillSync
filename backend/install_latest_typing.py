#!/usr/bin/env python3
"""
 Installation version LATEST de typing_extensions
=================================================
Solution finale pour avoir Sentinel + TypeAliasType
"""
import subprocess
import sys
def run_command(cmd, description):
    """Excute une commande avec gestion d'erreur"""
    print(f"\n {description}...")
    print(f"   Commande: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        
        if result.stdout:
            print(f" Sortie:\n{result.stdout}")
        if result.stderr and result.returncode != 0:
            print(f" Erreurs:\n{result.stderr}")
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f" Timeout aprs 120 secondes")
        return False
    except Exception as e:
        print(f" Erreur: {e}")
        return False
def main():
    print(" INSTALLATION VERSION LATEST")
    print("=" * 40)
    
    # tape 1: Installer la version la plus rcente disponible
    print("\n Installation typing_extensions LATEST...")
    success = run_command(
        "pip install typing_extensions --upgrade --force-reinstall --no-cache-dir",
        "Installation typing_extensions LATEST"
    )
    
    if not success:
        print("\n Tentative version spcifique 4.9.0...")
        success = run_command(
            "pip install typing_extensions==4.9.0 --force-reinstall --no-cache-dir",
            "Installation typing_extensions 4.9.0"
        )
    
    # tape 2: Test complet immdiat
    print("\n TEST COMPLET...")
    try:
        import importlib
        
        # Forcer le rechargement du module
        if 'typing_extensions' in sys.modules:
            importlib.reload(sys.modules['typing_extensions'])
        
        import typing_extensions
        print(f" typing_extensions recharg")
        
        # Test TypeAliasType
        try:
            from typing_extensions import TypeAliasType
            print(" TypeAliasType disponible")
        except ImportError as e:
            print(f" TypeAliasType: {e}")
            
        # Test Sentinel
        try:
            from typing_extensions import Sentinel
            print(" Sentinel disponible")
        except ImportError as e:
            print(f" Sentinel: {e}")
            
        # Test pydantic_core
        try:
            # Forcer rechargement pydantic_core aussi
            if 'pydantic_core' in sys.modules:
                importlib.reload(sys.modules['pydantic_core'])
            
            import pydantic_core
            print(" pydantic_core import")
        except ImportError as e:
            print(f" pydantic_core: {e}")
            return False
            
        # Test FastAPI
        try:
            # Recharger pydantic et FastAPI
            for module in ['pydantic', 'fastapi']:
                if module in sys.modules:
                    del sys.modules[module]
            
            from fastapi import FastAPI
            print(" FastAPI import!")
            
            print("\n SUCCS COMPLET!")
            print("\n PROCHAINES TAPES:")
            print("   1. Redmarrer Python (important!)")
            print("   2. Relancer: python main_simple_for_frontend.py")
            print("   3. Tester: http://localhost:8000")
            
            return True
            
        except ImportError as e:
            print(f" FastAPI: {e}")
            return False
            
    except Exception as e:
        print(f" Erreur gnrale: {e}")
        return False
if __name__ == "__main__":
    main()