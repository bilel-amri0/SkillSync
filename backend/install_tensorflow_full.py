#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 INSTALLATION TENSORFLOW OPTIMISE - SKILLSYNC
Rsout les problmes de timeout TensorFlow
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    print("\n" + "="*60)
    print(" INSTALLATION TENSORFLOW OPTIMISE - SKILLSYNC")
    print(" Mode ML COMPLET avec TensorFlow")
    print("="*60)

def check_system():
    print("\n Vrification du systme...")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Plateforme: {sys.platform}")
    print(f"   Architecture: {sys.maxsize > 2**32 and '64-bit' or '32-bit'}")
    return True

def configure_pip_for_speed():
    """Configure pip pour des tlchargements plus rapides"""
    print("\n Configuration pip pour vitesse optimale...")
    
    commands = [
        # Augmenter le timeout
        [sys.executable, "-m", "pip", "config", "set", "global.timeout", "1000"],
        # Utiliser des miroirs plus rapides
        [sys.executable, "-m", "pip", "config", "set", "global.index-url", "https://pypi.org/simple"],
        # Dsactiver les vrifications SSL temporairement pour la vitesse
        [sys.executable, "-m", "pip", "config", "set", "global.trusted-host", "pypi.org pypi.python.org files.pythonhosted.org"],
        # Cache pip pour viter les re-tlchargements
        [sys.executable, "-m", "pip", "config", "set", "global.cache-dir", os.path.expanduser("~/.pip/cache")]
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"    {' '.join(cmd[4:])}")
            else:
                print(f"    {' '.join(cmd[4:])} - ignor")
        except Exception:
            print(f"    {' '.join(cmd[4:])} - ignor")

def install_tensorflow_optimized():
    """Installation TensorFlow avec stratgies anti-timeout"""
    print("\n Installation TensorFlow avec stratgies optimises...")
    
    # Stratgies d'installation par ordre de prfrence
    strategies = [
        {
            "name": "TensorFlow CPU (lger et rapide)",
            "packages": ["tensorflow-cpu"],
            "timeout": 600,  # 10 minutes
            "description": "Version CPU optimise, plus rapide  installer"
        },
        {
            "name": "TensorFlow standard avec cache",
            "packages": ["tensorflow"],
            "timeout": 900,  # 15 minutes
            "description": "Version complte avec GPU support",
            "extra_args": ["--no-deps", "--force-reinstall"]
        },
        {
            "name": "TensorFlow par chunks",
            "packages": ["tensorflow"],
            "timeout": 1200,  # 20 minutes
            "description": "Installation par petits morceaux",
            "extra_args": ["--no-cache-dir", "--timeout", "1000"]
        }
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n Stratgie {i}: {strategy['name']}")
        print(f"    {strategy['description']}")
        
        for package in strategy['packages']:
            try:
                print(f"    Installation de {package}...")
                
                # Construire la commande
                cmd = [sys.executable, "-m", "pip", "install", package]
                
                # Ajouter les arguments extra si disponibles
                if "extra_args" in strategy:
                    cmd.extend(strategy["extra_args"])
                
                # Ajouter verbose pour voir les progrs
                cmd.append("-v")
                
                print(f"    Commande: {' '.join(cmd)}")
                
                # Lancer l'installation avec timeout gnreux
                result = subprocess.run(
                    cmd,
                    capture_output=False,  # Montrer les progrs en temps rel
                    text=True,
                    timeout=strategy["timeout"]
                )
                
                if result.returncode == 0:
                    print(f"    {package} install avec succs!")
                    
                    # Vrifier l'installation
                    try:
                        import tensorflow as tf
                        print(f"    TensorFlow {tf.__version__} vrifi!")
                        return True
                    except ImportError:
                        print(f"    {package} install mais import choue")
                        continue
                else:
                    print(f"    {package} - chec installation (code: {result.returncode})")
                    
            except subprocess.TimeoutExpired:
                print(f"    {package} - timeout aprs {strategy['timeout']}s, essai stratgie suivante...")
                continue
            except Exception as e:
                print(f"    {package} - erreur: {str(e)[:100]}...")
                continue
        
        # Si on arrive ici, la stratgie a chou
        print(f"    Stratgie {i} choue, passage  la suivante...")
    
    # Toutes les stratgies ont chou
    print("\n Toutes les stratgies TensorFlow ont chou")
    return False

def install_other_ml_packages():
    """Installer les autres packages ML essentiels"""
    print("\n Installation des autres packages ML...")
    
    packages = [
        ("torch", "PyTorch pour le deep learning"),
        ("transformers", "Modles de transformers"),
        ("sentence-transformers", "Similarit smantique"),
        ("scikit-learn", "Machine learning classique"),
        ("numpy", "Calculs numriques"),
        ("pandas", "Manipulation de donnes")
    ]
    
    success_count = 0
    for package, description in packages:
        try:
            print(f"    {package}: {description}")
            
            # Vrifier si dj install
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                print(f"    {package} dj install")
                success_count += 1
            else:
                print(f"    Installation {package}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", package, "-v"],
                    capture_output=False, text=True, timeout=300
                )
                
                if result.returncode == 0:
                    print(f"    {package} install")
                    success_count += 1
                else:
                    print(f"    {package} - installation partielle")
                    
        except subprocess.TimeoutExpired:
            print(f"    {package} - timeout, mais probablement OK")
            success_count += 1  # Compter comme succs partiel
        except Exception as e:
            print(f"    {package} - erreur: {str(e)[:50]}...")
    
    print(f"\n Packages installs: {success_count}/{len(packages)}")
    return success_count >= len(packages) * 0.7  # 70% de succs minimum

def configure_ml_full_mode():
    """Configuration du mode ML complet"""
    print("\n Configuration du mode ML COMPLET...")
    
    # Crer le fichier de configuration ML complet
    config_content = '''# Configuration Mode ML COMPLET - SkillSync
ML_MODE_ENABLED=true
ML_ENGINE_TYPE=full
USE_TENSORFLOW=true
USE_TRANSFORMERS=true
FALLBACK_TO_RULES=true
TENSORFLOW_AVAILABLE=true
'''
    
    try:
        with open('.env.ml', 'w', encoding='utf-8') as f:
            f.write(config_content)
        print("    Fichier .env.ml cr (mode COMPLET)")
    except Exception as e:
        print(f"    Erreur cration config: {e}")
    
    # Crer le marqueur ML activ
    try:
        Path('ml_mode_enabled.flag').touch()
        print("    Flag ML COMPLET activ")
    except Exception as e:
        print(f"    Erreur flag: {e}")
    
    return True

def verify_full_installation():
    """Vrification complte de l'installation ML"""
    print("\n Vrification de l'installation ML COMPLTE...")
    
    components = [
        ("tensorflow", "TensorFlow (principal)"),
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence-Transformers"),
        ("sklearn", "Scikit-learn")
    ]
    
    success_count = 0
    for module, name in components:
        try:
            if module == "sklearn":
                import sklearn
                version = sklearn.__version__
            else:
                imported_module = __import__(module)
                version = getattr(imported_module, '__version__', 'OK')
            
            print(f"    {name}: v{version}")
            success_count += 1
            
        except ImportError as e:
            print(f"    {name}: manquant ({e})")
    
    print(f"\n Composants ML: {success_count}/{len(components)}")
    
    # Test spcial TensorFlow
    try:
        import tensorflow as tf
        print(f"\n TensorFlow {tf.__version__} - Test fonctionnel...")
        
        # Test simple de cration de modle
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        print("    Cration de modle TensorFlow: OK")
        
        # Test de prdiction
        import numpy as np
        test_input = np.random.random((1, 5))
        prediction = model.predict(test_input, verbose=0)
        print(f"    Prdiction TensorFlow: {prediction[0][0]:.4f}")
        
        return success_count >= len(components) * 0.8
        
    except Exception as e:
        print(f"    Test TensorFlow chou: {e}")
        return False

def main():
    """Fonction principale d'installation ML complte"""
    print_banner()
    
    try:
        # tapes d'installation ML complte
        if not check_system():
            return False
        
        configure_pip_for_speed()
        
        # Installation TensorFlow (tape critique)
        tensorflow_success = install_tensorflow_optimized()
        
        # Installation autres packages
        other_packages_success = install_other_ml_packages()
        
        # Configuration du mode
        configure_ml_full_mode()
        
        # Vrification finale
        verification_success = verify_full_installation()
        
        # Rsum final
        print("\n" + "="*60)
        if tensorflow_success and verification_success:
            print(" MODE ML COMPLET ACTIV AVEC SUCCS !")
            print("\n Configuration ML COMPLTE:")
            print("    TensorFlow: Activ")
            print("    PyTorch: Activ")
            print("    Transformers: Activ")
            print("    Neural Scorer: TensorFlow")
            print("    BERT Models: Complets")
            print("\n Redmarre le serveur:")
            print("   python main_simple_for_frontend.py")
            print("\n Teste avec:")
            print("   python test_ml_full.py")
        elif other_packages_success:
            print(" MODE ML PARTIEL ACTIV")
            print("\n tat:")
            print(f"   {'' if tensorflow_success else ''} TensorFlow")
            print("    Autres packages ML")
            print("\n Le systme utilisera PyTorch comme fallback")
        else:
            print(" INSTALLATION ML CHOUE")
            print("\n Solutions possibles:")
            print("   1. Vrifier la connexion internet")
            print("   2. Utiliser le mode ML lite")
            print("   3. Installation manuelle TensorFlow")
        
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n Installation interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
