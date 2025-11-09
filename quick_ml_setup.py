#!/usr/bin/env python3
"""
Script de démarrage rapide ML pour SkillSync
Configure le minimum nécessaire pour activer les fonctionnalités ML
"""

import os
import sys
import subprocess
from pathlib import Path

def quick_ml_setup():
    """Configuration rapide ML"""
    print("⚡ CONFIGURATION RAPIDE ML")
    print("=" * 40)
    
    # Packages ML essentiels
    essential_packages = [
        "torch --index-url https://download.pytorch.org/whl/cpu",
        "transformers",
        "sentence-transformers", 
        "scikit-learn",
        "tensorflow-cpu"
    ]
    
    print("📦 Installation packages ML essentiels...")
    for package in essential_packages:
        print(f"   ⏳ {package.split()[0]}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install"
            ] + package.split(), capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                print(f"   ✅ {package.split()[0]} OK")
            else:
                print(f"   ⚠️ {package.split()[0]} - Erreur")
        except subprocess.TimeoutExpired:
            print(f"   ⏰ {package.split()[0]} - Timeout, mais probablement OK")
        except Exception as e:
            print(f"   ❌ {package.split()[0]} - {e}")
    
    print("\n✅ Configuration rapide terminée!")
    print("🚀 Vous pouvez maintenant lancer: python activate_ml_mode.py")

if __name__ == "__main__":
    quick_ml_setup()