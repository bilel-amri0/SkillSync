#!/usr/bin/env python3
"""
🔧 SCRIPT DE RÉPARATION TENSORFLOW + NUMPY
Résout les conflits de versions et permissions Windows
"""
import subprocess
import sys
import os
import time
from pathlib import Path
def print_header(message, emoji="🔧"):
    print(f"\n{'='*60}")
    print(f"{emoji} {message}")
    print(f"{'='*60}")
def print_step(message, emoji="📦"):
    print(f"\n{emoji} {message}")
def run_command(command, description="", timeout=600):
    """Exécute une commande avec gestion d'erreur et timeout"""
    print(f"   🔧 Commande: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print(f"   ✅ {description} - succès!")
            if result.stdout.strip():
                print(f"   📄 Sortie: {result.stdout.strip()[:200]}...")
            return True
        else:
            print(f"   ❌ {description} - échec (code: {result.returncode})")
            if result.stderr:
                print(f"   ⚠️ Erreur: {result.stderr.strip()[:300]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ⏱️ {description} - timeout après {timeout}s")
        return False
    except Exception as e:
        print(f"   ❌ {description} - erreur: {e}")
        return False
def check_admin_rights():
    """Vérifie si le script s'exécute avec des droits administrateur"""
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False
def install_with_user_flag():
    """Essaie d'installer avec le flag --user pour éviter les permissions"""
    print_step("Installation avec flag --user (évite permissions)", "🔑")
    
    # Étape 1: Downgrade NumPy
    print("   📉 Étape 1: Downgrade NumPy vers version compatible")
    success = run_command([
        sys.executable, "-m", "pip", "install", 
        "numpy==1.24.3", "--user", "--force-reinstall"
    ], "downgrade numpy")
    
    if not success:
        return False
    
    # Étape 2: Installation TensorFlow
    print("   🧠 Étape 2: Installation TensorFlow avec NumPy compatible")
    success = run_command([
        sys.executable, "-m", "pip", "install", 
        "tensorflow==2.13.0", "--user"
    ], "tensorflow stable")
    
    return success
def fix_dependencies():
    """Répare les dépendances conflictuelles"""
    print_step("Réparation des dépendances conflictuelles", "🔨")
    
    # Packages à réinstaller avec versions compatibles
    compatible_packages = [
        "numpy==1.24.3",
        "pandas==1.5.3", 
        "scikit-learn==1.2.2",
        "matplotlib==3.7.2"
    ]
    
    for package in compatible_packages:
        print(f"   📦 Réinstallation: {package}")
        success = run_command([
            sys.executable, "-m", "pip", "install", 
            package, "--force-reinstall", "--no-deps"
        ], f"réparation {package}")
        
        if not success:
            print(f"   ⚠️ Échec réparation {package}, continuation...")
def test_imports():
    """Teste les imports critiques"""
    print_step("Test des imports critiques", "🧪")
    
    test_modules = [
        ("numpy", "NumPy"),
        ("tensorflow", "TensorFlow"),
        ("torch", "PyTorch"),
        ("sklearn", "Scikit-learn"),
        ("pandas", "Pandas")
    ]
    
    results = {}
    for module, name in test_modules:
        try:
            imported_module = __import__(module)
            version = getattr(imported_module, '__version__', 'inconnue')
            print(f"   ✅ {name}: v{version}")
            results[module] = True
        except Exception as e:
            print(f"   ❌ {name}: {str(e)[:100]}...")
            results[module] = False
    
    return results
def create_ml_config():
    """Crée la configuration ML appropriée"""
    print_step("Configuration du mode ML", "⚙️")
    
    # Créer fichier .env.ml
    ml_config = """# Configuration ML SkillSync
ML_MODE=HYBRID
TENSORFLOW_AVAILABLE=true
PYTORCH_AVAILABLE=true
NUMPY_VERSION=1.24.3
BACKEND_TYPE=hybrid
"""
    
    try:
        with open('.env.ml', 'w', encoding='utf-8') as f:
            f.write(ml_config)
        print("   ✅ Fichier .env.ml créé (mode HYBRID)")
        return True
    except Exception as e:
        print(f"   ❌ Erreur création config: {e}")
        return False
def main():
    print_header("RÉPARATION TENSORFLOW + NUMPY SKILLSYNC", "🚀")
    print("💪 Résolution des conflits de versions et permissions")
    
    # Vérification des droits admin
    if check_admin_rights():
        print("🔑 Droits administrateur détectés")
    else:
        print("⚠️ Pas de droits admin - utilisation flag --user")
    
    # Étape 1: Nettoyage et réparation des dépendances
    print_step("Nettoyage des packages conflictuels", "🧹")
    
    # Désinstaller les packages problématiques
    problematic_packages = ["tensorflow", "tensorflow-cpu"]
    for package in problematic_packages:
        print(f"   🗑️ Suppression: {package}")
        run_command([
            sys.executable, "-m", "pip", "uninstall", 
            package, "-y"
        ], f"suppression {package}")
    
    # Étape 2: Installation avec compatibilité
    success = install_with_user_flag()
    
    if not success:
        print_step("Tentative alternative avec versions spécifiques", "🔄")
        
        # Version alternative plus stable
        run_command([
            sys.executable, "-m", "pip", "install", 
            "tensorflow==2.12.0", "numpy==1.24.3", 
            "--force-reinstall", "--no-deps"
        ], "tensorflow version alternative")
    
    # Étape 3: Réparation des dépendances
    fix_dependencies()
    
    # Étape 4: Configuration ML
    create_ml_config()
    
    # Étape 5: Tests finaux
    print_step("Vérification finale", "🔍")
    results = test_imports()
    
    # Résumé
    print_header("RÉSUMÉ DE LA RÉPARATION", "📊")
    
    working_modules = sum(1 for working in results.values() if working)
    total_modules = len(results)
    
    print(f"📈 Modules fonctionnels: {working_modules}/{total_modules}")
    
    if results.get('tensorflow', False) and results.get('numpy', False):
        print("🎉 RÉPARATION RÉUSSIE! TensorFlow + NumPy opérationnels")
        print("\n🚀 Prochaines étapes:")
        print("   1. cd backend")
        print("   2. python test_ml_full.py")
        print("   3. python main_simple_for_frontend.py")
    else:
        print("⚠️ Réparation partielle - voir détails ci-dessus")
        print("\n🔧 Actions suggérées:")
        print("   1. Redémarrer le terminal en tant qu'administrateur")
        print("   2. Réexécuter ce script")
if __name__ == "__main__":
    main()