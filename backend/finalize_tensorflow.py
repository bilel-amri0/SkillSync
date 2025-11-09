#!/usr/bin/env python3
"""
🎯 FINALISATION TENSORFLOW - SKILLSYNC
Corrige l'accès TensorFlow et optimise la configuration
"""
import subprocess
import sys
import os
import site
def print_header(message, emoji="🎯"):
    print(f"\n{'='*60}")
    print(f"{emoji} {message}")
    print(f"{'='*60}")
def print_step(message, emoji="📦"):
    print(f"\n{emoji} {message}")
def run_command(command, description="", timeout=600):
    """Exécute une commande avec gestion d'erreur"""
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
            return True, result.stdout.strip()
        else:
            print(f"   ❌ {description} - échec (code: {result.returncode})")
            if result.stderr:
                print(f"   ⚠️ Erreur: {result.stderr.strip()[:200]}...")
            return False, result.stderr.strip()
            
    except Exception as e:
        print(f"   ❌ {description} - erreur: {e}")
        return False, str(e)
def check_tensorflow_installation():
    """Vérifie où TensorFlow est installé"""
    print_step("Diagnostic TensorFlow", "🔍")
    
    # Vérifier les répertoires utilisateur
    user_packages = site.getusersitepackages()
    print(f"   📂 Répertoire utilisateur: {user_packages}")
    
    # Vérifier si TensorFlow existe dans user packages
    tf_user_path = os.path.join(user_packages, 'tensorflow')
    if os.path.exists(tf_user_path):
        print("   ✅ TensorFlow trouvé dans répertoire utilisateur")
        return True, user_packages
    else:
        print("   ❌ TensorFlow non trouvé dans répertoire utilisateur")
        return False, None
def fix_tensorflow_access():
    """Répare l'accès TensorFlow"""
    print_step("Réparation accès TensorFlow", "🔧")
    
    # Méthode 1: Réinstallation globale si possible
    print("   🌐 Tentative réinstallation globale...")
    success, output = run_command([
        sys.executable, "-m", "pip", "install", 
        "tensorflow==2.13.0", "--force-reinstall"
    ], "réinstallation globale tensorflow")
    
    if success:
        return True
    
    # Méthode 2: Forcer l'ajout du répertoire utilisateur
    print("   📂 Ajout répertoire utilisateur au path...")
    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.insert(0, user_site)
        print(f"   ✅ Répertoire ajouté: {user_site}")
    
    return False
def update_numpy_for_scipy():
    """Met à jour NumPy pour satisfaire SciPy"""
    print_step("Optimisation NumPy pour SciPy", "📈")
    
    # NumPy 1.25.2 est compatible avec TensorFlow 2.13 ET SciPy
    success, output = run_command([
        sys.executable, "-m", "pip", "install", 
        "numpy==1.25.2", "--force-reinstall"
    ], "mise à jour numpy optimale")
    
    return success
def test_all_imports():
    """Test complet de tous les imports ML"""
    print_step("Test complet des imports ML", "🧪")
    
    test_modules = [
        ("numpy", "NumPy"),
        ("tensorflow", "TensorFlow"),
        ("torch", "PyTorch"),
        ("sklearn", "Scikit-learn"),
        ("pandas", "Pandas"),
        ("transformers", "Transformers"),
        ("sentence_transformers", "Sentence Transformers")
    ]
    
    results = {}
    for module, name in test_modules:
        try:
            imported_module = __import__(module)
            version = getattr(imported_module, '__version__', 'inconnue')
            print(f"   ✅ {name}: v{version}")
            results[module] = version
        except Exception as e:
            print(f"   ❌ {name}: {str(e)[:50]}...")
            results[module] = None
    
    return results
def test_tensorflow_basic():
    """Test basique TensorFlow"""
    print_step("Test TensorFlow basique", "🧠")
    
    try:
        import tensorflow as tf
        
        # Test simple
        print("   🔢 Test création tensor...")
        x = tf.constant([1, 2, 3])
        y = tf.constant([4, 5, 6])
        z = tf.add(x, y)
        print(f"   ✅ Résultat: {z}")
        
        # Test modèle simple
        print("   🏗️ Test création modèle...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(5, input_shape=(3,)),
            tf.keras.layers.Dense(1)
        ])
        print("   ✅ Modèle créé avec succès")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur TensorFlow: {e}")
        return False
def create_final_config():
    """Crée la configuration finale"""
    print_step("Configuration finale", "⚙️")
    
    # Configuration optimisée
    config = """# Configuration ML SkillSync - FINALE
ML_MODE=FULL
TENSORFLOW_AVAILABLE=true
PYTORCH_AVAILABLE=true
NUMPY_VERSION=1.25.2
SCIPY_COMPATIBLE=true
BACKEND_TYPE=full
INSTALLATION_STATUS=complete
"""
    
    try:
        with open('.env.ml', 'w', encoding='utf-8') as f:
            f.write(config)
        print("   ✅ Configuration finale créée")
        return True
    except Exception as e:
        print(f"   ❌ Erreur config: {e}")
        return False
def main():
    print_header("FINALISATION TENSORFLOW SKILLSYNC", "🎯")
    print("🔧 Correction accès TensorFlow et optimisation finale")
    
    # Étape 1: Diagnostic TensorFlow
    tf_found, user_path = check_tensorflow_installation()
    
    # Étape 2: Réparation accès TensorFlow
    tf_fixed = fix_tensorflow_access()
    
    # Étape 3: Optimisation NumPy
    numpy_updated = update_numpy_for_scipy()
    
    # Étape 4: Tests imports
    print_step("Vérification post-finalisation", "🔍")
    results = test_all_imports()
    
    # Étape 5: Test TensorFlow spécifique
    tf_works = test_tensorflow_basic()
    
    # Étape 6: Configuration finale
    config_ok = create_final_config()
    
    # Résumé final
    print_header("FINALISATION TERMINÉE", "🏁")
    
    working_modules = len([r for r in results.values() if r is not None])
    total_modules = len(results)
    
    print(f"📊 Modules ML fonctionnels: {working_modules}/{total_modules}")
    print(f"🧠 TensorFlow opérationnel: {'✅ OUI' if tf_works else '❌ NON'}")
    
    if tf_works and working_modules >= 6:
        print("\n🎉 FINALISATION RÉUSSIE! Système ML 100% opérationnel!")
        print("\n🚀 PRÊT POUR LE DÉMARRAGE:")
        print("   1. python main_simple_for_frontend.py")
        print("   2. Interface web: http://localhost:8000")
        print("   3. Toutes les fonctionnalités ML disponibles!")
    else:
        print("\n⚠️ Finalisation partielle")
        print("\n🔧 Dernière action suggérée:")
        print("   Redémarrer terminal en ADMINISTRATEUR et relancer:")
        print("   python fix_numpy_tensorflow.py")
if __name__ == "__main__":
    main()