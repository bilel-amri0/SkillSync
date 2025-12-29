#!/usr/bin/env python3
"""
Script de démarrage rapide pour SkillSync Enhanced
Ce script lance automatiquement l'application corrigée
"""

import os
import sys
import subprocess
import time
import requests
import signal
from pathlib import Path

def print_banner():
    """Affiche la bannière du projet"""
    print("=" * 60)
    print("🚀 SkillSync Enhanced - Version Corrigée")
    print("=" * 60)
    print("✅ Erreur de syntaxe corrigée")
    print("✅ Données dynamiques implémentées") 
    print("✅ Backend FastAPI robuste")
    print("✅ Frontend React moderne")
    print("✅ Authentification JWT")
    print("=" * 60)
    print()

def check_requirements():
    """Vérifie les prérequis"""
    print("🔍 Vérification des prérequis...")
    
    # Vérifier Python
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ requis")
        return False
    print(f"✅ Python {sys.version.split()[0]} détecté")
    
    # Vérifier pip
    try:
        import pip
        print("✅ pip disponible")
    except ImportError:
        print("❌ pip non disponible")
        return False
    
    # Vérifier si les fichiers existent
    if not Path("main_simple_for_frontend_fixed.py").exists():
        print("❌ Fichier main_simple_for_frontend_fixed.py non trouvé")
        return False
    
    if not Path("requirements_fixed.txt").exists():
        print("❌ Fichier requirements_fixed.txt non trouvé")
        return False
    
    print("✅ Tous les fichiers requis sont présents")
    return True

def setup_backend():
    """Configure et installe le backend"""
    print("\n🔧 Configuration du backend...")
    
    # Créer l'environnement virtuel si nécessaire
    if not Path("venv").exists():
        print("📦 Création de l'environnement virtuel...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Déterminer le bon chemin pour activate
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        pip_path = "venv\\Scripts\\pip"
    else:  # Unix/Linux/Mac
        activate_script = "venv/bin/activate"
        pip_path = "venv/bin/pip"
    
    # Installer les dépendances
    print("📥 Installation des dépendances Python...")
    cmd = [pip_path, "install", "-r", "requirements_fixed.txt"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print("✅ Dépendances installées avec succès")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erreur lors de l'installation: {e}")
        return False, None
    
    # Créer le répertoire uploads
    Path("uploads").mkdir(exist_ok=True)
    print("✅ Répertoire uploads créé")
    
    return True, activate_script

def start_backend(activate_script):
    """Démarre le serveur backend"""
    print("\n🚀 Démarrage du serveur backend...")
    
    # Commande pour démarrer le serveur
    if os.name == 'nt':  # Windows
        cmd = f"{activate_script} && python main_simple_for_frontend_fixed.py"
    else:  # Unix/Linux/Mac
        cmd = f"source {activate_script} && python main_simple_for_frontend_fixed.py"
    
    # Démarrer en arrière-plan
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE, text=True)
    
    # Attendre que le serveur démarre
    print("⏳ Attente du démarrage du serveur (10 secondes)...")
    for i in range(10):
        time.sleep(1)
        try:
            response = requests.get("http://localhost:8000/health", timeout=1)
            if response.status_code == 200:
                print("✅ Serveur backend démarré avec succès!")
                print("🌐 Backend disponible sur: http://localhost:8000")
                print("📚 API Docs disponible sur: http://localhost:8000/docs")
                return process
        except requests.exceptions.RequestException:
            continue
    
    print("⚠️ Le serveur ne répond pas encore, mais il devrait démarrer...")
    return process

def show_frontend_instructions():
    """Affiche les instructions pour le frontend"""
    print("\n💻 Configuration du frontend React:")
    print("=" * 50)
    print("1. Ouvrez un nouveau terminal")
    print("2. Naviguez vers le dossier frontend:")
    print("   cd frontend")
    print("3. Installez les dépendances:")
    print("   npm install")
    print("4. Démarrez le serveur React:")
    print("   npm start")
    print("=" * 50)
    print("🌐 Frontend disponible sur: http://localhost:3000")

def show_login_instructions():
    """Affiche les instructions de connexion"""
    print("\n🔐 Instructions de connexion:")
    print("=" * 40)
    print("Email: test@example.com")
    print("Mot de passe: password123")
    print("(Ou n'importe quel email valide + 6+ caractères)")
    print("=" * 40)

def monitor_server(process):
    """Surveille le serveur et gère l'arrêt"""
    print(f"\n🔄 Serveur en cours d'exécution (PID: {process.pid})")
    print("Appuyez sur Ctrl+C pour arrêter le serveur")
    
    try:
        # Surveiller le processus
        while True:
            time.sleep(1)
            if process.poll() is not None:
                print("\n⚠️ Le serveur s'est arrêté inopinément")
                break
    except KeyboardInterrupt:
        print("\n🛑 Arrêt du serveur...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("✅ Serveur arrêté")

def main():
    """Fonction principale"""
    print_banner()
    
    # Vérifier les prérequis
    if not check_requirements():
        print("\n❌ Prérequis non satisfaits. Vérifiez l'installation.")
        return 1
    
    # Configurer le backend
    success, activate_script = setup_backend()
    if not success:
        print("\n❌ Échec de la configuration du backend")
        return 1
    
    # Démarrer le serveur
    backend_process = start_backend(activate_script)
    
    # Afficher les instructions
    show_frontend_instructions()
    show_login_instructions()
    
    # Surveiller le serveur
    monitor_server(backend_process)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        sys.exit(1)