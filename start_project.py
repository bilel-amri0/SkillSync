#!/usr/bin/env python3
"""
🚀 SkillSync - Lanceur de Projet Simple
Démarre automatiquement le backend et ouvre le frontend
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def check_requirements():
    """Vérifie que les dépendances sont installées"""
    try:
        import uvicorn
        import fastapi
        print("✅ Dépendances Python OK")
    except ImportError:
        print("❌ Erreur: Installez les dépendances avec 'pip install -r backend/requirements.txt'")
        sys.exit(1)

def start_backend():
    """Démarre le serveur backend"""
    backend_path = Path(__file__).parent / "backend"
    main_file = backend_path / "main_simple_for_frontend.py"
    
    if not main_file.exists():
        print(f"❌ Erreur: {main_file} non trouvé")
        sys.exit(1)
    
    print("🚀 Démarrage du backend...")
    cmd = [sys.executable, str(main_file)]
    return subprocess.Popen(cmd, cwd=str(backend_path))

def start_frontend():
    """Démarre le serveur frontend"""
    frontend_path = Path(__file__).parent / "frontend"
    
    if not frontend_path.exists():
        print(f"❌ Erreur: Dossier {frontend_path} non trouvé")
        sys.exit(1)
    
    print("⚛️ Démarrage du frontend...")
    cmd = ["npm", "start"]
    return subprocess.Popen(cmd, cwd=str(frontend_path), shell=True)

def main():
    print("🎯 SkillSync - Démarrage du projet complet\n")
    
    # Vérifications
    check_requirements()
    
    try:
        # Démarrer backend
        backend_process = start_backend()
        
        # Attendre que le backend démarre
        print("⏳ Attente du démarrage du backend...")
        time.sleep(5)
        
        # Démarrer frontend
        frontend_process = start_frontend()
        
        # Attendre que le frontend démarre
        print("⏳ Attente du démarrage du frontend...")
        time.sleep(10)
        
        # Ouvrir le navigateur
        print("🌐 Ouverture du navigateur...")
        webbrowser.open("http://localhost:3000")
        
        print("\n✅ Projet démarré avec succès!")
        print("📱 Frontend: http://localhost:3000")
        print("🔧 Backend API: http://localhost:8001")
        print("\n⚠️ Pour arrêter: Ctrl+C dans chaque terminal\n")
        
        # Garder le script actif
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Arrêt du projet...")
            backend_process.terminate()
            frontend_process.terminate()
            
    except Exception as e:
        print(f"❌ Erreur lors du démarrage: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
