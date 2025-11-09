#!/usr/bin/env python3
"""
SkillSync Server Launcher - Version Finale
Lance le serveur FastAPI avec configuration optimisée
Author: MiniMax Agent
"""

import os
import sys
from dotenv import load_dotenv

def main():
    """Lance le serveur SkillSync"""
    
    # Charge les variables d'environnement
    if os.path.exists('.env'):
        load_dotenv()
        print("✅ Variables d'environnement chargées depuis .env")
    else:
        print("⚠️  Fichier .env non trouvé - utilisation des variables système")
    
    # Vérifie les APIs configurées
    apis = {
        'LINKEDIN': os.getenv('LINKEDIN_RAPIDAPI_KEY'),
        'JSEARCH': os.getenv('JSEARCH_RAPIDAPI_KEY'),
        'MUSE': os.getenv('THE_MUSE_API_KEY'),
        'FINDWORK': os.getenv('FINDWORK_API_KEY'),
        'ADZUNA': os.getenv('ADZUNA_APP_ID') and os.getenv('ADZUNA_APP_KEY')
    }
    
    configured_count = sum(1 for api, key in apis.items() if key)
    
    print("🚀 SkillSync Multi-API Backend")
    print("=" * 50)
    print(f"🔍 APIs configurées: {configured_count}/5")
    
    for api, key in apis.items():
        status = "✅" if key else "❌"
        print(f"   {status} {api}")
    
    if configured_count == 0:
        print("\n❌ Aucune API configurée ! Vérifiez votre fichier .env")
        return
    
    print(f"\n🌐 Serveur disponible sur: http://127.0.0.1:8001")
    print(f"📖 Documentation: http://127.0.0.1:8001/docs")
    print(f"🔧 Health check: http://127.0.0.1:8001/health")
    print("\n⏹️  Appuyez sur Ctrl+C pour arrêter\n")
    
    # Lance le serveur
    try:
        import uvicorn
        uvicorn.run(
            "main_final:app",
            host="127.0.0.1",
            port=8001,
            reload=True,
            log_level="info"
        )
    except ImportError:
        print("❌ uvicorn non installé. Installation: pip install uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Serveur arrêté")
    except Exception as e:
        print(f"❌ Erreur serveur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
