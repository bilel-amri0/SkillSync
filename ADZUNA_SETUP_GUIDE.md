🚀 Guide d'Installation Adzuna API pour Job Matching
📋 Étape 1: Inscription sur Adzuna Developer
1.
Allez sur le site Adzuna Developer:
URL: https://developer.adzuna.com/
Cliquez sur "Get API Key" ou "Sign Up"
2.
Créez votre compte:
Remplissez le formulaire d'inscription
Confirmez votre email
Connectez-vous à votre compte
3.
Obtenez vos identifiants API:
Une fois connecté, allez dans votre dashboard
Vous verrez vos identifiants:
App ID (ex: 12345678)
App Key (ex: abcdef123456789...)
📋 Étape 2: Configuration dans SkillSync
Option A: Modification du fichier .env (RECOMMANDÉ)
1.
Naviguez vers le dossier backend:
bash
cd /workspace/SkillSync_Project/backend
2.
Éditez le fichier .env:
bash
nano .env
# ou
code .env
3.
Remplacez les placeholders par vos vraies valeurs:
env
# ===== ADZUNA API CONFIGURATION =====
ADZUNA_APP_ID=votre_app_id_ici
ADZUNA_APP_KEY=votre_app_key_ici
Exemple concret:
env
ADZUNA_APP_ID=12345678
ADZUNA_APP_KEY=abcdef123456789abcdef123456789ab
Option B: Variables d'environnement système
Vous pouvez aussi définir les variables directement dans votre terminal:

bash
export ADZUNA_APP_ID="votre_app_id"
export ADZUNA_APP_KEY="votre_app_key"
📋 Étape 3: Test de la Configuration
1.
Redémarrez le serveur backend:
bash
cd /workspace/SkillSync_Project
python start_server.py
2.
Vérifiez les logs:
✅ Si configuré correctement: Aucun message d'erreur
❌ Si mal configuré: Vous verrez "⚠️ ADZUNA API credentials not found"
📋 Étape 4: Test de l'Endpoint Job Matching
Test avec curl:
bash
# Test avec skills directs
curl -X POST "http://localhost:8001/api/v1/jobs/search" \
     -H "Content-Type: application/json" \
     -d '{
       "skills": ["Python", "FastAPI", "React"],
       "location": "fr",
       "max_results": 10
     }'
# Test avec analysis_id existant
curl -X POST "http://localhost:8001/api/v1/jobs/search" \
     -H "Content-Type: application/json" \
     -d '{
       "analysis_id": "your-analysis-id-here",
       "location": "fr",
       "max_results": 10
     }'
Réponse attendue:
json
{
  "success": true,
  "total_jobs": 10,
  "search_parameters": {
    "skills_count": 3,
    "location": "fr",
    "max_results": 10
  },
  "jobs": [
    {
      "job_id": "123456",
      "title": "Développeur Python",
      "company": "TechCorp",
      "location": "Paris, France",
      "salary_min": 40000,
      "salary_max": 60000,
      "description": "Recherche développeur Python...",
      "url": "https://www.adzuna.fr/jobs/...",
      "match_score": 85.5,
      "matching_skills": ["Python", "FastAPI"],
      "created_date": "2025-01-15T10:30:00"
    }
  ]
}
🚨 Résolution des Problèmes Courants
Erreur: "ADZUNA API credentials not found"
✅ Vérifiez que le fichier .env existe dans /workspace/SkillSync_Project/backend/
✅ Vérifiez que les variables sont bien définies sans espaces
✅ Redémarrez le serveur après modification
Erreur: "HTTP error calling Adzuna API"
✅ Vérifiez que vos identifiants API sont corrects
✅ Vérifiez votre connexion internet
✅ Vérifiez les limites de votre compte Adzuna (gratuit = 1000 requêtes/mois)
Erreur: "No skills provided or found in analysis"
✅ Assurez-vous d'envoyer soit skills soit analysis_id valide
✅ Si vous utilisez analysis_id, vérifiez qu'il existe dans la base
📊 Limites API Adzuna (Compte Gratuit)
1000 requêtes par mois
Maximum 50 résultats par requête
Pays supportés: UK, US, AU, CA, FR, DE, etc.
🔄 Prochaines Étapes
Une fois configuré, vous pourrez:

1.
✅ Intégrer l'endpoint dans le frontend React
2.
✅ Créer une page Job Matching dans l'interface
3.
✅ Afficher les résultats avec scoring des compétences
📁 Fichiers Créés/Modifiés
✅ /workspace/SkillSync_Project/backend/job_matching_service.py - Service Adzuna
✅ /workspace/SkillSync_Project/backend/config.py - Configuration étendue
✅ /workspace/SkillSync_Project/backend/.env - Variables d'environnement
✅ /workspace/SkillSync_Project/backend/main.py - Nouvel endpoint /api/v1/jobs/search
✅ Configuration terminée ! Vous êtes prêt pour le Job Matching ! 🚀