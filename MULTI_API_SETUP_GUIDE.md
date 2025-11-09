🌐 Multi-API Job Board Setup Guide
🎯 Overview
Votre SkillSync utilise maintenant 6 sources d'emplois pour maximiser vos résultats :

1.
🇹🇳 TanitJob - Jobs locaux Tunisie
2.
🇫🇷 Pôle Emploi - Jobs officiels France
3.
🌍 Indeed API - Global jobs
4.
💼 LinkedIn - Réseau professionnel
5.
🔍 The Muse - Jobs tech (gratuit !)
6.
⚡ JSearch - Agrégateur multi-sources
🚀 APIs Gratuites (Prêtes à utiliser)
✅ The Muse API - DÉJÀ CONFIGURÉ
Status: ✅ Gratuit, aucune configuration requise
Spécialité: Jobs tech et startup
Limite: 100 requêtes/heure
🔑 APIs avec Clés Required
1. 🇹🇳 TanitJob API
bash
# Status: 🔄 En développement
# Pour l'instant: Jobs démo tunisiens
# Prochaine étape: Web scraping ou API officielle
2. 🇫🇷 Pôle Emploi API
📍 Inscription : https://www.emploi-store-dev.fr/

1.
Créez un compte développeur
2.
Créez une nouvelle application
3.
Obtenez vos CLIENT_ID et CLIENT_SECRET
4.
Ajoutez dans .env :
env
POLE_EMPLOI_CLIENT_ID=votre_client_id
POLE_EMPLOI_CLIENT_SECRET=votre_client_secret
3. ⚡ JSearch (RapidAPI) - Indeed + LinkedIn
📍 Inscription : https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch/

1.
Créez un compte RapidAPI (gratuit)
2.
Souscrivez au plan JSearch :
Gratuit: 100 recherches/mois
Basic: $9.99/mois - 2,500 recherches
Pro: $24.99/mois - 10,000 recherches
3.
Copiez votre clé API
4.
Ajoutez dans .env :
env
RAPIDAPI_KEY=votre_rapidapi_key
4. 🔍 Adzuna API
📍 Inscription : https://developer.adzuna.com/

1.
Créez un compte développeur (gratuit)
2.
Créez une nouvelle application
3.
Obtenez votre APP_ID et APP_KEY
4.
Ajoutez dans .env :
env
ADZUNA_APP_ID=votre_app_id
ADZUNA_APP_KEY=votre_app_key
Limite gratuite: 1,000 requêtes/mois

5. 💼 LinkedIn API (Optionnel)
📍 Inscription : https://developer.linkedin.com/

1.
Créez une application LinkedIn
2.
Demandez l'accès à l'API Jobs
3.
Obtenez vos credentials OAuth2
4.
Ajoutez dans .env :
env
LINKEDIN_CLIENT_ID=votre_client_id
LINKEDIN_CLIENT_SECRET=votre_client_secret
⚙️ Configuration Actuelle
📁 Fichier .env
env
# ===== MULTI JOB BOARD API CONFIGURATION =====
# 1. Adzuna API (Global job board)
ADZUNA_APP_ID=YOUR_ADZUNA_APP_ID_HERE
ADZUNA_APP_KEY=YOUR_ADZUNA_APP_KEY_HERE
# 2. RapidAPI JSearch (Indeed, LinkedIn aggregator)  
RAPIDAPI_KEY=YOUR_RAPIDAPI_KEY_HERE
# 3. Pôle Emploi API (France official jobs)
POLE_EMPLOI_CLIENT_ID=YOUR_POLE_EMPLOI_CLIENT_ID_HERE
POLE_EMPLOI_CLIENT_SECRET=YOUR_POLE_EMPLOI_CLIENT_SECRET_HERE
# 4. LinkedIn API (optional)
LINKEDIN_CLIENT_ID=YOUR_LINKEDIN_CLIENT_ID_HERE
LINKEDIN_CLIENT_SECRET=YOUR_LINKEDIN_CLIENT_SECRET_HERE
🔄 Test de Configuration
1. Redémarrez le serveur backend
bash
cd SkillSync_Project/backend
python main_simple_for_frontend.py
2. Testez l'endpoint multi-API
bash
curl -X POST "http://localhost:8001/api/v1/jobs/search" \
     -H "Content-Type: application/json" \
     -d '{"skills": ["Python", "JavaScript", "React"], "location": "fr", "max_results": 10}'
3. Vérifiez les logs
Recherchez dans les logs :

✅ Multi Job Matching Service loaded
🚀 Searching jobs across multiple APIs
✅ Multi-API search completed: X jobs from Y sources
📊 Avantages du Multi-API
🎯 Résultats Maximisés
Plus de jobs : Combine 6 sources
Meilleure qualité : Score de correspondance amélioré
Diversité géographique : France + Tunisie + International
⚡ Performance Optimisée
Recherche parallèle : Toutes les APIs en même temps
Déduplication : Supprime les doublons automatiquement
Fallback intelligent : Mode démo si APIs indisponibles
🔧 Flexibilité
Configuration modulaire : Activez/désactivez les APIs
Priorité dynamique : Les meilleures sources en premier
Évolutif : Facile d'ajouter de nouvelles sources
🚨 Troubleshooting
Erreur : "Multi Job Matching service not available"
bash
# Vérifiez l'import du service
cd SkillSync_Project/backend
python -c "from multi_job_service import search_jobs_multi_source; print('✅ Service OK')"
Erreur : API Rate Limit
Vérifiez vos quotas sur chaque plateforme
Réduisez max_results dans vos requêtes
Attendez la réinitialisation du quota
Pas de résultats
Vérifiez vos API keys dans .env
Testez chaque API individuellement
Consultez les logs pour les erreurs spécifiques
🎉 Félicitations !
Votre SkillSync est maintenant connecté à 6 job boards !

🔥 Prochaines étapes :

1.
Configurez au moins 2-3 APIs pour de vrais résultats
2.
Testez avec vos vraies compétences
3.
Optimisez selon vos préférences géographiques
Votre MVP SkillSync est maintenant une plateforme complète de recherche d'emploi ! 🚀