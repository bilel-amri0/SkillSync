# 🎯 SkillSync - Guide Rapide

## 🚀 Démarrage Ultra-Rapide

### Option 1: Script Automatique
```bash
python start_project.py
```

### Option 2: Manuel (2 terminaux)

**Terminal 1 - Backend:**
```bash
cd backend
python main_simple_for_frontend.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

## 🌐 Accès
- **Application:** http://localhost:3000
- **API Backend:** http://localhost:8001

## 📋 Test Rapide
1. ✅ Ouvrir http://localhost:3000
2. ✅ Aller sur "Upload CV" 
3. ✅ Télécharger un CV PDF
4. ✅ Visiter "Recommendations" → Voir les recommandations personnalisées
5. ✅ Tester "Job Matching" → Voir les emplois (mode démo)
6. ✅ Essayer "Portfolio" → Générer un portfolio

## 🐛 Dépannage

### Recommandations vides:
- Vider le cache navigateur: `Ctrl+Shift+R`
- Vérifier la console (F12) pour les erreurs

### Port occupé:
- Backend: Changer le port dans `main_simple_for_frontend.py`
- Frontend: `PORT=3001 npm start`

### Erreurs de dépendances:
```bash
cd backend
pip install -r requirements.txt

cd ../frontend
npm install
```

## 📁 Fichiers Essentiels

**Ne pas supprimer:**
- `backend/main_simple_for_frontend.py` (serveur principal)
- `backend/recommendation_engine.py`
- `frontend/src/pages/Recommendations.js`
- `backend/requirements.txt`
- `frontend/package.json`

**Peut être supprimé:** Voir `CLEANUP_GUIDE.md`

## 🔧 API Endpoints
- `POST /api/v1/upload-cv` - Analyser CV
- `GET /api/v1/recommendations/{analysis_id}` - Recommandations
- `POST /api/v1/jobs/search` - Recherche d'emplois
- `POST /api/v1/generate-portfolio` - Générer portfolio

## 📊 Statut Fonctionnalités
- ✅ Upload et analyse CV
- ✅ Recommandations personnalisées 
- ✅ Recherche d'emplois (API + démo)
- ✅ Génération de portfolio
- ✅ Dashboard analytique

**Le projet est maintenant 100% fonctionnel!** 🎉
