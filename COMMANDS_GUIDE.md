# 🚀 SkillSync - Guide des Commandes

## 📋 Table des Matières
1. [Installation Initiale](#installation-initiale)
2. [Démarrage du Projet](#démarrage-du-projet)
3. [Commandes Backend](#commandes-backend)
4. [Commandes Frontend](#commandes-frontend)
5. [Tests](#tests)
6. [Maintenance](#maintenance)
7. [Déploiement](#déploiement)

---

## 🔧 Installation Initiale

### 1. Cloner le Projet
```bash
git clone https://github.com/bilel-amri0/SkillSync.git
cd SkillSync
```

### 2. Installer les Dépendances Backend
```bash
# Naviguer vers le dossier backend
cd backend

# Créer un environnement virtuel Python
python -m venv venv

# Activer l'environnement virtuel
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Installer les packages Python
pip install -r requirements.txt

# Retourner à la racine
cd ..
```

### 3. Installer les Dépendances Frontend
```bash
# Naviguer vers le dossier frontend
cd frontend

# Installer les packages npm
npm install

# Retourner à la racine
cd ..
```

### 4. Configuration de l'Environnement
```bash
# Créer le fichier .env dans le dossier backend
cd backend
copy .env.example .env  # Windows
# OU
cp .env.example .env    # Linux/Mac

# Éditer le fichier .env avec vos clés API
notepad .env            # Windows
# OU
nano .env               # Linux/Mac
```

**Variables d'environnement importantes:**
```env
# API Keys pour Job Search
JSEARCH_RAPIDAPI_KEY=your_jsearch_key_here
ADZUNA_APP_ID=your_adzuna_id_here
ADZUNA_APP_KEY=your_adzuna_key_here

# CORS Origins (ajuster selon vos besoins)
ALLOWED_ORIGINS=http://localhost:5175,http://localhost:5173

# Database (optionnel)
DATABASE_URL=sqlite:///./skillsync.db
```

---

## 🚀 Démarrage du Projet

### Démarrage Rapide (Tout en Une Fois)

**Option 1: Script Python (Recommandé)**
```bash
# Depuis la racine du projet
python start_project.py
```

**Option 2: Démarrage Manuel**

**Terminal 1 - Backend:**
```bash
cd backend
python main_simple_for_frontend.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### URLs d'Accès
```
Frontend: http://localhost:5175
Backend:  http://localhost:8001
API Docs: http://localhost:8001/docs
Health:   http://localhost:8001/health
```

---

## 🔙 Commandes Backend

### Démarrer le Serveur Backend

**Serveur Principal (Production-like):**
```bash
cd backend
python main_simple_for_frontend.py
```

**Serveur avec ML Avancé:**
```bash
cd backend
python start_enhanced.py
```

**Serveur Simple (Tests):**
```bash
cd backend
uvicorn main_simple_for_frontend:app --reload --port 8001
```

### Vérifier le Statut
```bash
# Health check
curl http://localhost:8001/health

# Status des APIs Job
curl http://localhost:8001/api/v1/jobs/status

# Version de l'API
curl http://localhost:8001/api/version
```

### Base de Données

**Créer les tables:**
```bash
cd backend
python -c "from database import init_db; init_db()"
```

**Réinitialiser la base de données:**
```bash
cd backend
rm skillsync.db
python -c "from database import init_db; init_db()"
```

**Voir les données:**
```bash
cd backend
sqlite3 skillsync.db
# Dans SQLite:
.tables
SELECT * FROM cv_analyses;
.exit
```

### Gestion des Dépendances

**Installer un nouveau package:**
```bash
cd backend
pip install package_name
pip freeze > requirements.txt
```

**Mettre à jour les packages:**
```bash
cd backend
pip install --upgrade -r requirements.txt
```

### Tests Backend

**Tests Unitaires:**
```bash
cd backend
pytest tests/
```

**Test API spécifique:**
```bash
cd backend
pytest tests/test_api.py -v
```

**Test avec couverture:**
```bash
cd backend
pytest --cov=. tests/
```

### Scripts Utiles Backend

**Tester le système ML:**
```bash
cd backend
python test_xai_system.py
```

**Tester les recommandations:**
```bash
cd backend
python test_recommendations.py
```

**Tester les APIs Job:**
```bash
cd backend
python test_job_apis.py
```

**Vérifier la configuration système:**
```bash
python check_system_ready.py
```

---

## 💻 Commandes Frontend

### Démarrer le Serveur Frontend

**Mode Développement:**
```bash
cd frontend
npm run dev
```

**Mode Production (Build + Preview):**
```bash
cd frontend
npm run build
npm run preview
```

**Avec un port spécifique:**
```bash
cd frontend
npm run dev -- --port 5175
```

### Build & Déploiement

**Build pour production:**
```bash
cd frontend
npm run build
```

**Analyser le build:**
```bash
cd frontend
npm run build -- --mode analyze
```

**Vérifier les erreurs TypeScript:**
```bash
cd frontend
npm run type-check
```

### Linting & Formatting

**Vérifier le code:**
```bash
cd frontend
npm run lint
```

**Corriger automatiquement:**
```bash
cd frontend
npm run lint:fix
```

**Formater le code (si Prettier configuré):**
```bash
cd frontend
npm run format
```

### Gestion des Dépendances

**Installer une dépendance:**
```bash
cd frontend
npm install package-name
```

**Installer une dépendance de développement:**
```bash
cd frontend
npm install --save-dev package-name
```

**Mettre à jour les packages:**
```bash
cd frontend
npm update
```

**Vérifier les packages obsolètes:**
```bash
cd frontend
npm outdated
```

**Nettoyer et réinstaller:**
```bash
cd frontend
rm -rf node_modules package-lock.json  # Linux/Mac
# OU
rmdir /s node_modules && del package-lock.json  # Windows
npm install
```

---

## 🧪 Tests

### Tests Complets

**Test Backend Seul:**
```bash
python test_backend_only.py
```

**Test Frontend (Ouvrir dans le navigateur):**
```
http://localhost:5175
```

**Test CORS:**
```bash
# Script Python
python test_cors.py

# OU ouvrir dans le navigateur:
test_cors.html
```

**Test ML Career Guidance:**
```bash
cd backend
python test_experience_translator.py
```

**Test Enhanced Recommendations:**
```bash
cd backend
python test_enhanced_recommendations.py
```

### Tests Frontend HTML

Ouvrir dans le navigateur:
```
test_frontend.html
test_frontend_clean.html
test_frontend_advanced.html
test_portfolio.html
test_remoteok_filter.html
test_themuse_filter_corrected.html
```

---

## 🔧 Maintenance

### Logs & Debugging

**Voir les logs backend:**
```bash
# Les logs s'affichent dans le terminal où vous avez lancé le backend
cd backend
python main_simple_for_frontend.py
# Ctrl+C pour arrêter
```

**Activer le mode debug:**
```bash
cd backend
# Éditer main_simple_for_frontend.py et chercher:
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8001, log_level="debug")
```

**Voir les requêtes réseau (Frontend):**
```
F12 dans le navigateur > Onglet Network
```

### Nettoyage

**Nettoyer les fichiers cache Python:**
```bash
cd backend
find . -type d -name __pycache__ -exec rm -rf {} +  # Linux/Mac
# OU
for /d /r %d in (__pycache__) do @if exist "%d" rd /s /q "%d"  # Windows
```

**Nettoyer les builds frontend:**
```bash
cd frontend
rm -rf dist node_modules/.vite  # Linux/Mac
# OU
rmdir /s dist  # Windows
```

**Nettoyer la base de données:**
```bash
cd backend
rm skillsync.db
python -c "from database import init_db; init_db()"
```

### Mise à Jour du Projet

**Pull les dernières modifications:**
```bash
git pull origin main
```

**Réinstaller les dépendances:**
```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install

cd ..
```

---

## 🌐 Déploiement

### Préparation pour la Production

**1. Build Frontend:**
```bash
cd frontend
npm run build
# Les fichiers sont dans: frontend/dist/
```

**2. Configurer les Variables d'Environnement:**
```bash
cd backend
# Créer .env.production
cp .env .env.production
# Éditer avec les valeurs de production
```

**3. Tester en Mode Production:**
```bash
# Backend
cd backend
python main_simple_for_frontend.py

# Frontend (preview)
cd frontend
npm run preview
```

### Déploiement Backend

**Option 1: Serveur VPS (Ubuntu):**
```bash
# Sur le serveur:
sudo apt update
sudo apt install python3-pip python3-venv nginx

# Cloner le projet
git clone https://github.com/bilel-amri0/SkillSync.git
cd SkillSync/backend

# Setup Python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Créer un service systemd
sudo nano /etc/systemd/system/skillsync.service
```

**Contenu du service:**
```ini
[Unit]
Description=SkillSync Backend
After=network.target

[Service]
User=your_user
WorkingDirectory=/path/to/SkillSync/backend
Environment="PATH=/path/to/SkillSync/backend/venv/bin"
ExecStart=/path/to/SkillSync/backend/venv/bin/python main_simple_for_frontend.py

[Install]
WantedBy=multi-user.target
```

**Démarrer le service:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable skillsync
sudo systemctl start skillsync
sudo systemctl status skillsync
```

**Option 2: Docker (Recommandé):**
```bash
# Créer Dockerfile pour backend
cd backend
nano Dockerfile
```

**Dockerfile Backend:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["python", "main_simple_for_frontend.py"]
```

**Build et Run:**
```bash
docker build -t skillsync-backend .
docker run -d -p 8001:8001 --name skillsync-backend skillsync-backend
```

### Déploiement Frontend

**Option 1: Vercel (Recommandé):**
```bash
cd frontend
npm install -g vercel
vercel login
vercel --prod
```

**Option 2: Netlify:**
```bash
cd frontend
npm run build
# Upload le dossier dist/ sur Netlify
```

**Option 3: Serveur VPS avec Nginx:**
```bash
# Build
cd frontend
npm run build

# Copier vers le serveur
scp -r dist/* user@server:/var/www/skillsync/

# Configurer Nginx
sudo nano /etc/nginx/sites-available/skillsync
```

**Configuration Nginx:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /var/www/skillsync;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Activer et redémarrer:**
```bash
sudo ln -s /etc/nginx/sites-available/skillsync /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 📊 Commandes de Monitoring

### Vérifier les Processus

**Backend:**
```bash
# Windows
netstat -ano | findstr :8001

# Linux/Mac
lsof -i :8001
ps aux | grep python
```

**Frontend:**
```bash
# Windows
netstat -ano | findstr :5175

# Linux/Mac
lsof -i :5175
ps aux | grep node
```

### Arrêter les Services

**Backend:**
```bash
# Windows
taskkill /F /IM python.exe

# Linux/Mac
pkill -f "python main_simple_for_frontend.py"
```

**Frontend:**
```bash
# Windows
taskkill /F /IM node.exe

# Linux/Mac
pkill -f "npm run dev"
```

### Performance

**Vérifier l'utilisation mémoire:**
```bash
# Backend
ps aux | grep python | awk '{print $4, $11}'

# Frontend
ps aux | grep node | awk '{print $4, $11}'
```

**Logs de performance:**
```bash
# Activer le profiling Python
cd backend
python -m cProfile -o output.prof main_simple_for_frontend.py
```

---

## 🔥 Commandes d'Urgence

### Redémarrage Complet

```bash
# 1. Arrêter tout
taskkill /F /IM python.exe  # Windows
taskkill /F /IM node.exe    # Windows
# OU
pkill python && pkill node  # Linux/Mac

# 2. Nettoyer
cd backend
rm -rf __pycache__
cd ../frontend
rm -rf node_modules/.vite

# 3. Redémarrer
# Terminal 1:
cd backend && python main_simple_for_frontend.py

# Terminal 2:
cd frontend && npm run dev
```

### Problèmes Courants

**Port déjà utilisé:**
```bash
# Windows - Libérer le port 8001
netstat -ano | findstr :8001
taskkill /F /PID <PID>

# Linux/Mac
lsof -ti:8001 | xargs kill -9
```

**Erreur CORS:**
```bash
# Vérifier la configuration CORS dans backend/main_simple_for_frontend.py
# Redémarrer le backend après modification
cd backend
python main_simple_for_frontend.py
```

**Modules Python manquants:**
```bash
cd backend
pip install -r requirements.txt --force-reinstall
```

**Packages npm manquants:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

---

## 📝 Scripts Utiles

### Script de Démarrage Automatique (Windows)

**start_all.bat:**
```batch
@echo off
echo Starting SkillSync Project...

start "Backend" cmd /k "cd backend && python main_simple_for_frontend.py"
timeout /t 5 /nobreak

start "Frontend" cmd /k "cd frontend && npm run dev"

echo SkillSync is starting...
echo Backend: http://localhost:8001
echo Frontend: http://localhost:5175
pause
```

### Script de Démarrage (Linux/Mac)

**start_all.sh:**
```bash
#!/bin/bash

echo "Starting SkillSync Project..."

# Start backend
cd backend
python main_simple_for_frontend.py &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

# Wait for backend
sleep 5

# Start frontend
cd ../frontend
npm run dev &
FRONTEND_PID=$!
echo "Frontend started (PID: $FRONTEND_PID)"

echo "SkillSync is running!"
echo "Backend: http://localhost:8001"
echo "Frontend: http://localhost:5175"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
```

**Rendre exécutable:**
```bash
chmod +x start_all.sh
./start_all.sh
```

---

## 🎓 Commandes pour Développeurs

### Git Workflow

**Créer une nouvelle branche:**
```bash
git checkout -b feature/nouvelle-fonctionnalite
```

**Committer les changements:**
```bash
git add .
git commit -m "Description des changements"
git push origin feature/nouvelle-fonctionnalite
```

**Mettre à jour depuis main:**
```bash
git checkout main
git pull origin main
git checkout feature/nouvelle-fonctionnalite
git merge main
```

### Configuration VS Code

**Installer les extensions recommandées:**
```bash
code --install-extension ms-python.python
code --install-extension dbaeumer.vscode-eslint
code --install-extension esbenp.prettier-vscode
code --install-extension bradlc.vscode-tailwindcss
```

---

## 📚 Documentation API

**Accéder à la documentation interactive:**
```
http://localhost:8001/docs          # Swagger UI
http://localhost:8001/redoc         # ReDoc
http://localhost:8001/openapi.json  # OpenAPI Schema
```

---

## ✅ Checklist Avant Production

- [ ] Tests backend passent: `pytest backend/tests/`
- [ ] Build frontend réussit: `npm run build`
- [ ] Variables d'environnement configurées
- [ ] CORS configuré pour le domaine de production
- [ ] Base de données sauvegardée
- [ ] SSL/HTTPS configuré
- [ ] Rate limiting activé
- [ ] Logs configurés
- [ ] Monitoring en place
- [ ] Backup automatique configuré

---

## 🆘 Support

**Documentation complète:**
- `README.md` - Vue d'ensemble
- `INSTALLATION_GUIDE.md` - Installation détaillée
- `API_DOCUMENTATION.md` - Documentation API
- `CORS_QUICK_FIX.md` - Fix erreurs CORS

**En cas de problème:**
1. Vérifier les logs dans le terminal
2. Consulter `TROUBLESHOOTING.md`
3. Ouvrir une issue sur GitHub

---

**Version:** 2.0.0  
**Dernière mise à jour:** 24 Novembre 2025  
**Auteur:** SkillSync Team
