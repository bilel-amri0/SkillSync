# 📦 GitHub Upload Guide

## ⚠️ Important: Large Files Excluded

This repository has been configured to **exclude large files** that exceed GitHub's 100MB limit.

### 🚫 Excluded Files

The following files are **NOT** uploaded to GitHub (see `.gitignore`):

#### 1. **AI/ML Models** (~411 MB)
- `models/resume-ner/model.safetensors` (411 MB)
- All model files (`.safetensors`, `.bin`, `.pt`, `.pth`, `.h5`)

#### 2. **Virtual Environment**
- `.venv/` directory
- All Python virtual environment files

#### 3. **Environment Variables**
- `.env` files (contains sensitive API keys and secrets)
- `backend/.env`
- `backend/.env.ml`

#### 4. **Databases**
- `*.db` files (SQLite databases)
- `skillsync.db` (~6-8 MB)
- `skillsync_enhanced.db`
- `test_skillsync.db`

#### 5. **Cache & Build Files**
- `__pycache__/`
- `node_modules/`
- `dist/`, `build/`
- `.pytest_cache/`

---

## 🚀 How to Upload to GitHub

### Step 1: Initialize Git (if not already done)
```bash
cd c:\Users\Lenovo\Downloads\SkillSync_Enhanced
git init
```

### Step 2: Check What Will Be Uploaded
```bash
git status
```

### Step 3: Add Files
```bash
git add .
```

### Step 4: Verify Large Files Are Excluded
```bash
# Check that models/ and .venv/ are NOT in the staging area
git status
```

### Step 5: Create Initial Commit
```bash
git commit -m "Initial commit: SkillSync v2.1.0 - Enterprise Career Platform"
```

### Step 6: Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `SkillSync` (or `SkillSync_Enhanced`)
3. Description: "AI-powered career development platform with CV analysis and job matching"
4. Choose: **Public** or **Private**
5. **DO NOT** initialize with README (you already have one)
6. Click "Create repository"

### Step 7: Connect to GitHub
```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/SkillSync.git
git branch -M main
git push -u origin main
```

---

## 📥 Setup Instructions for Others

When someone clones your repository, they need to:

### 1. Download the ML Model
```bash
# The model needs to be downloaded separately
# Create models directory
mkdir models
cd models

# Download from Hugging Face
git clone https://huggingface.co/dslim/bert-base-NER resume-ner
```

Or add this to your README:
```markdown
## 🤖 ML Model Setup

This project uses a BERT-based NER model (~411MB) that is not included in the repository.

**Download the model:**
```bash
pip install transformers
python -c "from transformers import AutoModelForTokenClassification; AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER').save_pretrained('models/resume-ner')"
```
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. Setup Environment Variables
```bash
cd backend
cp .env.example .env
# Edit .env with your API keys and secrets
```

### 4. Initialize Database
```bash
python -c "from database import init_db; init_db()"
```

---

## 🔍 Verify Before Pushing

Run this command to check file sizes:
```bash
git ls-files | ForEach-Object { Get-Item $_ -ErrorAction SilentlyContinue } | Where-Object { $_.Length -gt 50MB } | Select-Object Name, @{Name="SizeMB";Expression={[math]::Round($_.Length/1MB,2)}}
```

**Expected result:** No files over 50MB should be listed.

---

## 📝 Alternative: Git LFS (for Large Files)

If you want to include the model in GitHub, use **Git Large File Storage**:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "models/**/*.safetensors"
git lfs track "*.db"

# Add .gitattributes
git add .gitattributes

# Now add and commit normally
git add models/
git commit -m "Add ML models via Git LFS"
git push
```

**Note:** Git LFS has storage limits on free accounts (1GB storage, 1GB bandwidth/month).

---

## ✅ Checklist

- [ ] `.gitignore` updated
- [ ] Large files excluded (models, .venv, .env, *.db)
- [ ] `git status` shows no files > 100MB
- [ ] GitHub repository created
- [ ] Remote origin added
- [ ] Code pushed successfully
- [ ] README updated with model download instructions

---

## 🆘 Troubleshooting

### Error: "file exceeds GitHub's file size limit of 100 MB"
```bash
# Remove the file from Git history
git rm --cached path/to/large/file
git commit -m "Remove large file"
```

### Already committed large files?
```bash
# Use BFG Repo Cleaner
# Download from: https://rtyley.github.io/bfg-repo-cleaner/
java -jar bfg.jar --strip-blobs-bigger-than 100M
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

---

**Created:** 2025-12-29  
**Project:** SkillSync v2.1.0
