# 🚀 SkillSync Installation Guide

## 🎯 Complete AI-Powered Job Search Revolution Platform

**SkillSync** is a comprehensive AI-powered platform that revolutionizes job search with transparent CV analysis, automatic portfolio generation, and personalized career recommendations.

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Backend Installation](#backend-installation)
- [Frontend Installation](#frontend-installation)
- [Quick Start](#quick-start)
- [Features](#features)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)

## 💻 System Requirements

### Backend Requirements
- Python 3.8+
- pip (Python package installer)
- 8GB RAM minimum (16GB recommended)
- 5GB free disk space

### Frontend Requirements
- Node.js 16+
- npm or yarn
- Modern web browser

## 🔧 Backend Installation

### 1. Navigate to Backend Directory
```bash
cd SkillSync_Project/backend
```

### 2. Create Virtual Environment
```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Download NLP Models
```bash
python -m spacy download en_core_web_sm
```

### 6. Start Backend Server
```bash
# From SkillSync_Project directory
python start_server.py
```

**Backend will be available at:** `http://localhost:8000`

## 🌐 Frontend Installation

### 1. Navigate to Frontend Directory
```bash
cd SkillSync_Project/frontend
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Start Development Server
```bash
npm start
```

**Frontend will be available at:** `http://localhost:3000`

## ⚡ Quick Start

### Option 1: Full Development Setup

1. **Start Backend:**
   ```bash
   cd SkillSync_Project
   python start_server.py
   ```

2. **Start Frontend (new terminal):**
   ```bash
   cd SkillSync_Project/frontend
   npm start
   ```

3. **Access Application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Option 2: Backend Only (API Testing)

```bash
cd SkillSync_Project
python start_server.py
```

Access interactive API documentation at http://localhost:8000/docs

## 🌟 Features

### 🤖 Core AI Features

#### 1. **Advanced CV Analysis (F1-F5)**
- 📄 Multi-format CV upload (PDF/DOCX)
- 🔍 NER-based skill extraction (ESCO/O*NET)
- 🧠 Semantic CV-job matching
- 📊 Gap analysis with visualization
- 🗓️ Explainable AI (XAI) insights

#### 2. **Portfolio Generator (F6)**
- 🎨 Automatic portfolio generation
- 📱 Responsive modern templates
- ⚙️ Customizable themes and layouts
- 🗺️ Ready-to-deploy websites

#### 3. **Experience Translator (F7)**
- 🔄 Intelligent experience reformulation
- 🎯 Job-specific optimization
- 📝 NLG-powered content enhancement
- 📈 Keyword alignment scoring

#### 4. **Personalized Recommendations (F8)**
- 🛫 Custom development paths
- 🏆 Certification roadmaps
- 📚 Learning resource suggestions
- 🗺️ Career progression timeline

#### 5. **Interactive Dashboard (F9)**
- 📈 Progress tracking and analytics
- 💡 Skill development insights
- 📊 Visual performance metrics
- 🎯 Goal setting and monitoring

### 🔍 Technical Specifications

- **NLP Engine:** Transformer-based models (BERT/RoBERTa)
- **Similarity Calculation:** Cosine similarity on embeddings
- **Skill Taxonomy:** ESCO/O*NET integration
- **XAI Framework:** SHAP/LIME-inspired explanations
- **Architecture:** FastAPI microservices
- **Frontend:** React with Tailwind CSS

## 📚 API Documentation

### Core Endpoints

#### CV Analysis
```http
POST /api/v1/upload-cv
Content-Type: multipart/form-data

Parameters:
- file: CV file (PDF/DOCX)
- job_description: Target job (optional)
```

#### Portfolio Generation
```http
POST /api/v1/generate-portfolio
Content-Type: application/json

{
  "analysis_id": "uuid",
  "template": "modern",
  "customizations": {}
}
```

#### Experience Translation
```http
POST /api/v1/translate-experience
Content-Type: application/json

{
  "analysis_id": "uuid",
  "target_job_description": "string"
}
```

#### Recommendations
```http
GET /api/v1/recommendations/{analysis_id}
```

#### Dashboard
```http
GET /api/v1/dashboard/{user_id}
```

### Interactive API Documentation
Access full API documentation at: `http://localhost:8000/docs`

## 🛠️ Configuration

### Environment Variables

Create `.env` file in backend directory:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Model Configuration
SEMANTIC_MODEL=sentence-transformers/all-MiniLM-L6-v2
SUMMARIZATION_MODEL=facebook/bart-large-cnn

# Security
SECRET_KEY=your-secret-key-here

# Database (Production)
DATABASE_URL=postgresql://user:password@localhost/skillsync
```

### Frontend Configuration

Create `.env` file in frontend directory:

```env
REACT_APP_API_URL=http://localhost:8000/api/v1
REACT_APP_TITLE=SkillSync - AI-Powered Job Search
```

## 🐛 Troubleshooting

### Common Backend Issues

**1. Module Import Errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

**2. Model Download Issues**
```bash
# Download spaCy model manually
python -m spacy download en_core_web_sm

# If transformers models fail to download
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"
```

**3. Port Already in Use**
```bash
# Change port in config.py or use environment variable
export PORT=8001
python start_server.py
```

### Common Frontend Issues

**1. npm Install Failures**
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**2. API Connection Issues**
```bash
# Verify backend is running
curl http://localhost:8000/

# Check frontend proxy configuration in package.json
"proxy": "http://localhost:8000"
```

**3. Build Issues**
```bash
# Increase Node.js memory limit
export NODE_OPTIONS=--max_old_space_size=4096
npm run build
```

## 📦 Production Deployment

### Backend Deployment

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend Deployment

```bash
# Build for production
npm run build

# Serve static files (example with serve)
npm install -g serve
serve -s build -l 3000
```

### Docker Deployment

```bash
# Build backend container
docker build -t skillsync-backend ./backend

# Build frontend container
docker build -t skillsync-frontend ./frontend

# Run with docker-compose
docker-compose up -d
```

## 📞 Support

For issues and questions:

1. **Check the logs:** Both backend and frontend provide detailed logging
2. **API Documentation:** Visit http://localhost:8000/docs for API details
3. **GitHub Issues:** Report bugs and feature requests
4. **Troubleshooting:** Follow the troubleshooting guide above

## 🎆 Success!

Your SkillSync platform is now ready to revolutionize job search with AI! 

🔗 **Access Points:**
- **Main Application:** http://localhost:3000
- **API Documentation:** http://localhost:8000/docs
- **Backend Health:** http://localhost:8000

🎯 **Ready to transform careers with transparent AI!**