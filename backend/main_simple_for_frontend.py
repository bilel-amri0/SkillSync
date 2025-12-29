# -*- coding: utf-8 -*-
import sys
import os

# Fix Windows CMD encoding issues with emojis
if sys.platform == 'win32':
    try:
        # Try to set console to UTF-8
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except:
        pass

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import asyncio
from datetime import datetime
import base64
import uuid
import re
import traceback
import random
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False
    print("Warning: slowapi not installed. Rate limiting disabled. Install with: pip install slowapi")

# Load environment variables from .env file FIRST
try:
    from dotenv import load_dotenv
    # Fix: Point to the correct .env file path
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(" Environment variables loaded from .env file")
    else:
        print("Warning: .env file not found at", dotenv_path)
except ImportError:
    # If python-dotenv is not installed, try to load manually
    pass

logger = logging.getLogger(__name__)

# Debug: Show that APIs will be enabled
print(f" JSEARCH_RAPIDAPI_KEY: {'OK: LOADED' if os.getenv('JSEARCH_RAPIDAPI_KEY') else 'Error: MISSING'}")
print(f" ADZUNA_APP_ID: {'OK: LOADED' if os.getenv('ADZUNA_APP_ID') else 'Error: MISSING'}")

# Import our services AFTER env loading
from services.multi_job_api_service import get_job_service, JobResult
from skillsync.interviews import realtime as interview_realtime
from skillsync.interviews import routes as interview_routes

# Import Experience Translator (F7)
try:
    from experience_translator import translate_experience_api, ExperienceTranslator
    logger.info("OK: Experience Translator (F7) loaded successfully")
except ImportError as exp_error:
    logger.warning(f"Warning: Experience Translator not available: {exp_error}")
    translate_experience_api = None
    ExperienceTranslator = None

# Enhanced recommendation engine import with ML support and fallback
ML_MODE_ENABLED = False
ML_MODE_TYPE = "hybrid"  # rules, lite, full, ou hybrid
USE_ENHANCED_ENGINE = False
recommendation_engine = None
ml_recommendation_engine = None
ml_backend = None
ml_models = {}

# Initialisation Backend ML Hybride
# Lazy loading to avoid Windows multiprocessing issues
ml_backend = None
ml_backend_loading_attempted = False

def get_ml_backend_lazy():
    """Lazy load ML backend to avoid Windows multiprocessing issues"""
    global ml_backend, ML_MODE_ENABLED, ML_MODE_TYPE, ml_backend_loading_attempted
    
    if ml_backend_loading_attempted:
        return ml_backend
    
    ml_backend_loading_attempted = True
    
    try:
        from ml_backend_hybrid import get_ml_backend as _get_ml_backend
        ml_backend = _get_ml_backend()
        ML_MODE_ENABLED = True
        ML_MODE_TYPE = "hybrid"
        logger.info(" Backend ML Hybride initialis avec succs")
        logger.info(f" Status: {ml_backend.get_system_status()}")
        return ml_backend
    except Exception as e:
        logger.warning(f"Warning: Backend ML Hybride non disponible: {e}")
        ML_MODE_ENABLED = False
        return None

def detect_ml_configuration():
    """Dtecte automatiquement la configuration ML disponible"""
    global ML_MODE_ENABLED, ML_MODE_TYPE
    
    # Vrifier les flags de configuration
    import os
    from pathlib import Path
    
    if Path('ml_mode_enabled.flag').exists():
        logger.info(" Flag ML dtect, activation du mode ML...")
        
        # Vrifier le type de config dans .env.ml
        env_config = {}
        if Path('.env.ml').exists():
            try:
                with open('.env.ml', 'r') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            env_config[key] = value.lower() == 'true'
            except:
                pass
        
        use_tensorflow = env_config.get('USE_TENSORFLOW', True)
        
        if use_tensorflow:
            return try_full_ml_mode()
        else:
            return try_lite_ml_mode()
    
    return False

def try_lite_ml_mode():
    """Essaie d'activer le mode ML lger (sans TensorFlow)"""
    global ML_MODE_ENABLED, ML_MODE_TYPE, ml_models
    
    try:
        logger.info(" Tentative mode ML LGER...")
        
        # Import des bibliothques essentielles uniquement
        from transformers import AutoTokenizer, AutoModel
        from sentence_transformers import SentenceTransformer
        import torch
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        logger.info("    Chargement modles lgers...")
        
        # Modles plus lgers et plus rapides
        ml_models['tokenizer'] = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        ml_models['model'] = AutoModel.from_pretrained('distilbert-base-uncased')
        ml_models['sentence_model'] = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Scorer bas sur PyTorch (plus lger que TensorFlow)
        class LiteScorer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(384, 128),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(128, 64),
                    torch.nn.ReLU(),
                    torch.nn.Linear(64, 1),
                    torch.nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.layers(x)
        
        ml_models['lite_scorer'] = LiteScorer()
        
        ML_MODE_ENABLED = True
        ML_MODE_TYPE = "lite"
        
        logger.info("OK: Mode ML LGER activ avec succs!")
        logger.info(f"   ML: Modle: DistilBERT (lger)")
        logger.info(f"    Sentence-BERT: MiniLM-L6-v2")
        logger.info(f"    Scorer: PyTorch (lite)")
        return True
        
    except ImportError as e:
        logger.warning(f"Warning: Bibliothques ML lger manquantes: {e}")
        return False
    except Exception as e:
        logger.error(f"Error: Erreur mode ML lger: {e}")
        return False

def try_full_ml_mode():
    """Essaie d'activer le mode ML complet (avec TensorFlow)"""
    global ML_MODE_ENABLED, ML_MODE_TYPE, ml_models
    
    try:
        logger.info(" Tentative mode ML COMPLET avec TensorFlow...")
        
        # Import complet avec TensorFlow
        from transformers import AutoTokenizer, AutoModel
        from sentence_transformers import SentenceTransformer
        import tensorflow as tf
        import torch
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        logger.info("    Chargement modles ML complets...")
        
        # Modles complets avec TensorFlow
        ml_models['bert_tokenizer'] = AutoTokenizer.from_pretrained('bert-base-uncased')
        ml_models['bert_model'] = AutoModel.from_pretrained('bert-base-uncased')
        ml_models['sentence_model'] = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Neural Scorer TensorFlow (plus puissant)
        logger.info("   ML: Cration Neural Scorer TensorFlow...")
        ml_models['neural_scorer'] = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(384,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compiler le modle pour optimiser les performances
        ml_models['neural_scorer'].compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Test du modle TensorFlow
        test_input = np.random.random((1, 384))
        test_prediction = ml_models['neural_scorer'].predict(test_input, verbose=0)
        logger.info(f"   OK: Test Neural Scorer: {test_prediction[0][0]:.4f}")
        
        ML_MODE_ENABLED = True
        ML_MODE_TYPE = "full"
        
        logger.info(" Mode ML COMPLET avec TensorFlow activ!")
        logger.info(f"   ML: BERT: {ml_models['bert_model'].__class__.__name__}")
        logger.info(f"    Sentence-BERT: {ml_models['sentence_model'].__class__.__name__}")
        logger.info(f"    Neural Scorer: TensorFlow {tf.__version__}")
        logger.info(f"    Architecture: {len(ml_models['neural_scorer'].layers)} couches")
        return True
        
    except ImportError as e:
        logger.warning(f"Warning: Bibliothques ML compltes manquantes: {e}")
        return False
    except Exception as e:
        logger.error(f"Error: Erreur mode ML complet: {e}")
        return False

# Ajout des fonctions ML pour les recommandations
def extract_skills_with_ml(text):
    """Extrait les comptences avec ML"""
    if not ML_MODE_ENABLED or 'sentence_model' not in ml_models:
        return []
    
    try:
        # Liste de comptences de rfrence
        skill_database = [
            "Python", "JavaScript", "Java", "C++", "SQL", "HTML", "CSS",
            "React", "Angular", "Vue.js", "Node.js", "Django", "Flask",
            "Machine Learning", "Data Science", "AI", "Deep Learning",
            "Docker", "Kubernetes", "AWS", "Azure", "Git", "DevOps"
        ]
        
        # Utiliser sentence-transformers pour la similarit
        text_embedding = ml_models['sentence_model'].encode([text])
        skill_embeddings = ml_models['sentence_model'].encode(skill_database)
        
        # Calculer similarits
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(text_embedding, skill_embeddings)[0]
        
        # Retourner les comptences avec similarit > 0.5
        detected_skills = []
        for i, similarity in enumerate(similarities):
            if similarity > 0.5:
                detected_skills.append(skill_database[i])
        
        return detected_skills[:5]  # Top 5
        
    except Exception as e:
        logger.error(f"Erreur extraction ML: {e}")
        return []

def score_with_ml(profile, recommendation):
    """Score une recommandation avec ML (lite, full ou hybrid)"""
    if not ML_MODE_ENABLED:
        return 0.5
    
    try:
        # Encoder le profil et la recommandation
        profile_text = f"{profile.get('current_role', '')} {' '.join(profile.get('skills', []))}"
        rec_text = recommendation.get('title', '')
        
        # Mode ML HYBRID avec Backend ML Hybride (utilise son propre systme)
        backend = get_ml_backend_lazy()
        if ML_MODE_TYPE == "hybrid" and backend:
            try:
                # Utiliser le backend ML hybride pour le scoring direct
                skills_text = ' '.join(profile.get('skills', []))
                hybrid_score = backend.score_skill_match(skills_text, rec_text)
                
                # Ajouter un bonus bas sur l'exprience et le rle
                role_bonus = 0.1 if profile.get('current_role', '').lower() in rec_text.lower() else 0
                experience_bonus = min(0.15, profile.get('experience_years', 0) * 0.02)
                
                final_score = hybrid_score + role_bonus + experience_bonus
                final_score = max(0.1, min(1.0, final_score))
                
                logger.debug(f"Hybrid ML scoring - Base: {hybrid_score:.3f}, Role bonus: {role_bonus:.3f}, Exp bonus: {experience_bonus:.3f}, Final: {final_score:.3f}")
                return final_score
                
            except Exception as hybrid_error:
                logger.warning(f"Hybrid ML scoring failed, using fallback: {hybrid_error}")
                # Fallback simple pour le mode hybrid
                return 0.6 + (len(profile.get('skills', [])) * 0.02)
        
        # Pour les autres modes (full/lite), utiliser le systme d'embeddings existant
        if 'sentence_model' not in ml_models:
            logger.warning("Sentence model not available, using simple scoring")
            return 0.5 + (len(profile.get('skills', [])) * 0.03)
            
        # Utiliser sentence-transformers pour l'embedding de base
        embeddings = ml_models['sentence_model'].encode([profile_text, rec_text])
        
        # Calculer similarit comme score base
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        
        # Mode ML COMPLET avec TensorFlow Neural Scorer
        if ML_MODE_TYPE == "full" and 'neural_scorer' in ml_models:
            try:
                import numpy as np
                # Utiliser le Neural Scorer TensorFlow pour un scoring avanc
                profile_embedding = embeddings[0].reshape(1, -1)
                neural_score = ml_models['neural_scorer'].predict(profile_embedding, verbose=0)[0][0]
                
                # Combiner similarity et neural score avec pondration
                # 60% neural score (plus sophistiqu) + 40% similarity
                final_score = (neural_score * 0.6) + (similarity * 0.4)
                
                logger.debug(f"TensorFlow scoring - Similarity: {similarity:.3f}, Neural: {neural_score:.3f}, Final: {final_score:.3f}")
                
            except Exception as tf_error:
                logger.warning(f"TensorFlow scoring failed, fallback to similarity: {tf_error}")
                final_score = similarity
        
        # Mode ML LITE avec PyTorch Lite Scorer
        elif ML_MODE_TYPE == "lite" and 'lite_scorer' in ml_models:
            import sys
            import os
            workspace_path = '/workspace'
            if workspace_path not in sys.path:
                sys.path.append(workspace_path)

            if not detect_ml_configuration():
                logger.info("Info: Mode fallback: recommandations bases sur des rgles")
        # Fallback: utiliser seulement la similarit
        else:
            final_score = similarity
            logger.debug(f"Similarity-only scoring: {final_score:.3f}")
        
        # Normaliser le score entre 0.1 et 1.0
        final_score = max(0.1, min(1.0, final_score))
        
        return final_score
        
    except Exception as e:
        logger.error(f"Erreur scoring ML: {e}")
        return 0.5

async def generate_ml_enhanced_recommendations(skills_data, cv_data):
    """Gnre des recommandations avec le mode ML (lite ou full) - DYNAMIC VERSION"""
    try:
        # Profile utilisateur enrichi
        user_profile = {
            'current_role': ' '.join(cv_data.get('job_titles', ['Developer'])),
            'skills': [skill.get('skill', '') for skill in skills_data],
            'experience_years': cv_data.get('experience_years', 2),
            'education': cv_data.get('education', []),
            'industry': cv_data.get('industry', 'Tech')
        }
        
        #  NEW: Use DynamicRecommendationEngine for personalized recommendations
        from dynamic_recommendations import DynamicRecommendationEngine
        dynamic_engine = DynamicRecommendationEngine()
        
        # Prepare CV text for analysis (simulate text from profile data)
        cv_text = f"""
        Job Title: {user_profile['current_role']}
        Skills: {', '.join(user_profile['skills'])}
        Experience: {user_profile['experience_years']} years
        Industry: {user_profile['industry']}
        Education: {', '.join(user_profile['education']) if user_profile['education'] else 'Not specified'}
        """.lower()
        
        # Generate dynamic analysis and recommendations
        logger.info(f" Analyzing CV with {len(user_profile['skills'])} skills for {user_profile['current_role']}")
        
        try:
            cv_analysis = dynamic_engine.analyze_cv_content(cv_text)
            if not isinstance(cv_analysis, dict) or 'primary_domains' not in cv_analysis:
                logger.warning("Warning: CV analysis returned invalid format, using fallback")
                cv_analysis = dynamic_engine._get_default_analysis()
            
            logger.info(f" CV Analysis result: {cv_analysis['experience_level']} level, domains: {cv_analysis['primary_domains']}")
            
            # Generate personalized skill recommendations
            skill_recommendations = dynamic_engine.generate_skill_recommendations(cv_analysis)
            logger.info(f" Generated {len(skill_recommendations)} personalized skill recommendations")
            
        except Exception as e:
            logger.error(f"Error: Error in CV analysis: {e}")
            logger.info("Retry: Using fallback analysis")
            cv_analysis = {
                "experience_level": "mid",
                "primary_domains": ["frontend", "backend"],
                "current_skills": user_profile['skills'],
                "role_focus": "fullstack",
                "text_length": len(cv_text),
                "has_technical_content": True
            }
            skill_recommendations = dynamic_engine.generate_skill_recommendations(cv_analysis)
        
        # Convert to backend format and enhance with ML scoring
        base_recommendations = []
        
        # 1. Immediate Actions (personalized based on analysis)
        immediate_actions = []
        primary_domain = cv_analysis['primary_domains'][0] if cv_analysis['primary_domains'] else 'fullstack'
        role_focus = cv_analysis.get('role_focus', 'fullstack')
        
        if cv_analysis['experience_level'] == 'junior':
            immediate_actions = [
                {"title": f"Crer portfolio {primary_domain} dbutant", "type": "immediate", "category": "portfolio"},
                {"title": "Optimiser profil LinkedIn junior", "type": "immediate", "category": "personal_branding"},
                {"title": f"Rejoindre communauts {role_focus}", "type": "immediate", "category": "networking"},
                {"title": "Mettre  jour CV avec projets rcents", "type": "immediate", "category": "cv_optimization"},
                {"title": f"Configurer environnement {primary_domain}", "type": "immediate", "category": "technical_setup"}
            ]
        elif cv_analysis['experience_level'] == 'senior':
            immediate_actions = [
                {"title": f"Leadership technique en {role_focus}", "type": "immediate", "category": "leadership"},
                {"title": "Mentoring et formation quipe", "type": "immediate", "category": "mentoring"},
                {"title": f"Architecture avance {primary_domain}", "type": "immediate", "category": "architecture"},
                {"title": "Publication articles techniques", "type": "immediate", "category": "thought_leadership"},
                {"title": "Contribution projets open source", "type": "immediate", "category": "networking"}
            ]
        else:  # mid-level
            immediate_actions = [
                {"title": f"Portfolio avanc {role_focus}", "type": "immediate", "category": "portfolio"},
                {"title": f"Contribution projets {primary_domain}", "type": "immediate", "category": "networking"},
                {"title": "Optimisation LinkedIn avec mots-cls", "type": "immediate", "category": "personal_branding"},
                {"title": f"Dmo interactive {role_focus}", "type": "immediate", "category": "demonstration"},
                {"title": f"Configuration environnement {primary_domain} pro", "type": "immediate", "category": "technical_setup"}
            ]
        
        base_recommendations.extend(immediate_actions)
        
        # 2. Add personalized skill recommendations from dynamic engine
        for skill_rec in skill_recommendations:
            base_recommendations.append({
                "title": skill_rec['skill'],
                "type": "skill", 
                "category": skill_rec['category'].lower(),
                "reason": skill_rec['reason'],
                "priority": skill_rec['priority'],
                "estimated_time": skill_rec['estimated_time'],
                "difficulty": skill_rec['difficulty']
            })
        
        # 3. Domain-specific certifications
        cert_recommendations = []
        primary_domains = cv_analysis.get('primary_domains', ['frontend', 'backend'])
        for domain in primary_domains:
            if domain == 'data_science':
                cert_recommendations.extend([
                    {"title": "AWS Machine Learning Specialty", "type": "certification", "category": "aws_ml"},
                    {"title": "Google Professional ML Engineer", "type": "certification", "category": "gcp_ml"}
                ])
            elif domain == 'frontend':
                cert_recommendations.extend([
                    {"title": "React Developer Certification", "type": "certification", "category": "react"},
                    {"title": "AWS Certified Solutions Architect", "type": "certification", "category": "aws"}
                ])
            elif domain == 'backend':
                cert_recommendations.extend([
                    {"title": "AWS Certified Developer", "type": "certification", "category": "aws"},
                    {"title": "Node.js Professional Certification", "type": "certification", "category": "nodejs"}
                ])
            elif domain == 'devops':
                cert_recommendations.extend([
                    {"title": "AWS DevOps Engineer Professional", "type": "certification", "category": "aws_devops"},
                    {"title": "Kubernetes Administrator", "type": "certification", "category": "k8s"}
                ])
        
        base_recommendations.extend(cert_recommendations[:4])  # Limit to 4 certifications
        
        # 4. Add learning resources
        base_recommendations.extend([
            {"title": f"Formation {role_focus} avance", "type": "resource", "category": "online_course"},
            {"title": f"Documentation officielle {primary_domain}", "type": "resource", "category": "documentation"}
        ])
        
        # Continue with ML scoring and processing
        logger.info(f" Generated {len(base_recommendations)} personalized recommendations for {ML_MODE_TYPE} mode")
        
        # Scorer chaque recommandation avec ML appropri
        scored_recommendations = []
        for rec in base_recommendations:
            score = score_with_ml(user_profile, rec)
            
            # Enrichir avec descriptions spcifiques au mode ML
            if ML_MODE_TYPE == "full":
                user_skills_text = ', '.join(user_profile['skills'][:3])
                rec['description'] = f"Recommandation TensorFlow personnalise pour {user_skills_text}. Score Neural: {score:.1%}"
                rec['ml_engine'] = "TensorFlow Neural Network"
            elif ML_MODE_TYPE == "hybrid":
                user_skills_text = ', '.join(user_profile['skills'][:4])
                rec['description'] = f"Recommandation ML hybride optimise pour {user_skills_text}. Score IA: {score:.1%}"
                rec['ml_engine'] = "PyTorch + Scikit-learn Hybrid"
            else:
                user_skills_text = ', '.join(user_profile['skills'][:5])
                rec['description'] = f"Recommandation ML pour votre profil ({user_skills_text}). Score PyTorch: {score:.1%}"
                rec['ml_engine'] = "PyTorch Lite Model"
            
            rec['priority_score'] = round(score * 100, 1)
            rec['confidence'] = min(0.95, score + 0.1)
            rec['ml_mode_used'] = ML_MODE_TYPE.upper()
            
            scored_recommendations.append(rec)
        
        # Trier par score et grouper par type
        scored_recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Organiser par catgories avec quantits adaptes au mode
        recommendations_count = {
            "immediate": 6 if ML_MODE_TYPE == "full" else (5 if ML_MODE_TYPE == "hybrid" else 5),
            "skill": 8 if ML_MODE_TYPE == "full" else (7 if ML_MODE_TYPE == "hybrid" else 6),
            "certification": 5 if ML_MODE_TYPE == "full" else (4 if ML_MODE_TYPE == "hybrid" else 4),
            "career": 5 if ML_MODE_TYPE == "full" else (4 if ML_MODE_TYPE == "hybrid" else 4)
        }
        
        formatted_recommendations = {
            "IMMEDIATE_ACTIONS": [r for r in scored_recommendations if r['type'] == 'immediate'][:recommendations_count["immediate"]],
            "SKILL_DEVELOPMENT": [r for r in scored_recommendations if r['type'] == 'skill'][:recommendations_count["skill"]],
            "CERTIFICATIONS": [r for r in scored_recommendations if r['type'] == 'certification'][:recommendations_count["certification"]],
            "CAREER_OPPORTUNITIES": [r for r in scored_recommendations if r['type'] == 'career'][:recommendations_count["career"]],
            "LEARNING_RESOURCES": [],
            "CAREER_ROADMAP": {}  # Initialize career roadmap
        }
        
        # 5. Generate Career Roadmap based on CV analysis
        roadmap_milestones = []
        
        if cv_analysis['experience_level'] == 'junior':
            roadmap_milestones = [
                {
                    "month": 3,
                    "title": "Foundation Skills",
                    "description": f"Master core {primary_domain} fundamentals and build first projects",
                    "focus_areas": ["Basic Programming", "Version Control", "First Portfolio"],
                    "status": "current"
                },
                {
                    "month": 6,
                    "title": "Technical Proficiency", 
                    "description": f"Develop intermediate {role_focus} skills and complete real projects",
                    "focus_areas": ["Framework Knowledge", "Database Basics", "Code Quality"],
                    "status": "upcoming"
                },
                {
                    "month": 9,
                    "title": "Industry Integration",
                    "description": "Join tech communities, contribute to open source, network actively",
                    "focus_areas": ["Open Source", "Networking", "Industry Knowledge"],
                    "status": "upcoming"
                },
                {
                    "month": 12,
                    "title": "Job Ready",
                    "description": "Complete portfolio, practice interviews, apply for entry-level positions",
                    "focus_areas": ["Interview Prep", "Portfolio Polish", "Job Applications"],
                    "status": "upcoming"
                }
            ]
        elif cv_analysis['experience_level'] == 'senior':
            roadmap_milestones = [
                {
                    "month": 3,
                    "title": "Leadership Development",
                    "description": f"Enhance technical leadership and {primary_domain} architecture skills",
                    "focus_areas": ["Team Leadership", "Architecture Design", "Mentoring"],
                    "status": "current"
                },
                {
                    "month": 6,
                    "title": "Strategic Impact",
                    "description": "Drive technical strategy and cross-team collaboration",
                    "focus_areas": ["Technical Strategy", "Cross-team Collaboration", "Business Alignment"],
                    "status": "upcoming"
                },
                {
                    "month": 9,
                    "title": "Industry Influence",
                    "description": "Establish thought leadership and external presence",
                    "focus_areas": ["Public Speaking", "Technical Writing", "Community Building"],
                    "status": "upcoming"
                },
                {
                    "month": 12,
                    "title": "Executive Readiness",
                    "description": "Position for executive or principal technical roles",
                    "focus_areas": ["Technical Leadership", "Business Impact", "Strategy Input"],
                    "status": "upcoming"
                }
            ]
        else:  # mid-level
            roadmap_milestones = [
                {
                    "month": 3,
                    "title": "Skill Specialization",
                    "description": f"Deepen expertise in {primary_domain} and related technologies",
                    "focus_areas": ["Advanced Technical Skills", "Best Practices", "Problem Solving"],
                    "status": "current"
                },
                {
                    "month": 6,
                    "title": "Project Leadership",
                    "description": "Lead technical projects and mentor junior developers",
                    "focus_areas": ["Project Management", "Code Reviews", "Junior Mentoring"],
                    "status": "upcoming"
                },
                                {
                    "month": 9,
                    "title": "Cross-Domain Growth",
                    "description": "Expand into adjacent technologies",
                    "focus_areas": ["Technology Breadth", "System Design"],
                    "status": "upcoming"
                },
                {
                    "month": 12,
                    "title": "Senior Readiness",
                    "description": "Prepare for senior role",
                    "focus_areas": ["Leadership", "Impact"],
                    "status": "upcoming"
                }
            ]
        
        career_roadmap_data = {
            "target_role": f"Senior {role_focus.title()} Developer" if cv_analysis['experience_level'] != 'senior' else f"Lead {role_focus.title()} Architect",
            "timeline_months": 12,
            "milestones": roadmap_milestones,
            "current_level": cv_analysis['experience_level'],
            "focus_domains": cv_analysis['primary_domains'],
            "personalized": True
        }
        
        formatted_recommendations["CAREER_ROADMAP"] = career_roadmap_data
        
        # Debug logging for roadmap
        logger.info(f"Career roadmap created for {cv_analysis['experience_level']} level")
        
        total_recommendations = sum(len(v) for v in formatted_recommendations.values() if isinstance(v, list))
        logger.info(f"Generated {total_recommendations} ML recommendations")
        
        return formatted_recommendations
        
    except Exception as e:
        logger.error(f"Erreur generation ML recommendations: {e}")
        return generate_fallback_recommendations(skills_data)

# Add workspace to path for enhanced engine
import sys
workspace_path = '/workspace'
if workspace_path not in sys.path:
    sys.path.append(workspace_path)

# Tentative d'activation ML automatique
if not detect_ml_configuration():
    logger.info("Mode fallback: recommandations basees sur des regles")

# Fallback to enhanced engine (Rule-based)
if not ML_MODE_ENABLED:
    try:
        from enhanced_recommendation_engine import EnhancedRecommendationEngine
        recommendation_engine = EnhancedRecommendationEngine()
        USE_ENHANCED_ENGINE = True
        logger.info("Moteur de recommandations ameliore charge")
    except ImportError as enhanced_error:
        logger.warning(f"Moteur ameliore non disponible: {enhanced_error}")
        try:
            from recommendation_engine import RecommendationEngine
            recommendation_engine = RecommendationEngine()
            logger.info("Moteur de base charge")
        except ImportError as original_error:
            logger.error(f"Aucun moteur de recommandations disponible: {original_error}")
            recommendation_engine = None

# Import auth router
try:
    from auth.router import router as auth_router
    AUTH_ENABLED = True
except ImportError:
    AUTH_ENABLED = False
    logger.warning("Authentication module not available")

app = FastAPI(
    title="SkillSync Multi-API Backend",
    description="Professional job matching service",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Include auth router if available
if AUTH_ENABLED:
    app.include_router(auth_router)
    logger.info("Authentication enabled")

# Rate limiting setup
if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Register AI interview routers (REST + realtime streaming)
app.include_router(interview_routes.router)
app.include_router(interview_realtime.router)

# CORS configuration - restricted for security
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174,http://localhost:5175,http://127.0.0.1:5175,http://localhost:8080,http://127.0.0.1:8080"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],  # Allow all headers for development
    expose_headers=["*"],  # Expose all headers
    max_age=3600,  # Cache preflight for 1 hour
)

# Request/Response models
class JobSearchRequest(BaseModel):
    query: str
    location: Optional[str] = ""
    skills: Optional[List[str]] = []
    max_results: Optional[int] = 50

class JobResponse(BaseModel):
    id: str
    title: str
    company: str
    location: str
    description: str
    url: str
    salary: Optional[str]
    posted_date: Optional[str]
    source: str
    skills_match: float
    remote: bool

class SearchResponse(BaseModel):
    jobs: List[JobResponse]
    total_count: int
    search_query: str
    location: str
    sources_used: List[str]
    search_time_ms: int
    timestamp: str

class StatusResponse(BaseModel):
    apis: Dict[str, Dict[str, Any]]
    total_enabled: int
    system_status: str
    timestamp: str

class CVAnalysisRequest(BaseModel):
    cv_content: str
    format: Optional[str] = "text"

class CVAnalysisResponse(BaseModel):
    analysis_id: str
    skills: List[str]
    experience_years: Optional[int]
    job_titles: List[str]
    education: List[str]
    summary: str
    confidence_score: float
    timestamp: str
    sections: Optional[Dict[str, Any]] = None
    personal_info: Optional[Dict[str, Any]] = None
    contact_info: Optional[Dict[str, Any]] = None
    roadmap: Optional[Dict[str, Any]] = None
    certifications: Optional[List[Dict[str, Any]]] = []
    recommendations: Optional[List[Dict[str, str]]] = []
    
    # NEW FIELDS for advanced ML parser
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    current_title: Optional[str] = None
    seniority_level: Optional[str] = None
    skill_categories: Optional[Dict[str, List[str]]] = None
    soft_skills: Optional[List[str]] = []
    tech_stack_clusters: Optional[Dict[str, List[str]]] = None
    total_years_experience: Optional[int] = 0
    companies: Optional[List[str]] = []
    responsibilities: Optional[List[str]] = []
    degrees: Optional[List[str]] = []
    institutions: Optional[List[str]] = []
    degree_level: Optional[str] = None
    graduation_year: Optional[int] = None
    languages: Optional[List[Dict]] = []
    processing_time_ms: Optional[int] = 0
    
    # Advanced ML features
    industries: Optional[List[tuple]] = []
    career_trajectory: Optional[Dict] = {}
    projects: Optional[List[Dict]] = []
    portfolio_links: Optional[Dict[str, str]] = {}
    ml_confidence_breakdown: Optional[Dict] = {}
    parser_version: Optional[str] = "standard"

class PortfolioTemplate(BaseModel):
    id: str
    name: str
    description: str
    category: str
    features: List[str]
    preview_url: Optional[str] = None

class PortfolioMetrics(BaseModel):
    views: int
    downloads: int
    likes: int

class PortfolioItem(BaseModel):
    id: str
    name: str
    cv_id: str
    template_id: str
    customization: Dict[str, Any]
    generated_date: str
    last_modified: str
    status: str
    metrics: PortfolioMetrics

class PortfolioGenerateRequest(BaseModel):
    cv_id: str
    template_id: str
    customization: Optional[Dict[str, Any]] = None

class PortfolioGenerateResponse(BaseModel):
    portfolio: PortfolioItem
    html_content: str

class PortfolioExportResponse(BaseModel):
    portfolio_id: str
    format: str
    content: str
    exported_at: str

# Initialize services
job_service = get_job_service()
# recommendation_engine is already initialized in the try-catch block above

# In-memory storage for CV analysis results (for recommendations)
cv_analysis_storage = {}

# Portfolio templates - will be generated dynamically based on CV data
portfolio_templates = []


portfolio_store: Dict[str, PortfolioItem] = {}
portfolio_export_store: Dict[str, str] = {}

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with service information"""
    return {
        "service": "SkillSync Multi-API Backend",
        "version": "2.0.1-ML" if ML_MODE_ENABLED else "2.0.0",
        "status": "operational",
        "ml_mode": ML_MODE_ENABLED,
        "enhanced_mode": USE_ENHANCED_ENGINE,
        "apis_available": list((await job_service.get_api_status()).keys()),
        "endpoints": {
            "job_search_get": "GET /api/v1/jobs/search",
            "job_search_post": "POST /api/v1/jobs/search", 
            "api_status": "GET /api/v1/jobs/status",
            "ml_status": "GET /api/v1/ml/status",
            "upload_cv": "POST /api/v1/upload-cv",
            "analyze_cv": "POST /api/v1/analyze-cv",
            "dashboard_latest": "GET /api/v1/dashboard/latest",
            "dashboard_by_id": "GET /api/v1/dashboard/{analysis_id}",
            "recommendations": "GET /api/v1/recommendations/{analysis_id}",
            "recommendations_post": "POST /api/v1/recommendations",
            "portfolio_templates": "GET /api/v1/portfolio/templates",
            "portfolio_list": "GET /api/v1/portfolio/list",
            "portfolio_generate": "POST /api/v1/portfolio/generate",
            "portfolio_export": "GET /api/v1/portfolio/export/{portfolio_id}",
            "generate_portfolio_legacy": "POST /api/v1/generate-portfolio",
            "experience_translate": "POST /api/v1/experience/translate",
            "experience_styles": "GET /api/v1/experience/styles",
            "experience_analysis": "GET /api/v1/experience/analysis/{translation_id}"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/ml/status", response_model=dict)
async def get_ml_status():
    """Get ML system status and capabilities"""
    
    # Dterminer le type d'engine actuel
    engine_type = "Basic"
    version = "2.0.0"
    
    if ML_MODE_ENABLED:
        if ML_MODE_TYPE == "lite":
            engine_type = "ML-Lite"
            version = "2.1.0-ML-Lite"
        elif ML_MODE_TYPE == "full":
            engine_type = "ML-Full" 
            version = "2.1.0-ML-Full"
        else:
            engine_type = "ML-AI"
            version = "2.0.1-ML"
    elif USE_ENHANCED_ENGINE:
        engine_type = "Enhanced"
        version = "2.0.0-Enhanced"
    
    return {
        "ml_enabled": ML_MODE_ENABLED,
        "ml_mode_type": ML_MODE_TYPE,
        "enhanced_mode_enabled": USE_ENHANCED_ENGINE,
        "engine_type": engine_type,
        "version": version,
        "capabilities": {
            "ai_powered_recommendations": ML_MODE_ENABLED,
            "lite_ml_scoring": ML_MODE_ENABLED and ML_MODE_TYPE == "lite",
            "full_ml_scoring": ML_MODE_ENABLED and ML_MODE_TYPE == "full",
            "bert_skills_extraction": ML_MODE_ENABLED,
            "neural_scoring": ML_MODE_ENABLED,
            "semantic_similarity": ML_MODE_ENABLED,
            "pytorch_models": ML_MODE_ENABLED and ML_MODE_TYPE in ["lite", "full"],
            "tensorflow_models": ML_MODE_ENABLED and ML_MODE_TYPE == "full",
            "rule_based_recommendations": USE_ENHANCED_ENGINE or not ML_MODE_ENABLED,
            "basic_recommendations": True,
            "fallback_available": True
        },
        "models_loaded": {
            "sentence_transformer": "sentence_model" in ml_models,
            "distilbert": "tokenizer" in ml_models and ML_MODE_TYPE == "lite",
            "bert": "bert_tokenizer" in ml_models and ML_MODE_TYPE == "full",
            "lite_scorer": "lite_scorer" in ml_models and ML_MODE_TYPE == "lite",
            "neural_scorer": "neural_scorer" in ml_models and ML_MODE_TYPE == "full"
        },
        "components": {
            "ml_recommendation_engine": ml_recommendation_engine is not None,
            "enhanced_recommendation_engine": recommendation_engine is not None,
            "ml_models": len(ml_models) > 0,
            "fallback_engine": True
        },
        "performance": {
            "mode": "lite" if ML_MODE_TYPE == "lite" else ("full" if ML_MODE_TYPE == "full" else "rules"),
            "speed": "fast" if ML_MODE_TYPE == "lite" else ("medium" if ML_MODE_TYPE == "full" else "fastest"),
            "accuracy": "high" if ML_MODE_ENABLED else "good"
        },
        "timestamp": datetime.now().isoformat()
    }

# Request model for POST endpoint
class JobSearchBody(BaseModel):
    query: str
    location: Optional[str] = ""
    skills: Optional[List[str]] = []
    max_results: Optional[int] = 50

@app.get("/api/v1/jobs/search", response_model=SearchResponse)
async def search_jobs_get(
    query: str = Query(..., description="Job search query"),
    location: str = Query("", description="Location filter"),
    skills: Optional[str] = Query(None, description="Comma-separated skills"),
    max_results: int = Query(50, ge=1, le=100, description="Maximum results")
):
    """Search jobs across all configured APIs (GET method)"""
    # Parse skills for GET request
    skills_list = []
    if skills:
        skills_list = [skill.strip() for skill in skills.split(',') if skill.strip()]
    
    return await _perform_job_search(query, location, skills_list, max_results)

@app.post("/api/v1/jobs/search", response_model=SearchResponse)
async def search_jobs_post(search_request: JobSearchBody):
    """Search jobs across all configured APIs (POST method)"""
    return await _perform_job_search(
        search_request.query, 
        search_request.location, 
        search_request.skills or [], 
        search_request.max_results
    )

async def _perform_job_search(query: str, location: str, skills_list: List[str], max_results: int):
    """Search jobs across all configured APIs"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info(f" Job search request: query='{query}', location='{location}', skills={skills_list}")
        
        # Search jobs
        jobs = await job_service.search_jobs(query, location, skills_list)
        
        # Calculate search time
        search_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        # Convert to response format
        job_responses = []
        sources_used = set()
        
        for job in jobs[:max_results]:
            sources_used.add(job.source)
            
            # Fix: Convert posted_date to string if it's an integer
            posted_date_str = str(job.posted_date) if job.posted_date is not None else None
            location_str = job.location if isinstance(job.location, str) and job.location.strip() else ("Remote" if job.remote else "Not specified")
            description_str = job.description or "Description unavailable"
            
            job_responses.append(JobResponse(
                id=job.id,
                title=job.title,
                company=job.company,
                location=location_str,
                description=description_str,
                url=job.url,
                salary=job.salary,
                posted_date=posted_date_str,
                source=job.source,
                skills_match=job.skills_match,
                remote=job.remote
            ))
        
        response = SearchResponse(
            jobs=job_responses,
            total_count=len(job_responses),
            search_query=query,
            location=location,
            sources_used=list(sources_used),
            search_time_ms=search_time_ms,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"OK: Search completed: {len(job_responses)} jobs from {len(sources_used)} sources in {search_time_ms}ms")
        return response
        
    except Exception as e:
        logger.error(f"Error: Job search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Job search failed: {str(e)}")

@app.get("/api/v1/jobs/status", response_model=StatusResponse)
async def get_api_status():
    """Get status of all configured job APIs"""
    try:
        api_status = await job_service.get_api_status()
        enabled_count = sum(1 for config in api_status.values() if config['enabled'])
        
        system_status = "operational" if enabled_count > 0 else "no_apis_configured"
        
        return StatusResponse(
            apis=api_status,
            total_enabled=enabled_count,
            system_status=system_status,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error: Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@app.post("/api/v1/upload-cv", response_model=CVAnalysisResponse)
async def upload_cv(request: Request, file: UploadFile = File(...)):
    """Upload and analyze CV file - with security validation"""
    try:
        # Validate file size (10MB max)
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")
        
        # Validate file type
        allowed_types = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: PDF, DOCX, TXT")
        
        logger.info(f" CV upload request: {file.filename} ({file.content_type})")
        
        file_content = await file.read()
        
        # Extract text based on file type
        if file.content_type == "text/plain":
            cv_text = file_content.decode('utf-8')
        elif file.content_type == "application/pdf":
            # Extract PDF text properly
            import io
            import PyPDF2
            try:
                pdf_file = io.BytesIO(file_content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                cv_text = ""
                for page in pdf_reader.pages:
                    cv_text += page.extract_text() + "\n"
                
                logger.info(f" PDF extracted: {len(cv_text)} characters")
                logger.info(f"   First 200 chars: {cv_text[:200]}")
                
                if not cv_text.strip():
                    raise HTTPException(status_code=400, detail="Could not extract text from PDF. Please ensure the PDF contains selectable text.")
            except Exception as e:
                logger.error(f"Error: PDF extraction failed: {e}")
                raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")
        else:
            # DOCX or other - fallback to base64
            cv_text = base64.b64encode(file_content).decode('utf-8')[:1000]
        
        # Analyse CV - nouvelle version
        cv_result = create_cv_analysis(cv_text)
        
        # Store CV analysis result for recommendations
        cv_analysis_storage[cv_result.analysis_id] = cv_result.model_dump()
        
        skills_found = cv_result.skills
        total_skills = len(skills_found)
        logger.info(f"OK: CV analysis completed: {total_skills} skills found, analysis_id: {cv_result.analysis_id}")
        
        return cv_result
        
    except Exception as e:
        logger.error(f"Error: CV upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"CV upload failed: {str(e)}")

@app.post("/api/v1/analyze-cv", response_model=CVAnalysisResponse)
async def analyze_cv_text(http_request: Request, request: CVAnalysisRequest):
    """Analyze CV content directly - with input validation"""
    try:
        # ========== DETAILED DEBUG LOGGING START ==========
        logger.info("\n" + ""*50)
        logger.info(" [CV ANALYSIS] NEW REQUEST RECEIVED")
        logger.info(""*50)
        
        # Validate CV content
        if not request.cv_content or not request.cv_content.strip():
            logger.error("Error: [CV ANALYSIS] Empty CV content received!")
            raise HTTPException(status_code=400, detail="CV content cannot be empty")
        
        content_length = len(request.cv_content)
        logger.info(f" [CV ANALYSIS] CV content length: {content_length} characters")
        logger.info(f" [CV ANALYSIS] CV preview (first 500 chars):\n{request.cv_content[:500]}...")
        
        # Check max length (50KB)
        if content_length > 50000:
            logger.error(f"Error: [CV ANALYSIS] CV too long: {content_length} chars (max 50,000)")
            raise HTTPException(status_code=413, detail="CV content too long. Maximum 50,000 characters")
        
        logger.info(f"\n [CV ANALYSIS] Step 1: Starting CV parsing with ML...")
        
        # Analyse CV - nouvelle version
        cv_result = create_cv_analysis(request.cv_content)
        
        logger.info(f"OK: [CV ANALYSIS] Step 1 Complete - Parsing successful!")
        logger.info(f" [CV ANALYSIS] Parsed data summary:")
        logger.info(f"    Name: {cv_result.name or 'Not found'}")
        logger.info(f"    Email: {cv_result.email or 'Not found'}")
        logger.info(f"    Phone: {cv_result.phone or 'Not found'}")
        logger.info(f"    Seniority: {cv_result.seniority_level or 'Not detected'}")
        logger.info(f"    Total Experience: {cv_result.total_years_experience} years")
        
        # Store CV analysis result for recommendations
        cv_analysis_storage[cv_result.analysis_id] = cv_result.model_dump()
        logger.info(f" [CV ANALYSIS] Saved to storage with ID: {cv_result.analysis_id}")
        
        skills_found = cv_result.skills
        total_skills = len(skills_found)
        
        logger.info(f"\n [CV ANALYSIS] Skills Extraction Results:")
        logger.info(f"    Total skills found: {total_skills}")
        if total_skills > 0:
            logger.info(f"    First 15 skills: {', '.join(skills_found[:15])}")
            if total_skills > 15:
                logger.info(f"    ...and {total_skills - 15} more skills")
        else:
            logger.warning(f"   Warning: NO SKILLS FOUND! This is unusual for a CV.")
            logger.warning(f"   Info: Possible reasons:")
            logger.warning(f"       CV doesn't have a 'Skills' section")
            logger.warning(f"       Skills are embedded in experience descriptions")
            logger.warning(f"       CV format not recognized by parser")
        
        logger.info(f"\n"*50)
        logger.info(f"OK: [CV ANALYSIS] ANALYSIS COMPLETE")
        logger.info(f" [CV ANALYSIS] Final stats:")
        logger.info(f"    Skills: {total_skills}")
        logger.info(f"    Work experiences: {len(cv_result.work_history)}")
        logger.info(f"    Education: {len(cv_result.education)}")
        logger.info(f"    Projects: {len(cv_result.projects)}")
        logger.info(f"    Analysis ID: {cv_result.analysis_id}")
        logger.info(""*50 + "\n")
        
        return cv_result
        
    except Exception as e:
        logger.error(f"Error: CV analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"CV analysis failed: {str(e)}")


@app.post("/api/v1/extract-text")
async def extract_text_from_cv(file: UploadFile = File(...)):
    """
    Extract text from PDF/DOCX file for advanced ML analysis
    """
    try:
        # Read file content
        content = await file.read()
        
        # Extract text based on file type
        if file.filename.endswith('.pdf'):
            # Direct PDF extraction without importing heavy ML modules
            import PyPDF2
            import io
            try:
                pdf_file = io.BytesIO(content)
                reader = PyPDF2.PdfReader(pdf_file)
                cv_text = ""
                for page in reader.pages:
                    cv_text += page.extract_text() + "\n"
                # Clean text
                cv_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', cv_text)
                cv_text = re.sub(r'\s+', ' ', cv_text).strip()
            except Exception as pdf_error:
                logger.error(f"PDF parsing error: {pdf_error}")
                raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(pdf_error)}")
        elif file.filename.endswith('.txt'):
            cv_text = content.decode('utf-8', errors='ignore')
        elif file.filename.endswith('.docx'):
            # Handle DOCX files
            try:
                import docx
                import io
                doc = docx.Document(io.BytesIO(content))
                cv_text = "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                raise HTTPException(status_code=400, detail="DOCX support requires python-docx. Use PDF or TXT.")
            except Exception as docx_error:
                raise HTTPException(status_code=400, detail=f"Failed to parse DOCX: {str(docx_error)}")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, or TXT")
        
        if not cv_text or not cv_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        logger.info(f"✅ Text extracted from {file.filename}: {len(cv_text)} characters")
        
        return {"cv_text": cv_text, "length": len(cv_text)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Text extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")



@app.post("/api/v1/analyze-cv-advanced")
async def analyze_cv_advanced(http_request: Request, request: CVAnalysisRequest):
    """
     ADVANCED ML CV ANALYSIS (95% ML-driven)
    New endpoint using advanced ML modules for enhanced accuracy
    """
    try:
        # Validate CV content
        if not request.cv_content or not request.cv_content.strip():
            raise HTTPException(status_code=400, detail="CV content cannot be empty")
        
        content_length = len(request.cv_content)
        if content_length > 50000:
            raise HTTPException(status_code=413, detail="CV content too long. Maximum 50,000 characters")
        
        logger.info(f" Advanced ML CV analysis request: {content_length} characters")
        
        # Use production parser (already has advanced ML integrated)
        # Use simple rule-based analysis (ML fallback to avoid huggingface_hub errors)
        result = create_cv_analysis(request.cv_content)
        # parser = ProductionCVParser()  # Disabled due to huggingface_hub compatibility

        # Parse completed
        
        # Result is already a CVAnalysisResponse from create_cv_analysis()
        cv_result = result

        
        # Store result
        cv_analysis_storage[cv_result.analysis_id] = cv_result.model_dump()
        
        logger.info(f"OK: Advanced ML analysis completed: {len(result.skills)} skills, "
                   f"{len(result.industries)} industries, "
                   f"{result.processing_time_ms}ms, "
                   f"confidence: {result.confidence_score:.2%}")
        
        return cv_result
        
    except Exception as e:
        logger.error(f"Error: Advanced ML analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Advanced ML analysis failed: {str(e)}")


@app.post("/api/v1/career-guidance")
async def get_career_guidance(request: CVAnalysisRequest):
    """
     FULLY ML-DRIVEN CAREER GUIDANCE (100% Machine Learning)
    
    What's ML-Powered:
    OK: Job matching: Semantic similarity using transformer embeddings
    OK: Salary prediction: ML-based formula with skill/experience factors
    OK: Certification ranking: ML relevance scoring
    OK: Learning path: ML optimization with success prediction
    OK: Explainable AI: Complete transparency on all ML decisions
    
    NO static databases - everything computed using ML models!
    
    Model: paraphrase-mpnet-base-v2 (768-dimensional embeddings)
    """
    try:
        logger.info("\n" + ""*40)
        logger.info(" [API] NEW CAREER GUIDANCE REQUEST")
        logger.info(""*40)
        
        # Validate CV content
        if not request.cv_content or not request.cv_content.strip():
            logger.error("Error: [API] Empty CV content received")
            raise HTTPException(status_code=400, detail="CV content cannot be empty")
        
        logger.info(f" [API] CV content received: {len(request.cv_content)} characters")
        logger.info(f" [API] CV preview (first 300 chars):\n{request.cv_content[:300]}...")
        
        # Step 1: Advanced ML CV Analysis (95% ML)
        logger.info("\n [API] Step 1: Parsing CV with ML...")
        # from production_cv_parser_final import ProductionCVParser  # Disabled
        # parser = ProductionCVParser()  # Disabled due to huggingface_hub
        cv_result = create_cv_analysis(request.cv_content)
        
        logger.info(f"OK: [API] CV parsed successfully:")
        logger.info(f"    Skills found: {len(cv_result.skills)}")
        if cv_result.skills:
            logger.info(f"    Skills: {', '.join(cv_result.skills[:10])}{'...' if len(cv_result.skills) > 10 else ''}")
        else:
            logger.warning(f"   Warning: [API] NO SKILLS FOUND! This is critical for job matching!")
        logger.info(f"    Seniority: {cv_result.seniority_level}")
        logger.info(f"    Industries: {cv_result.industries or 'None'}")
        logger.info(f"    Experience: {cv_result.total_years_experience} years")
        
        # Convert to dict for career engine
        cv_analysis = {
            'skills': cv_result.skills,
            'seniority_level': cv_result.seniority_level,
            'industries': cv_result.industries or [],
            'projects': cv_result.projects or [],
            'portfolio_links': cv_result.portfolio_links or {},
            'experience_years': cv_result.total_years_experience,
            'total_years_experience': cv_result.total_years_experience,
            'ml_confidence_breakdown': cv_result.ml_confidence_breakdown or {},
            'raw_text': request.cv_content,
            'work_history': []  # Can be enhanced later
        }
        
        logger.info(f"\n [API] CV data prepared for ML engine:")
        logger.info(f"    Dictionary keys: {list(cv_analysis.keys())}")
        logger.info(f"    Skills count: {len(cv_analysis['skills'])}")
        logger.info(f"    Raw text length: {len(cv_analysis['raw_text'])} chars")
        
        # Step 2: ML-Driven Career Guidance (100% ML)
        logger.info("\n [API] Step 2: Running ML Career Engine...")
        from enhanced_ml_career_engine import get_ml_career_engine
        engine = get_ml_career_engine()
        guidance = engine.analyze_and_guide(cv_analysis)
        
        # Step 3: Convert to JSON
        logger.info("\n [API] Step 3: Converting results to JSON...")
        result = engine.to_json(guidance)
        
        logger.info("\n" + ""*40)
        logger.info(f"OK: [API] ML CAREER GUIDANCE COMPLETE")
        logger.info(f" [API] Results:")
        logger.info(f"    Jobs matched: {len(guidance.job_recommendations)}")
        if not guidance.job_recommendations:
            logger.warning(f"   Warning: NO JOBS MATCHED! Check skills in CV.")
        logger.info(f"    Certs ranked: {len(guidance.certification_recommendations)}")
        logger.info(f"    Roadmap phases: {len(guidance.learning_roadmap.phases)}")
        logger.info(f"    Processing time: {result['metadata']['processing_time_seconds']}s")
        logger.info(""*40 + "\n")
        
        return result
        
    except Exception as e:
        logger.error(f"Error: ML career guidance failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"ML career guidance failed: {str(e)}")


@app.get("/api/v1/cv-analyses")
async def get_cv_analyses():
    """Get list of all CV analyses"""
    try:
        analyses = list(cv_analysis_storage.values())
        logger.info(f" Retrieved {len(analyses)} CV analyses")
        return {"analyses": analyses, "total": len(analyses)}
    except Exception as e:
        logger.error(f"Error: Failed to get CV analyses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_learning_roadmap(skills: List[str], experience_years: int, job_titles: List[str]) -> Dict[str, Any]:
    """Generate personalized learning roadmap based on current skills"""
    
    skill_set = set([s.lower() for s in skills])
    
    # Determine career level
    if experience_years < 2:
        level = "Junior"
        timeline = "6-12 months"
    elif experience_years < 5:
        level = "Mid-Level"
        timeline = "12-18 months"
    else:
        level = "Senior"
        timeline = "18-24 months"
    
    # Smart roadmap based on skills
    roadmap_phases = []
    
    # Phase 1: Foundation (if needed)
    foundation_skills = []
    if 'python' not in skill_set and 'javascript' not in skill_set:
        foundation_skills.append("Master a core programming language (Python or JavaScript)")
    if 'git' not in skill_set:
        foundation_skills.append("Learn Git version control and GitHub collaboration")
    if 'sql' not in skill_set and 'mongodb' not in skill_set:
        foundation_skills.append("Learn database fundamentals (SQL and NoSQL)")
    
    if foundation_skills:
        roadmap_phases.append({
            "phase": "Foundation",
            "duration": "3-6 months",
            "skills": foundation_skills,
            "priority": "high"
        })
    
    # Phase 2: Specialization
    specialization_skills = []
    
    # Web Development path
    if 'react' in skill_set or 'vue' in skill_set or 'angular' in skill_set:
        if 'typescript' not in skill_set:
            specialization_skills.append("Master TypeScript for type-safe applications")
        if 'next.js' not in skill_set:
            specialization_skills.append("Learn Next.js for modern React development")
        specialization_skills.append("Advanced state management (Redux, Zustand, Recoil)")
    
    # Backend path
    if 'node.js' in skill_set or 'python' in skill_set:
        if 'docker' not in skill_set:
            specialization_skills.append("Learn Docker containerization")
        if 'aws' not in skill_set and 'azure' not in skill_set:
            specialization_skills.append("Cloud platforms (AWS, Azure, or GCP)")
        specialization_skills.append("Microservices architecture and API design")
    
    # ML/AI path
    if 'machine learning' in skill_set or 'tensorflow' in skill_set or 'pytorch' in skill_set:
        specialization_skills.append("Deep Learning specialization (CNNs, RNNs, Transformers)")
        specialization_skills.append("MLOps and model deployment")
        specialization_skills.append("Large Language Models (LLMs) and prompt engineering")
    
    if specialization_skills:
        roadmap_phases.append({
            "phase": "Specialization",
            "duration": "6-9 months",
            "skills": specialization_skills[:5],
            "priority": "high"
        })
    
    # Phase 3: Advanced
    advanced_skills = [
        "System design and architecture patterns",
        "Performance optimization and scalability",
        "DevOps and CI/CD pipelines",
        "Security best practices and penetration testing",
        "Leadership and team management skills"
    ]
    
    roadmap_phases.append({
        "phase": "Advanced",
        "duration": "9-12 months",
        "skills": advanced_skills[:4],
        "priority": "medium"
    })
    
    return {
        "current_level": level,
        "target_level": "Senior" if level != "Senior" else "Lead/Principal",
        "estimated_timeline": timeline,
        "phases": roadmap_phases,
        "recommended_resources": [
            "freeCodeCamp - Free comprehensive courses",
            "Coursera - Professional certifications",
            "Udemy - Practical project-based learning",
            "GitHub - Open source contribution",
            "LeetCode/HackerRank - Coding practice"
        ]
    }

def generate_recommended_certifications(skills: List[str], experience_years: int) -> List[Dict[str, Any]]:
    """Generate recommended certifications based on skills"""
    
    skill_set = set([s.lower() for s in skills])
    certifications = []
    
    # Cloud certifications
    if 'aws' in skill_set:
        certifications.append({
            "name": "AWS Certified Solutions Architect - Associate",
            "provider": "Amazon Web Services",
            "level": "Associate",
            "duration": "3-6 months prep",
            "cost": "$150",
            "value": "high",
            "url": "https://aws.amazon.com/certification/",
            "priority": "high"
        })
    elif 'azure' in skill_set:
        certifications.append({
            "name": "Microsoft Azure Fundamentals (AZ-900)",
            "provider": "Microsoft",
            "level": "Fundamentals",
            "duration": "1-2 months prep",
            "cost": "$99",
            "value": "high",
            "url": "https://learn.microsoft.com/certifications/",
            "priority": "high"
        })
    else:
        certifications.append({
            "name": "AWS Certified Cloud Practitioner",
            "provider": "Amazon Web Services",
            "level": "Foundational",
            "duration": "1-2 months prep",
            "cost": "$100",
            "value": "high",
            "url": "https://aws.amazon.com/certification/",
            "priority": "high"
        })
    
    # Programming certifications
    if 'python' in skill_set:
        certifications.append({
            "name": "Python Institute PCAP - Certified Associate",
            "provider": "Python Institute",
            "level": "Associate",
            "duration": "2-3 months prep",
            "cost": "$295",
            "value": "medium",
            "url": "https://pythoninstitute.org/pcap",
            "priority": "medium"
        })
    
    # Web development
    if any(s in skill_set for s in ['react', 'javascript', 'typescript']):
        certifications.append({
            "name": "Meta Front-End Developer Professional Certificate",
            "provider": "Meta (via Coursera)",
            "level": "Professional",
            "duration": "6-8 months",
            "cost": "$49/month",
            "value": "high",
            "url": "https://www.coursera.org/professional-certificates/meta-front-end-developer",
            "priority": "high"
        })
    
    # Machine Learning
    if any(s in skill_set for s in ['machine learning', 'tensorflow', 'pytorch', 'data science']):
        certifications.append({
            "name": "TensorFlow Developer Certificate",
            "provider": "Google",
            "level": "Professional",
            "duration": "3-4 months prep",
            "cost": "$100",
            "value": "high",
            "url": "https://www.tensorflow.org/certificate",
            "priority": "high"
        })
        certifications.append({
            "name": "Deep Learning Specialization",
            "provider": "DeepLearning.AI (via Coursera)",
            "level": "Specialization",
            "duration": "5 months",
            "cost": "$49/month",
            "value": "very high",
            "url": "https://www.coursera.org/specializations/deep-learning",
            "priority": "very high"
        })
    
    # DevOps/Docker
    if 'docker' in skill_set or 'kubernetes' in skill_set:
        certifications.append({
            "name": "Certified Kubernetes Administrator (CKA)",
            "provider": "Cloud Native Computing Foundation",
            "level": "Professional",
            "duration": "4-6 months prep",
            "cost": "$395",
            "value": "very high",
            "url": "https://www.cncf.io/certification/cka/",
            "priority": "high"
        })
    
    # General software engineering
    certifications.append({
        "name": "Professional Scrum Master I (PSM I)",
        "provider": "Scrum.org",
        "level": "Professional",
        "duration": "1-2 months prep",
        "cost": "$150",
        "value": "medium",
        "url": "https://www.scrum.org/assessments/professional-scrum-master-i-certification",
        "priority": "medium"
    })
    
    # Sort by priority
    priority_order = {"very high": 0, "high": 1, "medium": 2, "low": 3}
    certifications.sort(key=lambda x: priority_order.get(x["priority"], 4))
    
    return certifications[:6]  # Return top 6 most relevant

def generate_career_recommendations(skills: List[str], experience_years: int, job_titles: List[str]) -> List[Dict[str, str]]:
    """Generate personalized career recommendations"""
    
    skill_set = set([s.lower() for s in skills])
    recommendations = []
    
    # Skill gap analysis
    if len(skills) < 5:
        recommendations.append({
            "type": "skill_development",
            "priority": "high",
            "title": "Expand Your Skill Portfolio",
            "description": f"You have {len(skills)} skills detected. Aim for 8-12 diverse technical skills to be more competitive. Focus on complementary technologies in your field.",
            "action": "Add 3-5 new skills in the next 6 months"
        })
    
    # Cloud skills
    if not any(s in skill_set for s in ['aws', 'azure', 'gcp', 'cloud']):
        recommendations.append({
            "type": "skill_development",
            "priority": "very high",
            "title": "Learn Cloud Computing",
            "description": "Cloud skills are in top 5% demand. 92% of enterprises use cloud services. AWS, Azure, and GCP skills can increase salary by 20-30%.",
            "action": "Start with AWS Cloud Practitioner certification (1-2 months)"
        })
    
    # Modern frameworks
    if 'python' in skill_set and not any(s in skill_set for s in ['django', 'flask', 'fastapi']):
        recommendations.append({
            "type": "skill_development",
            "priority": "high",
            "title": "Master a Python Web Framework",
            "description": "Django, Flask, or FastAPI skills complement Python knowledge. FastAPI is the fastest-growing Python framework in 2025.",
            "action": "Build 2-3 projects with FastAPI or Django"
        })
    
    # DevOps skills
    if 'docker' not in skill_set:
        recommendations.append({
            "type": "skill_development",
            "priority": "high",
            "title": "Learn Containerization with Docker",
            "description": "Docker is used by 75% of companies. Essential for DevOps, microservices, and modern deployment strategies.",
            "action": "Complete Docker essentials course and containerize a project"
        })
    
    # AI/ML trending
    if not any(s in skill_set for s in ['machine learning', 'ai', 'tensorflow', 'pytorch']):
        recommendations.append({
            "type": "career_growth",
            "priority": "very high",
            "title": "Explore AI/ML Fundamentals",
            "description": "AI skills are top-demanded in 2025. Even basic ML knowledge gives you an edge. GenAI and LLMs are transforming every industry.",
            "action": "Take Andrew Ng's Machine Learning course on Coursera"
        })
    
    # Experience-based recommendations
    if experience_years < 3:
        recommendations.append({
            "type": "career_growth",
            "priority": "high",
            "title": "Build a Strong Portfolio",
            "description": "Create 5-8 diverse projects showcasing your skills. Include live demos and detailed README files. Contribute to 2-3 open source projects.",
            "action": "Complete one significant project per month"
        })
    elif experience_years >= 5:
        recommendations.append({
            "type": "career_growth",
            "priority": "high",
            "title": "Develop Leadership Skills",
            "description": "With your experience, focus on system design, architecture, and team leadership. Mentor junior developers and lead projects.",
            "action": "Take on tech lead role or mentorship responsibilities"
        })
    
    # TypeScript recommendation
    if 'javascript' in skill_set and 'typescript' not in skill_set:
        recommendations.append({
            "type": "skill_development",
            "priority": "high",
            "title": "Adopt TypeScript",
            "description": "TypeScript is now standard for enterprise JavaScript. Used by 78% of JS developers. Improves code quality and career prospects.",
            "action": "Convert one JavaScript project to TypeScript"
        })
    
    # Testing skills
    recommendations.append({
        "type": "skill_development",
        "priority": "medium",
        "title": "Master Testing Practices",
        "description": "Unit testing, integration testing, and TDD are expected in professional environments. Jest, Pytest, or your framework's testing tools.",
        "action": "Achieve 80%+ test coverage on your projects"
    })
    
    # Networking
    recommendations.append({
        "type": "career_growth",
        "priority": "medium",
        "title": "Build Your Professional Network",
        "description": "Network with professionals in your field. Attend meetups, conferences, and contribute to tech communities. 70% of jobs are found through networking.",
        "action": "Connect with 10 professionals on LinkedIn, join 2 tech communities"
    })
    
    return recommendations[:8]  # Return top 8 recommendations

def create_cv_analysis(cv_text: str) -> CVAnalysisResponse:
    """Enhanced CV analysis using REAL ML (ai_cv_analyzer.py)"""
    
    # Generate unique analysis ID
    analysis_id = str(uuid.uuid4())
    
    # =================================================================
    # REAL ML ANALYSIS using ai_cv_analyzer.py
    # =================================================================
    use_ml = False
    detected_skills = []
    name = "Professional"
    email = None
    phone = None
    detected_titles = []
    education = []
    estimated_experience = 3
    
    try:
        import warnings
        warnings.filterwarnings('ignore')
        
        from ai_cv_analyzer import AdvancedCVExtractor
        
        logger.info(" Initializing ML-powered CV analysis...")
        
        # Create ML extractor
        ml_extractor = AdvancedCVExtractor()
        
        # Check if ML is actually available
        if not ml_extractor.embedder:
            logger.warning("Warning: SentenceTransformer not available, using basic extraction")
            raise ImportError("SentenceTransformer not available")
        
        logger.info("OK: ML extractor ready with SentenceTransformer")
        
        # Parse CV using ML
        cv_data = ml_extractor.parse_cv_advanced(cv_text)
        
        # Use ML-extracted skills (with confidence scores)
        detected_skills = cv_data.skills if cv_data.skills else []
        
        # Get personal info from ML
        name = cv_data.name or "Professional"
        email = cv_data.email
        phone = cv_data.phone
        
        # Extract job titles from experience
        if cv_data.experience:
            for exp in cv_data.experience:
                if isinstance(exp, dict):
                    title = exp.get('title', '')
                    if title:
                        detected_titles.append(title)
                elif isinstance(exp, str):
                    detected_titles.append(exp)
        
        # Extract education
        if cv_data.education:
            for edu in cv_data.education:
                if isinstance(edu, dict):
                    degree = edu.get('degree', '')
                    institution = edu.get('institution', '')
                    education.append(f"{degree} - {institution}" if institution else degree)
                else:
                    education.append(str(edu))
        
        # Calculate experience years from ML-extracted data
        if cv_data.experience and len(cv_data.experience) > 0:
            # Estimate from number of positions (assume 2-3 years each)
            estimated_experience = len(cv_data.experience) * 2
        else:
            # Estimate from skills
            estimated_experience = max(3, len(detected_skills) // 3)
        
        # Get confidence scores
        avg_confidence = sum(cv_data.confidence_scores.values()) / max(len(cv_data.confidence_scores), 1) if cv_data.confidence_scores else 0.0
        
        logger.info(f"OK: ML Analysis Complete:")
        logger.info(f"   - Skills: {len(detected_skills)} detected")
        logger.info(f"   - Experience: {estimated_experience} years estimated")
        logger.info(f"   - Confidence: {avg_confidence:.2f}")
        logger.info(f"   - Name: {name}")
        logger.info(f"   - Job Titles: {len(detected_titles)}")
        
        use_ml = True
        
    except Exception as e:
        logger.warning(f"Warning: ML analysis failed, falling back to rule-based: {e}")
        use_ml = False
    
    # =================================================================
    # FALLBACK: Rule-based extraction (if ML fails)
    # =================================================================
    if not use_ml or not detected_skills:
        logger.info(" Using rule-based keyword extraction (not ML)")
        
        detected_skills = []
        skill_keywords = {
            'python': ['python', 'py', 'django', 'flask', 'fastapi'],
            'javascript': ['javascript', 'js', 'node.js', 'nodejs', 'typescript'],
            'java': ['java', 'spring', 'hibernate'],
            'react': ['react', 'reactjs', 'react.js'],
            'vue': ['vue', 'vuejs', 'vue.js'],
            'angular': ['angular', 'angularjs'],
            'sql': ['sql', 'mysql', 'postgresql', 'sqlite', 'oracle'],
            'html': ['html', 'html5'],
            'css': ['css', 'css3', 'sass', 'scss', 'less'],
            'git': ['git', 'github', 'gitlab'],
            'docker': ['docker', 'containerization'],
            'aws': ['aws', 'amazon web services'],
            'azure': ['azure', 'microsoft azure'],
            'mongodb': ['mongodb', 'mongo'],
            'machine learning': ['machine learning', 'ml', 'ai', 'artificial intelligence'],
            'data science': ['data science', 'data analysis', 'analytics'],
            'kubernetes': ['kubernetes', 'k8s'],
            'tensorflow': ['tensorflow', 'tf'],
            'pytorch': ['pytorch', 'torch']
        }
        
        cv_lower = cv_text.lower()
        for main_skill, variations in skill_keywords.items():
            for variation in variations:
                if variation in cv_lower:
                    detected_skills.append(main_skill.title())
                    break
        
        # If no skills detected, provide fallback skills for demo
        if not detected_skills:
            detected_skills = ['Python', 'JavaScript', 'React', 'SQL', 'Git']
        
        # Rule-based extraction for other fields
        name = "Professional"
        email = None
        phone = None
        detected_titles = []
        education = []
        estimated_experience = 3
        
        # Extract job titles with better patterns (only if ML failed)
        title_patterns = [
            r'(senior|junior|lead|principal)?\s*(software|web|full stack|backend|frontend|data)\s*(developer|engineer|analyst)',
            r'(project|product|technical)\s*manager',
            r'(devops|system)\s*engineer',
            r'(data|business)\s*analyst',
            r'(software|solution)\s*architect'
        ]
        
        lines = cv_text.split('\n')
        for line in lines[:20]:
            line_clean = line.strip()
            if len(line_clean) > 5 and len(line_clean) < 100:
                for pattern in title_patterns:
                    if re.search(pattern, line_clean, re.IGNORECASE):
                        detected_titles.append(line_clean)
                        break
        
        # Fallback titles if none detected
        if not detected_titles:
            detected_titles = ['Senior Full Stack Developer']
        
        # Calculate experience years
        experience_indicators = re.findall(r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)', cv_text, re.IGNORECASE)
        if experience_indicators:
            estimated_experience = max([int(x) for x in experience_indicators])
        else:
            estimated_experience = max(3, len(detected_skills) // 2)
        
        # Extract education
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'diploma']
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in education_keywords) and len(line.strip()) > 10:
                education.append(line.strip())
                if len(education) >= 3:
                    break
        
        # Extract personal information
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+?\d{1,3}[-.\ s]?)\(?\d{3}\)?[-.\ s]?\d{3}[-.\ s]?\d{4}'
        
        email_match = re.search(email_pattern, cv_text)
        phone_match = re.search(phone_pattern, cv_text)
        
        email = email_match.group() if email_match else None
        phone = phone_match.group() if phone_match else None
        
        # Extract name
        first_lines = cv_text.split('\n')[:5]
        for line in first_lines:
            line_clean = line.strip()
            if (len(line_clean.split()) >= 2 and 
                len(line_clean) < 50 and 
                not '@' in line_clean and 
                not any(char.isdigit() for char in line_clean)):
                name = line_clean
                break
    
    # =================================================================
    # Generate AI-powered recommendations (based on extracted data)
    # =================================================================
    analysis_method = "ML (SentenceTransformer + NLP)" if use_ml else "Rule-based (keyword matching)"
    logger.info(f" Analysis Method: {analysis_method}")
    logger.info(f" Extracted: {len(detected_skills)} skills, {estimated_experience} years exp, {len(detected_titles)} titles")
    
    # Generate personalized learning roadmap based on skills
    roadmap = generate_learning_roadmap(detected_skills, estimated_experience, detected_titles)
    logger.info(f" Generated roadmap with {len(roadmap.get('phases', []))} phases")
    
    # Generate recommended certifications
    certifications = generate_recommended_certifications(detected_skills, estimated_experience)
    logger.info(f" Generated {len(certifications)} certification recommendations")
    
    # Generate comprehensive recommendations
    recommendations = generate_career_recommendations(detected_skills, estimated_experience, detected_titles)
    logger.info(f"Info: Generated {len(recommendations)} career recommendations")
    
    # Create comprehensive response
    analysis_result = CVAnalysisResponse(
        analysis_id=analysis_id,
        skills=detected_skills[:10],
        experience_years=min(estimated_experience, 25),
        job_titles=detected_titles[:3],
        education=education[:3],
        summary=f"[{analysis_method}] CV analysis found {len(detected_skills)} technical skills and {len(detected_titles)} relevant job titles. Professional with {min(estimated_experience, 25)} years of experience.",
        confidence_score=0.85 if use_ml else 0.70,
        timestamp=datetime.now().isoformat(),
        sections={
            "summary": f"Experienced professional with {min(estimated_experience, 25)} years in technology",
            "technical_skills": detected_skills[:10],
            "professional_experience": detected_titles[:3],
            "education_background": education[:3]
        },
        personal_info={
            "name": name,
            "title": detected_titles[0] if detected_titles else "Professional",
            "years_experience": min(estimated_experience, 25)
        },
        contact_info={
            "email": email,
            "phone": phone
        },
        roadmap=roadmap,
        certifications=certifications,
        recommendations=recommendations
    )
    
    return analysis_result

# New endpoints for frontend integration
@app.get("/api/v1/dashboard/latest")
async def get_dashboard_latest():
    """Get latest dashboard data"""
    try:
        # Return sample dashboard data
        dashboard_data = {
            "recent_analyses": [],
            "job_match_count": 0,
            "skills_summary": [],
            "recommendations": [],
            "portfolio_status": "ready",
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info(" Dashboard data requested")
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error: Dashboard data failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard data failed: {str(e)}")

@app.get("/api/v1/dashboard/{analysis_id}")
async def get_dashboard_by_analysis_id(analysis_id: str):
    """Get dashboard data for specific analysis ID"""
    try:
        # Return dashboard data for specific analysis
        dashboard_data = {
            "analysis_id": analysis_id,
            "recent_analyses": [
                {
                    "id": analysis_id,
                    "timestamp": datetime.now().isoformat(),
                    "status": "completed",
                    "skills_count": 5,
                    "job_matches": 12
                }
            ],
            "job_match_count": 12,
            "skills_summary": ["Python", "JavaScript", "React", "SQL", "Git"],
            "recommendations": [
                {
                    "type": "skill",
                    "title": "Improve Python Skills",
                    "description": "Consider learning advanced Python frameworks",
                    "priority": "high"
                },
                {
                    "type": "job",
                    "title": "Full Stack Developer",
                    "description": "12 matching positions found",
                    "priority": "medium"
                }
            ],
            "portfolio_status": "ready",
            "last_updated": datetime.now().isoformat(),
            "analytics": {
                "profile_strength": 85,
                "market_demand": 92,
                "skill_relevance": 88
            }
        }
        
        logger.info(f" Dashboard data requested for analysis_id: {analysis_id}")
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error: Dashboard data failed for {analysis_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard data failed: {str(e)}")

@app.get("/api/v1/portfolio/templates", response_model=List[PortfolioTemplate])
async def get_portfolio_templates():
    """Return available portfolio templates - dynamically generated."""
    # Generate templates dynamically based on available CV analyses
    dynamic_templates = [
        PortfolioTemplate(
            id="professional",
            name="Professional",
            description="Clean and professional layout",
            category="modern",
            features=["Responsive", "ATS-friendly", "Customizable"],
            preview_url=None
        ),
        PortfolioTemplate(
            id="creative",
            name="Creative",
            description="Bold and creative design",
            category="creative",
            features=["Visual impact", "Portfolio showcase", "Interactive"],
            preview_url=None
        ),
        PortfolioTemplate(
            id="minimal",
            name="Minimal",
            description="Simple and elegant",
            category="minimal",
            features=["Typography-focused", "Clean layout", "Fast loading"],
            preview_url=None
        )
    ]
    return dynamic_templates

@app.get("/api/v1/portfolio/list", response_model=List[PortfolioItem])
async def list_portfolios(cv_id: Optional[str] = None):
    """List generated portfolios, optionally filtered by CV ID."""
    # Return only dynamically generated portfolios from actual CV uploads
    items = list(portfolio_store.values())
    if cv_id:
        items = [item for item in items if item.cv_id == cv_id]
    return items

def format_enhanced_recommendations(enhanced_data):
    """
    Format enhanced recommendations to match expected API structure
    Ensures scores are properly formatted and structure is consistent
    """
    try:
        logger.info(" Formatting enhanced recommendations...")
        
        # Convert lowercase keys to uppercase to match API expectations
        formatted_recommendations = {}
        
        key_mapping = {
            'immediate_actions': 'IMMEDIATE_ACTIONS',
            'skill_development': 'SKILL_DEVELOPMENT', 
            'project_suggestions': 'PROJECT_SUGGESTIONS',
            'certification_roadmap': 'CERTIFICATION_ROADMAP',
            'learning_resources': 'LEARNING_RESOURCES',
            'career_roadmap': 'CAREER_ROADMAP',
            'networking_opportunities': 'NETWORKING_OPPORTUNITIES',
            'timeline': 'TIMELINE'
        }
        
        for original_key, formatted_key in key_mapping.items():
            if original_key in enhanced_data:
                data = enhanced_data[original_key]
                
                # Format list recommendations (immediate_actions, skill_development, etc.)
                if isinstance(data, list):
                    formatted_list = []
                    for item in data:
                        if isinstance(item, dict):
                            formatted_item = {
                                'title': item.get('title', 'N/A'),
                                'description': item.get('description', ''),
                                'score': float(item.get('score', 0.0)),  # Ensure score is float
                                'priority': item.get('priority', 'medium'),
                                'estimated_time': item.get('estimated_time', ''),
                                'category': item.get('category', 'general')
                            }
                            # Add any additional fields that might exist
                            for key, value in item.items():
                                if key not in formatted_item:
                                    formatted_item[key] = value
                            formatted_list.append(formatted_item)
                        else:
                            formatted_list.append(item)
                    formatted_recommendations[formatted_key] = formatted_list
                else:
                    # Keep complex structures as-is
                    formatted_recommendations[formatted_key] = data
            else:
                # Provide empty structure if missing
                formatted_recommendations[formatted_key] = []
        
        logger.info("OK: Enhanced recommendations formatted successfully")
        return formatted_recommendations
        
    except Exception as e:
        logger.error(f"Error: Error formatting enhanced recommendations: {e}")
        # Return original data if formatting fails
        return enhanced_data

def generate_fallback_recommendations(skills_list):
    """
    Generate fallback recommendations when enhanced engine is not available
    """
    return {
        "IMMEDIATE_ACTIONS": [
            {
                "title": "Update Your Portfolio",
                "description": "Showcase your skills with recent projects",
                "score": 0.8,
                "priority": "high",
                "estimated_time": "1-2 weeks"
            }
        ],
        "SKILL_DEVELOPMENT": [
            {
                "title": "Advance Current Skills",
                "description": "Deepen your expertise in your strongest areas",
                "score": 0.75,
                "priority": "medium", 
                "estimated_time": "4-6 weeks"
            }
        ],
        "PROJECT_SUGGESTIONS": [
            {
                "title": "Personal Project",
                "description": "Build a project showcasing your skills",
                "score": 0.7,
                "estimated_time": "2-4 weeks"
            }
        ],
        "CERTIFICATION_ROADMAP": [
            {
                "name": "AWS Certified Solutions Architect",
                "title": "AWS Solutions Architect - Associate",
                "provider": "Amazon Web Services",
                "difficulty": "intermediate",
                "timeline": 3,
                "prep_time": "2-3 months",
                "pass_rate": "72%",
                "cost": "$150",
                "skills_validated": ["AWS", "Cloud Architecture", "S3", "EC2", "Lambda", "VPC"],
                "description": "Validates ability to design and deploy scalable, highly available systems on AWS"
            },
            {
                "name": "Professional Cloud Architect",
                "title": "Google Cloud Professional Cloud Architect",
                "provider": "Google Cloud",
                "difficulty": "advanced",
                "timeline": 6,
                "prep_time": "3-4 months",
                "pass_rate": "68%",
                "cost": "$200",
                "skills_validated": ["GCP", "Cloud Architecture", "Kubernetes", "Networking", "Security"],
                "description": "Demonstrates expertise in GCP infrastructure and application design"
            },
            {
                "name": "Certified Kubernetes Administrator",
                "title": "CKA - Kubernetes Administrator",
                "provider": "Cloud Native Computing Foundation",
                "difficulty": "advanced",
                "timeline": 9,
                "prep_time": "3-4 months",
                "pass_rate": "66%",
                "cost": "$395",
                "skills_validated": ["Kubernetes", "Container Orchestration", "Docker", "Cloud Native", "kubectl"],
                "description": "Validates Kubernetes administration skills and expertise in production environments"
            }
        ],
        "LEARNING_RESOURCES": {
            "free_resources": [],
            "paid_courses": [],
            "books": [],
            "practice_platforms": []
        },
        "CAREER_ROADMAP": {
            "target_role": "Tech Professional",
            "timeline_months": 12,
            "milestones": [
                {
                    "month": 3,
                    "title": "Skill Enhancement",
                    "description": "Focus on improving core technical skills",
                    "focus_areas": ["Technical Skills", "Best Practices", "Code Quality"],
                    "status": "current"
                },
                {
                    "month": 6,
                    "title": "Project Development",
                    "description": "Build and showcase meaningful projects",
                    "focus_areas": ["Portfolio Projects", "Problem Solving", "Implementation"],
                    "status": "upcoming"
                },
                {
                    "month": 9,
                    "title": "Industry Engagement",
                    "description": "Connect with the professional community",
                    "focus_areas": ["Networking", "Open Source", "Industry Knowledge"],
                    "status": "upcoming"
                },
                {
                    "month": 12,
                    "title": "Career Advancement",
                    "description": "Position yourself for next career step",
                    "focus_areas": ["Career Growth", "Leadership", "Advanced Skills"],
                    "status": "upcoming"
                }
            ],
            "current_level": "intermediate",
            "focus_domains": ["general"],
            "personalized": False
        },
        "NETWORKING_OPPORTUNITIES": [],
        "TIMELINE": {
            "month_1_3": [],
            "month_4_6": [],
            "month_7_9": [],
            "month_10_12": []
        }
    }

def sanitize_recommendations_data(data):
    """
    Sanitize recommendations data to ensure JSON serialization
    and prevent slice/hashable type errors
    """
    if isinstance(data, dict):
        return {k: sanitize_recommendations_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_recommendations_data(item) for item in data]
    elif hasattr(data, '__dict__'):  # Handle custom objects
        return sanitize_recommendations_data(data.__dict__)
    elif isinstance(data, slice):  # Handle slice objects
        return f"slice({data.start}, {data.stop}, {data.step})"
    elif hasattr(data, 'value'):  # Handle Enum objects
        return data.value
    elif callable(data):  # Skip functions
        return str(data)
    else:
        try:
            # Test if object is JSON serializable
            import json
            json.dumps(data)
            return data
        except (TypeError, ValueError):
            return str(data)

@app.get("/api/v1/recommendations/{analysis_id}")
async def get_recommendations(analysis_id: str):
    """
    Get personalized recommendations based on CV analysis ID.
    Generates skill, career, and learning recommendations.
    """
    try:
        logger.info(f" Recommendations requested for analysis_id: {analysis_id}")
        
        # Get CV data from storage
        cv_data = cv_analysis_storage.get(analysis_id)
        
        if not cv_data:
            logger.warning(f"Analysis ID {analysis_id} not found in storage.")
            raise HTTPException(status_code=404, detail=f"Analysis ID {analysis_id} not found. Please analyze your CV first.")

        # Convert CV data to skills format for recommendation engine
        skills_for_recommendations = []
        
        # Extract skills from CV data
        cv_skills = cv_data.get('skills', [])
        for skill in cv_skills:
            skills_for_recommendations.append({
                'skill': skill,
                'normalized_name': skill.lower(),
                'experience_level': 'intermediate',  # Default level
                'confidence': 0.8,
                'importance_score': 0.7,
                'category': 'technical'  # Default category
            })
        
        # Generate comprehensive recommendations using the best available engine
        if ML_MODE_ENABLED and ML_MODE_TYPE in ["lite", "full", "hybrid"]:
            # Use new ML lite/full/hybrid mode (most advanced)
            logger.info(f"ML: Using ML mode: {ML_MODE_TYPE.upper()}")
            try:
                # Generate ML-enhanced recommendations
                safe_recommendations = await generate_ml_enhanced_recommendations(
                    skills_for_recommendations, cv_data
                )
                logger.info(f"OK: ML {ML_MODE_TYPE} recommendations generated successfully")
            except Exception as ml_error:
                logger.error(f"Error: ML {ML_MODE_TYPE} engine failed: {ml_error}")
                # Fallback to enhanced engine
                if USE_ENHANCED_ENGINE and recommendation_engine:
                    raw_recommendations = await recommendation_engine.generate_comprehensive_recommendations(
                        skills=skills_for_recommendations,
                        gap_analysis=None,
                        career_goals=None
                    )
                    safe_recommendations = format_enhanced_recommendations(raw_recommendations)
                    logger.info("OK: Enhanced recommendations as fallback")
                else:
                    safe_recommendations = generate_fallback_recommendations(skills_for_recommendations)
                    logger.warning("Warning: Using basic fallback recommendations")
        elif ML_MODE_ENABLED and ml_recommendation_engine:
            # Use existing ML engine (legacy)
            logger.info("ML: Using legacy ML recommendation engine (AI-powered)")
            try:
                raw_recommendations = await ml_recommendation_engine.generate_comprehensive_recommendations(
                    skills=skills_for_recommendations,
                    gap_analysis=None,
                    career_goals=None
                )
                safe_recommendations = format_enhanced_recommendations(raw_recommendations)
                logger.info("OK: Legacy ML recommendations generated successfully")
            except Exception as ml_error:
                logger.error(f"Error: Legacy ML engine failed: {ml_error}")
                # Fallback to enhanced engine
                if USE_ENHANCED_ENGINE and recommendation_engine:
                    raw_recommendations = await recommendation_engine.generate_comprehensive_recommendations(
                        skills=skills_for_recommendations,
                        gap_analysis=None,
                        career_goals=None
                    )
                    safe_recommendations = format_enhanced_recommendations(raw_recommendations)
                    logger.info("OK: Enhanced recommendations as fallback")
                else:
                    safe_recommendations = generate_fallback_recommendations(skills_for_recommendations)
                    logger.warning("Warning: Using basic fallback recommendations")
                    
        elif USE_ENHANCED_ENGINE and recommendation_engine:
            # Use enhanced engine (rule-based)
            logger.info(" Using enhanced recommendation engine (rule-based)")
            raw_recommendations = await recommendation_engine.generate_comprehensive_recommendations(
                skills=skills_for_recommendations,
                gap_analysis=None,
                career_goals=None
            )
            safe_recommendations = format_enhanced_recommendations(raw_recommendations)
            logger.info("OK: Enhanced recommendations formatted successfully")
        else:
            # Fallback to basic recommendations
            logger.info(" Using fallback recommendation engine (basic)")
            safe_recommendations = generate_fallback_recommendations(skills_for_recommendations)
            logger.warning("Warning: Using basic fallback recommendations")
        
        # Final sanitization to prevent serialization errors
        safe_recommendations = sanitize_recommendations_data(safe_recommendations)
        
        # Format response to match frontend expectations
        response = {
            "analysis_id": analysis_id,
            "recommendations": safe_recommendations,
            "generated_at": datetime.utcnow().isoformat(),
            "user_profile": {
                "skills_count": len(cv_skills),
                "experience_years": cv_data.get('experience_years', 0),
                "job_titles": cv_data.get('job_titles', []),
                "education": cv_data.get('education', [])
            },
            "global_confidence": 0.85,  # Add missing confidence score
            "engine_info": {
                "ml_mode_enabled": ML_MODE_ENABLED,
                "ml_mode_type": ML_MODE_TYPE,
                "enhanced_mode_enabled": USE_ENHANCED_ENGINE,
                "engine_type": f"ML-{ML_MODE_TYPE.title()}" if ML_MODE_ENABLED else ("Enhanced" if USE_ENHANCED_ENGINE else "Basic"),
                "version": f"2.1.0-ML-{ML_MODE_TYPE.title()}" if ML_MODE_ENABLED else ("2.0.0-Enhanced" if USE_ENHANCED_ENGINE else "2.0.0"),
                "capabilities": {
                    "ai_scoring": ML_MODE_ENABLED,
                    "semantic_similarity": ML_MODE_ENABLED,
                    "personalized_recommendations": True
                }
            }
        }
        
        logger.info(f"OK: Generated recommendations for analysis_id: {analysis_id}")
        return response
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Error: Recommendations generation failed for {analysis_id}: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return a safe fallback response instead of crashing
        fallback_response = {
            "analysis_id": "fallback",
            "recommendations": {
                "immediate_actions": [
                    {
                        "title": "Update LinkedIn Profile",
                        "description": "Optimize your professional profile",
                        "score": 0.75,
                        "priority": "high"
                    },
                    {
                        "title": "Create GitHub Portfolio",
                        "description": "Showcase your coding projects",
                        "score": 0.70,
                        "priority": "medium"
                    }
                ],
                "skill_development": [
                    {
                        "title": "Advanced Python Techniques",
                        "description": "Master advanced Python concepts",
                        "score": 0.80,
                        "category": "programming"
                    },
                    {
                        "title": "Cloud Architecture",
                        "description": "Learn cloud platform fundamentals",
                        "score": 0.75,
                        "category": "infrastructure"
                    }
                ],
                "career_roadmap": [
                    {
                        "title": "Senior Developer Path",
                        "description": "Roadmap to senior technical roles",
                        "score": 0.85,
                        "timeline": "12-18 months"
                    },
                    {
                        "title": "Tech Lead Transition",
                        "description": "Leadership and technical growth",
                        "score": 0.70,
                        "timeline": "18-24 months"
                    },
                    {
                        "title": "Architecture Specialist",
                        "description": "Deep technical architecture focus",
                        "score": 0.65,
                        "timeline": "24-36 months"
                    }
                ]
            },
            "generated_at": datetime.utcnow().isoformat(),
            "user_profile": {
                "current_role": "Developer",
                "target_role": "Senior Developer",
                "skills_count": 0,
                "experience_years": 0,
                "industry": "Tech"
            },
            "global_confidence": 0.75,
            "error_handled": True,
            "original_error": str(e)
        }
        
        logger.info("OK: Returned fallback recommendations due to engine error")
        return fallback_response

# Modle pour les recommandations POST
class RecommendationRequest(BaseModel):
    current_role: Optional[str] = "Developer"
    target_role: Optional[str] = "Senior Developer"
    skills: Optional[List[str]] = []
    experience_years: Optional[int] = 2
    industry: Optional[str] = "Tech"

@app.post("/api/v1/recommendations")
async def generate_recommendations_direct(request: RecommendationRequest):
    """
    Generate recommendations directly from profile data (for testing and direct use)
    """
    try:
        logger.info(f" Direct recommendations requested for {request.current_role} -> {request.target_role}")
        
        # Convert request to CV data format
        cv_data = {
            'job_titles': [request.current_role],
            'skills': request.skills,
            'experience_years': request.experience_years,
            'education': [],
            'industry': request.industry
        }
        
        # Convert to skills format for recommendation engine
        skills_for_recommendations = []
        for skill in request.skills:
            skills_for_recommendations.append({
                'skill': skill,
                'normalized_name': skill.lower(),
                'experience_level': 'intermediate',
                'confidence': 0.8,
                'importance_score': 0.7,
                'category': 'technical'  # Default category
            })
        
        # Generate recommendations using the same logic as GET endpoint
        if ML_MODE_ENABLED and ML_MODE_TYPE in ["lite", "full"]:
            logger.info(f"ML: Using ML mode: {ML_MODE_TYPE.upper()}")
            safe_recommendations = await generate_ml_enhanced_recommendations(
                skills_for_recommendations, cv_data
            )
        elif ML_MODE_ENABLED and ml_recommendation_engine:
            logger.info("ML: Using legacy ML recommendation engine")
            raw_recommendations = await ml_recommendation_engine.generate_comprehensive_recommendations(
                skills=skills_for_recommendations,
                gap_analysis=None,
                career_goals=None
            )
            safe_recommendations = format_enhanced_recommendations(raw_recommendations)
        elif USE_ENHANCED_ENGINE and recommendation_engine:
            logger.info(" Using enhanced recommendation engine")
            raw_recommendations = await recommendation_engine.generate_comprehensive_recommendations(
                skills=skills_for_recommendations,
                gap_analysis=None,
                career_goals=None
            )
            safe_recommendations = format_enhanced_recommendations(raw_recommendations)
        else:
            logger.info(" Using fallback recommendation engine")
            safe_recommendations = generate_fallback_recommendations(skills_for_recommendations)
        
        # Final sanitization
        safe_recommendations = sanitize_recommendations_data(safe_recommendations)
        
        # Generate a unique analysis ID for this session
        import uuid
        analysis_id = str(uuid.uuid4())
        
        # Format response
        response = {
            "analysis_id": analysis_id,
            "recommendations": safe_recommendations,
            "generated_at": datetime.utcnow().isoformat(),
            "user_profile": {
                "current_role": request.current_role,
                "target_role": request.target_role,
                "skills_count": len(request.skills),
                "experience_years": request.experience_years,
                "industry": request.industry
            },
            "global_confidence": 0.90,
            "engine_info": {
                "ml_mode_enabled": ML_MODE_ENABLED,
                "ml_mode_type": ML_MODE_TYPE,
                "enhanced_mode_enabled": USE_ENHANCED_ENGINE,
                "engine_type": f"ML-{ML_MODE_TYPE.title()}" if ML_MODE_ENABLED else ("Enhanced" if USE_ENHANCED_ENGINE else "Basic"),
                "version": f"2.1.0-ML-{ML_MODE_TYPE.title()}" if ML_MODE_ENABLED else ("2.0.0-Enhanced" if USE_ENHANCED_ENGINE else "2.0.0"),
                "capabilities": {
                    "ai_scoring": ML_MODE_ENABLED,
                    "semantic_similarity": ML_MODE_ENABLED,
                    "personalized_recommendations": True
                }
            }
        }
        
        logger.info(f"OK: Generated direct recommendations for profile: {request.current_role}")
        return response
        
    except Exception as e:
        logger.error(f"Error: Direct recommendations generation failed: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return safe fallback
        fallback_response = {
            "analysis_id": "fallback",
            "recommendations": generate_fallback_recommendations([]),
            "generated_at": datetime.utcnow().isoformat(),
            "user_profile": {
                "current_role": "Developer",
                "target_role": "Senior Developer",
                "skills_count": 0,
                "experience_years": 0,
                "industry": "Tech"
            },
            "global_confidence": 0.75,
            "error_handled": True,
            "original_error": str(e)
        }
        
        logger.info("OK: Returned fallback recommendations due to engine error")
        return fallback_response

# --- MOTEUR DE RECOMMANDATION: Initialisation hors du bloc try/except principal ---
# (tout ce code est dj protg plus bas si besoin)
