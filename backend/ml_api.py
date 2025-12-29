"""
Advanced ML API Endpoints for SkillSync
Provides endpoints for the new ML-powered features
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Import ML components
from ml_models.advanced_recommendation_engine import AdvancedRecommendationEngine
from ml_models.ml_trainer import MLTrainer
from ml_models.skills_extractor import SkillsExtractorModel
from ml_models.similarity_engine import SimilarityEngine
from ml_models.neural_scorer import NeuralScorer

logger = logging.getLogger(__name__)

# Initialize ML components
ml_models_path = Path("models")
ml_models_path.mkdir(exist_ok=True)

advanced_engine = AdvancedRecommendationEngine(models_path=str(ml_models_path))
ml_trainer = MLTrainer(output_dir=str(ml_models_path))

# Create router
router = APIRouter(prefix="/api/v1/ml", tags=["Advanced ML"])

# Pydantic models
class CVAnalysisRequest(BaseModel):
    cv_data: Dict[str, Any]
    analysis_depth: Optional[str] = "comprehensive"

class JobMatchingRequest(BaseModel):
    cv_data: Dict[str, Any]
    job_list: List[Dict[str, Any]]
    top_k: Optional[int] = 10

class RecommendationRequest(BaseModel):
    cv_data: Dict[str, Any]
    recommendation_types: Optional[List[str]] = ["jobs", "courses", "certifications", "projects"]

class TrainingRequest(BaseModel):
    training_mode: Optional[str] = "quick"  # "quick" or "full"
    epochs: Optional[int] = 2
    batch_size: Optional[int] = 8

@router.post("/analyze-cv")
async def analyze_cv_advanced(request: CVAnalysisRequest):
    """
    Advanced CV analysis using BERT NER and ML models
    """
    try:
        logger.info("Starting advanced CV analysis")
        
        # Perform comprehensive CV analysis
        analysis_result = advanced_engine.analyze_cv_profile(request.cv_data)
        
        # Add metadata
        analysis_result["analysis_metadata"] = {
            "analysis_depth": request.analysis_depth,
            "ml_models_used": ["BERT_NER", "Rule_Based_Fallback"],
            "confidence_level": analysis_result.get("ml_confidence", "medium")
        }
        
        logger.info("CV analysis completed successfully")
        return {
            "success": True,
            "analysis": analysis_result
        }
        
    except Exception as e:
        logger.error(f"Error in CV analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/extract-skills")
async def extract_skills(cv_text: str):
    """
    Extract skills from CV text using BERT NER
    """
    try:
        logger.info("Extracting skills using BERT NER")
        
        # Extract skills
        skills_result = advanced_engine.skills_extractor.extract_skills(cv_text)
        
        # Categorize skills
        categorized = advanced_engine.skills_extractor.categorize_skills(
            skills_result.get('skills', [])
        )
        
        # Get suggestions
        suggestions = advanced_engine.skills_extractor.get_skill_suggestions(
            skills_result.get('skills', [])
        )
        
        return {
            "success": True,
            "extracted_skills": skills_result,
            "categorized_skills": categorized,
            "skill_suggestions": suggestions
        }
        
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        raise HTTPException(status_code=500, detail=f"Skill extraction failed: {str(e)}")

@router.post("/job-matching")
async def advanced_job_matching(request: JobMatchingRequest):
    """
    Advanced job matching using semantic similarity and neural scoring
    """
    try:
        logger.info(f"Matching CV against {len(request.job_list)} jobs")
        
        # Score job matches
        job_matches = advanced_engine.score_job_matches(
            request.cv_data,
            request.job_list,
            request.top_k
        )
        
        # Add summary statistics
        scores = [match['scores']['combined'] for match in job_matches]
        
        summary = {
            "total_jobs_analyzed": len(request.job_list),
            "top_matches_returned": len(job_matches),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "best_score": max(scores) if scores else 0,
            "score_distribution": {
                "excellent": len([s for s in scores if s >= 0.8]),
                "good": len([s for s in scores if 0.6 <= s < 0.8]),
                "fair": len([s for s in scores if 0.4 <= s < 0.6]),
                "poor": len([s for s in scores if s < 0.4])
            }
        }
        
        return {
            "success": True,
            "matches": job_matches,
            "summary": summary
        }
        
    except Exception as e:
        logger.error(f"Error in job matching: {e}")
        raise HTTPException(status_code=500, detail=f"Job matching failed: {str(e)}")

@router.post("/personalized-recommendations")
async def get_personalized_recommendations(request: RecommendationRequest):
    """
    Generate personalized recommendations using ML models
    """
    try:
        logger.info("Generating personalized recommendations")
        
        # Generate recommendations
        recommendations = advanced_engine.get_personalized_recommendations(
            request.cv_data,
            request.recommendation_types
        )
        
        # Add metadata
        recommendations["recommendation_metadata"] = {
            "types_requested": request.recommendation_types,
            "ml_analysis_method": recommendations.get("profile_summary", {}).get("analysis_method", "hybrid"),
            "recommendation_count": {
                rec_type: len(recs) 
                for rec_type, recs in recommendations.get("recommendations", {}).items()
            }
        }
        
        return {
            "success": True,
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation generation failed: {str(e)}")

@router.post("/calculate-similarity")
async def calculate_similarity(cv_text: str, job_text: str):
    """
    Calculate semantic similarity between CV and job description
    """
    try:
        # Calculate similarity
        similarity_score = advanced_engine.similarity_engine.calculate_similarity(cv_text, job_text)
        
        # Get detailed similarity if we have structured data
        cv_data = {"text": cv_text}
        job_data = {"text": job_text}
        
        detailed_similarity = advanced_engine.similarity_engine.calculate_cv_job_similarity(
            cv_data, job_data
        )
        
        return {
            "success": True,
            "similarity_score": similarity_score,
            "detailed_similarity": detailed_similarity,
            "interpretation": {
                "score": similarity_score,
                "level": "high" if similarity_score > 0.7 else "medium" if similarity_score > 0.4 else "low",
                "explanation": f"Semantic similarity of {similarity_score:.2f} indicates " +
                             ("strong alignment" if similarity_score > 0.7 else 
                              "moderate alignment" if similarity_score > 0.4 else "limited alignment")
            }
        }
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity calculation failed: {str(e)}")

@router.post("/explain-recommendation")
async def explain_recommendation(cv_data: Dict[str, Any], recommendation: Dict[str, Any]):
    """
    Provide detailed explanation for a specific recommendation
    """
    try:
        # Generate explanation
        explanation = advanced_engine.explain_recommendation(cv_data, recommendation)
        
        return {
            "success": True,
            "explanation": explanation
        }
        
    except Exception as e:
        logger.error(f"Error explaining recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

@router.post("/train-models")
async def train_ml_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train or retrain ML models (runs in background)
    """
    try:
        logger.info(f"Starting ML training in {request.training_mode} mode")
        
        def run_training():
            try:
                if request.training_mode == "quick":
                    result = ml_trainer.quick_setup()
                    logger.info("Quick ML setup completed")
                else:
                    result = ml_trainer.train_all_models(
                        epochs=request.epochs,
                        batch_size=request.batch_size
                    )
                    logger.info("Full ML training completed")
                
                # Save training results
                with open(ml_models_path / "latest_training_result.json", "w") as f:
                    import json
                    json.dump(result, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Training failed: {e}")
                # Save error information
                with open(ml_models_path / "training_error.json", "w") as f:
                    import json
                    json.dump({"error": str(e), "status": "failed"}, f, indent=2)
        
        # Add training task to background
        background_tasks.add_task(run_training)
        
        return {
            "success": True,
            "message": f"ML training started in {request.training_mode} mode",
            "training_config": {
                "mode": request.training_mode,
                "epochs": request.epochs,
                "batch_size": request.batch_size
            },
            "status_check_endpoint": "/api/v1/ml/training-status"
        }
        
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=f"Training initiation failed: {str(e)}")

@router.get("/training-status")
async def get_training_status():
    """
    Get the status of the latest training session
    """
    try:
        # Check for latest training result
        result_file = ml_models_path / "latest_training_result.json"
        error_file = ml_models_path / "training_error.json"
        
        if result_file.exists():
            with open(result_file, "r") as f:
                import json
                result = json.load(f)
            
            return {
                "success": True,
                "status": "completed",
                "result": result
            }
        elif error_file.exists():
            with open(error_file, "r") as f:
                import json
                error_info = json.load(f)
            
            return {
                "success": False,
                "status": "failed",
                "error": error_info
            }
        else:
            return {
                "success": True,
                "status": "no_training_found",
                "message": "No training session found. Run /train-models first."
            }
        
    except Exception as e:
        logger.error(f"Error checking training status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.get("/model-info")
async def get_model_info():
    """
    Get information about loaded ML models
    """
    try:
        info = {
            "models_loaded": {
                "skills_extractor": {
                    "available": advanced_engine.skills_extractor.model is not None,
                    "method": "BERT_NER" if advanced_engine.skills_extractor.model is not None else "rule_based",
                    "model_path": advanced_engine.skills_extractor.model_path
                },
                "similarity_engine": {
                    "available": advanced_engine.similarity_engine.model is not None,
                    "method": "sentence_transformers" if advanced_engine.similarity_engine.model is not None else "fallback",
                    "model_name": advanced_engine.similarity_engine.model_name
                },
                "neural_scorer": {
                    "available": advanced_engine.neural_scorer.model is not None,
                    "method": "neural_network" if advanced_engine.neural_scorer.model is not None else "rule_based",
                    "model_path": advanced_engine.neural_scorer.model_path
                }
            },
            "dependencies": {
                "transformers": "available" if hasattr(advanced_engine.skills_extractor, 'tokenizer') else "not_available",
                "sentence_transformers": "available" if advanced_engine.similarity_engine.model is not None else "not_available",
                "tensorflow": "available" if advanced_engine.neural_scorer.model is not None else "not_available"
            },
            "models_directory": str(ml_models_path)
        }
        
        return {
            "success": True,
            "model_info": info
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Model info retrieval failed: {str(e)}")

@router.post("/test-complete-system")
async def test_complete_system():
    """
    Test the complete ML recommendation system with sample data
    """
    try:
        logger.info("Testing complete ML system")
        
        # Sample CV data
        sample_cv = {
            "skills": ["Python", "React", "FastAPI", "Machine Learning", "AWS", "Docker", "SQL"],
            "experience_years": 5,
            "domain": "Backend Developer",
            "industry": "FinTech",
            "level": "mid",
            "text": "Experienced backend developer with 5 years in Python, React, and cloud technologies"
        }
        
        # Sample job data
        sample_jobs = [
            {
                "title": "Senior React Developer",
                "required_skills": ["React", "JavaScript", "Node.js", "Testing"],
                "domain": "Frontend",
                "min_experience": 4,
                "description": "Looking for senior React developer for fintech company"
            },
            {
                "title": "Machine Learning Engineer", 
                "required_skills": ["Python", "Machine Learning", "TensorFlow", "AWS"],
                "domain": "ML/AI",
                "min_experience": 3,
                "description": "ML engineer role focusing on production ML systems"
            },
            {
                "title": "AWS Certified Solutions Architect",
                "required_skills": ["AWS", "Cloud Architecture", "Security"],
                "domain": "Cloud",
                "min_experience": 4,
                "description": "Cloud architect role with AWS focus",
                "type": "certification"
            }
        ]
        
        # Test CV analysis
        cv_analysis = advanced_engine.analyze_cv_profile(sample_cv)
        
        # Test job matching
        job_matches = advanced_engine.score_job_matches(sample_cv, sample_jobs[:2], top_k=5)
        
        # Test personalized recommendations
        recommendations = advanced_engine.get_personalized_recommendations(sample_cv)
        
        # Compile test results
        test_results = {
            "cv_analysis": {
                "skills_extracted": cv_analysis.get("extracted_skills", {}).get("skills", []),
                "confidence": cv_analysis.get("ml_confidence", "unknown"),
                "method": cv_analysis.get("extracted_skills", {}).get("method", "unknown")
            },
            "job_matching": {
                "jobs_analyzed": len(sample_jobs[:2]),
                "top_match": job_matches[0] if job_matches else None,
                "average_score": sum(match['scores']['combined'] for match in job_matches) / len(job_matches) if job_matches else 0
            },
            "recommendations": {
                "recommendation_types": list(recommendations.get("recommendations", {}).keys()),
                "total_recommendations": sum(len(recs) for recs in recommendations.get("recommendations", {}).values())
            }
        }
        
        # Add formatted output for user
        formatted_output = f"""
 Test du systme de recommandations complet...

 Profil CV analys:
   Comptences: {sample_cv['skills']}
   Exprience: {sample_cv['experience_years']} ans
   Rle: {sample_cv['domain']}
   Industrie: {sample_cv['industry']}
   Niveau: {sample_cv['level']}

 Recommandations personnalises:
"""
        
        # Add top recommendations
        if job_matches:
            formatted_output += "\n"
            for i, match in enumerate(job_matches[:3], 1):
                job = match['job']
                scores = match['scores']
                explanation = match['explanation']
                
                formatted_output += f"""
{i}. {job['title']} ({job.get('type', 'job')})
   Score combin: {scores['combined']:.3f}
   Similarit: {scores['similarity']:.3f}
   Score neural: {scores['neural']:.3f}
   Domaine: {job.get('domain', 'N/A')}
   Explications:
      Vos comptences en {', '.join(list(set(sample_cv['skills']).intersection(set(job.get('required_skills', []))))[:2])} correspondent parfaitement"""
                
                # Add experience level info if available
                if sample_cv.get('level'):
                    formatted_output += f"\n      Adapt  votre niveau d'exprience ({sample_cv['level']})"
                
                # Add learning opportunities
                missing_skills = list(set(job.get('required_skills', [])) - set(sample_cv['skills']))[:2]
                if missing_skills:
                    formatted_output += f"\n      Vous apprendrez {', '.join(missing_skills)} pour progresser"
        
        formatted_output += "\n Test termin avec succs !"
        
        return {
            "success": True,
            "test_results": test_results,
            "formatted_output": formatted_output,
            "sample_data": {
                "cv": sample_cv,
                "jobs": sample_jobs
            }
        }
        
    except Exception as e:
        logger.error(f"Error in system test: {e}")
        raise HTTPException(status_code=500, detail=f"System test failed: {str(e)}")

# Health check endpoint
@router.get("/health")
async def ml_health_check():
    """
    Health check for ML components
    """
    return {
        "status": "healthy",
        "ml_components": {
            "advanced_engine": "initialized",
            "skills_extractor": "ready",
            "similarity_engine": "ready", 
            "neural_scorer": "ready"
        },
        "message": "Advanced ML API is running"
    }