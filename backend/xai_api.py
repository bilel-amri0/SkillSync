"""
XAI API Endpoints - FastAPI endpoints for Explainable AI functionality
Provides API access to SHAP/LIME explanations for frontend integration
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import time
import asyncio
from datetime import datetime
import uuid

# Import our XAI explainer
from xai_explainer import XAIExplainer

logger = logging.getLogger(__name__)

# Initialize XAI explainer
xai_explainer = XAIExplainer()

# Pydantic models for request/response
class CVAnalysisRequest(BaseModel):
    cv_content: Dict[str, Any]
    extracted_skills: List[Dict[str, Any]]
    matching_score: Optional[Dict[str, Any]] = None
    gap_analysis: Optional[Dict[str, Any]] = None
    models: Optional[Dict[str, Any]] = None

class ExplanationResponse(BaseModel):
    explanation_type: str
    explanation_text: str
    confidence: float
    supporting_evidence: List[str]
    explanation_time: float
    visual_data: Optional[Dict[str, Any]] = None
    lime_data: Optional[Dict[str, Any]] = None
    shap_data: Optional[Dict[str, Any]] = None

class XAIMetricsResponse(BaseModel):
    total_explanations: int
    average_explanation_time: float
    explanation_breakdown: Dict[str, int]
    average_accuracy: float
    explainability_percentage: float

def create_xai_api() -> FastAPI:
    """Create XAI API application"""
    
    app = FastAPI(
        title="SkillSync XAI API",
        description="Explainable AI endpoints for transparent decision making",
        version="2.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "SkillSync XAI API",
            "version": "2.0.0",
            "endpoints": {
                "explanations": "/api/xai/explain-analysis",
                "metrics": "/api/xai/metrics",
                "feedback": "/api/xai/feedback",
                "health": "/api/xai/health"
            },
            "documentation": "/docs"
        }
    
    @app.get("/api/xai/health")
    async def health_check():
        """Health check for XAI system"""
        try:
            # Test XAI functionality
            metrics = xai_explainer.get_metrics()
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "shap_available": hasattr(xai_explainer.shap_explainer, 'is_fitted') and xai_explainer.shap_explainer.is_fitted,
                "lime_available": hasattr(xai_explainer.lime_explainer, 'is_fitted') and xai_explainer.lime_explainer.is_fitted,
                "explainability_percentage": metrics.get('explainability_percentage', 0.0),
                "uptime": time.time()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail="XAI system unhealthy")
    
    @app.post("/api/xai/explain-analysis", response_model=List[ExplanationResponse])
    async def explain_cv_analysis(request: CVAnalysisRequest):
        """
        Generate comprehensive explanations for CV analysis results
        
        This endpoint provides SHAP/LIME explanations for:
        - Skill extraction decisions
        - Job matching scores
        - Gap analysis results
        - Overall profile assessment
        """
        try:
            start_time = time.time()
            
            # Generate explanations using XAI system
            explanations = await xai_explainer.explain_analysis(
                cv_content=request.cv_content,
                extracted_skills=request.extracted_skills,
                matching_score=request.matching_score,
                gap_analysis=request.gap_analysis,
                models=request.models
            )
            
            total_time = time.time() - start_time
            
            logger.info(f"Generated {len(explanations)} explanations in {total_time:.2f}s")
            
            # Ensure we return proper response format
            response_explanations = []
            for explanation in explanations:
                response_explanation = ExplanationResponse(
                    explanation_type=explanation.get('explanation_type', 'unknown'),
                    explanation_text=explanation.get('explanation_text', 'No explanation available'),
                    confidence=explanation.get('confidence', 0.0),
                    supporting_evidence=explanation.get('supporting_evidence', []),
                    explanation_time=explanation.get('explanation_time', 0.0),
                    visual_data=explanation.get('visual_data'),
                    lime_data=explanation.get('lime_data'),
                    shap_data=explanation.get('shap_data')
                )
                response_explanations.append(response_explanation)
            
            return response_explanations
            
        except Exception as e:
            logger.error(f"Error in explanation endpoint: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to generate explanations: {str(e)}"
            )
    
    @app.get("/api/xai/metrics", response_model=XAIMetricsResponse)
    async def get_xai_metrics():
        """Get XAI system performance metrics"""
        try:
            metrics = xai_explainer.get_metrics()
            
            return XAIMetricsResponse(
                total_explanations=metrics.get('total_explanations', 0),
                average_explanation_time=metrics.get('average_explanation_time', 0.0),
                explanation_breakdown=metrics.get('explanation_breakdown', {}),
                average_accuracy=metrics.get('average_accuracy', 0.0),
                explainability_percentage=metrics.get('explainability_percentage', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve metrics")
    
    @app.post("/api/xai/feedback")
    async def submit_feedback(
        explanation_id: str = Query(..., description="Unique ID of the explanation"),
        helpful: bool = Query(..., description="Whether the explanation was helpful"),
        feedback_text: Optional[str] = Query(None, description="Optional feedback text"),
        explanation_type: Optional[str] = Query(None, description="Type of explanation")
    ):
        """Submit user feedback on explanations for continuous improvement"""
        try:
            xai_explainer.record_feedback(explanation_id, helpful, feedback_text)
            
            logger.info(f"Received feedback for explanation {explanation_id}: helpful={helpful}")
            
            return {
                "status": "success",
                "message": "Feedback recorded successfully",
                "explanation_id": explanation_id,
                "helpful": helpful,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
            raise HTTPException(status_code=500, detail="Failed to record feedback")
    
    @app.post("/api/xai/set-models")
    async def set_models_for_explanation(
        neural_scorer_path: Optional[str] = Query(None, description="Path to trained neural scorer model"),
        skills_extractor_config: Optional[Dict[str, Any]] = Query(None, description="Skills extractor configuration"),
        background_tasks: BackgroundTasks
    ):
        """Set models for SHAP/LIME explanation (admin endpoint)"""
        try:
            # This would load actual models in production
            # For now, we'll just validate the request
            
            models = {}
            
            # Mock neural scorer model loading
            if neural_scorer_path:
                models['neural_scorer'] = {"path": neural_scorer_path, "loaded": True}
            
            # Mock skills extractor configuration
            if skills_extractor_config:
                models['skills_extractor'] = skills_extractor_config
            
            # Set models in background to avoid blocking
            background_tasks.add_task(xai_explainer.set_models, 
                                    models.get('neural_scorer'), 
                                    models.get('skills_extractor'))
            
            return {
                "status": "success",
                "message": "Models queued for loading",
                "models": models,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error setting models: {e}")
            raise HTTPException(status_code=500, detail="Failed to set models")
    
    @app.get("/api/xai/explanation/{explanation_id}")
    async def get_explanation_by_id(explanation_id: str):
        """Retrieve a specific explanation by ID (for caching/reuse)"""
        try:
            # In production, this would fetch from a database
            # For now, return a placeholder
            
            return {
                "explanation_id": explanation_id,
                "status": "not_found",
                "message": "Explanation caching not yet implemented",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error retrieving explanation {explanation_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve explanation")
    
    @app.get("/api/xai/feature-importance/{analysis_id}")
    async def get_feature_importance(analysis_id: str):
        """Get feature importance explanations for a specific analysis"""
        try:
            # Mock feature importance data
            # In production, this would retrieve actual SHAP/LIME results
            
            feature_importance = [
                {"feature": "semantic_similarity", "importance": 0.85, "percentage": 35.2},
                {"feature": "experience_years", "importance": 0.72, "percentage": 29.8},
                {"feature": "skills_match_ratio", "importance": 0.65, "percentage": 26.9},
                {"feature": "education_level", "importance": 0.21, "percentage": 8.7}
            ]
            
            return {
                "analysis_id": analysis_id,
                "feature_importance": feature_importance,
                "total_features": len(feature_importance),
                "method": "SHAP",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting feature importance for {analysis_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to get feature importance")
    
    @app.get("/api/xai/visualization/{explanation_id}")
    async def get_explanation_visualization(explanation_id: str):
        """Get visualization data for frontend charts"""
        try:
            # Mock visualization data
            # In production, this would return actual SHAP/LIME plot data
            
            visualization_data = {
                "explanation_id": explanation_id,
                "waterfall_plot": {
                    "expected_value": 0.65,
                    "features": [
                        {"name": "semantic_similarity", "value": 0.85, "shap_value": 0.15, "impact": "positive"},
                        {"name": "experience_years", "value": 5, "shap_value": 0.08, "impact": "positive"},
                        {"name": "skills_match", "value": 0.75, "shap_value": -0.05, "impact": "negative"}
                    ]
                },
                "summary_plot": {
                    "feature_names": ["semantic_similarity", "experience_years", "skills_match", "education"],
                    "shap_values": [[0.15, 0.08, -0.05, 0.02]],
                    "base_value": 0.65
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return visualization_data
            
        except Exception as e:
            logger.error(f"Error getting visualization for {explanation_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to get visualization data")
    
    @app.post("/api/xai/explain-skill-extraction")
    async def explain_skill_extraction(
        cv_text: str = Query(..., description="CV text content"),
        extracted_skills: List[Dict[str, Any]] = Query(..., description="Extracted skills"),
        skills_extractor_model: Optional[Dict[str, Any]] = Query(None, description="Skills extractor model")
    ):
        """Generate LIME explanation for specific skill extraction"""
        try:
            # This would use LIME to explain why specific skills were extracted
            # For now, return a mock explanation
            
            lime_explanation = {
                "method": "LIME",
                "cv_text_length": len(cv_text),
                "skills_found": len(extracted_skills),
                "explanation": f"LIME analysis of {len(extracted_skills)} skills extracted from {len(cv_text)} characters of text",
                "confidence": 0.78,
                "top_contributing_factors": [
                    {"factor": "keyword_patterns", "weight": 0.45, "impact": "positive"},
                    {"factor": "technical_terms", "weight": 0.32, "impact": "positive"},
                    {"factor": "context_analysis", "weight": 0.23, "impact": "positive"}
                ],
                "timestamp": datetime.now().isoformat()
            }
            
            return lime_explanation
            
        except Exception as e:
            logger.error(f"Error in skill extraction explanation: {e}")
            raise HTTPException(status_code=500, detail="Failed to explain skill extraction")
    
    @app.post("/api/xai/explain-job-matching")
    async def explain_job_matching(
        cv_features: List[float] = Query(..., description="CV feature vector"),
        job_features: List[float] = Query(..., description="Job feature vector"),
        prediction_score: float = Query(..., description="Matching prediction score"),
        neural_model: Optional[Dict[str, Any]] = Query(None, description="Neural network model")
    ):
        """Generate SHAP explanation for job matching prediction"""
        try:
            # This would use SHAP to explain job matching decisions
            # For now, return a mock explanation
            
            shap_explanation = {
                "method": "SHAP",
                "prediction_score": prediction_score,
                "base_value": 0.45,
                "feature_contributions": [
                    {"feature": "semantic_similarity", "value": 0.85, "shap_value": 0.15, "impact": "increases"},
                    {"feature": "experience_years", "value": 5, "shap_value": 0.08, "impact": "increases"},
                    {"feature": "skills_match_ratio", "value": 0.75, "shap_value": -0.03, "impact": "decreases"},
                    {"feature": "education_level", "value": 3, "shap_value": 0.02, "impact": "increases"}
                ],
                "confidence": 0.82,
                "explanation": "SHAP analysis shows semantic similarity and experience are the main positive contributors",
                "timestamp": datetime.now().isoformat()
            }
            
            return shap_explanation
            
        except Exception as e:
            logger.error(f"Error in job matching explanation: {e}")
            raise HTTPException(status_code=500, detail="Failed to explain job matching")
    
    @app.get("/api/xai/validate-explainability")
    async def validate_explainability():
        """Validate that explainability requirements are met"""
        try:
            metrics = xai_explainer.get_metrics()
            explainability_percentage = metrics.get('explainability_percentage', 0.0)
            
            # Check if we meet the 80% requirement
            meets_requirement = explainability_percentage >= 80.0
            
            validation_result = {
                "requirement": "80% of AI decisions must be explainable",
                "current_percentage": explainability_percentage,
                "meets_requirement": meets_requirement,
                "status": "compliant" if meets_requirement else "non_compliant",
                "recommendations": [],
                "timestamp": datetime.now().isoformat()
            }
            
            if not meets_requirement:
                validation_result["recommendations"].extend([
                    "Increase explanation coverage for skill extraction",
                    "Add explanations for job matching decisions",
                    "Implement LIME for CV text analysis",
                    "Enhance SHAP integration for feature importance"
                ])
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating explainability: {e}")
            raise HTTPException(status_code=500, detail="Failed to validate explainability")
    
    return app

# Create the XAI API instance
xai_api = create_xai_api()

# For direct usage
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(xai_api, host="0.0.0.0", port=8001)