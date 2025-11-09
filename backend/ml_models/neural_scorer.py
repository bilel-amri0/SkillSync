"""
Neural Scoring Model for Advanced Recommendations
Adapted from notebook for production use
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import pickle
from pathlib import Path

# TensorFlow with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except ImportError as e:
    logging.warning(f"TensorFlow/Sklearn not available: {e}")
    tf = None
    Sequential = None
    StandardScaler = None

logger = logging.getLogger(__name__)

class NeuralScorer:
    """
    Neural network-based scoring for job recommendations
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_path = model_path
        
        # Feature configuration
        self.feature_config = {
            'numerical_features': [
                'experience_years', 'similarity_score', 'skills_match_ratio',
                'education_level', 'industry_match', 'location_match',
                'salary_expectation_match', 'company_size_preference'
            ],
            'categorical_features': [
                'role_level', 'work_type', 'domain'
            ]
        }
        
        # Initialize if TensorFlow available
        if tf is not None:
            self._initialize_model()
        else:
            logger.warning("TensorFlow not available, using rule-based scoring")
    
    def _initialize_model(self):
        """Initialize or load neural network model"""
        try:
            if self.model_path and Path(self.model_path).exists():
                logger.info(f"Loading trained model from {self.model_path}")
                self.model = load_model(self.model_path)
                
                # Load scaler
                scaler_path = Path(self.model_path).parent / "scaler.pkl"
                if scaler_path.exists():
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
            else:
                logger.info("Creating new neural network model")
                self._create_model()
                
        except Exception as e:
            logger.error(f"Error initializing neural model: {e}")
            self.model = None
    
    def _create_model(self, input_dim: int = 15) -> None:
        """Create new neural network architecture"""
        if tf is None:
            return
            
        try:
            self.model = Sequential([
                Input(shape=(input_dim,)),
                Dense(128, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(32, activation='relu'),
                Dropout(0.2),
                
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')  # Output score 0-1
            ])
            
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy', 'mae']
            )
            
            # Initialize scaler
            if StandardScaler is not None:
                self.scaler = StandardScaler()
                
            logger.info("Neural network model created successfully")
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            self.model = None
    
    def extract_features(self, cv_data: Dict, job_data: Dict, similarity_score: float = 0.0) -> np.ndarray:
        """
        Extract features for neural network input
        """
        features = []
        
        try:
            # Numerical features
            features.append(cv_data.get('experience_years', 0))  # Experience years
            features.append(similarity_score)  # Semantic similarity
            
            # Skills match ratio
            cv_skills = set(cv_data.get('skills', []))
            job_skills = set(job_data.get('required_skills', []))
            skills_match = len(cv_skills.intersection(job_skills)) / max(len(job_skills), 1)
            features.append(skills_match)
            
            # Education level (0-4: none, high school, bachelor, master, phd)
            education_map = {'none': 0, 'high school': 1, 'bachelor': 2, 'master': 3, 'phd': 4}
            education = cv_data.get('education_level', 'bachelor').lower()
            features.append(education_map.get(education, 2))
            
            # Industry match (0-1)
            cv_industry = cv_data.get('industry', '').lower()
            job_industry = job_data.get('industry', '').lower()
            industry_match = 1.0 if cv_industry == job_industry else 0.0
            features.append(industry_match)
            
            # Location match (simplified)
            cv_location = cv_data.get('location', '').lower()
            job_location = job_data.get('location', '').lower()
            location_match = 1.0 if cv_location == job_location else 0.5  # Remote work consideration
            features.append(location_match)
            
            # Salary expectation match (0-1)
            cv_salary_expectation = cv_data.get('salary_expectation', 0)
            job_salary = job_data.get('salary', 0)
            if cv_salary_expectation > 0 and job_salary > 0:
                salary_match = min(job_salary / cv_salary_expectation, 1.0)
            else:
                salary_match = 0.5  # Neutral if no salary info
            features.append(salary_match)
            
            # Company size preference (0-1)
            cv_company_pref = cv_data.get('company_size_preference', 'medium')
            job_company_size = job_data.get('company_size', 'medium')
            size_match = 1.0 if cv_company_pref == job_company_size else 0.5
            features.append(size_match)
            
            # Role level encoding (0-3: junior, mid, senior, lead)
            role_map = {'junior': 0, 'mid': 1, 'senior': 2, 'lead': 3}
            cv_level = cv_data.get('level', 'mid').lower()
            job_level = job_data.get('level', 'mid').lower()
            features.append(role_map.get(cv_level, 1))
            features.append(role_map.get(job_level, 1))
            
            # Work type (0-2: onsite, hybrid, remote)
            work_type_map = {'onsite': 0, 'hybrid': 1, 'remote': 2}
            cv_work_type = cv_data.get('work_preference', 'hybrid').lower()
            job_work_type = job_data.get('work_type', 'hybrid').lower()
            features.append(work_type_map.get(cv_work_type, 1))
            features.append(work_type_map.get(job_work_type, 1))
            
            # Domain match
            cv_domain = cv_data.get('domain', '').lower()
            job_domain = job_data.get('domain', '').lower()
            domain_match = 1.0 if cv_domain == job_domain else 0.0
            features.append(domain_match)
            
            # Years since last relevant experience
            last_experience = cv_data.get('last_relevant_experience_years', 0)
            features.append(min(last_experience, 10))  # Cap at 10 years
            
            # Ensure we have the right number of features
            while len(features) < 15:
                features.append(0.0)
            
            return np.array(features[:15], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.zeros(15, dtype=np.float32)
    
    def predict_score(self, cv_data: Dict, job_data: Dict, similarity_score: float = 0.0) -> Dict[str, Any]:
        """
        Predict job match score using neural network
        """
        if self.model is None:
            return self._fallback_score(cv_data, job_data, similarity_score)
        
        try:
            # Extract features
            features = self.extract_features(cv_data, job_data, similarity_score)
            features = features.reshape(1, -1)
            
            # Scale features if scaler available
            if self.scaler is not None:
                features = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(features, verbose=0)[0][0]
            confidence = abs(prediction - 0.5) * 2  # Distance from neutral
            
            return {
                'neural_score': float(prediction),
                'confidence': float(confidence),
                'method': 'neural_network',
                'features_used': features.flatten().tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in neural prediction: {e}")
            return self._fallback_score(cv_data, job_data, similarity_score)
    
    def _fallback_score(self, cv_data: Dict, job_data: Dict, similarity_score: float) -> Dict[str, Any]:
        """
        Fallback scoring when neural network is not available
        """
        try:
            # Simple rule-based scoring
            score = 0.0
            
            # Base similarity contribution (40%)
            score += similarity_score * 0.4
            
            # Skills match (30%)
            cv_skills = set(cv_data.get('skills', []))
            job_skills = set(job_data.get('required_skills', []))
            skills_match = len(cv_skills.intersection(job_skills)) / max(len(job_skills), 1)
            score += skills_match * 0.3
            
            # Experience level match (20%)
            cv_exp = cv_data.get('experience_years', 0)
            job_exp_required = job_data.get('min_experience', 0)
            if cv_exp >= job_exp_required:
                exp_score = min(1.0, cv_exp / max(job_exp_required + 2, 1))
            else:
                exp_score = cv_exp / max(job_exp_required, 1)
            score += exp_score * 0.2
            
            # Industry match (10%)
            cv_industry = cv_data.get('industry', '').lower()
            job_industry = job_data.get('industry', '').lower()
            industry_match = 1.0 if cv_industry == job_industry else 0.0
            score += industry_match * 0.1
            
            # Normalize to 0-1
            score = max(0.0, min(1.0, score))
            
            return {
                'neural_score': score,
                'confidence': 0.7,  # Medium confidence for rule-based
                'method': 'rule_based',
                'components': {
                    'similarity': similarity_score * 0.4,
                    'skills_match': skills_match * 0.3,
                    'experience': exp_score * 0.2,
                    'industry': industry_match * 0.1
                }
            }
            
        except Exception as e:
            logger.error(f"Error in fallback scoring: {e}")
            return {
                'neural_score': 0.5,
                'confidence': 0.1,
                'method': 'default',
                'error': str(e)
            }
    
    def batch_predict(self, cv_data: Dict, job_list: List[Dict], similarity_scores: List[float]) -> List[Dict]:
        """
        Predict scores for multiple jobs
        """
        results = []
        
        for i, job in enumerate(job_list):
            similarity = similarity_scores[i] if i < len(similarity_scores) else 0.0
            score_result = self.predict_score(cv_data, job, similarity)
            
            result = {
                'job': job,
                'score_details': score_result,
                'combined_score': (similarity + score_result['neural_score']) / 2
            }
            results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results
    
    def explain_prediction(self, cv_data: Dict, job_data: Dict, similarity_score: float) -> Dict[str, Any]:
        """
        Provide detailed explanation of the scoring
        """
        features = self.extract_features(cv_data, job_data, similarity_score)
        score_result = self.predict_score(cv_data, job_data, similarity_score)
        
        explanation = {
            'final_score': score_result['neural_score'],
            'confidence': score_result['confidence'],
            'method': score_result['method'],
            'key_factors': [],
            'improvement_suggestions': []
        }
        
        # Analyze key contributing factors
        cv_skills = set(cv_data.get('skills', []))
        job_skills = set(job_data.get('required_skills', []))
        matching_skills = cv_skills.intersection(job_skills)
        missing_skills = job_skills - cv_skills
        
        if matching_skills:
            explanation['key_factors'].append(f"✅ Matching skills: {', '.join(list(matching_skills)[:3])}")
        
        if similarity_score > 0.7:
            explanation['key_factors'].append(f"✅ High semantic similarity: {similarity_score:.2f}")
        
        cv_exp = cv_data.get('experience_years', 0)
        job_exp = job_data.get('min_experience', 0)
        if cv_exp >= job_exp:
            explanation['key_factors'].append(f"✅ Sufficient experience: {cv_exp} years")
        
        # Improvement suggestions
        if missing_skills:
            explanation['improvement_suggestions'].append(
                f"Consider learning: {', '.join(list(missing_skills)[:3])}"
            )
        
        if cv_exp < job_exp:
            explanation['improvement_suggestions'].append(
                f"Gain more experience (need {job_exp - cv_exp} more years)"
            )
        
        if similarity_score < 0.5:
            explanation['improvement_suggestions'].append(
                "Consider tailoring your profile to better match this role"
            )
        
        return explanation
    
    def save_model(self, path: str) -> bool:
        """
        Save trained model and scaler
        """
        try:
            if self.model is None:
                logger.warning("No model to save")
                return False
            
            model_path = Path(path)
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self.model.save(model_path / "neural_scorer.h5")
            
            # Save scaler
            if self.scaler is not None:
                with open(model_path / "scaler.pkl", 'wb') as f:
                    pickle.dump(self.scaler, f)
            
            logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
