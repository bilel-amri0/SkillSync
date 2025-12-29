"""
F2: Skills Extraction & Normalization (NER fine-tuned on ESCO/O*NET)
F3: Semantic CV-Job Matching (cosine similarity on embeddings)
F4: Gap Analysis + Visualization
"""

import spacy
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import re
import logging

logger = logging.getLogger(__name__)

class SemanticAnalyzer:
    """Advanced semantic analysis for CV and job matching"""
    
    def __init__(self):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = None
        self.model = None
        self.nlp = None
        self.skill_taxonomy = self._load_skill_taxonomy()
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models"""
        try:
            # Load spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using basic tokenization")
                self.nlp = None
            
            # Load transformer model for embeddings
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            logger.info("Semantic analyzer models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def _load_skill_taxonomy(self) -> Dict[str, List[str]]:
        """Load ESCO/O*NET skill taxonomy for normalization"""
        
        # Simplified skill taxonomy (in production, load from ESCO/O*NET databases)
        return {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
                'php', 'ruby', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql'
            ],
            'web_technologies': [
                'html', 'css', 'react', 'vue', 'angular', 'node.js', 'express',
                'django', 'flask', 'spring', 'bootstrap', 'sass', 'webpack'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                'oracle', 'sqlite', 'cassandra', 'dynamodb'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'google cloud', 'docker', 'kubernetes',
                'terraform', 'jenkins', 'gitlab', 'github actions'
            ],
            'data_science': [
                'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
                'matplotlib', 'seaborn', 'jupyter', 'tableau', 'power bi'
            ],
            'soft_skills': [
                'leadership', 'communication', 'problem solving', 'teamwork',
                'project management', 'agile', 'scrum', 'analytical thinking'
            ],
            'methodologies': [
                'agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'tdd', 'bdd',
                'microservices', 'rest api', 'graphql'
            ]
        }
    
    async def extract_skills(self, cv_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract skills from CV using NER and pattern matching"""
        
        text = cv_data.get('raw_text', '')
        skills_section = cv_data.get('sections', {}).get('skills', '')
        
        # Combine relevant text
        combined_text = f"{text} {skills_section}"
        
        extracted_skills = []
        
        # Method 1: Pattern-based extraction using skill taxonomy
        pattern_skills = self._extract_skills_by_patterns(combined_text)
        extracted_skills.extend(pattern_skills)
        
        # Method 2: NER extraction (if spaCy available)
        if self.nlp:
            ner_skills = self._extract_skills_by_ner(combined_text)
            extracted_skills.extend(ner_skills)
        
        # Method 3: Section-specific extraction
        section_skills = self._extract_skills_from_sections(cv_data.get('sections', {}))
        extracted_skills.extend(section_skills)
        
        # Remove duplicates and return
        unique_skills = self._deduplicate_skills(extracted_skills)
        
        return unique_skills
    
    def _extract_skills_by_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract skills using pattern matching against skill taxonomy"""
        
        skills = []
        text_lower = text.lower()
        
        for category, skill_list in self.skill_taxonomy.items():
            for skill in skill_list:
                # Use word boundaries for better matching
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    skills.append({
                        'skill': skill,
                        'category': category,
                        'confidence': 0.9,
                        'extraction_method': 'pattern_matching',
                        'normalized_name': skill
                    })
        
        return skills
    
    def _extract_skills_by_ner(self, text: str) -> List[Dict[str, Any]]:
        """Extract skills using Named Entity Recognition"""
        
        skills = []
        
        try:
            doc = self.nlp(text)
            
            # Extract entities that might be skills
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'GPE']:  # Organizations, products, etc.
                    # Check if entity matches known skills
                    normalized_skill = self._normalize_skill(ent.text)
                    if normalized_skill:
                        skills.append({
                            'skill': ent.text,
                            'category': normalized_skill['category'],
                            'confidence': 0.7,
                            'extraction_method': 'ner',
                            'normalized_name': normalized_skill['normalized_name']
                        })
        
        except Exception as e:
            logger.error(f"Error in NER extraction: {str(e)}")
        
        return skills
    
    def _extract_skills_from_sections(self, sections: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract skills from specific CV sections"""
        
        skills = []
        
        # Focus on skills and experience sections
        relevant_sections = ['skills', 'experience', 'projects']
        
        for section_name in relevant_sections:
            if section_name in sections:
                section_text = sections[section_name]
                
                # Extract bullet points or comma-separated items
                items = self._extract_list_items(section_text)
                
                for item in items:
                    normalized_skill = self._normalize_skill(item)
                    if normalized_skill:
                        skills.append({
                            'skill': item.strip(),
                            'category': normalized_skill['category'],
                            'confidence': 0.8,
                            'extraction_method': f'section_{section_name}',
                            'normalized_name': normalized_skill['normalized_name'],
                            'context': section_name
                        })
        
        return skills
    
    def _extract_list_items(self, text: str) -> List[str]:
        """Extract list items from text (bullet points, commas, etc.)"""
        
        items = []
        
        # Split by common separators
        separators = [',', '', '', '-', '*', '\n']
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove bullet point characters
            line = re.sub(r'^[\-\*]\s*', '', line)
            
            # Split by commas if line contains multiple items
            if ',' in line and len(line.split(',')) > 1:
                items.extend([item.strip() for item in line.split(',')])
            else:
                items.append(line)
        
        return [item for item in items if len(item.strip()) > 2]
    
    def _normalize_skill(self, skill_text: str) -> Dict[str, str]:
        """Normalize skill against ESCO/O*NET taxonomy"""
        
        skill_lower = skill_text.lower().strip()
        
        for category, skill_list in self.skill_taxonomy.items():
            for normalized_skill in skill_list:
                # Exact match
                if skill_lower == normalized_skill.lower():
                    return {
                        'category': category,
                        'normalized_name': normalized_skill
                    }
                
                # Partial match
                if normalized_skill.lower() in skill_lower or skill_lower in normalized_skill.lower():
                    return {
                        'category': category,
                        'normalized_name': normalized_skill
                    }
        
        return None
    
    def _deduplicate_skills(self, skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate skills and merge information"""
        
        unique_skills = {}
        
        for skill in skills:
            key = skill['normalized_name'].lower()
            
            if key not in unique_skills:
                unique_skills[key] = skill
            else:
                # Merge with higher confidence
                if skill['confidence'] > unique_skills[key]['confidence']:
                    unique_skills[key] = skill
        
        return list(unique_skills.values())
    
    async def normalize_skills(self, extracted_skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Additional normalization and categorization"""
        
        normalized_skills = []
        
        for skill in extracted_skills:
            # Add experience level estimation
            skill['experience_level'] = self._estimate_experience_level(skill)
            
            # Add skill importance score
            skill['importance_score'] = self._calculate_skill_importance(skill)
            
            normalized_skills.append(skill)
        
        # Sort by importance
        normalized_skills.sort(key=lambda x: x['importance_score'], reverse=True)
        
        return normalized_skills
    
    def _estimate_experience_level(self, skill: Dict[str, Any]) -> str:
        """Estimate experience level based on context"""
        
        # Simple heuristics (in production, use more sophisticated methods)
        context = skill.get('context', '')
        
        if 'lead' in context.lower() or 'senior' in context.lower():
            return 'expert'
        elif 'project' in context.lower() or 'experience' in context.lower():
            return 'intermediate'
        else:
            return 'beginner'
    
    def _calculate_skill_importance(self, skill: Dict[str, Any]) -> float:
        """Calculate skill importance score"""
        
        base_score = skill['confidence']
        
        # Boost important categories
        category_weights = {
            'programming_languages': 1.2,
            'web_technologies': 1.1,
            'cloud_platforms': 1.3,
            'data_science': 1.2,
            'soft_skills': 0.9
        }
        
        weight = category_weights.get(skill['category'], 1.0)
        
        return base_score * weight
    
    async def calculate_semantic_similarity(
        self, 
        cv_content: Dict[str, Any], 
        job_description: str
    ) -> Dict[str, Any]:
        """Calculate semantic similarity between CV and job description"""
        
        try:
            # Prepare texts
            cv_text = cv_content.get('raw_text', '')
            
            # Generate embeddings
            cv_embedding = self._get_text_embedding(cv_text)
            job_embedding = self._get_text_embedding(job_description)
            
            # Calculate cosine similarity
            similarity_score = cosine_similarity(
                cv_embedding.reshape(1, -1),
                job_embedding.reshape(1, -1)
            )[0][0]
            
            # Calculate section-wise similarities
            section_similarities = {}
            for section_name, section_content in cv_content.get('sections', {}).items():
                if section_content.strip():
                    section_embedding = self._get_text_embedding(section_content)
                    section_similarity = cosine_similarity(
                        section_embedding.reshape(1, -1),
                        job_embedding.reshape(1, -1)
                    )[0][0]
                    section_similarities[section_name] = float(section_similarity)
            
            return {
                'overall_similarity': float(similarity_score),
                'section_similarities': section_similarities,
                'compatibility_score': self._calculate_compatibility_score(similarity_score),
                'matching_strength': self._get_matching_strength(similarity_score)
            }
        
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return {
                'overall_similarity': 0.0,
                'section_similarities': {},
                'compatibility_score': 'low',
                'matching_strength': 'weak'
            }
    
    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Generate text embedding using transformer model"""
        
        try:
            # Tokenize and encode
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return embeddings.numpy().flatten()
        
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Fallback to TF-IDF
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([text])
            return tfidf_matrix.toarray().flatten()
    
    def _calculate_compatibility_score(self, similarity: float) -> str:
        """Convert numeric similarity to categorical score"""
        
        if similarity >= 0.8:
            return 'excellent'
        elif similarity >= 0.6:
            return 'good'
        elif similarity >= 0.4:
            return 'moderate'
        elif similarity >= 0.2:
            return 'low'
        else:
            return 'poor'
    
    def _get_matching_strength(self, similarity: float) -> str:
        """Get matching strength description"""
        
        if similarity >= 0.8:
            return 'very strong'
        elif similarity >= 0.6:
            return 'strong'
        elif similarity >= 0.4:
            return 'moderate'
        elif similarity >= 0.2:
            return 'weak'
        else:
            return 'very weak'
    
    async def analyze_skill_gaps(
        self, 
        user_skills: List[Dict[str, Any]], 
        job_description: str
    ) -> Dict[str, Any]:
        """Analyze skill gaps between user skills and job requirements"""
        
        try:
            # Extract required skills from job description
            job_skills = await self._extract_job_requirements(job_description)
            
            # Create skill sets
            user_skill_names = {skill['normalized_name'].lower() for skill in user_skills}
            job_skill_names = {skill['normalized_name'].lower() for skill in job_skills}
            
            # Calculate gaps
            missing_skills = job_skill_names - user_skill_names
            matching_skills = job_skill_names & user_skill_names
            extra_skills = user_skill_names - job_skill_names
            
            # Categorize missing skills by importance
            critical_missing = []
            important_missing = []
            nice_to_have_missing = []
            
            for skill in job_skills:
                if skill['normalized_name'].lower() in missing_skills:
                    if skill.get('importance_score', 0.5) >= 0.8:
                        critical_missing.append(skill)
                    elif skill.get('importance_score', 0.5) >= 0.6:
                        important_missing.append(skill)
                    else:
                        nice_to_have_missing.append(skill)
            
            return {
                'missing_skills': {
                    'critical': critical_missing,
                    'important': important_missing,
                    'nice_to_have': nice_to_have_missing
                },
                'matching_skills': list(matching_skills),
                'extra_skills': list(extra_skills),
                'gap_score': len(missing_skills) / len(job_skill_names) if job_skill_names else 0,
                'match_percentage': len(matching_skills) / len(job_skill_names) * 100 if job_skill_names else 0,
                'total_job_requirements': len(job_skill_names),
                'user_skill_coverage': len(matching_skills)
            }
        
        except Exception as e:
            logger.error(f"Error analyzing skill gaps: {str(e)}")
            return {
                'missing_skills': {'critical': [], 'important': [], 'nice_to_have': []},
                'matching_skills': [],
                'extra_skills': [],
                'gap_score': 1.0,
                'match_percentage': 0.0,
                'total_job_requirements': 0,
                'user_skill_coverage': 0
            }
    
    async def _extract_job_requirements(self, job_description: str) -> List[Dict[str, Any]]:
        """Extract required skills from job description"""
        
        # Use same extraction methods as CV processing
        fake_cv_data = {'raw_text': job_description, 'sections': {'skills': job_description}}
        job_skills = await self.extract_skills(fake_cv_data)
        
        # Enhance with job-specific importance scoring
        for skill in job_skills:
            skill['importance_score'] = self._calculate_job_skill_importance(
                skill, job_description
            )
        
        return job_skills
    
    def _calculate_job_skill_importance(self, skill: Dict[str, Any], job_text: str) -> float:
        """Calculate skill importance based on context in job description"""
        
        base_importance = 0.5
        skill_name = skill['normalized_name'].lower()
        job_text_lower = job_text.lower()
        
        # Check for emphasis keywords
        emphasis_patterns = [
            f"required.*{skill_name}",
            f"must.*{skill_name}",
            f"essential.*{skill_name}",
            f"{skill_name}.*required",
            f"{skill_name}.*essential"
        ]
        
        for pattern in emphasis_patterns:
            if re.search(pattern, job_text_lower):
                base_importance += 0.3
                break
        
        # Check frequency of mention
        frequency = job_text_lower.count(skill_name)
        if frequency > 2:
            base_importance += 0.2
        elif frequency > 1:
            base_importance += 0.1
        
        return min(base_importance, 1.0)