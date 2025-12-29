"""
ADVANCED ML MODULES FOR CV PARSER
Pure ML-driven extraction with minimal rules
CPU-optimized, uses embeddings + transformers
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ==================== ADVANCED SKILL EXTRACTION ====================

class SemanticSkillExtractor:
    """
    Pure ML skill extraction - no keyword matching
    Uses context windows and semantic similarity
    """
    
    def __init__(self, embedder, skill_database: List[Tuple[str, str]]):
        self.embedder = embedder
        self.skill_database = skill_database
        
        # Precompute skill embeddings
        skill_names = [s for s, _ in skill_database]
        self.skill_embeddings = embedder.encode(skill_names, show_progress_bar=False)
        self.skill_names = skill_names
        self.skill_categories = {s: cat for s, cat in skill_database}
        
        # Technical context indicators
        self.tech_contexts = [
            'experience with', 'proficient in', 'skilled in', 'knowledge of',
            'expertise in', 'worked with', 'using', 'utilized', 'implemented',
            'developed with', 'built using', 'technologies:', 'tools:',
            'stack:', 'languages:', 'frameworks:'
        ]
    
    def extract_skills_semantic(self, text: str, threshold: float = 0.72) -> Dict[str, Tuple[str, float, str]]:
        """
        Pure semantic skill extraction
        Returns: {skill: (category, confidence, context)}
        """
        found_skills = {}
        
        # Step 1: Extract skill contexts (sentences mentioning technical work)
        sentences = self._split_into_sentences(text)
        skill_contexts = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            # Check if sentence contains technical context
            if any(ctx in sent_lower for ctx in self.tech_contexts):
                skill_contexts.append(sent)
            # Also include sentences with capitalized tech terms
            elif re.search(r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)*\b', sent):
                skill_contexts.append(sent)
        
        # Step 2: Extract n-grams from contexts
        candidates = self._extract_candidate_phrases(text, max_words=3)
        
        # Step 3: Filter candidates that appear in technical contexts
        context_text = ' '.join(skill_contexts).lower()
        filtered_candidates = [
            c for c in candidates 
            if c.lower() in context_text or c.lower() in text.lower()
        ]
        
        if not filtered_candidates:
            return found_skills
        
        # Step 4: Semantic matching with context awareness
        candidate_embeddings = self.embedder.encode(
            filtered_candidates[:200],  # Limit for performance
            show_progress_bar=False
        )
        
        similarities = cosine_similarity(candidate_embeddings, self.skill_embeddings)
        
        for i, candidate in enumerate(filtered_candidates[:200]):
            best_idx = np.argmax(similarities[i])
            best_sim = similarities[i][best_idx]
            
            if best_sim > threshold:
                matched_skill = self.skill_names[best_idx]
                category = self.skill_categories[matched_skill]
                
                # Get context snippet
                context = self._get_skill_context(text, candidate)
                
                # Calculate context-boosted confidence
                context_boost = self._calculate_context_boost(context)
                final_confidence = min(0.98, best_sim * context_boost)
                
                if matched_skill not in found_skills or found_skills[matched_skill][1] < final_confidence:
                    found_skills[matched_skill] = (category, final_confidence, context)
        
        # Step 5: Multi-sentence skill detection (skills in paragraphs)
        paragraph_skills = self._extract_paragraph_skills(text, threshold=0.75)
        for skill, (cat, conf, ctx) in paragraph_skills.items():
            if skill not in found_skills or found_skills[skill][1] < conf:
                found_skills[skill] = (cat, conf, ctx)
        
        return found_skills
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _extract_candidate_phrases(self, text: str, max_words: int = 3) -> List[str]:
        """Extract n-gram candidates"""
        candidates = set()
        
        # Single words (capitalized or tech-like)
        words = re.findall(r'\b[A-Z][a-zA-Z0-9+#.]*\b', text)
        candidates.update(words)
        
        # Acronyms
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        candidates.update(acronyms)
        
        # Tech patterns (with dots, hyphens, plus)
        tech_patterns = re.findall(r'\b\w+[.#+-]\w+\b', text)
        candidates.update(tech_patterns)
        
        # N-grams from sentences
        sentences = self._split_into_sentences(text)
        for sent in sentences:
            tokens = sent.split()
            for i in range(len(tokens)):
                for n in range(1, min(max_words + 1, len(tokens) - i + 1)):
                    phrase = ' '.join(tokens[i:i+n])
                    if len(phrase) > 2:
                        candidates.add(phrase)
        
        return list(candidates)
    
    def _get_skill_context(self, text: str, skill: str, window: int = 50) -> str:
        """Get context around skill mention"""
        skill_lower = skill.lower()
        text_lower = text.lower()
        
        idx = text_lower.find(skill_lower)
        if idx == -1:
            return ""
        
        start = max(0, idx - window)
        end = min(len(text), idx + len(skill) + window)
        
        return text[start:end].strip()
    
    def _calculate_context_boost(self, context: str) -> float:
        """Calculate confidence boost from context"""
        context_lower = context.lower()
        boost = 1.0
        
        # Positive indicators
        positive_words = ['expert', 'proficient', 'experienced', 'skilled', 'mastery', 'advanced']
        boost += sum(0.02 for word in positive_words if word in context_lower)
        
        # Years of experience mentioned
        if re.search(r'\d+\+?\s*years?', context_lower):
            boost += 0.05
        
        return min(1.15, boost)
    
    def _extract_paragraph_skills(self, text: str, threshold: float = 0.75) -> Dict:
        """Extract skills mentioned across multiple sentences"""
        paragraphs = text.split('\n\n')
        paragraph_skills = {}
        
        for para in paragraphs[:10]:  # Limit to first 10 paragraphs
            if len(para) < 50:
                continue
            
            # Encode entire paragraph
            para_embedding = self.embedder.encode([para], show_progress_bar=False)[0]
            
            # Compare with all skills
            sims = cosine_similarity([para_embedding], self.skill_embeddings)[0]
            
            # Find high-confidence matches
            for idx, sim in enumerate(sims):
                if sim > threshold:
                    skill = self.skill_names[idx]
                    category = self.skill_categories[skill]
                    
                    if skill not in paragraph_skills:
                        paragraph_skills[skill] = (category, sim, para[:100])
        
        return paragraph_skills


# ==================== ML-BASED JOB TITLE & SENIORITY ====================

class MLJobTitleExtractor:
    """
    ML-based job title extraction and seniority prediction
    Uses embeddings to understand roles and levels
    """
    
    def __init__(self, embedder):
        self.embedder = embedder
        
        # Seniority reference embeddings
        self.seniority_profiles = {
            'Executive': [
                'Chief Technology Officer', 'Vice President Engineering', 'Director of Engineering',
                'Head of Data Science', 'Chief Executive Officer', 'C-level executive'
            ],
            'Lead': [
                'Lead Software Engineer', 'Team Lead', 'Technical Lead', 'Lead Developer',
                'Principal Engineer', 'Staff Engineer', 'Lead Architect'
            ],
            'Senior': [
                'Senior Software Engineer', 'Senior Developer', 'Senior Data Scientist',
                'Senior Analyst', 'Senior Consultant', 'experienced professional'
            ],
            'Mid': [
                'Software Engineer', 'Developer', 'Data Analyst', 'Consultant',
                'Engineer II', 'intermediate professional'
            ],
            'Junior': [
                'Junior Developer', 'Associate Engineer', 'Entry-level Developer',
                'Graduate Engineer', 'Junior Analyst', 'beginner professional'
            ]
        }
        
        # Precompute seniority embeddings
        self.seniority_embeddings = {}
        for level, examples in self.seniority_profiles.items():
            embeddings = self.embedder.encode(examples, show_progress_bar=False)
            self.seniority_embeddings[level] = np.mean(embeddings, axis=0)
    
    def extract_job_titles_ml(self, text: str) -> Tuple[Optional[str], List[str], str, List[Dict]]:
        """
        Extract job titles and predict seniority using ML
        Returns: (current_title, all_titles, predicted_seniority, career_progression)
        """
        # Extract title candidates
        title_candidates = self._extract_title_candidates(text)
        
        if not title_candidates:
            return None, [], 'Unknown', []
        
        # Score each candidate using semantic similarity
        scored_titles = []
        for candidate, context in title_candidates:
            score = self._score_title_candidate(candidate, context)
            if score > 0.4:  # Threshold for title-like text
                scored_titles.append((candidate, score, context))
        
        scored_titles.sort(key=lambda x: x[1], reverse=True)
        
        # Extract titles
        titles = [t[0] for t in scored_titles[:10]]
        current_title = titles[0] if titles else None
        
        # Predict seniority using embeddings
        predicted_seniority = self._predict_seniority_ml(text, titles)
        
        # Detect career progression
        career_progression = self._detect_career_progression(text, titles)
        
        return current_title, titles, predicted_seniority, career_progression
    
    def _extract_title_candidates(self, text: str) -> List[Tuple[str, str]]:
        """Extract potential job titles with context"""
        candidates = []
        lines = text.split('\n')
        
        # Look for title patterns in first 40 lines
        for i, line in enumerate(lines[:40]):
            line_stripped = line.strip()
            
            if len(line_stripped) < 5 or len(line_stripped) > 80:
                continue
            
            # Get context (previous and next lines)
            context_lines = []
            if i > 0:
                context_lines.append(lines[i-1])
            context_lines.append(line)
            if i < len(lines) - 1:
                context_lines.append(lines[i+1])
            
            context = ' '.join(context_lines)
            
            # Add candidate
            candidates.append((line_stripped, context))
        
        return candidates
    
    def _score_title_candidate(self, candidate: str, context: str) -> float:
        """Score how likely a candidate is a job title"""
        # Job title reference examples
        title_examples = [
            'Software Engineer', 'Data Scientist', 'Product Manager',
            'Senior Developer', 'Technical Lead', 'Business Analyst',
            'DevOps Engineer', 'Full Stack Developer', 'UX Designer'
        ]
        
        # Encode candidate and references
        candidate_emb = self.embedder.encode([candidate], show_progress_bar=False)[0]
        title_embs = self.embedder.encode(title_examples, show_progress_bar=False)
        
        # Calculate similarity
        similarities = cosine_similarity([candidate_emb], title_embs)[0]
        max_similarity = np.max(similarities)
        
        # Context boost
        context_lower = context.lower()
        if any(word in context_lower for word in ['experience', 'position', 'role', 'worked as']):
            max_similarity *= 1.1
        
        return float(max_similarity)
    
    def _predict_seniority_ml(self, text: str, titles: List[str]) -> str:
        """Predict seniority level using embeddings"""
        # Encode CV text and titles
        cv_text = ' '.join(titles) + ' ' + text[:1000]
        cv_embedding = self.embedder.encode([cv_text], show_progress_bar=False)[0]
        
        # Calculate similarity to each seniority level
        seniority_scores = {}
        for level, ref_embedding in self.seniority_embeddings.items():
            similarity = cosine_similarity([cv_embedding], [ref_embedding])[0][0]
            seniority_scores[level] = similarity
        
        # Get best match
        predicted_level = max(seniority_scores.items(), key=lambda x: x[1])[0]
        
        return predicted_level
    
    def _detect_career_progression(self, text: str, titles: List[str]) -> List[Dict]:
        """Detect promotions and career moves"""
        progression = []
        
        # Find date-title pairs
        lines = text.split('\n')
        for i, line in enumerate(lines):
            # Look for year ranges
            date_match = re.search(r'(\d{4})\s*[-]\s*(\d{4}|present|current)', line, re.IGNORECASE)
            if date_match:
                # Check nearby lines for title
                context_lines = lines[max(0, i-2):min(len(lines), i+3)]
                context = ' '.join(context_lines)
                
                # Find best matching title
                for title in titles:
                    if title.lower() in context.lower():
                        progression.append({
                            'title': title,
                            'period': date_match.group(0),
                            'start_year': int(date_match.group(1)),
                            'context': context[:100]
                        })
                        break
        
        # Sort by start year
        progression.sort(key=lambda x: x['start_year'])
        
        return progression


# ==================== ML-BASED RESPONSIBILITY EXTRACTION ====================

class SemanticResponsibilityExtractor:
    """
    Extract responsibilities using transformers instead of bullet regex
    Distinguishes impact statements from routine tasks
    """
    
    def __init__(self, embedder):
        self.embedder = embedder
        
        # Impact vs routine reference embeddings
        self.impact_examples = [
            'increased revenue by 40%', 'reduced costs by $2M', 'improved performance by 50%',
            'led team of 10 engineers', 'launched product used by 1M users',
            'architected scalable system', 'saved 100 hours per week'
        ]
        
        self.routine_examples = [
            'attended meetings', 'wrote documentation', 'fixed bugs',
            'responded to emails', 'participated in standups', 'reviewed code'
        ]
        
        self.impact_embeddings = self.embedder.encode(self.impact_examples, show_progress_bar=False)
        self.routine_embeddings = self.embedder.encode(self.routine_examples, show_progress_bar=False)
    
    def extract_responsibilities_ml(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract responsibilities with ML-based classification
        Returns: {'impact': [...], 'routine': [...], 'technical': [...]}
        """
        # Extract all potential responsibility statements
        statements = self._extract_responsibility_candidates(text)
        
        classified = {
            'impact': [],
            'routine': [],
            'technical': []
        }
        
        if not statements:
            return classified
        
        # Encode all statements
        statement_embeddings = self.embedder.encode(
            [s['text'] for s in statements],
            show_progress_bar=False
        )
        
        # Classify each statement
        for i, stmt in enumerate(statements):
            # Calculate similarity to impact vs routine
            impact_sims = cosine_similarity([statement_embeddings[i]], self.impact_embeddings)[0]
            routine_sims = cosine_similarity([statement_embeddings[i]], self.routine_embeddings)[0]
            
            max_impact = np.max(impact_sims)
            max_routine = np.max(routine_sims)
            
            # Detect metrics (numbers, percentages, dollar amounts)
            has_metrics = bool(re.search(r'\d+[%$kKmM]|\d+\s*(percent|users|hours|dollars)', stmt['text']))
            
            # Classify
            stmt['impact_score'] = float(max_impact)
            stmt['routine_score'] = float(max_routine)
            stmt['has_metrics'] = has_metrics
            
            if has_metrics or max_impact > 0.65:
                classified['impact'].append(stmt)
            elif max_routine > 0.60:
                classified['routine'].append(stmt)
            else:
                classified['technical'].append(stmt)
        
        # Sort by impact score
        for category in classified:
            classified[category].sort(
                key=lambda x: x.get('impact_score', 0),
                reverse=True
            )
        
        return classified
    
    def _extract_responsibility_candidates(self, text: str) -> List[Dict]:
        """Extract potential responsibility statements"""
        candidates = []
        lines = text.split('\n')
        
        # Action verbs that indicate responsibilities
        action_verbs = [
            'led', 'managed', 'developed', 'built', 'designed', 'implemented',
            'created', 'launched', 'improved', 'optimized', 'reduced', 'increased',
            'architected', 'established', 'coordinated', 'delivered', 'achieved'
        ]
        
        for line in lines:
            line_stripped = line.strip()
            
            # Skip very short or very long lines
            if len(line_stripped) < 15 or len(line_stripped) > 300:
                continue
            
            line_lower = line_stripped.lower()
            
            # Check for action verbs
            has_action = any(verb in line_lower for verb in action_verbs)
            
            # Check for bullet point (but don't rely on it)
            is_bullet = bool(re.match(r'^[\-\*]', line_stripped))
            
            if has_action or is_bullet:
                # Clean the text
                clean_text = re.sub(r'^[\-\*]\s*', '', line_stripped)
                
                candidates.append({
                    'text': clean_text,
                    'is_bullet': is_bullet,
                    'has_action': has_action
                })
        
        return candidates


# ==================== ML-BASED EDUCATION & CERTIFICATION ====================

class SemanticEducationExtractor:
    """
    Semantic detection of degrees, certifications, and institutions
    """
    
    def __init__(self, embedder):
        self.embedder = embedder
        
        # Degree reference examples
        self.degree_examples = {
            'PhD': ['Doctor of Philosophy', 'PhD in Computer Science', 'Doctorate in Engineering'],
            'Master': ['Master of Science', 'Masters in Business', 'MBA', 'MSc in Data Science'],
            'Bachelor': ['Bachelor of Science', 'Bachelor of Arts', 'BSc in Engineering', 'Undergraduate degree']
        }
        
        # Certification examples
        self.cert_examples = [
            'AWS Certified Solutions Architect', 'Google Cloud Professional',
            'PMP Certification', 'Certified Scrum Master', 'Microsoft Certified',
            'CompTIA Security+', 'Cisco CCNA'
        ]
        
        # Precompute embeddings
        self.degree_embeddings = {}
        for level, examples in self.degree_examples.items():
            embs = self.embedder.encode(examples, show_progress_bar=False)
            self.degree_embeddings[level] = np.mean(embs, axis=0)
        
        self.cert_embeddings = self.embedder.encode(self.cert_examples, show_progress_bar=False)
    
    def extract_education_ml(self, text: str) -> Dict:
        """
        ML-based education extraction
        Returns: {degrees: [...], level: str, institutions: [...], year: int, gpa: float}
        """
        # Find education section
        edu_section = self._extract_education_section(text)
        
        # Extract degree candidates
        degree_candidates = self._extract_degree_candidates(edu_section)
        
        # Classify degree level
        degrees = []
        degree_level = None
        
        for candidate in degree_candidates:
            # Encode candidate
            cand_emb = self.embedder.encode([candidate], show_progress_bar=False)[0]
            
            # Compare with degree levels
            best_level = None
            best_sim = 0.0
            
            for level, ref_emb in self.degree_embeddings.items():
                sim = cosine_similarity([cand_emb], [ref_emb])[0][0]
                if sim > best_sim and sim > 0.60:
                    best_sim = sim
                    best_level = level
            
            if best_level:
                degrees.append({
                    'text': candidate,
                    'level': best_level,
                    'confidence': float(best_sim)
                })
                if not degree_level or best_sim > 0.75:
                    degree_level = best_level
        
        # Extract institutions, year, GPA
        institutions = self._extract_institutions_ml(edu_section)
        graduation_year = self._extract_graduation_year(edu_section)
        gpa = self._extract_gpa(edu_section)
        
        return {
            'degrees': degrees,
            'level': degree_level,
            'institutions': institutions,
            'graduation_year': graduation_year,
            'gpa': gpa
        }
    
    def extract_certifications_ml(self, text: str) -> List[Dict]:
        """ML-based certification detection"""
        # Extract cert candidates
        cert_candidates = self._extract_cert_candidates(text)
        
        certifications = []
        
        if cert_candidates:
            # Encode candidates
            cand_embs = self.embedder.encode(cert_candidates, show_progress_bar=False)
            
            # Compare with certification examples
            sims = cosine_similarity(cand_embs, self.cert_embeddings)
            
            for i, candidate in enumerate(cert_candidates):
                max_sim = np.max(sims[i])
                
                if max_sim > 0.65:
                    # Extract issuer and year
                    issuer = self._extract_issuer(candidate)
                    year = self._extract_year(candidate)
                    
                    certifications.append({
                        'name': candidate,
                        'confidence': float(max_sim),
                        'issuer': issuer,
                        'year': year
                    })
        
        return certifications
    
    def _extract_education_section(self, text: str) -> str:
        """Find education section"""
        lines = text.split('\n')
        edu_section = []
        in_edu = False
        
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in ['education', 'academic', 'qualification', 'degree']):
                in_edu = True
            elif in_edu and any(kw in line_lower for kw in ['experience', 'employment', 'skills', 'projects']):
                break
            
            if in_edu:
                edu_section.append(line)
        
        return '\n'.join(edu_section)
    
    def _extract_degree_candidates(self, text: str) -> List[str]:
        """Extract potential degree mentions"""
        candidates = []
        lines = text.split('\n')
        
        for line in lines:
            if 10 < len(line.strip()) < 150:
                candidates.append(line.strip())
        
        return candidates
    
    def _extract_institutions_ml(self, text: str) -> List[str]:
        """Extract university/institution names"""
        # University indicators
        indicators = ['university', 'college', 'institute', 'school', 'academy']
        
        institutions = []
        lines = text.split('\n')
        
        for line in lines:
            if any(ind in line.lower() for ind in indicators):
                institutions.append(line.strip())
        
        return institutions[:3]
    
    def _extract_graduation_year(self, text: str) -> Optional[int]:
        """Extract graduation year"""
        years = re.findall(r'\b(19|20)\d{2}\b', text)
        if years:
            return int(max(years))
        return None
    
    def _extract_gpa(self, text: str) -> Optional[float]:
        """Extract GPA"""
        gpa_match = re.search(r'gpa[:\s]+(\d+\.?\d*)', text.lower())
        if gpa_match:
            return float(gpa_match.group(1))
        return None
    
    def _extract_cert_candidates(self, text: str) -> List[str]:
        """Extract certification candidates"""
        candidates = []
        lines = text.split('\n')
        
        cert_keywords = ['certified', 'certification', 'certificate', 'credential']
        
        for line in lines:
            line_lower = line.lower()
            if any(kw in line_lower for kw in cert_keywords):
                clean_line = re.sub(r'^[\-\*]\s*', '', line.strip())
                if 10 < len(clean_line) < 100:
                    candidates.append(clean_line)
        
        return candidates
    
    def _extract_issuer(self, cert_text: str) -> Optional[str]:
        """Extract certification issuer"""
        issuers = ['AWS', 'Microsoft', 'Google', 'Cisco', 'Oracle', 'PMI', 'CompTIA', 'Coursera', 'Udemy']
        for issuer in issuers:
            if issuer.lower() in cert_text.lower():
                return issuer
        return None
    
    def _extract_year(self, cert_text: str) -> Optional[int]:
        """Extract certification year"""
        years = re.findall(r'\b(20\d{2})\b', cert_text)
        if years:
            return int(years[-1])
        return None


# ==================== ML CONFIDENCE SCORING ====================

class MLConfidenceScorer:
    """
    Probabilistic confidence scoring using embedding similarity
    Replaces static weights with dynamic ML-based scoring
    """
    
    def __init__(self, embedder):
        self.embedder = embedder
        
        # High-quality CV examples for reference
        self.high_quality_cv_features = [
            'detailed work experience with metrics',
            'multiple technical skills listed',
            'education from recognized institution',
            'professional certifications',
            'quantifiable achievements and impact',
            'clear career progression',
            'complete contact information'
        ]
        
        self.hq_embeddings = self.embedder.encode(
            self.high_quality_cv_features,
            show_progress_bar=False
        )
    
    def calculate_ml_confidence(self, cv_data: Dict) -> Dict[str, float]:
        """
        Calculate confidence scores using ML
        Returns: {overall: float, per_field: {field: confidence}}
        """
        field_confidences = {}
        
        # Name confidence (semantic check)
        if cv_data.get('name'):
            field_confidences['name'] = self._score_name_ml(cv_data['name'])
        
        # Skills confidence (based on quantity and diversity)
        if cv_data.get('skills'):
            field_confidences['skills'] = self._score_skills_ml(cv_data['skills'])
        
        # Experience confidence (based on detail and metrics)
        if cv_data.get('responsibilities'):
            field_confidences['experience'] = self._score_experience_ml(cv_data['responsibilities'])
        
        # Education confidence
        if cv_data.get('degrees'):
            field_confidences['education'] = self._score_education_ml(cv_data['degrees'])
        
        # Overall confidence (weighted average)
        weights = {
            'name': 0.10,
            'skills': 0.35,
            'experience': 0.30,
            'education': 0.15,
            'contact': 0.10
        }
        
        overall = 0.0
        for field, weight in weights.items():
            if field in field_confidences:
                overall += field_confidences[field] * weight
            else:
                overall += 0.3 * weight  # Default low score for missing
        
        return {
            'overall': overall,
            'per_field': field_confidences
        }
    
    def _score_name_ml(self, name: str) -> float:
        """Score name quality"""
        if not name or len(name) < 3:
            return 0.2
        
        # Check if looks like a real name
        parts = name.split()
        if len(parts) >= 2 and all(p[0].isupper() for p in parts):
            return 0.95
        elif len(parts) >= 2:
            return 0.80
        else:
            return 0.50
    
    def _score_skills_ml(self, skills: List) -> float:
        """Score skills section quality"""
        skill_count = len(skills)
        
        if skill_count == 0:
            return 0.1
        elif skill_count < 5:
            return 0.4
        elif skill_count < 15:
            return 0.7
        elif skill_count < 30:
            return 0.9
        else:
            return 0.95
    
    def _score_experience_ml(self, responsibilities: List) -> float:
        """Score experience section quality"""
        if not responsibilities:
            return 0.2
        
        # Count high-impact statements
        impact_count = 0
        for resp in responsibilities[:10]:
            text = resp if isinstance(resp, str) else resp.get('text', '')
            if re.search(r'\d+[%$kKmM]|\d+\s*percent', text):
                impact_count += 1
        
        base_score = min(0.6, len(responsibilities) * 0.05)
        impact_bonus = min(0.35, impact_count * 0.1)
        
        return base_score + impact_bonus
    
    def _score_education_ml(self, degrees: List) -> float:
        """Score education section quality"""
        if not degrees:
            return 0.3
        
        # Check degree level
        degree_levels = {'PhD': 1.0, 'Master': 0.9, 'Bachelor': 0.8}
        
        max_score = 0.5
        for deg in degrees:
            level = deg.get('level') if isinstance(deg, dict) else None
            if level in degree_levels:
                max_score = max(max_score, degree_levels[level])
        
        return max_score


# ==================== INDUSTRY CLASSIFICATION ====================

class IndustryClassifier:
    """
    Automatically classify CV into industries using semantic embeddings
    """
    
    def __init__(self, embedder):
        self.embedder = embedder
        
        # 25 industry profiles
        self.industries = {
            'Software_Engineering': ['software development', 'coding', 'programming', 'web applications', 'APIs'],
            'Data_Science': ['machine learning', 'data analysis', 'statistics', 'predictive modeling', 'AI'],
            'DevOps': ['CI/CD', 'Docker', 'Kubernetes', 'cloud infrastructure', 'automation'],
            'Product_Management': ['product strategy', 'roadmap', 'stakeholder management', 'user research'],
            'Finance': ['financial analysis', 'accounting', 'investment', 'risk management', 'trading'],
            'Marketing': ['digital marketing', 'SEO', 'content strategy', 'brand management', 'campaigns'],
            'Healthcare': ['medical', 'patient care', 'clinical', 'healthcare systems', 'diagnostics'],
            'Education': ['teaching', 'curriculum', 'pedagogy', 'academic', 'learning'],
            'Consulting': ['business strategy', 'client engagement', 'advisory', 'transformation'],
            'Sales': ['business development', 'revenue generation', 'client acquisition', 'account management'],
            'Design': ['UX design', 'UI design', 'visual design', 'prototyping', 'user experience'],
            'Research': ['scientific research', 'publications', 'experiments', 'academic research'],
            'HR': ['human resources', 'recruitment', 'talent management', 'employee relations'],
            'Legal': ['legal counsel', 'contracts', 'compliance', 'litigation', 'regulatory'],
            'Manufacturing': ['production', 'supply chain', 'operations', 'quality control', 'assembly'],
            'Retail': ['customer service', 'store operations', 'merchandise', 'sales floor'],
            'Real_Estate': ['property management', 'real estate', 'leasing', 'property development'],
            'Media': ['journalism', 'content creation', 'broadcasting', 'publishing', 'media production'],
            'Cybersecurity': ['security', 'penetration testing', 'threat analysis', 'InfoSec', 'vulnerabilities'],
            'Mobile_Development': ['iOS', 'Android', 'mobile apps', 'React Native', 'Swift', 'Kotlin'],
            'Cloud_Engineering': ['AWS', 'Azure', 'GCP', 'cloud architecture', 'serverless'],
            'QA_Testing': ['quality assurance', 'test automation', 'Selenium', 'testing frameworks'],
            'Project_Management': ['project planning', 'Agile', 'Scrum', 'project delivery', 'PMO'],
            'Blockchain': ['cryptocurrency', 'smart contracts', 'DeFi', 'blockchain technology'],
            'IoT': ['Internet of Things', 'embedded systems', 'sensors', 'connected devices']
        }
        
        # Precompute industry embeddings
        self.industry_embeddings = {}
        for industry, keywords in self.industries.items():
            industry_text = ' '.join(keywords)
            emb = self.embedder.encode([industry_text], show_progress_bar=False)[0]
            self.industry_embeddings[industry] = emb
    
    def classify_industry(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Classify CV into industries
        Returns: [(industry, confidence), ...]
        """
        # Encode CV text
        cv_embedding = self.embedder.encode([text[:2000]], show_progress_bar=False)[0]
        
        # Calculate similarity to each industry
        industry_scores = []
        for industry, industry_emb in self.industry_embeddings.items():
            similarity = cosine_similarity([cv_embedding], [industry_emb])[0][0]
            industry_scores.append((industry, float(similarity)))
        
        # Sort and return top K
        industry_scores.sort(key=lambda x: x[1], reverse=True)
        
        return industry_scores[:top_k]


# ==================== CAREER TRAJECTORY ANALYSIS ====================

class CareerTrajectoryAnalyzer:
    """
    Analyze career progression, detect gaps, predict next roles
    """
    
    def __init__(self, embedder):
        self.embedder = embedder
    
    def analyze_trajectory(self, career_progression: List[Dict]) -> Dict:
        """
        Analyze career trajectory
        Returns: {speed: str, gaps: List, predicted_next: List[str]}
        """
        if not career_progression:
            return {'speed': 'Unknown', 'gaps': [], 'predicted_next': []}
        
        # Calculate progression speed
        speed = self._calculate_progression_speed(career_progression)
        
        # Detect gaps
        gaps = self._detect_career_gaps(career_progression)
        
        # Predict next roles
        predicted_next = self._predict_next_roles(career_progression)
        
        return {
            'speed': speed,
            'gaps': gaps,
            'predicted_next': predicted_next
        }
    
    def _calculate_progression_speed(self, progression: List[Dict]) -> str:
        """Calculate how fast someone is progressing"""
        if len(progression) < 2:
            return 'Insufficient Data'
        
        years_span = progression[-1]['start_year'] - progression[0]['start_year']
        role_changes = len(progression) - 1
        
        if years_span == 0:
            return 'Fast'
        
        avg_years_per_role = years_span / role_changes
        
        if avg_years_per_role < 2:
            return 'Very Fast'
        elif avg_years_per_role < 3:
            return 'Fast'
        elif avg_years_per_role < 5:
            return 'Moderate'
        else:
            return 'Slow'
    
    def _detect_career_gaps(self, progression: List[Dict]) -> List[Dict]:
        """Detect gaps in employment"""
        gaps = []
        
        for i in range(len(progression) - 1):
            current_end = progression[i].get('end_year', progression[i]['start_year'])
            next_start = progression[i+1]['start_year']
            
            gap_years = next_start - current_end
            
            if gap_years > 1:
                gaps.append({
                    'period': f"{current_end}-{next_start}",
                    'duration_years': gap_years
                })
        
        return gaps
    
    def _predict_next_roles(self, progression: List[Dict]) -> List[str]:
        """Predict likely next roles based on trajectory"""
        if not progression:
            return []
        
        latest_title = progression[-1].get('title', '')
        
        # Simple rule-based prediction (can be enhanced with ML)
        predictions = []
        
        if 'junior' in latest_title.lower():
            predictions = ['Mid-Level Engineer', 'Software Engineer', 'Engineer II']
        elif 'senior' in latest_title.lower():
            predictions = ['Staff Engineer', 'Principal Engineer', 'Tech Lead', 'Engineering Manager']
        elif 'lead' in latest_title.lower() or 'principal' in latest_title.lower():
            predictions = ['Director of Engineering', 'VP Engineering', 'Chief Architect']
        elif 'manager' in latest_title.lower():
            predictions = ['Senior Manager', 'Director', 'VP']
        else:
            predictions = ['Senior Engineer', 'Tech Lead', 'Specialist']
        
        return predictions[:3]


# ==================== PROJECT & ACHIEVEMENT EXTRACTION ====================

class ProjectExtractor:
    """
    ML-based extraction of projects, publications, achievements
    """
    
    def __init__(self, embedder):
        self.embedder = embedder
        
        # Project indicators
        self.project_keywords = [
            'project', 'built', 'developed', 'created', 'launched',
            'implemented', 'designed', 'architected'
        ]
    
    def extract_projects(self, text: str) -> List[Dict]:
        """Extract projects with technologies and impact"""
        projects = []
        
        # Find project section
        project_section = self._extract_project_section(text)
        
        # Extract project candidates
        candidates = self._extract_project_candidates(project_section if project_section else text)
        
        for candidate in candidates:
            # Extract technologies mentioned
            technologies = self._extract_technologies(candidate)
            
            # Detect impact metrics
            impact = self._extract_impact_metrics(candidate)
            
            projects.append({
                'description': candidate[:200],
                'technologies': technologies,
                'impact': impact
            })
        
        return projects[:10]
    
    def _extract_project_section(self, text: str) -> Optional[str]:
        """Find projects section"""
        lines = text.split('\n')
        section = []
        in_section = False
        
        for line in lines:
            if any(kw in line.lower() for kw in ['project', 'portfolio']):
                in_section = True
            elif in_section and any(kw in line.lower() for kw in ['experience', 'education', 'skills']):
                break
            
            if in_section:
                section.append(line)
        
        return '\n'.join(section) if section else None
    
    def _extract_project_candidates(self, text: str) -> List[str]:
        """Extract project descriptions"""
        candidates = []
        lines = text.split('\n')
        
        current_project = []
        for line in lines:
            line_stripped = line.strip()
            
            # Start of new project
            if any(kw in line.lower() for kw in self.project_keywords):
                if current_project:
                    candidates.append(' '.join(current_project))
                current_project = [line_stripped]
            elif current_project and len(line_stripped) > 10:
                current_project.append(line_stripped)
            elif not line_stripped and current_project:
                candidates.append(' '.join(current_project))
                current_project = []
        
        if current_project:
            candidates.append(' '.join(current_project))
        
        return candidates
    
    def _extract_technologies(self, project_text: str) -> List[str]:
        """Extract technologies from project description"""
        # Common tech terms
        tech_patterns = [
            r'\b[A-Z][a-z]*(?:[A-Z][a-z]*)*\b',  # CamelCase
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+\.js\b',  # JavaScript frameworks
            r'\b\w+[+-]\b'  # Special chars
        ]
        
        technologies = set()
        for pattern in tech_patterns:
            matches = re.findall(pattern, project_text)
            technologies.update(matches)
        
        return list(technologies)[:10]
    
    def _extract_impact_metrics(self, project_text: str) -> Optional[str]:
        """Extract impact metrics"""
        # Look for numbers with units
        metrics = re.findall(
            r'\d+[%$kKmMbB]|\d+\s*(percent|users|customers|hours|million|thousand)',
            project_text
        )
        
        return metrics[0] if metrics else None


# ==================== PORTFOLIO & LINKS DETECTOR ====================

def extract_portfolio_links(text: str) -> Dict[str, str]:
    """
    Detect GitHub, LinkedIn, portfolio URLs
    Infer tech stack from links
    """
    links = {
        'github': None,
        'linkedin': None,
        'portfolio': None,
        'other': []
    }
    
    # GitHub
    github_match = re.search(r'github\.com/[\w-]+', text, re.IGNORECASE)
    if github_match:
        links['github'] = 'https://' + github_match.group(0)
    
    # LinkedIn
    linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', text, re.IGNORECASE)
    if linkedin_match:
        links['linkedin'] = 'https://' + linkedin_match.group(0)
    
    # General URLs
    urls = re.findall(r'https?://[\w\.-]+(?:/[\w\.-]*)*', text)
    for url in urls:
        if 'github' not in url and 'linkedin' not in url:
            if not links['portfolio']:
                links['portfolio'] = url
            else:
                links['other'].append(url)
    
    return links
