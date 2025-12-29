"""
Experience Translator Module (F7)
Advanced NLG-powered experience reformulation system that analyzes and rewrites
professional experience descriptions to match specific job requirements.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import uuid
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ExperienceAnalysis:
    """Analysis results of original experience description"""
    analysis_id: str
    original_text: str
    key_skills: List[str]
    action_verbs: List[str]
    quantified_achievements: List[str]
    experience_level: str
    industry_focus: List[str]
    content_length: int
    clarity_score: float
    improvement_areas: List[str]
    timestamp: str

@dataclass
class TargetAlignment:
    """Analysis of job description alignment"""
    job_requirements: List[str]
    matching_keywords: Dict[str, int]
    missing_keywords: List[str]
    keyword_density: float
    alignment_score: float
    priority_skills: List[str]
    tone_requirements: str

@dataclass
class RewrittenExperience:
    """Result of experience rewriting"""
    rewritten_text: str
    rewriting_style: str
    confidence_score: float
    keyword_matches: Dict[str, int]
    enhancements_made: List[str]
    improvement_suggestions: List[str]
    version_comparison: Dict[str, Any]
    export_formats: Dict[str, str]

class ExperienceTranslator:
    """
    Advanced Experience Translator using NLG techniques for intelligent
    experience reformulation and job requirement alignment.
    """
    
    def __init__(self):
        """Initialize the Experience Translator with NLG models and patterns"""
        self.rewriting_styles = {
            'professional': {
                'tone': 'formal',
                'structure': 'bullet_points',
                'focus': 'achievements',
                'keywords': ['demonstrated', 'achieved', 'delivered', 'managed', 'developed']
            },
            'technical': {
                'tone': 'precise',
                'structure': 'technical_format',
                'focus': 'skills_tools',
                'keywords': ['implemented', 'architected', 'optimized', 'integrated', 'designed']
            },
            'creative': {
                'tone': 'engaging',
                'structure': 'narrative',
                'focus': 'impact_innovation',
                'keywords': ['innovated', 'pioneered', 'transformed', 'revolutionized', 'created']
            }
        }
        
        # Action verbs by category for better rewriting
        self.action_verbs = {
            'leadership': ['led', 'directed', 'coordinated', 'supervised', 'managed'],
            'development': ['developed', 'built', 'created', 'designed', 'implemented'],
            'improvement': ['optimized', 'enhanced', 'improved', 'refined', 'streamlined'],
            'collaboration': ['collaborated', 'partnered', 'worked_with', 'coordinated'],
            'analysis': ['analyzed', 'evaluated', 'assessed', 'investigated', 'reviewed']
        }
        
        # Industry-specific terminology patterns
        self.industry_patterns = {
            'technology': {
                'keywords': ['agile', 'scrum', 'api', 'microservices', 'cloud', 'devops'],
                'frameworks': ['react', 'angular', 'vue', 'node', 'django', 'flask']
            },
            'finance': {
                'keywords': ['roi', 'compliance', 'risk', 'portfolio', 'revenue', 'profit'],
                'frameworks': ['agile', 'compliance', 'risk_management']
            },
            'healthcare': {
                'keywords': ['patient', 'clinical', 'compliance', 'hipaa', 'medical'],
                'frameworks': ['clinical', 'patient_care', 'quality_assurance']
            },
            'marketing': {
                'keywords': ['campaign', 'conversion', 'engagement', 'roi', 'kpi'],
                'frameworks': ['digital_marketing', 'seo', 'content_strategy']
            }
        }
        
        # Quantification patterns for achievements
        self.quantification_patterns = [
            r'(\d+(?:\.\d+)?)\%?\s*(?:increase|decrease|improvement|growth|reduction)',
            r'(\d+(?:\.\d+)?)\s*(?:x|times|fold)\s*(?:increase|growth|improvement)',
            r'\$(\d+(?:\.\d+)?)(?:\s*(?:million|k|thousand))?\s*(?:revenue|savings|budget)',
            r'(\d+(?:\.\d+)?)\s*(?:users|customers|clients|employees)',
            r'(\d+(?:\.\d+)?)\s*(?:projects|implementations|deployments|systems)'
        ]

    def analyze_experience(self, experience_text: str) -> ExperienceAnalysis:
        """
        Analyze original experience description to extract key information
        
        Args:
            experience_text: Original experience description
            
        Returns:
            ExperienceAnalysis object with detailed analysis
        """
        logger.info("Starting experience analysis...")
        
        analysis_id = str(uuid.uuid4())
        text_lower = experience_text.lower()
        
        # Extract key skills using pattern matching
        key_skills = self._extract_skills(experience_text)
        
        # Extract action verbs
        action_verbs = self._extract_action_verbs(text_lower)
        
        # Extract quantified achievements
        quantified_achievements = self._extract_quantified_achievements(experience_text)
        
        # Determine experience level
        experience_level = self._determine_experience_level(experience_text)
        
        # Identify industry focus
        industry_focus = self._identify_industry_focus(text_lower)
        
        # Calculate clarity score
        clarity_score = self._calculate_clarity_score(experience_text)
        
        # Identify improvement areas
        improvement_areas = self._identify_improvement_areas(experience_text, key_skills)
        
        logger.info(f"Experience analysis completed. Found {len(key_skills)} skills, clarity score: {clarity_score:.2f}")
        
        return ExperienceAnalysis(
            analysis_id=analysis_id,
            original_text=experience_text,
            key_skills=key_skills,
            action_verbs=action_verbs,
            quantified_achievements=quantified_achievements,
            experience_level=experience_level,
            industry_focus=industry_focus,
            content_length=len(experience_text),
            clarity_score=clarity_score,
            improvement_areas=improvement_areas,
            timestamp=datetime.now().isoformat()
        )

    def analyze_job_alignment(self, job_description: str, experience_analysis: ExperienceAnalysis) -> TargetAlignment:
        """
        Analyze job description and alignment with current experience
        
        Args:
            job_description: Target job description
            experience_analysis: Analysis of original experience
            
        Returns:
            TargetAlignment object with alignment analysis
        """
        logger.info("Starting job alignment analysis...")
        
        # Extract job requirements from description
        job_requirements = self._extract_job_requirements(job_description)
        
        # Identify matching keywords
        matching_keywords = self._find_matching_keywords(
            job_description, experience_analysis.key_skills
        )
        
        # Find missing keywords
        missing_keywords = self._find_missing_keywords(
            job_description, experience_analysis.key_skills
        )
        
        # Calculate keyword density
        keyword_density = self._calculate_keyword_density(job_description, matching_keywords)
        
        # Calculate alignment score
        alignment_score = self._calculate_alignment_score(
            matching_keywords, missing_keywords, job_requirements
        )
        
        # Identify priority skills
        priority_skills = self._identify_priority_skills(job_description)
        
        # Determine tone requirements
        tone_requirements = self._analyze_tone_requirements(job_description)
        
        logger.info(f"Job alignment analysis completed. Alignment score: {alignment_score:.2f}")
        
        return TargetAlignment(
            job_requirements=job_requirements,
            matching_keywords=matching_keywords,
            missing_keywords=missing_keywords,
            keyword_density=keyword_density,
            alignment_score=alignment_score,
            priority_skills=priority_skills,
            tone_requirements=tone_requirements
        )

    def rewrite_experience(self, 
                          experience_analysis: ExperienceAnalysis, 
                          target_alignment: TargetAlignment,
                          rewriting_style: str = 'professional') -> RewrittenExperience:
        """
        Rewrite experience using NLG techniques for target job alignment
        
        Args:
            experience_analysis: Analysis of original experience
            target_alignment: Analysis of job alignment
            rewriting_style: Style of rewriting ('professional', 'technical', 'creative')
            
        Returns:
            RewrittenExperience object with rewritten content
        """
        logger.info(f"Starting experience rewriting in '{rewriting_style}' style...")
        
        if rewriting_style not in self.rewriting_styles:
            raise ValueError(f"Invalid rewriting style: {rewriting_style}")
        
        style_config = self.rewriting_styles[rewriting_style]
        
        # Generate rewritten content using NLG techniques
        rewritten_text = self._generate_rewritten_content(
            experience_analysis, target_alignment, style_config
        )
        
        # Calculate confidence score
        confidence_score = self._calculate_rewrite_confidence(
            experience_analysis, target_alignment, rewritten_text
        )
        
        # Identify keyword matches in rewritten text
        keyword_matches = self._extract_keyword_matches(
            rewritten_text, target_alignment.job_requirements
        )
        
        # Identify enhancements made
        enhancements_made = self._identify_enhancements(
            experience_analysis.original_text, rewritten_text, target_alignment
        )
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            experience_analysis, target_alignment, rewritten_text
        )
        
        # Create version comparison
        version_comparison = self._create_version_comparison(
            experience_analysis, rewritten_text, keyword_matches
        )
        
        # Generate export formats
        export_formats = self._generate_export_formats(rewritten_text, keyword_matches)
        
        logger.info(f"Experience rewriting completed. Confidence: {confidence_score:.2f}")
        
        return RewrittenExperience(
            rewritten_text=rewritten_text,
            rewriting_style=rewriting_style,
            confidence_score=confidence_score,
            keyword_matches=keyword_matches,
            enhancements_made=enhancements_made,
            improvement_suggestions=improvement_suggestions,
            version_comparison=version_comparison,
            export_formats=export_formats
        )

    def _extract_skills(self, text: str) -> List[str]:
        """Extract technical and soft skills from text"""
        skills = []
        
        # Technical skills patterns
        tech_patterns = [
            r'\b(python|java|javascript|typescript|react|angular|vue|node\.?js)\b',
            r'\b(sql|mysql|postgresql|mongodb|redis)\b',
            r'\b(aws|azure|gcp|docker|kubernetes)\b',
            r'\b(git|github|gitlab|jenkins|ci/cd)\b',
            r'\b(ml|ai|machine learning|deep learning|tensorflow|pytorch)\b'
        ]
        
        # Soft skills patterns
        soft_patterns = [
            r'\b(leadership|teamwork|collaboration|communication|problem.solving)\b',
            r'\b(analytical|creative|innovative|detail.oriented|project.management)\b'
        ]
        
        for pattern in tech_patterns + soft_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend([match.title() for match in matches])
        
        # Remove duplicates and return
        return list(set(skills))

    def _extract_action_verbs(self, text: str) -> List[str]:
        """Extract action verbs from text"""
        found_verbs = []
        
        for category, verbs in self.action_verbs.items():
            for verb in verbs:
                if verb in text:
                    found_verbs.append(verb)
        
        return found_verbs

    def _extract_quantified_achievements(self, text: str) -> List[str]:
        """Extract quantified achievements from text"""
        achievements = []
        
        for pattern in self.quantification_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            achievements.extend(matches)
        
        return achievements

    def _determine_experience_level(self, text: str) -> str:
        """Determine experience level based on content"""
        text_lower = text.lower()
        
        senior_indicators = ['senior', 'lead', 'principal', 'architect', 'management']
        junior_indicators = ['junior', 'entry', 'intern', 'graduate', 'assistant']
        
        if any(indicator in text_lower for indicator in senior_indicators):
            return 'senior'
        elif any(indicator in text_lower for indicator in junior_indicators):
            return 'junior'
        else:
            return 'mid-level'

    def _identify_industry_focus(self, text: str) -> List[str]:
        """Identify industry focus from text"""
        industries = []
        
        for industry, patterns in self.industry_patterns.items():
            if any(keyword in text for keyword in patterns['keywords']):
                industries.append(industry)
        
        return industries if industries else ['general']

    def _calculate_clarity_score(self, text: str) -> float:
        """Calculate clarity score based on various factors"""
        score = 0.0
        
        # Length factor (optimal: 100-500 words)
        length = len(text.split())
        if 100 <= length <= 500:
            score += 0.3
        elif 50 <= length <= 1000:
            score += 0.2
        
        # Quantification factor
        quantified_count = len(self._extract_quantified_achievements(text))
        if quantified_count >= 3:
            score += 0.3
        elif quantified_count >= 1:
            score += 0.2
        
        # Structure factor (presence of bullet points or clear sections)
        if '' in text or '-' in text or '\n' in text:
            score += 0.2
        
        # Action verbs factor
        action_verb_count = len(self._extract_action_verbs(text.lower()))
        if action_verb_count >= 5:
            score += 0.2
        elif action_verb_count >= 3:
            score += 0.1
        
        return min(score, 1.0)

    def _identify_improvement_areas(self, text: str, skills: List[str]) -> List[str]:
        """Identify areas for improvement"""
        improvements = []
        
        # Check for quantification
        if len(self._extract_quantified_achievements(text)) < 2:
            improvements.append("Add more quantified achievements")
        
        # Check for action verbs
        if len(self._extract_action_verbs(text.lower())) < 3:
            improvements.append("Use more action-oriented language")
        
        # Check for skills mention
        if len(skills) < 5:
            improvements.append("Include more relevant technical skills")
        
        # Check for length
        if len(text.split()) < 100:
            improvements.append("Expand description with more details")
        
        return improvements

    def _extract_job_requirements(self, job_description: str) -> List[str]:
        """Extract job requirements from job description"""
        requirements = []
        
        # Look for requirement sections
        requirement_patterns = [
            r'required[:\s]*(.*?)(?:\n|$)',
            r'qualifications[:\s]*(.*?)(?:\n|$)',
            r'must.have[:\s]*(.*?)(?:\n|$)',
            r'responsibilities[:\s]*(.*?)(?:\n|$)'
        ]
        
        for pattern in requirement_patterns:
            matches = re.findall(pattern, job_description, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Split by common delimiters
                items = re.split(r'[\-\n]', match)
                requirements.extend([item.strip() for item in items if item.strip()])
        
        return requirements

    def _find_matching_keywords(self, job_description: str, skills: List[str]) -> Dict[str, int]:
        """Find matching keywords between job description and skills"""
        matches = {}
        job_lower = job_description.lower()
        
        for skill in skills:
            count = job_lower.count(skill.lower())
            if count > 0:
                matches[skill] = count
        
        return matches

    def _find_missing_keywords(self, job_description: str, skills: List[str]) -> List[str]:
        """Find keywords from job description not present in skills"""
        job_lower = job_description.lower()
        missing = []
        
        # Extract potential skills from job description
        job_skills = self._extract_skills(job_description)
        
        for skill in job_skills:
            if skill.lower() not in [s.lower() for s in skills]:
                missing.append(skill)
        
        return missing

    def _calculate_keyword_density(self, job_description: str, matches: Dict[str, int]) -> float:
        """Calculate keyword density in job description"""
        if not matches:
            return 0.0
        
        total_words = len(job_description.split())
        matched_words = sum(matches.values())
        
        return (matched_words / total_words) * 100 if total_words > 0 else 0.0

    def _calculate_alignment_score(self, matches: Dict[str, int], missing: List[str], requirements: List[str]) -> float:
        """Calculate overall alignment score"""
        if not requirements:
            return 0.5
        
        # Score based on matches and missing requirements
        match_score = len(matches) / len(requirements) if requirements else 0
        penalty = len(missing) / len(requirements) if requirements else 0
        
        return max(0, match_score - penalty * 0.5)

    def _identify_priority_skills(self, job_description: str) -> List[str]:
        """Identify priority skills from job description"""
        # Look for skills mentioned multiple times or in key sections
        skills = self._extract_skills(job_description)
        skill_counts = Counter(skills)
        
        # Skills mentioned more than once are likely priority
        priority_skills = [skill for skill, count in skill_counts.items() if count > 1]
        
        return priority_skills[:5]  # Top 5

    def _analyze_tone_requirements(self, job_description: str) -> str:
        """Analyze tone requirements from job description"""
        text_lower = job_description.lower()
        
        if any(word in text_lower for word in ['innovative', 'creative', 'visionary']):
            return 'innovative'
        elif any(word in text_lower for word in ['leadership', 'senior', 'management']):
            return 'leadership'
        elif any(word in text_lower for word in ['technical', 'detailed', 'specific']):
            return 'technical'
        else:
            return 'professional'

    def _generate_rewritten_content(self, experience: ExperienceAnalysis, alignment: TargetAlignment, style_config: Dict) -> str:
        """Generate rewritten content using NLG techniques"""
        
        # Start with structure based on style
        if style_config['structure'] == 'bullet_points':
            lines = []
            
            # Process each sentence/phrase
            original_sentences = re.split(r'[.!?]+', experience.original_text)
            
            for sentence in original_sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Enhance with action verbs if missing
                enhanced_sentence = self._enhance_with_action_verbs(sentence, style_config)
                
                # Add quantification if missing
                quantified_sentence = self._add_quantification(enhanced_sentence, alignment)
                
                # Ensure proper keywords are included
                final_sentence = self._integrate_keywords(quantified_sentence, alignment)
                
                lines.append(f" {final_sentence}")
            
            rewritten_text = '\n'.join(lines)
            
        else:
            # For other structures, use paragraph format
            rewritten_text = self._generate_paragraph_format(experience, alignment, style_config)
        
        return rewritten_text

    def _enhance_with_action_verbs(self, sentence: str, style_config: Dict) -> str:
        """Enhance sentence with appropriate action verbs"""
        # This is a simplified enhancement - in a real implementation,
        # this would use more sophisticated NLP techniques
        if not any(verb in sentence.lower() for verb in ['developed', 'created', 'managed', 'led']):
            # Add a leading action verb if none present
            action_verb = style_config['keywords'][0]
            sentence = f"{action_verb.capitalize()} {sentence.lower()}"
        
        return sentence

    def _add_quantification(self, sentence: str, alignment: TargetAlignment) -> str:
        """Add quantification to achievements where appropriate"""
        # This is a simplified approach - real implementation would be more sophisticated
        if 'increase' in sentence.lower() or 'improvement' in sentence.lower():
            if '25%' not in sentence and '40%' not in sentence:
                return sentence + ", resulting in 25% improvement"
        
        return sentence

    def _integrate_keywords(self, sentence: str, alignment: TargetAlignment) -> str:
        """Integrate relevant keywords from job description"""
        # Add priority skills if not present
        for skill in alignment.priority_skills[:2]:
            if skill.lower() not in sentence.lower():
                # Try to naturally integrate the skill
                if sentence.endswith('.'):
                    sentence = sentence[:-1] + f" using {skill.lower()}"
                else:
                    sentence += f", utilizing {skill.lower()}"
        
        return sentence

    def _generate_paragraph_format(self, experience: ExperienceAnalysis, alignment: TargetAlignment, style_config: Dict) -> str:
        """Generate content in paragraph format"""
        # Create a flowing narrative version
        enhanced_experience = experience.original_text
        
        # Add missing keywords naturally
        for skill in alignment.missing_keywords[:2]:
            if skill.lower() not in enhanced_experience.lower():
                # Insert skill mention naturally
                enhanced_experience += f" Additionally, gained hands-on experience with {skill.lower()}."
        
        return enhanced_experience

    def _calculate_rewrite_confidence(self, experience: ExperienceAnalysis, alignment: TargetAlignment, rewritten_text: str) -> float:
        """Calculate confidence score for the rewrite"""
        confidence = 0.0
        
        # Base score from alignment
        confidence += alignment.alignment_score * 0.4
        
        # Score from keyword integration
        if alignment.missing_keywords:
            integrated_count = sum(1 for skill in alignment.missing_keywords if skill.lower() in rewritten_text.lower())
            confidence += (integrated_count / len(alignment.missing_keywords)) * 0.3
        
        # Score from content enhancement
        if len(rewritten_text) > len(experience.original_text):
            confidence += 0.2
        
        # Score from structure improvement
        if '' in rewritten_text and '' not in experience.original_text:
            confidence += 0.1
        
        return min(confidence, 1.0)

    def _extract_keyword_matches(self, rewritten_text: str, requirements: List[str]) -> Dict[str, int]:
        """Extract keyword matches from rewritten text"""
        matches = {}
        text_lower = rewritten_text.lower()
        
        for requirement in requirements:
            # Extract potential keywords from requirements
            potential_skills = self._extract_skills(requirement)
            for skill in potential_skills:
                count = text_lower.count(skill.lower())
                if count > 0:
                    matches[skill] = matches.get(skill, 0) + count
        
        return matches

    def _identify_enhancements(self, original: str, rewritten: str, alignment: TargetAlignment) -> List[str]:
        """Identify enhancements made during rewriting"""
        enhancements = []
        
        # Check for keyword additions
        if len(alignment.missing_keywords) > 0:
            added_keywords = [skill for skill in alignment.missing_keywords if skill.lower() in rewritten.lower()]
            if added_keywords:
                enhancements.append(f"Added {len(added_keywords)} job-relevant keywords: {', '.join(added_keywords[:3])}")
        
        # Check for structure improvement
        if '' in rewritten and '' not in original:
            enhancements.append("Enhanced readability with bullet-point structure")
        
        # Check for quantification
        if rewritten.count('%') > original.count('%'):
            enhancements.append("Added quantified achievements")
        
        # Check for length improvement
        if len(rewritten) > len(original) * 1.2:
            enhancements.append("Expanded content with additional relevant details")
        
        return enhancements

    def _generate_improvement_suggestions(self, experience: ExperienceAnalysis, alignment: TargetAlignment, rewritten_text: str) -> List[str]:
        """Generate actionable improvement suggestions"""
        suggestions = []
        
        # Alignment-based suggestions
        if alignment.alignment_score < 0.7:
            suggestions.append("Focus on more closely matching job-specific requirements")
        
        # Keyword density suggestions
        if alignment.keyword_density < 5:
            suggestions.append("Increase usage of industry-specific terminology")
        
        # Content suggestions
        if len(rewritten_text.split()) < 150:
            suggestions.append("Expand descriptions with more specific examples")
        
        # Structure suggestions
        if experience.clarity_score < 0.6:
            suggestions.append("Improve clarity through better organization and structure")
        
        # Action verb suggestions
        if len(experience.action_verbs) < 3:
            suggestions.append("Use more dynamic action verbs to convey impact")
        
        return suggestions

    def _create_version_comparison(self, experience: ExperienceAnalysis, rewritten_text: str, matches: Dict[str, int]) -> Dict[str, Any]:
        """Create comparison between original and rewritten versions"""
        return {
            "original_length": len(experience.original_text),
            "rewritten_length": len(rewritten_text),
            "length_change": len(rewritten_text) - len(experience.original_text),
            "keyword_matches": len(matches),
            "clarity_improvement": self._calculate_clarity_score(rewritten_text) - experience.clarity_score,
            "readability_score": self._calculate_clarity_score(rewritten_text),
            "job_alignment_improvement": "Enhanced" if matches else "Basic"
        }

    def _generate_export_formats(self, rewritten_text: str, matches: Dict[str, int]) -> Dict[str, str]:
        """Generate different export formats"""
        return {
            "text": rewritten_text,
            "markdown": self._convert_to_markdown(rewritten_text, matches),
            "json": json.dumps({
                "rewritten_text": rewritten_text,
                "keyword_matches": matches,
                "export_timestamp": datetime.now().isoformat()
            }, indent=2),
            "html": self._convert_to_html(rewritten_text, matches)
        }

    def _convert_to_markdown(self, text: str, matches: Dict[str, int]) -> str:
        """Convert to markdown format"""
        markdown_text = text.replace('', '-')
        
        # Add keyword highlight section
        if matches:
            markdown_text += f"\n\n### Aligned Keywords\n"
            for keyword, count in matches.items():
                markdown_text += f"- {keyword}: {count} mentions\n"
        
        return markdown_text

    def _convert_to_html(self, text: str, matches: Dict[str, int]) -> str:
        """Convert to HTML format"""
        html_content = text.replace('', '<li>').replace('\n', '</li>\n<li>')
        if '<li>' in html_content:
            html_content = html_content.replace('\n\n', '</li>\n\n<li>')
            html_content += '</li>'
        
        # Add keyword section
        if matches:
            html_content += '<h3>Aligned Keywords</h3><ul>'
            for keyword, count in matches.items():
                html_content += f'<li>{keyword}: {count} mentions</li>'
            html_content += '</ul>'
        
        return f"<div class='experience-text'>{html_content}</div>"

    def translate_experience(self, original_experience: str, job_description: str, style: str = 'professional') -> Dict[str, Any]:
        """
        Main method to translate experience for job requirements
        
        Args:
            original_experience: Original experience description
            job_description: Target job description
            style: Rewriting style ('professional', 'technical', 'creative')
            
        Returns:
            Complete translation results dictionary
        """
        logger.info(f"Starting experience translation with style: {style}")
        
        try:
            # Step 1: Analyze original experience
            experience_analysis = self.analyze_experience(original_experience)
            
            # Step 2: Analyze job alignment
            target_alignment = self.analyze_job_alignment(job_description, experience_analysis)
            
            # Step 3: Rewrite experience
            rewritten_result = self.rewrite_experience(experience_analysis, target_alignment, style)
            
            # Step 4: Compile complete results
            results = {
                "translation_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "analysis": {
                    "experience_analysis": {
                        "analysis_id": experience_analysis.analysis_id,
                        "key_skills": experience_analysis.key_skills,
                        "action_verbs": experience_analysis.action_verbs,
                        "quantified_achievements": experience_analysis.quantified_achievements,
                        "experience_level": experience_analysis.experience_level,
                        "clarity_score": experience_analysis.clarity_score,
                        "improvement_areas": experience_analysis.improvement_areas
                    },
                    "job_alignment": {
                        "alignment_score": target_alignment.alignment_score,
                        "keyword_density": target_alignment.keyword_density,
                        "matching_keywords": target_alignment.matching_keywords,
                        "missing_keywords": target_alignment.missing_keywords,
                        "priority_skills": target_alignment.priority_skills,
                        "tone_requirements": target_alignment.tone_requirements
                    }
                },
                "rewritten_experience": {
                    "text": rewritten_result.rewritten_text,
                    "style": rewritten_result.rewriting_style,
                    "confidence_score": rewritten_result.confidence_score,
                    "keyword_matches": rewritten_result.keyword_matches,
                    "enhancements_made": rewritten_result.enhancements_made,
                    "improvement_suggestions": rewritten_result.improvement_suggestions,
                    "version_comparison": rewritten_result.version_comparison,
                    "export_formats": rewritten_result.export_formats
                },
                "metadata": {
                    "original_length": len(original_experience),
                    "rewritten_length": len(rewritten_result.rewritten_text),
                    "processing_time": "completed",
                    "style_used": style
                }
            }
            
            logger.info(f"Experience translation completed successfully. Confidence: {rewritten_result.confidence_score:.2f}")
            return results
            
        except Exception as e:
            logger.error(f"Experience translation failed: {e}")
            raise

# Initialize global instance
experience_translator = ExperienceTranslator()

def translate_experience_api(original_experience: str, job_description: str, style: str = 'professional') -> Dict[str, Any]:
    """
    API wrapper for experience translation
    
    Args:
        original_experience: Original experience description
        job_description: Target job description  
        style: Rewriting style
        
    Returns:
        Translation results
    """
    return experience_translator.translate_experience(original_experience, job_description, style)