"""
 Career Guidance Engine with Explainable AI (XAI)
Provides actionable career recommendations based on ML-powered CV analysis
"""
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class JobRecommendation:
    """Job recommendation with XAI reasoning"""
    title: str
    match_score: float
    reasons: List[str]
    required_skills: List[str]
    missing_skills: List[str]
    salary_range: str
    growth_potential: str


@dataclass
class CertificationRecommendation:
    """Certification recommendation with XAI reasoning"""
    name: str
    provider: str
    priority: str
    duration: str
    cost: str
    reasons: List[str]
    skills_gained: List[str]
    career_impact: str


@dataclass
class LearningRoadmap:
    """Step-by-step learning roadmap"""
    current_level: str
    target_level: str
    timeline: str
    phases: List[Dict[str, Any]]
    reasoning: List[str]


@dataclass
class CareerGuidance:
    """Complete career guidance with XAI"""
    # From CV analysis
    skills: List[str]
    seniority: str
    industries: List[Tuple[str, float]]
    projects: List[Dict]
    portfolio_links: Dict[str, str]
    experience_years: int
    
    # Recommendations
    recommended_jobs: List[JobRecommendation]
    recommended_certifications: List[CertificationRecommendation]
    learning_roadmap: LearningRoadmap
    
    # XAI Insights
    xai_insights: Dict[str, Any]


class CareerGuidanceEngine:
    """
    Advanced Career Guidance Engine with Explainable AI
    Analyzes CV using ML and provides actionable recommendations
    """
    
    def __init__(self):
        """Initialize career guidance engine"""
        self.job_database = self._load_job_database()
        self.cert_database = self._load_certification_database()
        self.skill_taxonomy = self._load_skill_taxonomy()
        logger.info(" Career Guidance Engine initialized")
    
    def _load_job_database(self) -> Dict[str, Dict]:
        """Load job role database with requirements"""
        return {
            # AI/ML Roles
            "ML Engineer": {
                "required_skills": ["Python", "Machine Learning", "TensorFlow", "PyTorch", "Deep Learning"],
                "preferred_skills": ["MLOps", "Docker", "Kubernetes", "AWS"],
                "min_experience": 2,
                "industries": ["Data_Science", "AI_ML", "Software_Engineering"],
                "salary_range": "$90k-$150k",
                "growth_potential": "Very High",
                "seniority_levels": ["Mid", "Senior", "Lead"]
            },
            "Data Scientist": {
                "required_skills": ["Python", "Machine Learning", "Statistics", "Pandas", "NumPy"],
                "preferred_skills": ["SQL", "Data Visualization", "R", "TensorFlow"],
                "min_experience": 1,
                "industries": ["Data_Science", "Analytics", "Business_Intelligence"],
                "salary_range": "$80k-$140k",
                "growth_potential": "Very High",
                "seniority_levels": ["Junior", "Mid", "Senior"]
            },
            "AI Research Engineer": {
                "required_skills": ["Python", "Deep Learning", "NLP", "Computer Vision", "Research"],
                "preferred_skills": ["PyTorch", "TensorFlow", "CUDA", "C++"],
                "min_experience": 3,
                "industries": ["AI_ML", "Research", "Data_Science"],
                "salary_range": "$100k-$180k",
                "growth_potential": "Very High",
                "seniority_levels": ["Senior", "Lead", "Principal"]
            },
            
            # Software Engineering Roles
            "Full Stack Developer": {
                "required_skills": ["JavaScript", "React", "Node.js", "HTML", "CSS"],
                "preferred_skills": ["TypeScript", "MongoDB", "PostgreSQL", "Docker"],
                "min_experience": 1,
                "industries": ["Software_Engineering", "Web_Development", "Startups"],
                "salary_range": "$70k-$130k",
                "growth_potential": "High",
                "seniority_levels": ["Junior", "Mid", "Senior"]
            },
            "Backend Engineer": {
                "required_skills": ["Python", "Django", "FastAPI", "SQL", "REST API"],
                "preferred_skills": ["Docker", "Redis", "PostgreSQL", "Microservices"],
                "min_experience": 1,
                "industries": ["Software_Engineering", "Cloud", "DevOps"],
                "salary_range": "$75k-$135k",
                "growth_potential": "High",
                "seniority_levels": ["Junior", "Mid", "Senior"]
            },
            "Frontend Developer": {
                "required_skills": ["JavaScript", "React", "HTML", "CSS", "TypeScript"],
                "preferred_skills": ["Vue", "Angular", "Next.js", "Webpack"],
                "min_experience": 1,
                "industries": ["Software_Engineering", "Web_Development", "UI_UX"],
                "salary_range": "$65k-$125k",
                "growth_potential": "High",
                "seniority_levels": ["Junior", "Mid", "Senior"]
            },
            
            # DevOps/Cloud Roles
            "DevOps Engineer": {
                "required_skills": ["Docker", "Kubernetes", "CI/CD", "AWS", "Linux"],
                "preferred_skills": ["Terraform", "Jenkins", "Ansible", "Python"],
                "min_experience": 2,
                "industries": ["DevOps", "Cloud", "Infrastructure"],
                "salary_range": "$85k-$145k",
                "growth_potential": "Very High",
                "seniority_levels": ["Mid", "Senior", "Lead"]
            },
            "Cloud Architect": {
                "required_skills": ["AWS", "Azure", "GCP", "Cloud Architecture", "Terraform"],
                "preferred_skills": ["Kubernetes", "Docker", "Microservices", "Security"],
                "min_experience": 4,
                "industries": ["Cloud", "Enterprise", "DevOps"],
                "salary_range": "$110k-$180k",
                "growth_potential": "Very High",
                "seniority_levels": ["Senior", "Lead", "Principal"]
            },
            
            # Entry Level Roles
            "Junior Software Developer": {
                "required_skills": ["Python", "JavaScript", "HTML", "CSS", "Git"],
                "preferred_skills": ["React", "SQL", "Linux", "Problem Solving"],
                "min_experience": 0,
                "industries": ["Software_Engineering", "Startups", "Tech"],
                "salary_range": "$50k-$80k",
                "growth_potential": "High",
                "seniority_levels": ["Junior", "Entry"]
            },
            "Junior Data Analyst": {
                "required_skills": ["Python", "SQL", "Excel", "Data Visualization"],
                "preferred_skills": ["Pandas", "Tableau", "Statistics", "Machine Learning"],
                "min_experience": 0,
                "industries": ["Data_Science", "Analytics", "Business_Intelligence"],
                "salary_range": "$45k-$75k",
                "growth_potential": "High",
                "seniority_levels": ["Junior", "Entry"]
            }
        }
    
    def _load_certification_database(self) -> List[Dict]:
        """Load certification recommendations database"""
        return [
            # Cloud Certifications
            {
                "name": "AWS Certified Solutions Architect",
                "provider": "Amazon Web Services",
                "skills": ["AWS", "Cloud", "Architecture"],
                "duration": "3-4 months",
                "cost": "$150",
                "difficulty": "Intermediate",
                "career_impact": "High - Opens cloud architecture roles",
                "industries": ["Cloud", "DevOps", "Enterprise"]
            },
            {
                "name": "Google Cloud Professional Data Engineer",
                "provider": "Google Cloud",
                "skills": ["GCP", "Data Engineering", "BigQuery"],
                "duration": "2-3 months",
                "cost": "$200",
                "difficulty": "Intermediate",
                "career_impact": "High - Strong for data engineering roles",
                "industries": ["Data_Science", "Cloud", "Big_Data"]
            },
            {
                "name": "Microsoft Azure Fundamentals",
                "provider": "Microsoft",
                "skills": ["Azure", "Cloud", "DevOps"],
                "duration": "1-2 months",
                "cost": "$99",
                "difficulty": "Beginner",
                "career_impact": "Medium - Good foundation for cloud roles",
                "industries": ["Cloud", "Enterprise", "DevOps"]
            },
            
            # AI/ML Certifications
            {
                "name": "TensorFlow Developer Certificate",
                "provider": "Google",
                "skills": ["TensorFlow", "Deep Learning", "ML"],
                "duration": "2-3 months",
                "cost": "$100",
                "difficulty": "Intermediate",
                "career_impact": "Very High - Essential for ML roles",
                "industries": ["AI_ML", "Data_Science", "Research"]
            },
            {
                "name": "Deep Learning Specialization",
                "provider": "Coursera (Andrew Ng)",
                "skills": ["Deep Learning", "Neural Networks", "Python"],
                "duration": "3-4 months",
                "cost": "$49/month",
                "difficulty": "Intermediate",
                "career_impact": "Very High - Industry gold standard",
                "industries": ["AI_ML", "Data_Science", "Research"]
            },
            {
                "name": "AWS Machine Learning Specialty",
                "provider": "AWS",
                "skills": ["AWS", "Machine Learning", "SageMaker"],
                "duration": "2-3 months",
                "cost": "$300",
                "difficulty": "Advanced",
                "career_impact": "Very High - Combines ML + Cloud",
                "industries": ["AI_ML", "Cloud", "MLOps"]
            },
            
            # DevOps Certifications
            {
                "name": "Certified Kubernetes Administrator (CKA)",
                "provider": "CNCF",
                "skills": ["Kubernetes", "Container Orchestration", "DevOps"],
                "duration": "2-3 months",
                "cost": "$395",
                "difficulty": "Advanced",
                "career_impact": "Very High - Critical for DevOps/Cloud roles",
                "industries": ["DevOps", "Cloud", "Infrastructure"]
            },
            {
                "name": "Docker Certified Associate",
                "provider": "Docker",
                "skills": ["Docker", "Containers", "DevOps"],
                "duration": "1-2 months",
                "cost": "$195",
                "difficulty": "Intermediate",
                "career_impact": "High - Essential DevOps skill",
                "industries": ["DevOps", "Cloud", "Software_Engineering"]
            },
            
            # Development Certifications
            {
                "name": "Meta Front-End Developer Professional",
                "provider": "Coursera (Meta)",
                "skills": ["React", "JavaScript", "HTML", "CSS"],
                "duration": "5-6 months",
                "cost": "$49/month",
                "difficulty": "Beginner",
                "career_impact": "High - Strong foundation for frontend roles",
                "industries": ["Web_Development", "Software_Engineering", "UI_UX"]
            },
            {
                "name": "Python for Everybody Specialization",
                "provider": "Coursera (University of Michigan)",
                "skills": ["Python", "Data Structures", "SQL"],
                "duration": "3-4 months",
                "cost": "$49/month",
                "difficulty": "Beginner",
                "career_impact": "Medium - Good Python foundation",
                "industries": ["Software_Engineering", "Data_Science", "General"]
            }
        ]
    
    def _load_skill_taxonomy(self) -> Dict[str, List[str]]:
        """Load skill taxonomy for skill mapping"""
        return {
            "AI_ML": ["Machine Learning", "Deep Learning", "NLP", "Computer Vision", "TensorFlow", 
                     "PyTorch", "Scikit-learn", "MLOps", "Neural Networks"],
            "Data_Science": ["Python", "Pandas", "NumPy", "Statistics", "Data Visualization", 
                            "Jupyter", "R", "SQL"],
            "Cloud": ["AWS", "Azure", "GCP", "Cloud Architecture", "Serverless"],
            "DevOps": ["Docker", "Kubernetes", "CI/CD", "Jenkins", "Terraform", "Ansible"],
            "Frontend": ["React", "Angular", "Vue", "JavaScript", "TypeScript", "HTML", "CSS"],
            "Backend": ["Python", "Django", "FastAPI", "Node.js", "Java", "Spring"],
            "Database": ["SQL", "PostgreSQL", "MongoDB", "Redis", "MySQL"],
            "Mobile": ["React Native", "Flutter", "iOS", "Android", "Swift", "Kotlin"]
        }
    
    def analyze_and_guide(self, cv_analysis: Dict[str, Any]) -> CareerGuidance:
        """
        Main method: Analyze CV and provide comprehensive career guidance
        
        Args:
            cv_analysis: Result from /api/v1/analyze-cv-advanced
            
        Returns:
            CareerGuidance with jobs, certs, roadmap, and XAI insights
        """
        logger.info(" Starting career guidance analysis...")
        
        # Extract CV data
        skills = cv_analysis.get('skills', [])
        seniority = cv_analysis.get('seniority_level', 'Unknown')
        industries = cv_analysis.get('industries', [])
        projects = cv_analysis.get('projects', [])
        portfolio_links = cv_analysis.get('portfolio_links', {})
        experience_years = cv_analysis.get('experience_years', 0) or cv_analysis.get('total_years_experience', 0)
        
        # Generate recommendations with XAI
        recommended_jobs = self._recommend_jobs(skills, seniority, industries, experience_years)
        recommended_certs = self._recommend_certifications(skills, industries, recommended_jobs)
        learning_roadmap = self._create_learning_roadmap(skills, seniority, recommended_jobs)
        xai_insights = self._generate_xai_insights(cv_analysis, recommended_jobs, recommended_certs)
        
        guidance = CareerGuidance(
            skills=skills,
            seniority=seniority,
            industries=industries,
            projects=projects,
            portfolio_links=portfolio_links,
            experience_years=experience_years,
            recommended_jobs=recommended_jobs,
            recommended_certifications=recommended_certs,
            learning_roadmap=learning_roadmap,
            xai_insights=xai_insights
        )
        
        logger.info(f" Career guidance complete: {len(recommended_jobs)} jobs, {len(recommended_certs)} certs")
        return guidance
    
    def _recommend_jobs(self, skills: List[str], seniority: str, 
                       industries: List[Tuple[str, float]], experience_years: int) -> List[JobRecommendation]:
        """Recommend jobs with XAI reasoning"""
        recommendations = []
        skills_lower = [s.lower() for s in skills]
        
        # Map seniority
        seniority_map = {
            "Junior": ["Junior", "Entry"],
            "Mid": ["Junior", "Mid"],
            "Senior": ["Mid", "Senior", "Lead"],
            "Lead": ["Senior", "Lead", "Principal"],
            "Unknown": ["Junior", "Mid"]  # Default for unknown
        }
        allowed_seniorities = seniority_map.get(seniority, ["Junior", "Mid"])
        
        for job_title, job_data in self.job_database.items():
            # Check seniority fit
            if not any(level in allowed_seniorities for level in job_data['seniority_levels']):
                continue
            
            # Check experience
            if experience_years < job_data['min_experience']:
                continue
            
            # Calculate skill match
            required_matched = sum(1 for skill in job_data['required_skills'] 
                                  if skill.lower() in skills_lower)
            preferred_matched = sum(1 for skill in job_data['preferred_skills'] 
                                   if skill.lower() in skills_lower)
            
            total_required = len(job_data['required_skills'])
            total_preferred = len(job_data['preferred_skills'])
            
            # Calculate match score
            required_score = (required_matched / total_required) * 0.7
            preferred_score = (preferred_matched / total_preferred) * 0.3 if total_preferred > 0 else 0
            match_score = required_score + preferred_score
            
            # Only recommend if at least 50% required skills match
            if required_matched / total_required < 0.5:
                continue
            
            # Check industry fit
            industry_boost = 0.0
            top_industries = [ind[0] for ind in industries[:3]]
            if any(ind in top_industries for ind in job_data['industries']):
                industry_boost = 0.1
            
            match_score += industry_boost
            match_score = min(match_score, 1.0)  # Cap at 1.0
            
            # Generate reasons
            reasons = []
            if required_matched == total_required:
                reasons.append(f" All {total_required} required skills matched")
            else:
                reasons.append(f" {required_matched}/{total_required} required skills matched")
            
            if preferred_matched > 0:
                reasons.append(f" {preferred_matched}/{total_preferred} preferred skills matched")
            
            if industry_boost > 0:
                reasons.append(f" Strong industry fit: {', '.join(job_data['industries'])}")
            
            if experience_years >= job_data['min_experience']:
                reasons.append(f" Experience requirement met ({experience_years} years)")
            
            # Find missing skills
            missing_required = [skill for skill in job_data['required_skills'] 
                              if skill.lower() not in skills_lower]
            missing_preferred = [skill for skill in job_data['preferred_skills'] 
                               if skill.lower() not in skills_lower]
            
            recommendations.append(JobRecommendation(
                title=job_title,
                match_score=match_score,
                reasons=reasons,
                required_skills=job_data['required_skills'],
                missing_skills=missing_required + missing_preferred[:3],
                salary_range=job_data['salary_range'],
                growth_potential=job_data['growth_potential']
            ))
        
        # Sort by match score
        recommendations.sort(key=lambda x: x.match_score, reverse=True)
        return recommendations[:5]  # Top 5 jobs
    
    def _recommend_certifications(self, skills: List[str], industries: List[Tuple[str, float]], 
                                  recommended_jobs: List[JobRecommendation]) -> List[CertificationRecommendation]:
        """Recommend certifications with XAI reasoning"""
        recommendations = []
        skills_lower = [s.lower() for s in skills]
        top_industries = [ind[0] for ind in industries[:3]]
        
        # Get missing skills from top jobs
        missing_skills = set()
        for job in recommended_jobs[:3]:
            missing_skills.update(job.missing_skills)
        
        for cert in self.cert_database:
            # Check if cert helps with missing skills
            helps_with_missing = any(skill in missing_skills for skill in cert['skills'])
            
            # Check industry relevance
            industry_relevant = any(ind in top_industries for ind in cert['industries'])
            
            # Check if builds on existing skills
            builds_on_existing = any(skill.lower() in skills_lower for skill in cert['skills'])
            
            # Calculate priority
            priority_score = 0
            if helps_with_missing:
                priority_score += 3
            if industry_relevant:
                priority_score += 2
            if builds_on_existing:
                priority_score += 1
            
            if priority_score < 2:
                continue  # Skip low-priority certs
            
            # Determine priority level
            if priority_score >= 5:
                priority = "Very High"
            elif priority_score >= 4:
                priority = "High"
            else:
                priority = "Medium"
            
            # Generate reasons
            reasons = []
            if helps_with_missing:
                missing_covered = [s for s in cert['skills'] if s in missing_skills]
                reasons.append(f" Covers missing skills: {', '.join(missing_covered)}")
            
            if industry_relevant:
                reasons.append(f" Relevant to your top industries: {', '.join(cert['industries'])}")
            
            if builds_on_existing:
                existing = [s for s in cert['skills'] if s.lower() in skills_lower]
                reasons.append(f" Builds on your existing skills: {', '.join(existing)}")
            
            reasons.append(f" {cert['career_impact']}")
            
            recommendations.append(CertificationRecommendation(
                name=cert['name'],
                provider=cert['provider'],
                priority=priority,
                duration=cert['duration'],
                cost=cert['cost'],
                reasons=reasons,
                skills_gained=cert['skills'],
                career_impact=cert['career_impact']
            ))
        
        # Sort by priority
        priority_order = {"Very High": 0, "High": 1, "Medium": 2, "Low": 3}
        recommendations.sort(key=lambda x: priority_order.get(x.priority, 4))
        return recommendations[:6]  # Top 6 certs
    
    def _create_learning_roadmap(self, skills: List[str], seniority: str, 
                                recommended_jobs: List[JobRecommendation]) -> LearningRoadmap:
        """Create step-by-step learning roadmap"""
        # Determine current and target level
        current_level = seniority if seniority != "Unknown" else "Junior"
        
        # Determine target based on top job
        if recommended_jobs:
            top_job = recommended_jobs[0]
            if "Senior" in top_job.title or "Lead" in top_job.title:
                target_level = "Senior"
                timeline = "12-18 months"
            elif "Mid" in top_job.title or "Engineer" in top_job.title:
                target_level = "Mid-Level"
                timeline = "6-12 months"
            else:
                target_level = "Junior"
                timeline = "3-6 months"
        else:
            target_level = "Mid-Level"
            timeline = "6-12 months"
        
        # Create phases based on missing skills
        phases = []
        
        # Phase 1: Foundation (missing critical skills)
        if recommended_jobs:
            critical_missing = recommended_jobs[0].missing_skills[:3]
            if critical_missing:
                phases.append({
                    "phase": "Foundation Building",
                    "duration": "1-2 months",
                    "priority": "Very High",
                    "skills": critical_missing,
                    "reason": f"These skills are required for {recommended_jobs[0].title}",
                    "resources": [
                        "Online courses (Coursera, Udemy)",
                        "Official documentation",
                        "Hands-on projects"
                    ]
                })
        
        # Phase 2: Practical Application
        phases.append({
            "phase": "Practical Application",
            "duration": "2-3 months",
            "priority": "High",
            "skills": ["Build real projects", "GitHub portfolio", "Open source contributions"],
            "reason": "Demonstrate skills to employers",
            "resources": [
                "Personal projects",
                "Contribute to open source",
                "Build portfolio website"
            ]
        })
        
        # Phase 3: Specialization
        if recommended_jobs and len(recommended_jobs[0].missing_skills) > 3:
            advanced_skills = recommended_jobs[0].missing_skills[3:6]
            if advanced_skills:
                phases.append({
                    "phase": "Specialization",
                    "duration": "2-4 months",
                    "priority": "Medium",
                    "skills": advanced_skills,
                    "reason": "Stand out in competitive job market",
                    "resources": [
                        "Advanced courses",
                        "Certifications",
                        "Industry conferences"
                    ]
                })
        
        # Generate reasoning
        reasoning = [
            f" Current Level: {current_level}",
            f" Target Level: {target_level}",
            f" Estimated Timeline: {timeline}",
            f" Top Target Role: {recommended_jobs[0].title if recommended_jobs else 'General Software Engineer'}"
        ]
        
        return LearningRoadmap(
            current_level=current_level,
            target_level=target_level,
            timeline=timeline,
            phases=phases,
            reasoning=reasoning
        )
    
    def _generate_xai_insights(self, cv_analysis: Dict, jobs: List[JobRecommendation], 
                             certs: List[CertificationRecommendation]) -> Dict[str, Any]:
        """Generate explainable AI insights"""
        insights = {
            "analysis_summary": {
                "total_skills": len(cv_analysis.get('skills', [])),
                "top_industries": [ind[0] for ind in cv_analysis.get('industries', [])[:3]],
                "seniority_confidence": cv_analysis.get('ml_confidence_breakdown', {}).get('seniority', 0.5),
                "skill_extraction_method": "Semantic + ML (paraphrase-mpnet-base-v2 + BERT-NER)"
            },
            "job_matching_logic": {
                "algorithm": "Multi-factor scoring: Required skills (70%) + Preferred skills (30%) + Industry fit (10%)",
                "threshold": "Minimum 50% required skills match",
                "top_match": {
                    "job": jobs[0].title if jobs else "None",
                    "score": f"{jobs[0].match_score*100:.1f}%" if jobs else "0%",
                    "reasons": jobs[0].reasons if jobs else []
                }
            },
            "certification_logic": {
                "algorithm": "Priority scoring: Missing skills (3pts) + Industry relevance (2pts) + Builds on existing (1pt)",
                "threshold": "Minimum 2 points for recommendation",
                "rationale": "Focus on filling skill gaps for target roles"
            },
            "roadmap_logic": {
                "phases": len(self._create_learning_roadmap(cv_analysis.get('skills', []), 
                                                           cv_analysis.get('seniority_level', 'Unknown'),
                                                           jobs).phases),
                "methodology": "Prioritize critical missing skills  Practical application  Specialization",
                "personalization": f"Tailored for {cv_analysis.get('seniority_level', 'Unknown')} level targeting {jobs[0].title if jobs else 'growth'}"
            },
            "confidence_scores": {
                "job_recommendations": "High" if jobs and jobs[0].match_score > 0.7 else "Medium",
                "certification_fit": "High" if certs and certs[0].priority in ["Very High", "High"] else "Medium",
                "roadmap_accuracy": "High - Based on industry standards and job requirements"
            },
            "ml_features_used": [
                "Semantic skill extraction (paraphrase-mpnet-base-v2)",
                "Industry classification (3-class confidence)",
                "Project detection (NER + pattern matching)",
                "Portfolio analysis (GitHub, LinkedIn)",
                "Seniority prediction (ML-based)"
            ]
        }
        
        return insights
    
    def to_json(self, guidance: CareerGuidance) -> Dict[str, Any]:
        """Convert CareerGuidance to JSON format"""
        return {
            "cv_analysis": {
                "skills": guidance.skills,
                "seniority": guidance.seniority,
                "industries": [{"name": ind[0], "confidence": ind[1]} for ind in guidance.industries],
                "projects": guidance.projects,
                "portfolio_links": guidance.portfolio_links,
                "experience_years": guidance.experience_years
            },
            "recommended_jobs": [
                {
                    "title": job.title,
                    "match_score": f"{job.match_score*100:.1f}%",
                    "salary_range": job.salary_range,
                    "growth_potential": job.growth_potential,
                    "reasons": job.reasons,
                    "required_skills": job.required_skills,
                    "missing_skills": job.missing_skills
                }
                for job in guidance.recommended_jobs
            ],
            "recommended_certifications": [
                {
                    "name": cert.name,
                    "provider": cert.provider,
                    "priority": cert.priority,
                    "duration": cert.duration,
                    "cost": cert.cost,
                    "reasons": cert.reasons,
                    "skills_gained": cert.skills_gained,
                    "career_impact": cert.career_impact
                }
                for cert in guidance.recommended_certifications
            ],
            "learning_roadmap": {
                "current_level": guidance.learning_roadmap.current_level,
                "target_level": guidance.learning_roadmap.target_level,
                "timeline": guidance.learning_roadmap.timeline,
                "phases": guidance.learning_roadmap.phases,
                "reasoning": guidance.learning_roadmap.reasoning
            },
            "xai_insights": guidance.xai_insights
        }


# Singleton instance
_career_guidance_engine = None

def get_career_guidance_engine() -> CareerGuidanceEngine:
    """Get singleton instance of career guidance engine"""
    global _career_guidance_engine
    if _career_guidance_engine is None:
        _career_guidance_engine = CareerGuidanceEngine()
    return _career_guidance_engine
