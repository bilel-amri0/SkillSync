"""
Job Matching Service - Integration with Adzuna API
Handles job search and matching based on CV analysis
"""

import os
import httpx
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class JobMatch:
    """Data class for job match results"""
    job_id: str
    title: str
    company: str
    location: str
    salary_min: Optional[float]
    salary_max: Optional[float]
    description: str
    url: str
    match_score: float
    matching_skills: List[str]
    created_date: str

class AdzunaJobService:
    """Service for Adzuna API integration"""
    
    def __init__(self):
        # Get API credentials from environment variables
        self.app_id = os.getenv("ADZUNA_APP_ID")
        self.app_key = os.getenv("ADZUNA_APP_KEY")
        self.base_url = "http://api.adzuna.com/v1/api/jobs"
        
        if not self.app_id or not self.app_key:
            logger.error("⚠️ ADZUNA API credentials not found in environment variables!")
            logger.error("Please set ADZUNA_APP_ID and ADZUNA_APP_KEY")
    
    async def search_jobs(self, 
                         skills: List[str], 
                         location: str = "fr", 
                         max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search jobs using Adzuna API
        
        Args:
            skills: List of skills from CV analysis
            location: Country code (default: fr for France)
            max_results: Maximum number of results
            
        Returns:
            List of job dictionaries
        """
        
        if not self.app_id or not self.app_key:
            logger.error("Cannot search jobs: Missing API credentials")
            return []
        
        try:
            # Construct search query from skills
            search_query = " OR ".join(skills[:5])  # Use top 5 skills
            
            # Adzuna API endpoint
            url = f"{self.base_url}/{location}/search/1"
            
            # API parameters
            params = {
                "app_id": self.app_id,
                "app_key": self.app_key,
                "results_per_page": min(max_results, 50),  # Adzuna max is 50
                "what": search_query,
                "content-type": "application/json"
            }
            
            logger.info(f"🔍 Searching jobs for skills: {skills[:5]}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                jobs = data.get("results", [])
                
                logger.info(f"✅ Found {len(jobs)} jobs from Adzuna API")
                return jobs
                
        except httpx.HTTPError as e:
            logger.error(f"❌ HTTP error calling Adzuna API: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return []
    
    def calculate_skill_match_score(self, job_description: str, user_skills: List[str]) -> tuple:
        """
        Calculate how well user skills match job description
        
        Returns:
            tuple: (match_score, matching_skills)
        """
        job_desc_lower = job_description.lower()
        matching_skills = []
        
        for skill in user_skills:
            if skill.lower() in job_desc_lower:
                matching_skills.append(skill)
        
        # Calculate score: (matching skills / total user skills) * 100
        match_score = (len(matching_skills) / len(user_skills)) * 100 if user_skills else 0
        
        return round(match_score, 2), matching_skills
    
    def format_job_results(self, jobs: List[Dict], user_skills: List[str]) -> List[JobMatch]:
        """
        Format raw Adzuna API results into JobMatch objects
        
        Args:
            jobs: Raw job data from Adzuna API
            user_skills: Skills extracted from user's CV
            
        Returns:
            List of formatted JobMatch objects
        """
        formatted_jobs = []
        
        for job in jobs:
            try:
                # Extract salary information
                salary_min = job.get("salary_min")
                salary_max = job.get("salary_max")
                
                # Calculate skill match
                description = job.get("description", "")
                match_score, matching_skills = self.calculate_skill_match_score(description, user_skills)
                
                # Create JobMatch object
                job_match = JobMatch(
                    job_id=job.get("id", ""),
                    title=job.get("title", "No title"),
                    company=job.get("company", {}).get("display_name", "Unknown company"),
                    location=job.get("location", {}).get("display_name", "Unknown location"),
                    salary_min=salary_min,
                    salary_max=salary_max,
                    description=description[:500] + "..." if len(description) > 500 else description,
                    url=job.get("redirect_url", ""),
                    match_score=match_score,
                    matching_skills=matching_skills,
                    created_date=job.get("created", datetime.now().isoformat())
                )
                
                formatted_jobs.append(job_match)
                
            except Exception as e:
                logger.error(f"Error formatting job: {e}")
                continue
        
        # Sort by match score (highest first)
        formatted_jobs.sort(key=lambda x: x.match_score, reverse=True)
        
        return formatted_jobs

# Global service instance
job_service = AdzunaJobService()

async def search_matching_jobs(user_skills: List[str], 
                              location: str = "fr", 
                              max_results: int = 20) -> List[JobMatch]:
    """
    Main function to search and match jobs for a user
    
    Args:
        user_skills: Skills extracted from CV analysis
        location: Country code for job search
        max_results: Maximum number of results to return
        
    Returns:
        List of JobMatch objects sorted by relevance
    """
    # Search jobs using Adzuna API
    raw_jobs = await job_service.search_jobs(user_skills, location, max_results)
    
    # Format and score the results
    job_matches = job_service.format_job_results(raw_jobs, user_skills)
    
    logger.info(f"🎯 Processed {len(job_matches)} job matches")
    
    return job_matches
