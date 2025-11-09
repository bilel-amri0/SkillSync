import asyncio
import aiohttp
import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from urllib.parse import quote
from dotenv import load_dotenv
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class JobResult:
    """Standardized job result format"""
    id: str
    title: str
    company: str
    location: str
    description: str
    url: str
    salary: Optional[str]
    posted_date: Optional[str]
    source: str
    skills_match: float = 0.0
    remote: bool = False

class MultiJobAPIService:
    """Asynchronous multi-API job search service"""
    
    def __init__(self):
        self.timeout = int(os.getenv('API_TIMEOUT', 30))
        self.max_jobs_per_source = int(os.getenv('MAX_JOBS_PER_SOURCE', 20))
        self.debug = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        
        # API configurations with priority order
        self.api_configs = {
            'linkedin': {
                'priority': 1,
                'enabled': bool(os.getenv('LINKEDIN_RAPIDAPI_KEY')),
                'headers': {
                    'X-RapidAPI-Key': os.getenv('LINKEDIN_RAPIDAPI_KEY'),
                    'X-RapidAPI-Host': os.getenv('LINKEDIN_RAPIDAPI_HOST', 'linkedin-data-api.p.rapidapi.com')
                }
            },
            'jsearch': {
                'priority': 2,
                'enabled': bool(os.getenv('JSEARCH_RAPIDAPI_KEY')),
                'headers': {
                    'X-RapidAPI-Key': os.getenv('JSEARCH_RAPIDAPI_KEY'),
                    'X-RapidAPI-Host': os.getenv('JSEARCH_RAPIDAPI_HOST', 'jsearch.p.rapidapi.com')
                }
            },
            'themuse': {
                'priority': 3,
                'enabled': bool(os.getenv('MUSE_API_KEY')),
                'api_key': os.getenv('MUSE_API_KEY')
            },
            'findwork': {
                'priority': 4,
                'enabled': bool(os.getenv('FINDWORK_API_KEY')),
                'headers': {
                    'Authorization': f"Token {os.getenv('FINDWORK_API_KEY')}"
                }
            },
            'adzuna': {
                'priority': 5,
                'enabled': bool(os.getenv('ADZUNA_APP_ID') and os.getenv('ADZUNA_APP_KEY')),
                'app_id': os.getenv('ADZUNA_APP_ID'),
                'app_key': os.getenv('ADZUNA_APP_KEY')
            },
            'arbeitnow': {
                'priority': 6,
                'enabled': True  # Always enabled (free)
            },
            'jobicy': {
                'priority': 7,
                'enabled': True  # Always enabled (free)
            },
            'remoteok': {
                'priority': 8,
                'enabled': True  # Always enabled (free, no API key required)
            }
        }
        
        logger.info(f"🚀 MultiJobAPIService initialized with {len([k for k, v in self.api_configs.items() if v['enabled']])} enabled APIs")
    
    async def search_jobs(self, query: str, location: str = "", skills: List[str] = None) -> List[JobResult]:
        """Search jobs across all enabled APIs"""
        if skills is None:
            skills = []
        
        logger.info(f"🔍 Starting multi-API job search for: '{query}' in '{location}'")
        
        # Create tasks for all enabled APIs
        tasks = []
        enabled_apis = [(name, config) for name, config in self.api_configs.items() if config['enabled']]
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            for api_name, config in enabled_apis:
                task = asyncio.create_task(
                    self._search_single_api(session, api_name, query, location, skills),
                    name=f"search_{api_name}"
                )
                tasks.append(task)
            
            # Wait for all tasks with timeout
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        all_jobs = []
        for i, result in enumerate(results):
            api_name = enabled_apis[i][0]
            if isinstance(result, Exception):
                logger.error(f"❌ {api_name} failed: {result}")
            elif isinstance(result, list):
                all_jobs.extend(result)
                logger.info(f"✅ {api_name}: {len(result)} jobs")
        
        # Remove duplicates and sort by priority/relevance
        unique_jobs = self._deduplicate_jobs(all_jobs)
        sorted_jobs = self._sort_jobs_by_priority(unique_jobs)
        
        logger.info(f"📊 Total unique jobs found: {len(sorted_jobs)}")
        return sorted_jobs[:100]  # Limit to 100 jobs
    
    async def _search_single_api(self, session: aiohttp.ClientSession, api_name: str, 
                               query: str, location: str, skills: List[str]) -> List[JobResult]:
        """Search jobs from a single API"""
        try:
            if api_name == 'linkedin':
                return await self._search_linkedin(session, query, location)
            elif api_name == 'jsearch':
                return await self._search_jsearch(session, query, location)
            elif api_name == 'themuse':
                return await self._search_themuse(session, query, location)
            elif api_name == 'findwork':
                return await self._search_findwork(session, query, location)
            elif api_name == 'adzuna':
                return await self._search_adzuna(session, query, location)
            elif api_name == 'arbeitnow':
                return await self._search_arbeitnow(session, query)
            elif api_name == 'jobicy':
                return await self._search_jobicy(session, query)
            elif api_name == 'remoteok':
                return await self._search_remoteok(session, query)
            else:
                return []
        except Exception as e:
            logger.error(f"🚨 Error searching {api_name}: {e}")
            return []
    
    async def _search_linkedin(self, session: aiohttp.ClientSession, query: str, location: str) -> List[JobResult]:
        """Search LinkedIn Jobs via RapidAPI"""
        url = "https://linkedin-data-api.p.rapidapi.com/search-jobs"
        params = {
            "keywords": query,
            "locationId": "103644278",  # Worldwide
            "datePosted": "anyTime",
            "sort": "mostRelevant"
        }
        if location:
            params["location"] = location
        
        headers = self.api_configs['linkedin']['headers']
        
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                jobs = []
                
                # Ensure data is valid and contains 'data' key
                if not data or not isinstance(data, dict):
                    logger.warning("LinkedIn API returned invalid data format")
                    return []
                
                job_list = data.get('data')
                if job_list is None:
                    logger.info("LinkedIn API returned no jobs (data is None)")
                    return []
                
                if not isinstance(job_list, list):
                    logger.warning(f"LinkedIn API 'data' field is not a list, got: {type(job_list)}")
                    return []
                
                for job in job_list[:self.max_jobs_per_source]:
                    # Safe extraction for company name
                    company_data = job.get('company')
                    company_name = ''
                    if isinstance(company_data, dict):
                        company_name = company_data.get('name', '')
                    elif isinstance(company_data, str):
                        company_name = company_data
                    
                    # Safe description handling
                    description = job.get('description') or ''
                    description_preview = (description[:500] + '...') if description else ''
                    
                    jobs.append(JobResult(
                        id=f"linkedin_{job.get('id', '')}",
                        title=job.get('title', ''),
                        company=company_name,
                        location=job.get('location', ''),
                        description=description_preview,
                        url=job.get('url', ''),
                        salary=None,
                        posted_date=job.get('postedAt', ''),
                        source='LinkedIn',
                        remote='remote' in job.get('title', '').lower()
                    ))
                
                return jobs
            else:
                logger.warning(f"LinkedIn API returned status {response.status}")
                return []
    
    async def _search_jsearch(self, session: aiohttp.ClientSession, query: str, location: str) -> List[JobResult]:
        """Search jobs via JSearch RapidAPI"""
        url = "https://jsearch.p.rapidapi.com/search"
        
        # Build search query - prioritize job keywords
        search_terms = []
        if query and query.strip():
            search_terms.append(query.strip())
        if location and location.strip():
            search_terms.append(f"in {location.strip()}")
            
        search_query = " ".join(search_terms) if search_terms else "software developer"
        
        params = {
            "query": search_query,
            "page": "1",
            "num_pages": "1",
            "date_posted": "today"
        }
        
        headers = self.api_configs['jsearch']['headers']
        
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                jobs = []
                
                job_list = data.get('data', [])
                logger.info(f"🔍 JSearch returned {len(job_list)} total jobs")
                
                for job in job_list[:self.max_jobs_per_source]:
                    # Safe string handling for location
                    job_city = job.get('job_city') or ''
                    job_country = job.get('job_country') or ''
                    job_state = job.get('job_state') or ''
                    
                    location_parts = [p for p in [job_city, job_state, job_country] if p]
                    location_str = ", ".join(location_parts) if location_parts else 'Not specified'
                    
                    # Safe string handling for description
                    description = job.get('job_description') or job.get('job_highlights', {}).get('Qualifications', [''])[0] or ''
                    description_preview = (description[:500] + '...') if len(description) > 500 else description
                    
                    # Extract salary information
                    salary_min = job.get('job_min_salary')
                    salary_max = job.get('job_max_salary')
                    salary_str = None
                    if salary_min and salary_max:
                        salary_str = f"${salary_min:,} - ${salary_max:,}"
                    elif salary_min:
                        salary_str = f"${salary_min:,}+"
                    
                    jobs.append(JobResult(
                        id=f"jsearch_{job.get('job_id', '')}_jsearch",
                        title=job.get('job_title', ''),
                        company=job.get('employer_name', ''),
                        location=location_str,
                        description=description_preview,
                        url=job.get('job_apply_link', ''),
                        salary=salary_str,
                        posted_date=job.get('job_posted_at_datetime_utc', ''),
                        source='JSearch',
                        remote=job.get('job_is_remote', False)
                    ))
                
                logger.info(f"🎯 JSearch matched {len(jobs)} jobs for query: '{search_query}'")
                return jobs
            else:
                error_msg = await response.text()
                logger.warning(f"JSearch API returned status {response.status}: {error_msg}")
                return []
    
    async def _search_themuse(self, session: aiohttp.ClientSession, query: str, location: str) -> List[JobResult]:
        """Search The Muse jobs"""
        url = "https://www.themuse.com/api/public/jobs"
        params = {
            "category": "Computer and IT",
            "level": "Entry Level,Mid Level,Senior Level",
            "page": 0
        }
        if location:
            params["location"] = location
        
        headers = {"api_key": self.api_configs['themuse']['api_key']}
        
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                jobs = []
                
                for job in data.get('results', [])[:self.max_jobs_per_source]:
                    jobs.append(JobResult(
                        id=f"muse_{job.get('id', '')}",
                        title=job.get('name', ''),
                        company=job.get('company', {}).get('name', ''),
                        location=', '.join([loc.get('name', '') for loc in job.get('locations', [])]),
                        description=job.get('contents', '')[:500] + '...',
                        url=job.get('refs', {}).get('landing_page', ''),
                        salary=None,
                        posted_date=job.get('publication_date', ''),
                        source='The Muse',
                        remote='remote' in job.get('name', '').lower()
                    ))
                
                return jobs
            else:
                logger.warning(f"The Muse API returned status {response.status}")
                return []
    
    async def _search_findwork(self, session: aiohttp.ClientSession, query: str, location: str) -> List[JobResult]:
        """Search FindWork.dev jobs"""
        url = "https://findwork.dev/api/jobs/"
        params = {}
        
        # FindWork API expects specific parameter names
        if query and query.strip():
            params["role"] = query  # Changed from "search" to "role"
        
        if location and location.strip():
            params["location"] = location
            
        # Use correct ordering parameter
        params["ordering"] = "-date_posted"  # Changed from "order_by" to "ordering"
        
        headers = self.api_configs['findwork']['headers']
        
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                jobs = []
                
                for job in data.get('results', [])[:self.max_jobs_per_source]:
                    jobs.append(JobResult(
                        id=f"findwork_{job.get('id', '')}",
                        title=job.get('role', ''),
                        company=job.get('company_name', ''),
                        location=job.get('location', ''),
                        description=job.get('text', '')[:500] + '...',
                        url=job.get('url', ''),
                        salary=None,
                        posted_date=job.get('date_posted', ''),
                        source='FindWork',
                        remote=job.get('remote', False)
                    ))
                
                return jobs
            else:
                logger.warning(f"FindWork API returned status {response.status}")
                return []
    
    async def _search_adzuna(self, session: aiohttp.ClientSession, query: str, location: str) -> List[JobResult]:
        """Search Adzuna jobs"""
        # Choose country based on location or default to US
        country = "fr" if "fr" in location.lower() else "us"
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/1"
        
        params = {
            "app_id": self.api_configs['adzuna']['app_id'],
            "app_key": self.api_configs['adzuna']['app_key'],
            "what": query or "developer",
            "results_per_page": self.max_jobs_per_source,
            "sort_by": "relevance"
        }
        
        if location and location.strip():
            params["where"] = location
        
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                jobs = []
                
                job_list = data.get('results', [])
                logger.info(f"🔍 Adzuna returned {len(job_list)} total jobs")
                
                for job in job_list:
                    # Safe extraction of nested data
                    company_data = job.get('company', {})
                    company_name = company_data.get('display_name', '') if isinstance(company_data, dict) else str(company_data)
                    
                    location_data = job.get('location', {})
                    location_name = location_data.get('display_name', '') if isinstance(location_data, dict) else str(location_data)
                    
                    # Clean description
                    description = job.get('description', '') or ''
                    description_preview = (description[:500] + '...') if len(description) > 500 else description
                    
                    # Handle salary
                    salary_min = job.get('salary_min')
                    salary_max = job.get('salary_max')
                    salary_str = None
                    if salary_min and salary_max:
                        salary_str = f"${salary_min:,.0f} - ${salary_max:,.0f}"
                    elif salary_min:
                        salary_str = f"${salary_min:,.0f}+"
                    
                    jobs.append(JobResult(
                        id=f"adzuna_{job.get('id', '')}_adzuna",
                        title=job.get('title', ''),
                        company=company_name,
                        location=location_name,
                        description=description_preview,
                        url=job.get('redirect_url', ''),
                        salary=salary_str,
                        posted_date=job.get('created', ''),
                        source='Adzuna',
                        remote='remote' in job.get('title', '').lower() or 'remote' in description.lower()
                    ))
                
                logger.info(f"🎯 Adzuna matched {len(jobs)} jobs for query: '{query}'")
                return jobs
            else:
                error_msg = await response.text()
                logger.warning(f"Adzuna API returned status {response.status}: {error_msg}")
                return []
    
    async def _search_arbeitnow(self, session: aiohttp.ClientSession, query: str) -> List[JobResult]:
        """Search Arbeitnow jobs (free API)"""
        url = "https://www.arbeitnow.com/api/job-board-api"
        
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                jobs = []
                
                # Get all jobs first, then filter (less restrictive)
                all_jobs = data.get('data', [])
                logger.info(f"🔍 Arbeitnow returned {len(all_jobs)} total jobs")
                
                query_keywords = query.lower().split() if query else []
                
                for job in all_jobs[:self.max_jobs_per_source]:
                    # More flexible keyword matching - if no query, return all jobs
                    job_text = f"{job.get('title', '')} {job.get('description', '')} {job.get('company_name', '')}".lower()
                    
                    if not query_keywords or any(keyword in job_text for keyword in query_keywords):
                        # Clean description
                        description = job.get('description', '') or job.get('summary', '') or ''
                        description_preview = (description[:500] + '...') if len(description) > 500 else description
                        
                        jobs.append(JobResult(
                            id=f"arbeit_{job.get('slug', job.get('id', ''))}_arbeitnow",
                            title=job.get('title', ''),
                            company=job.get('company_name', ''),
                            location=job.get('location', ''),
                            description=description_preview,
                            url=job.get('url', ''),
                            salary=None,
                            posted_date=job.get('created_at', ''),
                            source='Arbeitnow',
                            remote=job.get('remote', False)
                        ))
                
                logger.info(f"🎯 Arbeitnow matched {len(jobs)} jobs for query: '{query}'")
                return jobs
            else:
                logger.warning(f"Arbeitnow API returned status {response.status}")
                return []
    
    async def _search_jobicy(self, session: aiohttp.ClientSession, query: str) -> List[JobResult]:
        """Search Jobicy jobs (free API via GitHub)"""
        # Use the GitHub API endpoint which is more reliable
        url = "https://raw.githubusercontent.com/Jobicy/remote-jobs-api/main/jobs.json"
        
        async with session.get(url) as response:
            if response.status == 200:
                try:
                    data = await response.json()
                    jobs = []
                    
                    # Handle different response formats
                    job_list = data if isinstance(data, list) else data.get('jobs', [])
                    logger.info(f"🔍 Jobicy returned {len(job_list)} total jobs")
                    
                    query_keywords = query.lower().split() if query else []
                    
                    for job in job_list[:self.max_jobs_per_source]:
                        # More flexible keyword matching
                        job_text = f"{job.get('jobTitle', '')} {job.get('jobExcerpt', '')} {job.get('companyName', '')}".lower()
                        
                        if not query_keywords or any(keyword in job_text for keyword in query_keywords):
                            # Clean description
                            description = job.get('jobExcerpt', '') or job.get('jobDescription', '') or ''
                            description_preview = (description[:500] + '...') if len(description) > 500 else description
                            
                            jobs.append(JobResult(
                                id=f"jobicy_{job.get('id', job.get('jobSlug', ''))}_jobicy",
                                title=job.get('jobTitle', ''),
                                company=job.get('companyName', ''),
                                location="Remote",
                                description=description_preview,
                                url=job.get('url', job.get('jobUrl', '')),
                                salary=None,
                                posted_date=job.get('pubDate', job.get('publishedAt', '')),
                                source='Jobicy',
                                remote=True
                            ))
                    
                    logger.info(f"🎯 Jobicy matched {len(jobs)} jobs for query: '{query}'")
                    return jobs
                except Exception as e:
                    logger.error(f"🚨 Error parsing Jobicy response: {e}")
                    return []
            else:
                logger.warning(f"Jobicy API returned status {response.status}")
                return []
    
    async def _search_remoteok(self, session: aiohttp.ClientSession, query: str) -> List[JobResult]:
        """Search RemoteOK jobs (free API)"""
        url = "https://remoteok.io/api"
        
        async with session.get(url) as response:
            if response.status == 200:
                try:
                    data = await response.json()
                    jobs = []
                    
                    # RemoteOK returns an array, first item is usually metadata
                    job_list = data[1:] if isinstance(data, list) and len(data) > 1 else []
                    logger.info(f"🔍 RemoteOK returned {len(job_list)} total jobs")
                    
                    query_keywords = query.lower().split() if query else []
                    
                    for job in job_list[:self.max_jobs_per_source]:
                        if not isinstance(job, dict):
                            continue
                            
                        # More flexible keyword matching
                        job_text = f"{job.get('position', '')} {job.get('description', '')} {job.get('company', '')}".lower()
                        
                        if not query_keywords or any(keyword in job_text for keyword in query_keywords):
                            # Clean description
                            description = job.get('description', '') or ''
                            description_preview = (description[:500] + '...') if len(description) > 500 else description
                            
                            # Handle salary
                            salary_min = job.get('salary_min')
                            salary_max = job.get('salary_max')
                            salary_str = None
                            if salary_min and salary_max:
                                salary_str = f"${salary_min:,} - ${salary_max:,}"
                            elif salary_min:
                                salary_str = f"${salary_min:,}+"
                            
                            jobs.append(JobResult(
                                id=f"remoteok_{job.get('id', job.get('slug', ''))}_remoteok",
                                title=job.get('position', ''),
                                company=job.get('company', ''),
                                location="Remote",
                                description=description_preview,
                                url=job.get('url', f"https://remoteok.io/remote-jobs/{job.get('id', '')}"),
                                salary=salary_str,
                                posted_date=job.get('date', ''),
                                source='RemoteOK',
                                remote=True
                            ))
                    
                    logger.info(f"🎯 RemoteOK matched {len(jobs)} jobs for query: '{query}'")
                    return jobs
                except Exception as e:
                    logger.error(f"🚨 Error parsing RemoteOK response: {e}")
                    return []
            else:
                logger.warning(f"RemoteOK API returned status {response.status}")
                return []

    def _deduplicate_jobs(self, jobs: List[JobResult]) -> List[JobResult]:
        """Remove duplicate jobs based on title and company"""
        seen = set()
        unique_jobs = []
        
        for job in jobs:
            job_key = (job.title.lower().strip(), job.company.lower().strip())
            if job_key not in seen:
                seen.add(job_key)
                unique_jobs.append(job)
        
        return unique_jobs
    
    def _sort_jobs_by_priority(self, jobs: List[JobResult]) -> List[JobResult]:
        """Sort jobs by source priority and relevance"""
        priority_map = {config['priority']: name for name, config in self.api_configs.items()}
        source_priority = {v: k for k, v in priority_map.items()}
        
        return sorted(jobs, key=lambda job: (
            source_priority.get(job.source, 999),  # Source priority
            -len(job.description),  # Description length (more detailed first)
            job.title.lower()  # Alphabetical by title
        ))
    
    async def get_api_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all configured APIs"""
        status = {}
        
        for api_name, config in self.api_configs.items():
            status[api_name] = {
                'enabled': config['enabled'],
                'priority': config['priority'],
                'status': 'enabled' if config['enabled'] else 'disabled (missing keys)'
            }
        
        return status

# Singleton instance
_service_instance = None

def get_job_service() -> MultiJobAPIService:
    """Get singleton instance of the job service"""
    global _service_instance
    if _service_instance is None:
        _service_instance = MultiJobAPIService()
    return _service_instance