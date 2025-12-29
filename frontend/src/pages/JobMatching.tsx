import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search,
  MapPin,
  Clock,
  DollarSign,
  Building,
  AlertCircle,
  ExternalLink,
  Filter,
} from 'lucide-react';
import { jobApi, handleApiError } from '../services/api';
import toast from 'react-hot-toast';
import type { Job, SearchFilters } from '../types';

type SalaryRange = {
  min: number;
  max: number;
  currency: string;
};

interface RawJob {
  id?: string;
  job_id?: string;
  title?: string;
  company?: string;
  employer?: string;
  location?: string;
  remote?: boolean;
  description?: string;
  summary?: string;
  url?: string;
  application_url?: string;
  source?: string;
  salary?: string;
  salary_range?: SalaryRange;
  salary_min?: number | string;
  salary_max?: number | string;
  salary_currency?: string;
  skills_required?: string[];
  skills?: string[];
  requirements?: string[];
  skills_nice_to_have?: string[];
  employment_type?: string;
  job_type?: string;
  experience_level?: string;
  level?: string;
  posted_date?: string;
  date_posted?: string;
  [key: string]: unknown;
}

const stripHtml = (html: string): string => {
  if (!html) {
    return '';
  }

  let text = html.replace(/<[^>]*>/g, ' ');
  const entities: Record<string, string> = {
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&quot;': '"',
    '&#39;': "'",
    '&nbsp;': ' ',
    '&hellip;': '...',
    '&mdash;': '—',
    '&ndash;': '–',
  };

  Object.entries(entities).forEach(([entity, replacement]) => {
    text = text.replace(new RegExp(entity, 'g'), replacement);
  });

  return text.replace(/\s+/g, ' ').trim();
};

const normalizeJobResult = (job: RawJob, index: number): Job => {
  const skillsRequired = Array.isArray(job.skills_required)
    ? job.skills_required
    : Array.isArray(job.skills)
      ? job.skills
      : [];

  const requirements = Array.isArray(job.requirements)
    ? job.requirements
    : skillsRequired;

  const salaryRange = job.salary_range
    || (job.salary_min && job.salary_max
      ? {
          min: Number(job.salary_min),
          max: Number(job.salary_max),
          currency: job.salary_currency || 'USD',
        }
      : undefined);

  return {
    id: job.id || job.job_id || `job-${index}`,
    title: job.title || 'Untitled Role',
    company: job.company || job.employer || 'Confidential Company',
    location: job.location || (job.remote ? 'Remote' : 'Location not provided'),
    description: job.description || job.summary || 'No description provided.',
    url: job.url || job.application_url || '#',
    source: job.source || 'aggregator',
    salary: job.salary,
    salary_range: salaryRange,
    posted_date: job.posted_date || job.date_posted,
    remote: Boolean(job.remote),
    skills_required: skillsRequired,
    skills_nice_to_have: Array.isArray(job.skills_nice_to_have) ? job.skills_nice_to_have : [],
    requirements,
    employment_type: job.employment_type || job.job_type || 'full-time',
    experience_level: job.experience_level || job.level || 'mid',
    application_url: job.application_url || job.url,
  };
};

const formatSalary = (salary: SalaryRange) => {
  const formatter = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: salary.currency,
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  });

  return `${formatter.format(salary.min)} - ${formatter.format(salary.max)}`;
};

const formatDate = (dateString?: string) => {
  if (!dateString) {
    return 'Recently posted';
  }

  return new Date(dateString).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
};

const JOB_RESULTS_LIMIT = 60;

type JobSearchConfig = {
  query: string;
  location?: string;
  skills?: string[];
  remoteOnly: boolean;
};

const JobMatching = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState<SearchFilters>({});
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [remoteOnly, setRemoteOnly] = useState(false);
  const [searchConfig, setSearchConfig] = useState<JobSearchConfig | null>(null);
  const [initialFetchTriggered, setInitialFetchTriggered] = useState(false);

  const {
    data: jobSearch,
    isLoading: jobsLoading,
    error: jobsError,
  } = useQuery({
    queryKey: ['jobs', searchConfig?.query, searchConfig?.location, searchConfig?.remoteOnly],
    queryFn: async () => {
      if (!searchConfig) {
        return null;
      }

      const payload = await jobApi.search({
        query: searchConfig.query,
        location: searchConfig.remoteOnly ? 'Remote' : searchConfig.location,
        skills: searchConfig.skills,
        max_results: JOB_RESULTS_LIMIT,
      });

      const normalizedJobs = (payload.jobs ?? []).map((job, index) =>
        normalizeJobResult(job, index)
      );

      return {
        ...payload,
        jobs: normalizedJobs,
      };
    },
    retry: 0,
    enabled: Boolean(searchConfig),
    staleTime: 60_000,
  });

  const jobResults = jobSearch?.jobs ?? [];
  const jobs = useMemo(() => jobResults, [jobResults]);
  const totalResults = jobSearch?.total_count ?? jobs.length;
  const sourcesUsed = jobSearch?.sources_used ?? [];
  const searchTimeMs = jobSearch?.search_time_ms ?? 0;

  const filteredJobs = useMemo(() => {
    return jobs.filter((job) => {
      const matchesExperience = filters.experience_level
        ? (job.experience_level || '').toLowerCase().includes(filters.experience_level!.toLowerCase())
        : true;
      const matchesEmployment = filters.employment_type
        ? (job.employment_type || '').toLowerCase().includes(filters.employment_type!.toLowerCase())
        : true;
      return matchesExperience && matchesEmployment;
    });
  }, [jobs, filters.experience_level, filters.employment_type]);

  useEffect(() => {
    if (jobsError) {
      toast.error(handleApiError(jobsError));
    }
  }, [jobsError]);

  const selectedJob = useMemo(() => {
    if (filteredJobs.length === 0) {
      return null;
    }

    if (selectedJobId) {
      const match = filteredJobs.find(job => job.id === selectedJobId);
      if (match) {
        return match;
      }
    }

    return filteredJobs[0];
  }, [filteredJobs, selectedJobId]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedQuery = searchQuery.trim();
    setSelectedJobId(null);
    setSearchConfig({
      query: trimmedQuery,
      location: filters.location?.trim() || undefined,
      skills: filters.skills,
      remoteOnly,
    });
    toast.success('Fetching fresh listings...');
  };

  useEffect(() => {
    if (initialFetchTriggered || searchConfig) {
      return;
    }
    setSearchConfig({
      query: '',
      location: undefined,
      skills: undefined,
      remoteOnly: false,
    });
    setInitialFetchTriggered(true);
  }, [initialFetchTriggered, searchConfig]);

  useEffect(() => {
    setSearchConfig((previous) => {
      if (!previous || previous.remoteOnly === remoteOnly) {
        return previous;
      }
      return { ...previous, remoteOnly };
    });
  }, [remoteOnly]);

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Job Matching</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Choose your keywords, toggle remote mode if needed, and pull live roles whenever you hit search.
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700"
      >
        <form onSubmit={handleSearch} className="space-y-4">
          <div className="flex flex-wrap gap-4">
            <div className="flex-1 min-w-[200px] relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search jobs by title, company, or keywords..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white"
              />
            </div>
            <button
              type="button"
              onClick={() => setRemoteOnly((prev) => !prev)}
              className={`px-4 py-2 rounded-lg transition-colors flex items-center border ${
                remoteOnly
                  ? 'bg-green-600 text-white border-green-600 hover:bg-green-700'
                  : 'border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              <MapPin className="h-4 w-4 mr-2" />
              Remote roles only
            </button>
            <button
              type="button"
              onClick={() => setShowFilters(!showFilters)}
              className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors flex items-center"
            >
              <Filter className="h-4 w-4 mr-2" />
              Filters
            </button>
            <button
              type="submit"
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Search
            </button>
          </div>

          <AnimatePresence>
            {showFilters && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4 border-t border-gray-200 dark:border-gray-700"
              >
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Location
                  </label>
                  <input
                    type="text"
                    placeholder="City, State"
                    value={filters.location || ''}
                    onChange={(e) => setFilters((prev) => ({ ...prev, location: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Experience Level
                  </label>
                  <select
                    value={filters.experience_level || ''}
                    onChange={(e) => setFilters((prev) => ({ ...prev, experience_level: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  >
                    <option value="">All Levels</option>
                    <option value="entry">Entry Level</option>
                    <option value="mid">Mid Level</option>
                    <option value="senior">Senior Level</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Employment Type
                  </label>
                  <select
                    value={filters.employment_type || ''}
                    onChange={(e) => setFilters((prev) => ({ ...prev, employment_type: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  >
                    <option value="">All Types</option>
                    <option value="full-time">Full-time</option>
                    <option value="part-time">Part-time</option>
                    <option value="contract">Contract</option>
                  </select>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </form>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-4"
        >
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Job Listings</h2>
          {searchConfig && jobSearch && !jobsLoading && (
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Showing {filteredJobs.length} live roles · Sources: {sourcesUsed.length > 0 ? sourcesUsed.join(', ') : '—'}
            </p>
          )}

          {!searchConfig && (
            <div className="rounded-lg border border-dashed border-blue-200 dark:border-blue-900/40 bg-blue-50/60 dark:bg-slate-900/30 p-6 text-center text-sm text-blue-800 dark:text-blue-200">
              Start by entering a role (for example “software engineer”, “data analyst”, or “product manager”), choose whether you want remote roles, then click Search.
            </div>
          )}

          {jobsLoading ? (
            <div className="space-y-4">
              {[...Array(3)].map((_, i) => (
                <div
                  key={i}
                  className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 animate-pulse"
                >
                  <div className="h-4 bg-gray-200 rounded w-3/4 mb-2" />
                  <div className="h-3 bg-gray-200 rounded w-1/2 mb-4" />
                  <div className="space-y-2">
                    <div className="h-3 bg-gray-200 rounded" />
                    <div className="h-3 bg-gray-200 rounded w-2/3" />
                  </div>
                </div>
              ))}
            </div>
          ) : filteredJobs.length > 0 ? (
            <div className="space-y-4">
              {filteredJobs.map((job, index) => {
                const primarySkills = job.skills_required ?? [];

                return (
                  <motion.div
                    key={job.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 + index * 0.05 }}
                    className={`bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border cursor-pointer transition-all hover:shadow-xl ${
                      selectedJob?.id === job.id
                        ? 'border-blue-500 ring-2 ring-blue-200 dark:ring-blue-800'
                        : 'border-gray-200 dark:border-gray-700'
                    }`}
                    onClick={() => setSelectedJobId(job.id)}
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">{job.title}</h3>
                        <div className="flex flex-wrap items-center gap-3 text-sm text-gray-600 dark:text-gray-400">
                          <span className="flex items-center gap-1">
                            <Building className="h-4 w-4" /> {job.company}
                          </span>
                          <span className="flex items-center gap-1">
                            <MapPin className="h-4 w-4" /> {job.location}
                          </span>
                          <span className="flex items-center gap-1">
                            <Clock className="h-4 w-4" /> {formatDate(job.posted_date)}
                          </span>
                        </div>
                      </div>
                      {(job.salary_range || job.salary) && (
                        <div className="text-right">
                          <div className="flex items-center justify-end text-green-600 dark:text-green-400">
                            <DollarSign className="h-4 w-4 mr-1" />
                            <span className="font-medium">
                              {job.salary_range ? formatSalary(job.salary_range) : job.salary}
                            </span>
                          </div>
                          {job.salary_range && (
                            <span className="text-xs text-gray-500 dark:text-gray-400">per year</span>
                          )}
                        </div>
                      )}
                    </div>

                    <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2 mb-4">
                      {stripHtml(job.description)}
                    </p>

                    <div className="flex flex-wrap gap-2 mb-4">
                      {primarySkills.slice(0, 4).map((skill, skillIndex) => (
                        <span
                          key={skillIndex}
                          className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs rounded-full"
                        >
                          {skill}
                        </span>
                      ))}
                      {primarySkills.length > 4 && (
                        <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 text-xs rounded-full">
                          +{primarySkills.length - 4} more
                        </span>
                      )}
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-xs rounded-full">
                          {job.experience_level || 'Experience N/A'}
                        </span>
                        <span className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-xs rounded-full">
                          {job.employment_type || 'Type N/A'}
                        </span>
                      </div>
                      {job.application_url && (
                        <ExternalLink className="h-4 w-4 text-gray-400" />
                      )}
                    </div>
                  </motion.div>
                );
              })}
            </div>
          ) : (
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg border border-dashed border-gray-300 dark:border-gray-700 text-center">
              <AlertCircle className="h-10 w-10 text-gray-400 mx-auto mb-3" />
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                {searchConfig ? 'No jobs match your filters' : 'Ready when you are'}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                {searchConfig
                  ? 'Adjust your keywords or relax a filter, then run the search again.'
                  : 'Enter a search query to start pulling live listings.'}
              </p>
            </div>
          )}
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="space-y-4"
        >
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Job Details & Insights</h2>

          {selectedJob ? (
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 space-y-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h3 className="text-2xl font-semibold text-gray-900 dark:text-white">{selectedJob.title}</h3>
                  <p className="text-gray-600 dark:text-gray-400 flex items-center gap-2 mt-1">
                    <Building className="h-4 w-4" /> {selectedJob.company}
                  </p>
                  <p className="text-gray-600 dark:text-gray-400 flex items-center gap-2 mt-1">
                    <MapPin className="h-4 w-4" /> {selectedJob.location}
                    {selectedJob.remote && (
                      <span className="ml-2 px-2 py-0.5 text-xs rounded-full bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300">
                        Remote friendly
                      </span>
                    )}
                  </p>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                    Posted {formatDate(selectedJob.posted_date)} • Source: {selectedJob.source}
                  </p>
                </div>
                {(selectedJob.salary_range || selectedJob.salary) && (
                  <div className="text-right">
                    <div className="flex items-center justify-end text-green-600 dark:text-green-400">
                      <DollarSign className="h-4 w-4 mr-1" />
                      <span className="font-semibold">
                        {selectedJob.salary_range ? formatSalary(selectedJob.salary_range) : selectedJob.salary}
                      </span>
                    </div>
                    <span className="text-xs text-gray-500 dark:text-gray-400">Compensation</span>
                  </div>
                )}
              </div>

              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Key Skills</h4>
                <div className="flex flex-wrap gap-2">
                  {(selectedJob.skills_required ?? []).slice(0, 6).map((skill, index) => (
                    <span
                      key={index}
                      className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs rounded-full"
                    >
                      {skill}
                    </span>
                  ))}
                  {(!selectedJob.skills_required || selectedJob.skills_required.length === 0) && (
                    <span className="text-sm text-gray-500 dark:text-gray-400">No skill data from source.</span>
                  )}
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Job Overview</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400 whitespace-pre-line">
                  {stripHtml(selectedJob.description)}
                </p>
              </div>

              <div>
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Requirements</h4>
                {selectedJob.requirements && selectedJob.requirements.length > 0 ? (
                  <ul className="list-disc list-inside text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    {selectedJob.requirements.slice(0, 6).map((req, index) => (
                      <li key={index}>{req}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-sm text-gray-500 dark:text-gray-400">The source did not provide a requirements list.</p>
                )}
              </div>

              <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-gray-100 dark:border-gray-700">
                <span className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-xs rounded-full">
                  {selectedJob.experience_level || 'Experience unspecified'}
                </span>
                <span className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-xs rounded-full">
                  {selectedJob.employment_type || 'Employment type N/A'}
                </span>
                <a
                  href={selectedJob.application_url || selectedJob.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="ml-auto inline-flex items-center text-blue-600 dark:text-blue-400 font-medium hover:underline"
                >
                  View posting
                  <ExternalLink className="h-4 w-4 ml-1" />
                </a>
              </div>
            </div>
          ) : (
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 text-center">
              <AlertCircle className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">Select a job to view details</h3>
              <p className="text-gray-600 dark:text-gray-400">
                Choose a job from the list to see its complete description, skills, and requirements.
              </p>
            </div>
          )}

          {jobSearch && (
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Search Insights</h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Total results</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">{totalResults}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Search time</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">{searchTimeMs} ms</p>
                </div>
                <div>
                  <p className="text-sm text-gray-500 dark:text-gray-400">Sources used</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">{sourcesUsed.length}</p>
                </div>
              </div>
              {sourcesUsed.length > 0 && (
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-3">{sourcesUsed.join(', ')}</p>
              )}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
};

export default JobMatching;
