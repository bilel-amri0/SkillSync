// Shared SkillSync Types aligned with backend responses

export interface Job {
  id: string;
  title: string;
  company: string;
  location: string;
  description: string;
  url: string;
  source: string;
  salary?: string;
  posted_date?: string;
  skills_match?: number;
  remote?: boolean;
  match_score?: number;
  employment_type?: string;
  experience_level?: string;
  salary_range?: {
    min: number;
    max: number;
    currency: string;
  };
  requirements?: string[];
  skills_required?: string[];
  skills?: string[];
  skills_nice_to_have?: string[];
  application_url?: string;
}

export interface JobMatch {
  job: Job;
  match_score: number;
  matched_skills: string[];
  missing_skills: string[];
  match_reasons: string[];
  skill_gap_analysis: {
    current_skills: string[];
    required_skills: string[];
    gap_skills: string[];
    priority_gaps: string[];
    recommendations: string[];
  };
}

export interface SearchFilters {
  location?: string;
  employment_type?: string;
  experience_level?: string;
  salary_min?: number;
  salary_max?: number;
  remote_only?: boolean;
  skills?: string[];
}

export interface JobSearchParams {
  query: string;
  location?: string;
  skills?: string[];
  max_results?: number;
}

export interface JobSearchResponse {
  jobs: Job[];
  total_count: number;
  search_query: string;
  location: string;
  sources_used: string[];
  search_time_ms: number;
  timestamp: string;
}

export interface CVAnalysis {
  analysis_id: string;
  skills: string[];
  experience_years?: number;
  job_titles: string[];
  education: string[];
  summary: string;
  confidence_score: number;
  timestamp: string;
  raw_text?: string;
  total_years_experience?: number;
  personal_info?: {
    name?: string;
    email?: string;
    phone?: string;
    location?: string;
  };
  contact_info?: {
    email?: string;
    phone?: string;
    linkedin?: string;
    github?: string;
    portfolio?: string;
  };
  industries?: Array<[string, number]> | string[];
  job_matches?: JobMatch[];
  certifications?: Array<{
    name: string;
    provider: string;
    level: string;
    duration: string;
    cost: string;
    value: string;
    url?: string;
    priority: string;
  }>;
  recommendations?: Array<{
    type: string;
    priority: string;
    title: string;
    description: string;
    action: string;
  }>;
  roadmap?: {
    current_level: string;
    target_level: string;
    estimated_timeline: string;
    phases: Array<{
      phase: string;
      duration: string;
      skills: string[];
      priority: string;
    }>;
    recommended_resources?: string[];
  };
  projects?: Array<{
    name?: string;
    title?: string;
    summary?: string;
    impact?: string;
    tech_stack?: string[];
  }>;
  portfolio_links?: Record<string, string>;
  sections?: Record<string, unknown>;
}

export interface CVAnalysisSummary {
  analysis_id: string;
  skills: string[];
  experience_years: number;
  summary?: string;
  created_at: string;
}

export interface CVAnalysisListResponse {
  analyses: CVAnalysisSummary[];
  total: number;
}

export interface CVTextAnalysisResponse {
  analysis_id: string;
  skills: string[];
  experience_years: number;
  job_titles: string[];
  education: string[];
  summary: string;
  confidence_score: number;
  timestamp: string;
  learning_focus: string[];
  raw_text_length: number;
}

export interface CVUploadResponse {
  analysis_id: string;
  message: string;
  confidence_score: number;
}

export interface CVExtractTextResponse {
  cv_text: string;
  length: number;
}

export interface PortfolioSectionBlock {
  type: 'paragraph' | 'chips' | 'list';
  text?: string;
  items?: string[];
}

export interface PortfolioSection {
  id: string;
  title: string;
  content: PortfolioSectionBlock[];
}

export interface PortfolioStats {
  skills_count: number;
  top_skills: string[];
  experience_years: number;
  sections: string[];
}

export interface PortfolioMeta {
  id: string;
  name: string;
  template: string;
  color_scheme: string;
  generated_at: string;
  summary: string;
  stats: PortfolioStats;
  sections: PortfolioSection[];
  hero?: {
    name?: string;
    title?: string;
    location?: string;
    headline?: string;
    contact?: Record<string, string>;
  };
  skills?: Record<string, string[]>;
  experiences?: Array<{
    title: string;
    company: string;
    period: string;
    bullets: string[];
  }>;
  projects?: Array<{
    title: string;
    summary?: string;
    bullets?: string[];
    tech?: string[];
  }>;
  education?: string[];
}

export interface PortfolioCustomizationPayload {
  color_scheme: string;
  font_family: string;
  layout_style: string;
  sections_visible: string[];
  include_photo: boolean;
  include_projects: boolean;
  include_contact_form: boolean;
  dark_mode: boolean;
}

export interface PortfolioResponsePayload {
  portfolio: PortfolioMeta;
  html_content: string;
  preview_id: string;
  preview_url: string;
}

export interface LearningResource {
  title: string;
  name?: string;
  url?: string;
  description?: string;
  category?: string;
  cost?: string;
  time_commitment?: string;
  score?: number;
  platform?: string;
  focus?: string;
}

export interface RoadmapAction {
  action?: string;
  timeline?: string;
  priority?: string;
  description?: string;
  goal?: string;
  vision?: string;
  score?: number;
}

export interface DashboardMetrics {
  total_cvs: number;
  jobs_analyzed: number;
  skills_identified: number;
  match_score_average: number;
  growth_rate?: number;
}

export interface RecentActivity {
  id: string | number;
  type: string;
  description: string;
  timestamp: string;
  status: string;
  details?: string;
}

export interface AnalyticsOverviewMetrics {
  total_cvs: number;
  jobs_analyzed: number;
  skills_identified: number;
  match_score_avg: number;
  growth_rate: number;
}

export interface AnalyticsDashboardData {
  overview: AnalyticsOverviewMetrics;
  skill_progress: Array<{ skill: string; current: number; target: number }>;
  job_matching_trends: Array<{ month: string; matches: number; avg_score: number }>;
  skill_distribution: Array<{ category: string; count: number }>;
  recent_activities: RecentActivity[];
}
