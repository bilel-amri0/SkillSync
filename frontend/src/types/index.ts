// Core API Types
export interface ApiResponse<T = any> {
  data: T;
  message?: string;
  success: boolean;
}

// User Types
export interface User {
  id: string;
  email: string;
  name: string;
  created_at: string;
  updated_at: string;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterData {
  email: string;
  password: string;
  name: string;
}

// CV Analysis Types
export interface CVAnalysis {
  id: string;
  file_name: string;
  file_size: number;
  upload_date: string;
  skills: Skill[];
  experience: ExperienceEntry[];
  education: EducationEntry[];
  certifications: Certification[];
  summary: string;
  status: 'processing' | 'completed' | 'error';
}

export interface Skill {
  name: string;
  category: string;
  proficiency_level: number; // 1-10
  confidence: number; // 0-100%
  source: string;
  years_of_experience?: number;
  verified: boolean;
}

export interface ExperienceEntry {
  company: string;
  position: string;
  start_date: string;
  end_date: string | null;
  description: string;
  skills_used: string[];
  achievements: string[];
  duration_months: number;
}

export interface EducationEntry {
  institution: string;
  degree: string;
  field_of_study: string;
  start_date: string;
  end_date: string | null;
  gpa?: number;
}

export interface Certification {
  name: string;
  issuer: string;
  date_earned: string;
  expiry_date?: string;
  credential_id?: string;
  verification_url?: string;
}

// Job Matching Types
export interface Job {
  id: string;
  title: string;
  company: string;
  location: string;
  employment_type: string;
  experience_level: string;
  salary_range?: {
    min: number;
    max: number;
    currency: string;
  };
  description: string;
  requirements: string[];
  skills_required: string[];
  skills_nice_to_have: string[];
  posted_date: string;
  application_url?: string;
}

export interface JobMatch {
  job: Job;
  match_score: number; // 0-100%
  matched_skills: string[];
  missing_skills: string[];
  match_reasons: string[];
  skill_gap_analysis: SkillGapAnalysis;
}

export interface SkillGapAnalysis {
  current_skills: Skill[];
  required_skills: string[];
  gap_skills: string[];
  priority_gaps: string[];
  recommendations: string[];
}

// Experience Translation Types
export interface TranslationRequest {
  experience_text: string;
  target_role?: string;
  industry?: string;
  style?: 'professional' | 'technical' | 'creative';
}

export interface TranslationResult {
  id: string;
  original_text: string;
  translated_text: string;
  style: string;
  confidence_score: number; // 0-100%
  analysis: {
    skills_extracted: string[];
    action_verbs: string[];
    quantified_achievements: string[];
    experience_level: string;
    clarity_score: number;
  };
  improvements: {
    suggestions: string[];
    added_keywords: string[];
    enhanced_areas: string[];
  };
  created_at: string;
}

// Analytics Types
export interface DashboardMetrics {
  total_cvs: number;
  jobs_analyzed: number;
  skills_identified: number;
  match_score_average: number;
  recent_activity: Activity[];
  trends: {
    period: string;
    cv_uploads: number;
    job_matches: number;
    skill_improvements: number;
  }[];
}

export interface Activity {
  id: string;
  type: 'cv_upload' | 'job_match' | 'skill_improvement' | 'translation' | 'portfolio_generation';
  description: string;
  timestamp: string;
  metadata?: any;
}

export interface ProgressData {
  skill_name: string;
  current_level: number;
  target_level: number;
  progress_percentage: number;
  category: string;
  last_updated: string;
}

export interface SkillGap {
  skill: string;
  current_level: number;
  target_level: number;
  gap_size: number;
  priority: 'high' | 'medium' | 'low';
  recommendation: string;
}

// Portfolio Types
export interface Portfolio {
  id: string;
  name: string;
  cv_id: string;
  template_id: string;
  customization: PortfolioCustomization;
  generated_date: string;
  last_modified: string;
  status: 'draft' | 'published' | 'archived';
  metrics: {
    views: number;
    downloads: number;
    likes: number;
  };
}

export interface PortfolioCustomization {
  color_scheme: string;
  font_family: string;
  layout_style: string;
  sections_visible: string[];
  custom_sections?: any;
}

export interface PortfolioTemplate {
  id: string;
  name: string;
  description: string;
  preview_url: string;
  category: string;
  features: string[];
}

// XAI Explanation Types
export interface XAIExplanation {
  feature: string;
  explanation: string;
  factors: ExplanationFactor[];
  confidence: number;
  data_sources: string[];
  methodology: string;
}

export interface ExplanationFactor {
  name: string;
  impact: number; // -100 to 100
  description: string;
  weight: number;
}

export interface XAIRecommendation {
  id: string;
  title: string;
  description: string;
  rationale: string;
  confidence: number;
  related_skills: string[];
  expected_outcome: string;
}

// Search and Filter Types
export interface SearchFilters {
  keywords?: string;
  location?: string;
  experience_level?: string;
  salary_range?: {
    min: number;
    max: number;
  };
  skills?: string[];
  industry?: string;
  employment_type?: string;
}

// UI State Types
export interface LoadingState {
  isLoading: boolean;
  error?: string;
  message?: string;
}

export interface PaginationInfo {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
}

// Chart Data Types
export interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    backgroundColor?: string | string[];
    borderColor?: string | string[];
    borderWidth?: number;
  }[];
}

export interface TimeSeriesData {
  date: string;
  value: number;
  label?: string;
}