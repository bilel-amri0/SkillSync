// SkillSync API Client - Updated 2025-11-23
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8001';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types matching backend response
export interface CVAnalysisResponse {
  analysis_id: string;
  skills: string[];
  hard_skills?: string[];
  soft_skills?: string[];
  experience_years?: number;
  job_titles: string[];
  education: string[];
  summary: string;
  confidence_score: number;
  timestamp: string;
  raw_text?: string;
  work_history?: Array<Record<string, unknown>>;
  total_years_experience?: number;
  sections?: Record<string, unknown>;
  personal_info?: Record<string, unknown>;
  contact_info?: Record<string, unknown>;
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
  certifications?: Array<Record<string, unknown>>;
  recommendations?: Array<{
    type: string;
    priority: string;
    title: string;
    description: string;
    action: string;
  }>;
  // Advanced ML features
  name?: string;
  email?: string;
  phone?: string;
  location?: string;
  current_title?: string;
  seniority_level?: string;
  industries?: Array<[string, number]>;
  career_trajectory?: Record<string, unknown>;
  projects?: Array<Record<string, unknown>>;
  portfolio_links?: Record<string, unknown>;
  ml_confidence_breakdown?: Record<string, number>;
  parser_version?: string;
  skill_categories?: Record<string, string[]>;
  tech_stack_clusters?: Record<string, string[]>;
  languages?: Array<string | Record<string, unknown>>;
  processing_time_ms?: number;
}

export interface Template {
  id: string;
  name: string;
  description: string;
}

export interface ColorScheme {
  id: string;
  name: string;
  primary: string;
}

type PortfolioResponsePayload = {
  portfolio: {
    id?: string;
    name?: string;
    [key: string]: unknown;
  };
  html_content: string;
};

/**
 * Analyze CV file (Standard endpoint)
 * @param file - CV file (PDF, DOCX, or TXT)
 * @returns Parsed CV data
 */
export const analyzeCV = async (file: File): Promise<CVAnalysisResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('extract_skills', 'true');

  const response = await apiClient.post<CVAnalysisResponse>(
    '/api/v1/upload-cv',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );

  return response.data;
};

/**
 * Analyze CV file with Advanced ML (Industries, Projects, Seniority, Portfolio)
 * @param file - CV file (PDF, DOCX, or TXT)
 * @returns Advanced ML-powered CV analysis
 */
export const analyzeAdvancedCV = async (file: File): Promise<CVAnalysisResponse> => {
  // For PDF files, use FormData upload first to extract text
  if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
    // Upload PDF to get text extracted
    const formData = new FormData();
    formData.append('file', file);
    
    const uploadResponse = await apiClient.post<{ cv_text: string }>(
      '/api/v1/extract-text',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );
    
    // Now analyze with advanced ML
    const response = await apiClient.post<CVAnalysisResponse>(
      '/api/v1/analyze-cv-advanced',
      { cv_content: uploadResponse.data.cv_text },
      {
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );

    const enrichedResult: CVAnalysisResponse = {
      ...response.data,
      raw_text: response.data.raw_text ?? uploadResponse.data.cv_text,
    };
    
    return enrichedResult;
  }
  
  // For text files, read directly
  const text = await file.text();

  const response = await apiClient.post<CVAnalysisResponse>(
    '/api/v1/analyze-cv-advanced',
    { cv_content: text },
    {
      headers: {
        'Content-Type': 'application/json',
      },
    }
  );

  const enrichedResult: CVAnalysisResponse = {
    ...response.data,
    raw_text: response.data.raw_text ?? text,
  };

  return enrichedResult;
};

/**
 * Generate portfolio and download ZIP
 * @param cvData - Analyzed CV data from analyzeCV (contains analysis_id)
 * @param templateId - Template choice (modern, classic, creative, minimal, tech)
 * @param colorScheme - Color scheme choice (blue, green, purple, red, orange)
 * @returns Downloads ZIP file automatically
 */
export const generatePortfolio = async (
  cvData: CVAnalysisResponse,
  templateId: string,
  colorScheme: string
): Promise<void> => {
  const payload = {
    cv_id: cvData.analysis_id,
    template_id: templateId,
    customization: {
      color_scheme: colorScheme,
      font_family: 'Inter',
      layout_style: 'clean',
      sections_visible: ['about', 'experience', 'skills', 'projects', 'education'],
      include_photo: true,
      include_projects: true,
      include_contact_form: true,
      dark_mode: false,
    }
  };

  const response = await apiClient.post<PortfolioResponsePayload>(
    '/api/v1/generate-portfolio',
    payload
  );

  // Extract portfolio data
  const { portfolio, html_content } = response.data;
  
  // Create HTML blob and trigger download
  const blob = new Blob([html_content], { type: 'text/html' });
  const downloadUrl = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = downloadUrl;
  link.download = `portfolio-${portfolio.name || 'website'}.html`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(downloadUrl);
  
  console.log('✅ Portfolio generated:', portfolio.id);
};

/**
 * Get available templates
 */
export const getTemplates = async (): Promise<Record<string, string>> => {
  const response = await apiClient.get<{
    success: boolean;
    templates: Record<string, string>;
  }>('/templates');
  
  return response.data.templates;
};

/**
 * Get available color schemes
 */
export const getColorSchemes = async (): Promise<string[]> => {
  const response = await apiClient.get<{
    success: boolean;
    color_schemes: string[];
  }>('/color-schemes');
  
  return response.data.color_schemes;
};

/**
 * Search for jobs across multiple APIs
 * @param query - Job title or keywords
 * @param location - Location filter
 * @param skills - Array of skills from CV
 * @param maxResults - Maximum number of results
 */
export interface JobSearchParams {
  query: string;
  location?: string;
  skills?: string[];
  max_results?: number;
}

export interface Job {
  id: string;
  title: string;
  company: string;
  location: string;
  description: string;
  url: string;
  salary?: string;
  match_score?: number;
  source: string;
  posted_date?: string;
}

export interface JobSearchResponse {
  success?: boolean;
  total_count: number;
  jobs: Job[];
  sources_used: string[];
  search_query?: string;
  location?: string;
  search_time_ms?: number;
  timestamp?: string;
}

export const searchJobs = async (params: JobSearchParams): Promise<JobSearchResponse> => {
  const response = await apiClient.post<JobSearchResponse>(
    '/api/v1/jobs/search',
    params
  );
  
  return response.data;
};

/**
 * Check API health
 */
export const checkHealth = async (): Promise<boolean> => {
  try {
    const response = await apiClient.get('/api/v1/analyze/health');
    return response.data.status === 'healthy';
  } catch (error) {
    console.error('Health check failed:', error);
    return false;
  }
};

// ============================================================================
// Interview API
// ============================================================================

export interface StartInterviewRequest {
  user_id: string;
  cv_id?: string;
  cv_text: string;
  job_title: string;
  job_description?: string;
  difficulty: 'easy' | 'medium' | 'hard';
  skills: string[];
  interview_mode: 'text' | 'voice';
}

export interface InterviewQuestion {
  question_id: number;
  question_text: string;
  category: string;
  order: number;
}

export interface InterviewOut {
  interview_id: string;
  user_id: string;
  job_title: string;
  interview_mode: string;
  status: string;
  current_question?: InterviewQuestion;
  total_questions: number;
  answered_questions: number;
}

export interface SubmitAnswerRequest {
  interview_id: string;
  question_id: number;
  answer_text: string;
}

export interface NextQuestionResponse {
  interview_id: string;
  status: string;
  next_question?: InterviewQuestion;
  message?: string;
  progress?: {
    current: number;
    total: number;
  };
}

export interface InterviewReport {
  interview_id: string;
  status: string;
  overall_score?: number;
  analysis?: Record<string, unknown>;
  report?: Record<string, unknown>;
}

export interface SubmitAnswerResponse {
  interview_id: string;
  status: string;
  message?: string;
  next_question?: InterviewQuestion;
  progress?: {
    current: number;
    total: number;
  };
}

/**
 * Start a new interview session
 */
export const startInterview = async (payload: StartInterviewRequest): Promise<InterviewOut> => {
  const response = await apiClient.post<InterviewOut>('/api/v2/interviews/start', payload);
  return response.data;
};

/**
 * Submit an answer to a question
 */
export const submitAnswer = async (payload: SubmitAnswerRequest): Promise<SubmitAnswerResponse> => {
  const response = await apiClient.post<SubmitAnswerResponse>('/api/v2/interviews/answer', payload);
  return response.data;
};

/**
 * Get next question in the interview
 */
export const getNextQuestion = async (interviewId: string): Promise<NextQuestionResponse> => {
  const response = await apiClient.post<NextQuestionResponse>(
    '/api/v2/interviews/next-question',
    { interview_id: interviewId }
  );
  return response.data;
};

/**
 * Finish interview and generate report
 */
export const finishInterview = async (interviewId: string): Promise<InterviewReport> => {
  const response = await apiClient.post<InterviewReport>(
    '/api/v2/interviews/finish',
    { interview_id: interviewId }
  );
  return response.data;
};

/**
 * Get interview details
 */
export const getInterview = async (interviewId: string): Promise<InterviewOut> => {
  const response = await apiClient.get<InterviewOut>(`/api/v2/interviews/${interviewId}`);
  return response.data;
};

/**
 * Get interview report
 */
export const getInterviewReport = async (interviewId: string): Promise<InterviewReport> => {
  const response = await apiClient.get<InterviewReport>(
    `/api/v2/interviews/${interviewId}/report`
  );
  return response.data;
};

/**
 * List interviews
 */
export const listInterviews = async (userId?: string, limit: number = 20): Promise<InterviewOut[]> => {
  const params = new URLSearchParams();
  if (userId) params.append('user_id', userId);
  params.append('limit', limit.toString());
  
  const response = await apiClient.get<InterviewOut[]>(`/api/v2/interviews?${params.toString()}`);
  return response.data;
};
