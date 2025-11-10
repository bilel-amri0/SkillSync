import axios from 'axios';

// Create axios instance for interview API
const interviewApi = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
interviewApi.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor
interviewApi.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Types
export interface Question {
  question_id: number;
  question_text: string;
  category: string;
}

export interface StartInterviewRequest {
  cv_text: string;
  job_description: string;
  num_questions?: number;
}

export interface StartInterviewResponse {
  interview_id: string;
  questions: Question[];
  message: string;
}

export interface SubmitAnswerRequest {
  question_id: number;
  answer_text: string;
}

export interface SubmitAnswerResponse {
  message: string;
  next_question: Question | null;
  is_complete: boolean;
}

export interface InterviewTranscriptItem {
  question_id: number;
  question_text: string;
  answer_text: string;
  category: string;
}

export interface InterviewAnalysis {
  overall_score: number;
  summary: string;
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
}

export interface InterviewReportResponse {
  interview_id: string;
  cv_text: string;
  job_description: string;
  transcript: InterviewTranscriptItem[];
  analysis: InterviewAnalysis;
  created_at: string;
}

// Interview Service
export const interviewService = {
  /**
   * Start a new interview session
   */
  startInterview: async (data: StartInterviewRequest): Promise<StartInterviewResponse> => {
    const response = await interviewApi.post<StartInterviewResponse>('/interviews/start', data);
    return response.data;
  },

  /**
   * Submit an answer to a question
   */
  submitAnswer: async (interviewId: string, data: SubmitAnswerRequest): Promise<SubmitAnswerResponse> => {
    const response = await interviewApi.post<SubmitAnswerResponse>(
      `/interviews/interviews/${interviewId}/submit_answer`,
      data
    );
    return response.data;
  },

  /**
   * Get the complete interview report
   */
  getReport: async (interviewId: string): Promise<InterviewReportResponse> => {
    const response = await interviewApi.get<InterviewReportResponse>(
      `/interviews/interviews/${interviewId}/report`
    );
    return response.data;
  },
};

// Error handling helper
export const handleInterviewError = (error: any): string => {
  if (error.response) {
    // Server responded with error status
    return error.response.data?.detail || error.response.data?.message || 'Server error';
  } else if (error.code === 'ECONNREFUSED' || error.request?.message?.includes('Network Error') || error.message?.includes('timeout')) {
    // Connection refused or network timeout
    return 'Cannot connect to backend server. Please ensure your FastAPI server is running on localhost:8000.';
  } else if (error.request) {
    // Request was made but no response received
    return 'Network error - unable to connect to localhost:8000. Please check if the backend server is running.';
  } else {
    // Something else happened
    return error.message || 'Unknown error occurred';
  }
};

export default interviewService;
