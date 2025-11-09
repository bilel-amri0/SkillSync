import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
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
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;

// CV Analysis API
export const cvApi = {
  upload: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/cv/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  analyze: (cvId: string) => api.get(`/cv/analyze/${cvId}`),
  getSkills: (cvId: string) => api.get(`/cv/skills/${cvId}`),
  getExperience: (cvId: string) => api.get(`/cv/experience/${cvId}`),
  getRecommendations: (cvId: string) => api.get(`/cv/recommendations/${cvId}`),
};

// Job Matching API
export const jobApi = {
  search: (query: { keywords?: string; location?: string; experience_level?: string; skills?: string[] }) =>
    api.post('/jobs/search', query),
  getMatches: (cvId: string) => api.get(`/jobs/matches/${cvId}`),
  getJobDetails: (jobId: string) => api.get(`/jobs/${jobId}`),
  analyzeGap: (cvId: string, jobId: string) => api.post(`/jobs/gap-analysis`, { cv_id: cvId, job_id: jobId }),
};

// Experience Translation API
export const experienceApi = {
  translate: (data: { experience_text: string; target_role?: string; industry?: string; style?: string }) =>
    api.post('/experience/translate', data),
  getStyles: () => api.get('/experience/styles'),
  getAnalysis: (translationId: string) => api.get(`/experience/analysis/${translationId}`),
  export: (translationId: string, format: 'text' | 'markdown' | 'html' | 'json') =>
    api.get(`/experience/export/${translationId}?format=${format}`),
};

// Analytics API
export const analyticsApi = {
  getDashboard: () => api.get('/analytics/dashboard'),
  getProgress: () => api.get('/analytics/progress'),
  getSkillGaps: () => api.get('/analytics/skill-gaps'),
  getTrends: (timeRange: string) => api.get(`/analytics/trends?range=${timeRange}`),
  getPerformance: () => api.get('/analytics/performance'),
};

// Portfolio API
export const portfolioApi = {
  getTemplates: () => api.get('/portfolio/templates'),
  generate: (data: { cv_id: string; template_id: string; customization?: any }) =>
    api.post('/portfolio/generate', data),
  getPortfolios: () => api.get('/portfolio/list'),
  export: (portfolioId: string, format: 'pdf' | 'html') =>
    api.get(`/portfolio/export/${portfolioId}?format=${format}`),
};

// XAI Explanations API
export const xaiApi = {
  getExplanation: (feature: string, context: any) => api.post('/xai/explain', { feature, context }),
  getRecommendations: () => api.get('/xai/recommendations'),
  getTransparency: () => api.get('/xai/transparency'),
};

// User Management API
export const userApi = {
  login: (credentials: { email: string; password: string }) => api.post('/auth/login', credentials),
  register: (userData: { email: string; password: string; name: string }) => api.post('/auth/register', userData),
  getProfile: () => api.get('/user/profile'),
  updateProfile: (data: any) => api.put('/user/profile', data),
  logout: () => api.post('/auth/logout'),
};

// Health check
export const healthApi = {
  check: () => api.get('/health'),
};

// Error handling helper
export const handleApiError = (error: any) => {
  if (error.response) {
    // Server responded with error status
    const message = error.response.data?.detail || error.response.data?.message || 'Server error';
    throw new Error(message);
  } else if (error.code === 'ECONNREFUSED' || error.request?.message?.includes('Network Error') || error.message?.includes('timeout')) {
    // Connection refused or network timeout
    throw new Error('Cannot connect to backend server. Please ensure your FastAPI server is running on localhost:8000.');
  } else if (error.request) {
    // Request was made but no response received
    throw new Error('Network error - unable to connect to localhost:8000. Please check if the backend server is running.');
  } else {
    // Something else happened
    throw new Error(error.message || 'Unknown error occurred');
  }
};