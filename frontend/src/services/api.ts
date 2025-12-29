import axios from 'axios';
import type {
  JobSearchParams,
  JobSearchResponse,
  CVAnalysisListResponse,
  AnalyticsDashboardData,
  CVTextAnalysisResponse,
  CVUploadResponse,
  CVExtractTextResponse,
  CVAnalysis,
  PortfolioResponsePayload,
  PortfolioCustomizationPayload,
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8001';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const cvApi = {
  listAnalyses: async (limit = 10): Promise<CVAnalysisListResponse> => {
    const response = await apiClient.get<CVAnalysisListResponse>('/api/v1/cv-analyses', {
      params: { limit },
    });
    return response.data;
  },
  analyzeText: async (cvContent: string): Promise<CVTextAnalysisResponse> => {
    const response = await apiClient.post<CVTextAnalysisResponse>('/api/v1/analyze-cv', {
      cv_content: cvContent,
      format: 'text',
    });
    return response.data;
  },
  analyzeAdvanced: async (cvContent: string): Promise<CVAnalysis> => {
    const response = await apiClient.post<CVAnalysis>('/api/v1/analyze-cv-advanced', {
      cv_content: cvContent,
    });
    return response.data;
  },
  uploadCv: async (file: File, userId?: string): Promise<CVUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<CVUploadResponse>(
      '/api/v1/cv/upload',
      formData,
      {
        params: userId ? { user_id: userId } : undefined,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  },
  extractText: async (file: File): Promise<CVExtractTextResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post<CVExtractTextResponse>(
      '/api/v1/extract-text',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  },
  generatePortfolio: async (params: {
    cvId: string;
    templateId: string;
    customization: PortfolioCustomizationPayload;
  }): Promise<PortfolioResponsePayload> => {
    const response = await apiClient.post<PortfolioResponsePayload>(
      '/api/v1/generate-portfolio',
      {
        cv_id: params.cvId,
        template_id: params.templateId,
        customization: params.customization,
      }
    );
    return response.data;
  },
};

type AnalyticsDashboardResponse = {
  success: boolean;
  data: AnalyticsDashboardData;
  timestamp: string;
};

export const analyticsApi = {
  getDashboard: async (): Promise<AnalyticsDashboardData> => {
    const response = await apiClient.get<AnalyticsDashboardResponse>('/api/v1/analytics/dashboard');
    return response.data.data;
  },
};

export const handleApiError = (error: unknown): string => {
  if (axios.isAxiosError(error)) {
    if (error.response) {
      const normalizeDetail = (detail: unknown): string | undefined => {
        if (!detail) return undefined;
        if (typeof detail === 'string') return detail;
        if (Array.isArray(detail)) {
          const joined = detail
            .map(entry => {
              if (typeof entry === 'string') return entry;
              if (entry && typeof entry === 'object' && 'msg' in entry) {
                return String(entry.msg);
              }
              return JSON.stringify(entry);
            })
            .join(' | ');
          return joined || undefined;
        }
        if (typeof detail === 'object') {
          if ('msg' in (detail as Record<string, unknown>)) {
            return String((detail as Record<string, unknown>).msg);
          }
          return JSON.stringify(detail);
        }
        return undefined;
      };

      return (
        normalizeDetail(error.response.data?.detail) ||
        normalizeDetail(error.response.data?.message) ||
        'Server error'
      );
    }
    if (error.request) {
      return 'Unable to connect to server. Please verify the backend is running.';
    }
  }

  if (error instanceof Error) {
    return error.message;
  }

  if (typeof error === 'string') {
    return error;
  }

  return 'An unexpected error occurred';
};

export const jobApi = {
  search: async (params: JobSearchParams): Promise<JobSearchResponse> => {
    const response = await apiClient.post<JobSearchResponse>('/api/v1/jobs/search', params);
    return response.data;
  },
};

export const experienceApi = {
  translate: async (experience: string): Promise<{ original: string; translated: string; improvements: string[] }> => {
    try {
      const response = await apiClient.post<{ original: string; translated: string; improvements: string[] }>(
        '/api/v1/translate-experience',
        { experience_text: experience }
      );
      return response.data;
    } catch (error: unknown) {
      console.error('Experience translation failed:', error);
      return {
        original: experience,
        translated: experience,
        improvements: [],
      };
    }
  },
};

export default apiClient;
