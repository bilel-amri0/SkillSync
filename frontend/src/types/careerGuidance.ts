export interface JobRecommendation {
  title: string;
  similarity_score: number;
  confidence: number;
  predicted_salary: {
    min: number;
    max: number;
    currency: string;
  };
  matching_skills: string[];
  matched_skills?: string[];
  skill_gaps: string[];
  gap_skills?: string[];
  growth_potential: string;
  reasons: string[];
  description?: string;
  job_url: string;
  source?: string;
  source_logo?: string;
  posted_date?: string;
  posting_date?: string;
  location?: string;
  location_type?: string;
  tags?: string[];
}

export interface CertificationResource {
  title: string;
  url?: string;
  link?: string;
  provider?: string;
  cost?: string;
  format?: string;
  type?: string;
}

export interface CertificationRecommendation {
  name: string;
  relevance_score: number;
  skill_alignment: number;
  predicted_roi: string;
  estimated_time: string;
  career_boost: string;
  reasons: string[];
  url?: string;
  official_url?: string;
  provider?: string;
  provider_logo?: string;
  cost_type?: string;
  delivery_format?: string;
  cost_estimate?: string;
  prerequisites?: string[] | string;
  prerequisite_details?: string[] | string;
  exam_format?: string;
  examFormat?: string;
  resources?: CertificationResource[];
}

export interface ResourceItem {
  skill: string;
  title: string;
  provider: string;
  duration: string;
  rating: number;
  tier?: string;
  url?: string;
  link?: string;
  cost?: string;
  is_free?: boolean;
  estimated_time_hours?: number;
  time_hours?: number;
}

export interface SmartMilestone {
  title: string;
  type: string;
  target_metric: string;
  deadline_hours: number;
}

export interface RoadmapPhase {
  phase_name: string;
  duration_weeks: number;
  duration_months: number;
  total_time_estimate_hours: number;
  skills_to_learn: string[];
  learning_resources: ResourceItem[];
  resources?: ResourceItem[];
  success_probability: string;
  success_justification?: string;
  effort_level: string;
  milestones: string[];
  smart_milestones?: SmartMilestone[];
}

export interface RoadmapResponse {
  total_duration_weeks: number;
  total_duration_months: number;
  total_time_estimate_hours: number;
  predicted_success_rate: string;
  personalization_score: string;
  learning_strategy: string;
  phases: RoadmapPhase[];
}

export type LearningPhase = RoadmapPhase;
export type LearningRoadmap = RoadmapResponse;

export interface XAIInsights {
  how_we_analyzed_your_cv: Record<string, unknown>;
  job_matching_explanation: Record<string, unknown>;
  certification_ranking_explanation: Record<string, unknown>;
  learning_path_explanation: Record<string, unknown>;
  ml_confidence_scores: Record<string, string | number>;
  key_insights: string[];
}

export interface CareerGuidanceResponse {
  job_recommendations: JobRecommendation[];
  certification_recommendations: CertificationRecommendation[];
  learning_roadmap: LearningRoadmap;
  xai_insights: XAIInsights;
  metadata: {
    processing_time_seconds: number;
    cv_skills_count: number;
    jobs_recommended: number;
    certs_recommended: number;
    roadmap_phases: number;
    ml_model: string;
    engine_version: string;
    timestamp?: string;
    job_sources?: string[];
    live_job_total?: number;
    search_terms?: string[];
    job_matching_confidence?: number;
  };
}
