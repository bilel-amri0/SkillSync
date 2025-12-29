import { useRef, useState, type ChangeEvent } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  FileText,
  Target,
  CheckCircle,
  AlertCircle,
  Brain,
  Briefcase,
  Rocket,
  GraduationCap,
  Lightbulb,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  MapPin,
  Clock,
  Link,
  Tag
} from 'lucide-react';
import axios from 'axios';
import { analyzeAdvancedCV, type CVAnalysisResponse } from '../api';
import { handleApiError } from '../services/api';
import { saveLatestAnalysis, saveLatestGuidance } from '../services/careerStore';
import type {
  CareerGuidanceResponse,
  JobRecommendation,
} from '../types/careerGuidance';
import { Roadmap } from '../components/roadmap/Roadmap';

interface MLCareerGuidancePageProps {
  onCvAnalyzed?: (analysis: CVAnalysisResponse) => void;
  onGuidanceComplete?: (guidance: CareerGuidanceResponse | null) => void;
}

const API_BASE_URL = 'http://localhost:8001';

const formatSalaryRange = (salary?: JobRecommendation['predicted_salary']) => {
  if (!salary) return 'Salary range available';
  const formatter = new Intl.NumberFormat(undefined, {
    style: 'currency',
    currency: salary.currency || 'USD',
    maximumFractionDigits: 0,
  });
  return `${formatter.format(salary.min)} - ${formatter.format(salary.max)}`;
};

const formatPostedDate = (value?: string | null) => {
  if (!value) return 'Recently analyzed';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  const diffDays = Math.floor((Date.now() - parsed.getTime()) / (1000 * 60 * 60 * 24));
  if (diffDays <= 0) return 'Posted today';
  if (diffDays === 1) return 'Posted yesterday';
  if (diffDays < 30) return `Posted ${diffDays} days ago`;
  return parsed.toLocaleDateString();
};

const truncateText = (value: string, max = 260) => {
  if (!value) return '';
  if (value.length <= max) return value;
  return `${value.slice(0, max).trim()}...`;
};

const sanitizeRichText = (value?: string | null) => {
  if (!value) return '';
  const withoutTags = value.replace(/<[^>]*>/g, ' ');
  return withoutTags.replace(/\s+/g, ' ').trim();
};

const normalizeStringList = (value?: string | string[] | null) => {
  if (!value) return [];
  if (Array.isArray(value)) {
    return value
      .map(entry => sanitizeRichText(entry) || entry.trim())
      .filter(Boolean);
  }
  return value
    .split(/[,;\n•]+/)
    .map(entry => sanitizeRichText(entry) || entry.trim())
    .filter(Boolean);
};

const formatResourceRating = (rating?: number | null) => {
  if (rating === null || rating === undefined) {
    return 'N/A';
  }
  const numeric = Number(rating);
  if (!Number.isFinite(numeric)) {
    return 'N/A';
  }
  return parseFloat(numeric.toFixed(1)).toString();
};

const buildRoadmapMarkdown = (guidance: CareerGuidanceResponse) => {
  const lines: string[] = [];
  const totalEffort = guidance.learning_roadmap.total_time_estimate_hours
    ?? guidance.learning_roadmap.phases.reduce((sum, phase) => sum + (phase.total_time_estimate_hours || 0), 0);
  lines.push('# SkillSync Learning Roadmap');
  lines.push(`Generated on ${new Date(guidance.metadata.timestamp || Date.now()).toLocaleString()}`);
  lines.push('\n## Summary');
  lines.push(`- Total Duration: ${guidance.learning_roadmap.total_duration_weeks} weeks (${guidance.learning_roadmap.total_duration_months} months)`);
  lines.push(`- Total Effort: ${totalEffort} hours`);
  lines.push(`- Success Rate: ${guidance.learning_roadmap.predicted_success_rate}`);
  lines.push(`- Personalization: ${guidance.learning_roadmap.personalization_score}`);
  if (guidance.metadata.job_matching_confidence !== undefined) {
    lines.push(`- Job Matching Confidence: ${(guidance.metadata.job_matching_confidence * 100).toFixed(1)}%`);
  }
  lines.push('\n## Job Recommendations');
  if (guidance.job_recommendations.length === 0) {
    lines.push('- No live recommendations available.');
  } else {
    guidance.job_recommendations.forEach(job => {
      lines.push(`- [${job.title}](${job.job_url}) • ${job.source || 'Live Source'} • ${formatSalaryRange(job.predicted_salary)}`);
      const matchedSkills = job.matched_skills && job.matched_skills.length > 0 ? job.matched_skills : job.matching_skills;
      const gapSkills = job.gap_skills && job.gap_skills.length > 0 ? job.gap_skills : job.skill_gaps;
      if (matchedSkills.length) {
        lines.push(`  - Matched Skills: ${matchedSkills.join(', ')}`);
      }
      if (gapSkills.length) {
        lines.push(`  - Priority Gaps: ${gapSkills.join(', ')}`);
      }
    });
  }
  lines.push('\n## Learning Phases');
  guidance.learning_roadmap.phases.forEach((phase, index) => {
    lines.push(`### Phase ${index + 1}: ${phase.phase_name}`);
    lines.push(`- Duration: ${phase.duration_weeks} weeks (${phase.duration_months} months)`);
    lines.push(`- Effort: ${phase.total_time_estimate_hours} hours`);
    lines.push(`- Success Probability: ${phase.success_probability}`);
    lines.push(`- Effort: ${phase.effort_level}`);
    if (phase.success_justification) {
      lines.push(`- Success Justification: ${phase.success_justification}`);
    }
    if (phase.skills_to_learn.length) {
      lines.push(`- Skills: ${phase.skills_to_learn.join(', ')}`);
    }
    const resources = phase.resources?.length ? phase.resources : phase.learning_resources || [];
    if (resources.length) {
      lines.push('- Resources:');
      resources.forEach(resource => {
        const link = resource.link || resource.url;
        const tierLabel = resource.tier ? `${resource.tier} • ` : '';
        const base = `${tierLabel}${resource.title} — ${resource.provider} (${resource.duration})`;
        const hours = resource.estimated_time_hours || resource.time_hours;
        const suffix = hours ? `${base} • ~${hours}h` : base;
        lines.push(link ? `  - [${suffix}](${link})` : `  - ${suffix}`);
      });
    }
    if (phase.smart_milestones && phase.smart_milestones.length) {
      lines.push('- SMART Milestones:');
      phase.smart_milestones.forEach(milestone =>
        lines.push(`  - ${milestone.title} (${milestone.type}) → ${milestone.target_metric} within ${milestone.deadline_hours} hrs`)
      );
    }
    if (phase.milestones.length) {
      lines.push('- Additional Milestones:');
      phase.milestones.forEach(milestone => lines.push(`  - ${milestone}`));
    }
  });

  if (guidance.certification_recommendations.length) {
    lines.push('\n## Certifications');
    guidance.certification_recommendations.forEach((cert, index) => {
      const link = cert.official_url || cert.url;
      const label = `${index + 1}. ${cert.name}`;
      lines.push(link ? `- [${label}](${link})` : `- ${label}`);
      lines.push(`  - Provider: ${cert.provider || 'Top provider'}`);
      if (cert.cost_estimate) {
        lines.push(`  - Cost: ${cert.cost_estimate}`);
      }
      if (cert.resources && cert.resources.length) {
        lines.push('  - Resources:');
        cert.resources.forEach(resource => {
          const resourceLink = resource.link || resource.url;
          if (resourceLink) {
            lines.push(`    - [${resource.title}](${resourceLink})`);
          } else {
            lines.push(`    - ${resource.title}`);
          }
        });
      }
    });
  }

  return lines.join('\n');
};

const AnalysisSkeleton = () => (
  <motion.div
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    className="space-y-6"
  >
    {[1, 2, 3].map(section => (
      <div
        key={section}
        className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-6 border border-gray-100 dark:border-gray-700"
      >
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/3" />
          {[0, 1, 2].map(row => (
            <div key={row} className="flex gap-4">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2" />
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/3" />
            </div>
          ))}
          <div className="h-40 bg-gray-100 dark:bg-gray-700/70 rounded-lg" />
        </div>
      </div>
    ))}
  </motion.div>
);

const SectionSkeleton = ({ title }: { title: string }) => (
  <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-6 border border-gray-100 dark:border-gray-700">
    <div className="flex items-center gap-3 mb-4">
      <div className="w-8 h-8 rounded-full bg-gray-200 dark:bg-gray-700 animate-pulse" />
      <div className="h-6 w-48 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
    </div>
    <div className="space-y-3">
      {[1, 2, 3].map(item => (
        <div key={item} className="h-4 bg-gray-100 dark:bg-gray-700/60 rounded animate-pulse" />
      ))}
    </div>
    <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
      {[1, 2].map(item => (
        <div key={`${title}-${item}`} className="h-24 bg-gray-100 dark:bg-gray-700/40 rounded-lg animate-pulse" />
      ))}
    </div>
  </div>
);

const JobsSectionSkeleton = () => <SectionSkeleton title="Jobs" />;
const CertificationsSectionSkeleton = () => <SectionSkeleton title="Certifications" />;
const RoadmapSectionSkeleton = () => <SectionSkeleton title="Roadmap" />;

export const MLCareerGuidancePage = ({ onCvAnalyzed, onGuidanceComplete }: MLCareerGuidancePageProps) => {
  const [guidance, setGuidance] = useState<CareerGuidanceResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedXAI, setExpandedXAI] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [cvAnalysis, setCvAnalysis] = useState<CVAnalysisResponse | null>(null);
  const [expandedCerts, setExpandedCerts] = useState<Record<number, boolean>>({});
  const [certDetailPanels, setCertDetailPanels] = useState<Record<number, boolean>>({});
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const resetWorkflow = () => {
    setGuidance(null);
    setCvAnalysis(null);
    setSelectedFile(null);
    setExpandedXAI(false);
    setExpandedCerts({});
    setCertDetailPanels({});
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    onGuidanceComplete?.(null);
  };

  const toggleCertResources = (index: number) => {
    setExpandedCerts(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const toggleCertDetails = (index: number) => {
    setCertDetailPanels(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const handleDownloadRoadmap = () => {
    if (!guidance) return;
    const fileName = `skillsync-learning-roadmap-${guidance.metadata.timestamp || Date.now()}.md`;
    const markdown = buildRoadmapMarkdown(guidance);
    const blob = new Blob([markdown], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
    setError(null);
    setGuidance(null);
    setCvAnalysis(null);
    setExpandedXAI(false);
    setExpandedCerts({});
    setCertDetailPanels({});
  };

  const handleAnalyzeWithML = async () => {
    if (!selectedFile) {
      setError('Please upload a CV before launching the ML analysis.');
      return;
    }

    console.log('🔍 [ML Career] Starting full ML workflow with file:', selectedFile.name);
    setIsAnalyzing(true);
    setError(null);
    setGuidance(null);

    try {
      const cvResult = await analyzeAdvancedCV(selectedFile);
      setCvAnalysis(cvResult);
      onCvAnalyzed?.(cvResult);
      saveLatestAnalysis(cvResult);

      console.log('✅ [ML Career] Advanced CV analysis complete:', {
        analysisId: cvResult.analysis_id,
        skills: cvResult.skills?.length || 0,
        hard_skills: cvResult.hard_skills?.length || 0,
        work_history: cvResult.work_history?.length || 0,
      });

      const payload = {
        cv_analysis: cvResult,
        cv_content: cvResult.raw_text || undefined,
      };

      console.log('🧪 [ML Career] Payload readiness check:', {
        analysisId: cvResult.analysis_id,
        work_history_entries: cvResult.work_history?.length || 0,
        projects: cvResult.projects?.length || 0,
        soft_skills: cvResult.soft_skills?.length || 0,
        education_entries: cvResult.education?.length || 0,
        certifications: cvResult.certifications?.length || 0,
        industries: cvResult.industries?.length || 0,
        languages: cvResult.languages?.length || 0,
        tech_categories: cvResult.tech_stack_clusters ? Object.keys(cvResult.tech_stack_clusters).length : 0,
        hard_skills: cvResult.hard_skills?.length || 0,
        raw_text_length: cvResult.raw_text?.length || 0,
      });
      console.debug('📦 [ML Career] Serialized payload preview:', payload);
      console.log('🚀 [ML Career] Sending full payload to ML engine...');
      const response = await axios.post<CareerGuidanceResponse>(
        `${API_BASE_URL}/api/v1/career-guidance`,
        payload,
        {
          timeout: 120000,
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      console.log('✅ [ML Career] ML engine response:', {
        jobs: response.data.job_recommendations?.length || 0,
        certs: response.data.certification_recommendations?.length || 0,
        roadmapPhases: response.data.learning_roadmap?.phases?.length || 0,
      });

      setGuidance(response.data);
      setExpandedCerts({});
      setExpandedXAI(false);
      setCertDetailPanels({});
      onGuidanceComplete?.(response.data);
      saveLatestGuidance(response.data);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        console.error('❌ [ML Career] Error during analysis:', {
          message: err.message,
          response: err.response?.data,
          status: err.response?.status,
        });
      } else {
        console.error('❌ [ML Career] Error during analysis:', err);
      }
      setError(handleApiError(err));
    } finally {
      setIsAnalyzing(false);
      console.log('🏁 [ML Career] ML workflow completed');
    }
  };

  const getGrowthColor = (growth: string) => {
    if (growth.includes('Very High')) return 'text-green-600 bg-green-100 dark:bg-green-900/20';
    if (growth.includes('High')) return 'text-blue-600 bg-blue-100 dark:bg-blue-900/20';
    return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-blue-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center gap-3 mb-2">
            <Rocket className="w-10 h-10 text-purple-600" />
            <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
              🤖 ML Career Guidance System
            </h1>
          </div>
          <p className="text-gray-600 dark:text-gray-400">
            100% ML-Powered: Get job recommendations, certifications, and personalized learning roadmap with complete explainability
          </p>
        </motion.div>

        {!guidance ? (
          isAnalyzing ? (
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-8">
              <div className="flex items-center gap-3 mb-6">
                <Brain className="w-6 h-6 text-purple-600" />
                <div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white">Running ML pipeline...</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">We're fetching live jobs, certifications, and roadmap insights.</p>
                </div>
              </div>
              <AnalysisSkeleton />
              <div className="mt-8 space-y-4">
                <JobsSectionSkeleton />
                <CertificationsSectionSkeleton />
                <RoadmapSectionSkeleton />
              </div>
            </div>
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-8"
            >
              <div className="bg-gradient-to-r from-purple-100 to-blue-100 dark:from-purple-900/30 dark:to-blue-900/30 rounded-lg p-6 mb-6">
                <div className="flex items-start gap-3">
                  <Brain className="w-6 h-6 text-purple-600 flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="font-semibold text-gray-900 dark:text-white mb-2">
                      🚀 Fully ML-Driven Workflow
                    </h3>
                    <ul className="text-sm text-gray-700 dark:text-gray-300 space-y-1">
                      <li>1️⃣ Upload your CV (PDF/TXT)</li>
                      <li>2️⃣ Advanced parser extracts every field</li>
                      <li>3️⃣ Payload is forwarded directly to the ML engine</li>
                      <li>4️⃣ Real job matches, roadmap, and certifications returned</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-xl p-12 text-center">
                <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  Upload Your CV
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  Accepted formats: PDF or TXT • Max 10MB
                </p>

                <input
                  type="file"
                  accept=".pdf,.txt"
                  onChange={handleFileChange}
                  className="hidden"
                  id="ml-cv-upload"
                  ref={fileInputRef}
                />
                <label
                  htmlFor="ml-cv-upload"
                  className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer transition-colors"
                >
                  <FileText className="w-5 h-5" />
                  Choose File
                </label>

                {selectedFile && (
                  <div className="mt-4 flex items-center justify-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                    <CheckCircle className="w-5 h-5 text-green-600" />
                    {selectedFile.name}
                  </div>
                )}

                <button
                  onClick={handleAnalyzeWithML}
                  disabled={!selectedFile || isAnalyzing}
                  className="mt-6 w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 px-6 rounded-lg font-semibold hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all"
                >
                  {isAnalyzing ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent" />
                      <span>Analyzing with ML...</span>
                    </>
                  ) : (
                    <>
                      <Rocket className="w-5 h-5" />
                      <span>Analyze with ML</span>
                    </>
                  )}
                </button>

                {error && (
                  <div className="mt-4 flex items-center justify-center gap-2 text-sm text-red-600 dark:text-red-400">
                    <AlertCircle className="w-5 h-5" />
                    {error}
                  </div>
                )}
              </div>
            </motion.div>
          )
        ) : (
          <div className="space-y-6">
            {/* Metadata */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl p-6 shadow-xl"
            >
              <div className="flex items-center gap-3 mb-4">
                <Brain className="w-8 h-8" />
                <div>
                  <h2 className="text-2xl font-bold">ML Analysis Complete</h2>
                  <p className="text-purple-100">{guidance.metadata.ml_model}</p>
                </div>
                <button
                  onClick={resetWorkflow}
                  className="ml-auto px-4 py-2 bg-white/20 text-white rounded-lg hover:bg-white/30 transition"
                >
                  Analyze Another CV
                </button>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                <div>
                  <div className="text-3xl font-bold">{guidance.metadata.cv_skills_count}</div>
                  <div className="text-sm text-purple-100">Skills Found</div>
                </div>
                <div>
                  <div className="text-3xl font-bold">{guidance.metadata.jobs_recommended}</div>
                  <div className="text-sm text-purple-100">Jobs Matched</div>
                </div>
                <div>
                  <div className="text-3xl font-bold">{guidance.metadata.certs_recommended}</div>
                  <div className="text-sm text-purple-100">Certs Ranked</div>
                </div>
                <div>
                  <div className="text-3xl font-bold">{guidance.metadata.roadmap_phases}</div>
                  <div className="text-sm text-purple-100">Roadmap Phases</div>
                </div>
                <div>
                  <div className="text-3xl font-bold">{guidance.metadata.processing_time_seconds.toFixed(1)}s</div>
                  <div className="text-sm text-purple-100">Processing Time</div>
                </div>
              </div>
              {guidance.metadata.timestamp && (
                <p className="mt-4 text-sm text-purple-100">
                  Generated on {new Date(guidance.metadata.timestamp).toLocaleString()}
                </p>
              )}
            </motion.div>

            {/* Job Recommendations */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-6"
            >
              <div className="flex items-center gap-3 mb-6">
                <Briefcase className="w-6 h-6 text-blue-600" />
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                  💼 ML-Powered Job Recommendations
                </h2>
              </div>

              {guidance.job_recommendations.length === 0 ? (
                <div className="text-center text-gray-500 dark:text-gray-400 py-8">
                  No jobs matched the ML similarity threshold (60%). Try adding more skills to your CV.
                </div>
              ) : (
                <div className="space-y-4">
                  {guidance.job_recommendations.map((job, index) => {
                    const matchedSkills = job.matched_skills && job.matched_skills.length > 0 ? job.matched_skills : job.matching_skills;
                    const gapSkills = job.gap_skills && job.gap_skills.length > 0 ? job.gap_skills : job.skill_gaps;
                    const cleanDescription = sanitizeRichText(job.description);
                    const postedLabel = job.posting_date || job.posted_date ? formatPostedDate(job.posting_date || job.posted_date) : null;
                    return (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="border-2 border-gray-200 dark:border-gray-700 rounded-xl p-6 hover:border-blue-400 dark:hover:border-blue-600 transition-colors shadow-sm"
                      >
                        <div className="flex flex-col gap-4 md:flex-row md:justify-between mb-4">
                          <div className="flex items-start gap-4">
                            {job.source_logo ? (
                              <img src={job.source_logo} alt={job.source || 'Job source'} className="w-12 h-12 rounded-full border border-gray-200 dark:border-gray-700 object-contain bg-white" />
                            ) : (
                              <div className="w-12 h-12 rounded-full bg-blue-600 text-white flex items-center justify-center text-sm font-semibold">
                                {(job.source || 'Live')[0]}
                              </div>
                            )}
                            <div>
                              <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
                                {job.source || 'Live Job Feed'} • {(job.location_type || 'On-site')}
                              </p>
                              <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                                {job.title}
                              </h3>
                              <div className="flex flex-wrap gap-2 mt-2">
                                <span className={`px-3 py-1 rounded-full text-xs font-semibold ${getGrowthColor(job.growth_potential)}`}>
                                  {job.growth_potential} Growth
                                </span>
                                {job.tags?.slice(0, 2).map((tag, tagIndex) => (
                                  <span key={`${tag}-${tagIndex}`} className="px-3 py-1 rounded-full text-xs font-semibold bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300">
                                    {tag}
                                  </span>
                                ))}
                              </div>
                              <div className="mt-3 flex flex-wrap items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
                                <span className="inline-flex items-center gap-2">
                                  <MapPin className="w-4 h-4 text-blue-500" />
                                  {job.location || 'Global'}
                                </span>
                                {postedLabel && (
                                  <span className="inline-flex items-center gap-2">
                                    <Clock className="w-4 h-4 text-gray-500" />
                                    {postedLabel}
                                  </span>
                                )}
                              </div>
                            </div>
                          </div>
                          <div className="flex flex-col items-start md:items-end gap-2">
                            <div className="text-2xl font-bold text-green-600">
                              {formatSalaryRange(job.predicted_salary)}
                            </div>
                            <p className="text-xs text-gray-500 dark:text-gray-400">ML-predicted salary band</p>
                            {job.job_url ? (
                              <a
                                href={job.job_url}
                                target="_blank"
                                rel="noreferrer"
                                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-semibold hover:bg-blue-700"
                              >
                                <ExternalLink className="w-4 h-4" />
                                Apply on {job.source || 'Source'}
                              </a>
                            ) : (
                              <button
                                type="button"
                                className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300 text-sm font-semibold cursor-not-allowed"
                                disabled
                              >
                                <ExternalLink className="w-4 h-4" />
                                Listing Unavailable
                              </button>
                            )}
                          </div>
                        </div>

                        {cleanDescription && (
                          <p className="text-sm text-gray-600 dark:text-gray-300 mb-4 leading-relaxed">
                            {truncateText(cleanDescription)}
                          </p>
                        )}

                        {job.tags && job.tags.length > 0 && (
                          <div className="flex flex-wrap gap-2 mb-4">
                            {job.tags.map((tag, tagIndex) => (
                              <span key={`${tag}-${tagIndex}`} className="inline-flex items-center gap-1 px-3 py-1 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 text-xs font-semibold rounded-full">
                                <Tag className="w-3 h-3" />
                                {tag}
                              </span>
                            ))}
                          </div>
                        )}

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                          <div className="rounded-xl bg-green-50 dark:bg-green-900/10 p-4 border border-green-100 dark:border-green-900/40">
                            <div className="flex items-center justify-between">
                              <p className="text-sm font-semibold text-green-800 dark:text-green-300">Matched Skills</p>
                              <span className="text-xs text-green-600 dark:text-green-200 font-semibold">{matchedSkills.length}</span>
                            </div>
                            <div className="flex flex-wrap gap-2 mt-3">
                              {matchedSkills.slice(0, 8).map((skill, i) => (
                                <span key={i} className="px-2 py-1 bg-white/80 dark:bg-green-900/40 text-green-800 dark:text-green-200 text-xs rounded-full">
                                  {skill}
                                </span>
                              ))}
                            </div>
                          </div>
                          <div className="rounded-xl bg-orange-50 dark:bg-orange-900/10 p-4 border border-orange-100 dark:border-orange-900/40">
                            <div className="flex items-center justify-between">
                              <p className="text-sm font-semibold text-orange-800 dark:text-orange-300">Priority Gaps</p>
                              <span className="text-xs text-orange-600 dark:text-orange-200 font-semibold">{gapSkills.length}</span>
                            </div>
                            <div className="flex flex-wrap gap-2 mt-3">
                              {gapSkills.slice(0, 8).map((skill, i) => (
                                <span key={i} className="px-2 py-1 bg-white/80 dark:bg-orange-900/40 text-orange-800 dark:text-orange-200 text-xs rounded-full">
                                  {skill}
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                          <div>
                            <div className="flex items-center gap-2 text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                              <Target className="w-4 h-4" />
                              ML Similarity: {(job.similarity_score * 100).toFixed(1)}%
                            </div>
                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                              <div
                                className="bg-blue-600 h-2 rounded-full"
                                style={{ width: `${job.similarity_score * 100}%` }}
                              />
                            </div>
                          </div>
                          <div>
                            <div className="flex items-center gap-2 text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                              <Brain className="w-4 h-4" />
                              ML Confidence: {(job.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                              <div
                                className="bg-green-600 h-2 rounded-full"
                                style={{ width: `${job.confidence * 100}%` }}
                              />
                            </div>
                          </div>
                        </div>

                        <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
                          <div className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                            🤖 ML Reasoning:
                          </div>
                          <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                            {job.reasons.map((reason, i) => {
                              const cleanReason = sanitizeRichText(reason) || reason;
                              return <li key={i}>• {cleanReason}</li>;
                            })}
                          </ul>
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              )}
            </motion.div>

            {/* Certifications */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-6"
            >
              <div className="flex items-center gap-3 mb-6">
                <GraduationCap className="w-6 h-6 text-purple-600" />
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                  🎓 ML-Ranked Certifications
                </h2>
              </div>

              <div className="grid gap-4">
                {guidance.certification_recommendations.map((cert, index) => {
                  const primaryLink = cert.official_url || cert.url;
                  const relevancePercent = `${(cert.relevance_score * 100).toFixed(1)}%`;
                  const alignmentPercent = `${(cert.skill_alignment * 100).toFixed(1)}%`;
                  const resourceCount = cert.resources?.length || 0;
                  const providerLabel = cert.provider || 'Top Provider';
                  const prerequisiteSource = cert.prerequisites ?? cert.prerequisite_details ?? null;
                  const prerequisiteList = normalizeStringList(prerequisiteSource);
                  const examFormatLabel = sanitizeRichText(cert.exam_format || cert.examFormat || '') || '';
                  const hasDetailPanel = prerequisiteList.length > 0 || Boolean(examFormatLabel);
                  return (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="border-2 border-gray-200 dark:border-gray-700 rounded-lg p-5 hover:border-purple-400 dark:hover:border-purple-600 transition-colors"
                    >
                      <div className="flex justify-between items-start mb-4 gap-4">
                        <div className="flex items-start gap-3 flex-1">
                          {cert.provider_logo ? (
                            <img src={cert.provider_logo} alt={providerLabel} className="w-12 h-12 rounded-full border border-gray-200 dark:border-gray-700 object-contain bg-white" />
                          ) : (
                            <div className="w-12 h-12 rounded-full bg-purple-600 text-white flex items-center justify-center text-sm font-semibold">
                              {providerLabel[0]}
                            </div>
                          )}
                          <div>
                            <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400">
                              {providerLabel}{cert.delivery_format ? ` • ${cert.delivery_format}` : ''}
                            </p>
                            <h3 className="text-2xl font-bold text-gray-900 dark:text-white">
                              {cert.name}
                            </h3>
                            <div className="flex flex-wrap gap-2 mt-2">
                              {cert.career_boost && (
                                <span className="px-3 py-1 rounded-full bg-orange-50 dark:bg-orange-900/20 text-orange-700 dark:text-orange-300 text-xs font-semibold">
                                  {cert.career_boost}
                                </span>
                              )}
                              {cert.cost_estimate && (
                                <span className="px-3 py-1 rounded-full bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300 text-xs font-semibold">
                                  {cert.cost_estimate}
                                </span>
                              )}
                              {cert.cost_type && (
                                <span className="px-3 py-1 rounded-full bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 text-xs font-semibold">
                                  {cert.cost_type}
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                        <div className="text-right">
                          <span className="px-3 py-1 bg-purple-100 dark:bg-purple-900/20 text-purple-700 dark:text-purple-400 text-xs font-semibold rounded-full inline-flex items-center gap-1">
                            #{index + 1}
                          </span>
                          <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">{resourceCount} linked resources</p>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                        <div>
                          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">ML Relevance</div>
                          <div className="text-lg font-bold text-purple-600">{relevancePercent}</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Skill Alignment</div>
                          <div className="text-lg font-bold text-blue-600">{alignmentPercent}</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Predicted ROI</div>
                          <div className="text-sm font-semibold text-green-600">{cert.predicted_roi}</div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Est. Time</div>
                          <div className="text-sm font-semibold text-gray-700 dark:text-gray-300">{cert.estimated_time}</div>
                        </div>
                      </div>

                      <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3">
                        <div className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                          💡 Why this cert:
                        </div>
                        <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                          {cert.reasons.slice(0, 3).map((reason, i) => {
                            const cleanReason = sanitizeRichText(reason) || reason;
                            return <li key={i}>• {cleanReason}</li>;
                          })}
                        </ul>
                      </div>

                      {hasDetailPanel && (
                        <div className="mt-4 border border-dashed border-purple-200 dark:border-purple-700 rounded-lg bg-white dark:bg-gray-900/20">
                          <button
                            type="button"
                            onClick={() => toggleCertDetails(index)}
                            className="w-full flex items-center justify-between px-4 py-3 text-sm font-semibold text-purple-700 dark:text-purple-200"
                          >
                            <span className="inline-flex items-center gap-2">
                              <AlertCircle className="w-4 h-4" />
                              Prerequisites & Exam Format
                            </span>
                            {certDetailPanels[index] ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                          </button>
                          <AnimatePresence>
                            {certDetailPanels[index] && (
                              <motion.div
                                initial={{ opacity: 0, height: 0 }}
                                animate={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                className="px-4 pb-4 space-y-3 text-sm text-gray-700 dark:text-gray-300"
                              >
                                {prerequisiteList.length > 0 && (
                                  <div>
                                    <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">Prerequisites</p>
                                    <ul className="list-disc pl-5 space-y-1">
                                      {prerequisiteList.map((item, prereqIndex) => (
                                        <li key={prereqIndex}>{item}</li>
                                      ))}
                                    </ul>
                                  </div>
                                )}
                                {examFormatLabel && (
                                  <div>
                                    <p className="text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-1">Exam Format</p>
                                    <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-200 text-xs font-semibold">
                                      {examFormatLabel}
                                    </span>
                                  </div>
                                )}
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </div>
                      )}

                      <div className="flex flex-wrap items-center gap-3 mt-4">
                        {primaryLink && (
                          <a
                            href={primaryLink}
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg bg-purple-600 text-white text-sm font-semibold hover:bg-purple-700"
                          >
                            <ExternalLink className="w-4 h-4" />
                            Visit Official Page
                          </a>
                        )}
                        {resourceCount > 0 && (
                          <button
                            onClick={() => toggleCertResources(index)}
                            className="inline-flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 text-sm font-semibold text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700"
                          >
                            <Link className="w-4 h-4" />
                            {expandedCerts[index] ? 'Hide Resources' : 'Show Resources'}
                          </button>
                        )}
                      </div>

                      <AnimatePresence>
                        {expandedCerts[index] && cert.resources && cert.resources.length > 0 && (
                          <motion.div
                            initial={{ opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: 'auto' }}
                            exit={{ opacity: 0, height: 0 }}
                            className="mt-3 space-y-2"
                          >
                            {cert.resources.map((resource, resourceIndex) => {
                              const resourceLink = resource.link || resource.url;
                              const resourceTitle = sanitizeRichText(resource.title) || resource.title;
                              const resourceProvider = sanitizeRichText(resource.provider) || resource.provider || 'Trusted resource';
                              const resourceFormat = sanitizeRichText(resource.format) || resource.format || '';
                              const resourceType = sanitizeRichText(resource.type) || resource.type || '';
                              const resourceCost = sanitizeRichText(resource.cost) || resource.cost || '';
                              return (
                                <div key={resourceIndex} className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 bg-white dark:bg-gray-900/30 border border-dashed border-purple-200 dark:border-purple-800 rounded-lg p-3">
                                  <div>
                                    <p className="text-sm font-semibold text-gray-900 dark:text-white">{resourceTitle}</p>
                                    <p className="text-xs text-gray-500 dark:text-gray-400">
                                      {resourceProvider}{resourceFormat ? ` • ${resourceFormat}` : ''}
                                    </p>
                                    <div className="flex flex-wrap gap-2 mt-2">
                                      {resourceType && (
                                        <span className="px-2 py-0.5 rounded-full bg-purple-50 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 text-[10px] font-semibold">
                                          {resourceType}
                                        </span>
                                      )}
                                      {resourceCost && (
                                        <span className="px-2 py-0.5 rounded-full bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300 text-[10px] font-semibold">
                                          {resourceCost}
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                  <div className="flex items-center gap-3 text-xs text-gray-600 dark:text-gray-300">
                                    {resourceLink && (
                                      <a
                                        href={resourceLink}
                                        target="_blank"
                                        rel="noreferrer"
                                        className="inline-flex items-center gap-1 text-purple-600 hover:text-purple-700"
                                      >
                                        <ExternalLink className="w-3 h-3" />
                                        Open
                                      </a>
                                    )}
                                  </div>
                                </div>
                              );
                            })}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>

            <Roadmap roadmap={guidance.learning_roadmap} onDownload={handleDownloadRoadmap} />

            {cvAnalysis && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow"
              >
                <div className="flex flex-wrap items-center gap-6 text-sm text-gray-600 dark:text-gray-300">
                  <div>
                    <p className="font-semibold text-gray-900 dark:text-white">CV Analyzed</p>
                    <p>{cvAnalysis.personal_info?.name || cvAnalysis.name || 'Professional'}</p>
                  </div>
                  <div>
                    <p className="font-semibold text-gray-900 dark:text-white">Skills</p>
                    <p>{cvAnalysis.skills?.length || 0} total</p>
                  </div>
                  <div>
                    <p className="font-semibold text-gray-900 dark:text-white">Work History Entries</p>
                    <p>{cvAnalysis.work_history?.length || 0}</p>
                  </div>
                  <div>
                    <p className="font-semibold text-gray-900 dark:text-white">Certifications Detected</p>
                    <p>{cvAnalysis.certifications?.length || 0}</p>
                  </div>
                </div>
              </motion.div>
            )}

            {/* XAI Insights */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-white dark:bg-gray-800 rounded-xl shadow-xl p-6"
            >
              <button
                onClick={() => setExpandedXAI(!expandedXAI)}
                className="w-full flex items-center justify-between mb-4"
              >
                <div className="flex items-center gap-3">
                  <Lightbulb className="w-6 h-6 text-yellow-600" />
                  <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
                    🧠 Explainable AI (XAI) Insights
                  </h2>
                </div>
                {expandedXAI ? <ChevronUp /> : <ChevronDown />}
              </button>

              <AnimatePresence>
                {expandedXAI && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="space-y-4"
                  >
                    <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                      <div className="font-semibold text-gray-900 dark:text-white mb-2">
                        💡 Key Insights:
                      </div>
                      <ul className="space-y-1 text-sm text-gray-700 dark:text-gray-300">
                        {guidance.xai_insights.key_insights.map((insight, i) => (
                          <li key={i}>• {insight}</li>
                        ))}
                      </ul>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
                        <div className="font-semibold text-gray-900 dark:text-white mb-2 text-sm">
                          📊 ML Confidence:
                        </div>
                        <div className="space-y-1 text-xs text-gray-600 dark:text-gray-400">
                          {Object.entries(guidance.xai_insights.ml_confidence_scores).map(([key, value]) => {
                            const displayValue = typeof value === 'string' || typeof value === 'number'
                              ? value
                              : JSON.stringify(value);

                            return (
                              <div key={key}>• {key}: {displayValue}</div>
                            );
                          })}
                        </div>
                      </div>

                      <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
                        <div className="font-semibold text-gray-900 dark:text-white mb-2 text-sm">
                          🔍 ML Model:
                        </div>
                        <div className="text-xs text-gray-600 dark:text-gray-400">
                          {guidance.xai_insights.how_we_analyzed_your_cv.model}
                        </div>
                      </div>

                      <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
                        <div className="font-semibold text-gray-900 dark:text-white mb-2 text-sm">
                          📈 Engine:
                        </div>
                        <div className="text-xs text-gray-600 dark:text-gray-400">
                          {guidance.metadata.engine_version}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Reset Button */}
            <div className="flex justify-center">
              <button
                onClick={resetWorkflow}
                className="bg-gray-200 dark:bg-gray-700 text-gray-900 dark:text-white px-6 py-3 rounded-lg font-semibold hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
              >
                Analyze Another CV
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
