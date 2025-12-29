import { useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { useEffect } from 'react';
import {
  ResponsiveContainer,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend as RechartsLegend,
  BarChart,
  Bar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import { 
  FileText, 
  Briefcase, 
  TrendingUp, 
  Clock,
  Award,
  Target,
  BarChart3,
  ArrowRight,
  CheckCircle
} from 'lucide-react';
import { analyticsApi, cvApi, handleApiError } from '../services/api';
import toast from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';

// Quick actions configuration
const quickActions = [
  {
    title: 'Upload CV',
    description: 'Analyze your latest CV',
    icon: FileText,
    href: '/cv-analysis',
    color: 'bg-blue-500'
  },
  {
    title: 'Find Jobs',
    description: 'Search for matching opportunities',
    icon: Briefcase,
    href: '/job-matching',
    color: 'bg-green-500'
  },
  {
    title: 'Translate Experience',
    description: 'Enhance your experience descriptions',
    icon: Target,
    href: '/experience-translator',
    color: 'bg-purple-500'
  },
  {
    title: 'View Analytics',
    description: 'Track your progress',
    icon: BarChart3,
    href: '/analytics',
    color: 'bg-orange-500'
  }
];

const Dashboard = () => {
  const navigate = useNavigate();

  const {
    data: dashboardData,
    isLoading: isDashboardLoading,
    error: dashboardError,
  } = useQuery({
    queryKey: ['dashboard-latest'],
    queryFn: () => analyticsApi.getDashboard(),
    staleTime: 60_000,
  });

  const {
    data: cvAnalyses,
    isLoading: isAnalysesLoading,
    error: analysesError,
  } = useQuery({
    queryKey: ['cv-analyses'],
    queryFn: () => cvApi.listAnalyses(10),
    staleTime: 60_000,
  });

  const isLoading = isDashboardLoading || isAnalysesLoading;
  const error = dashboardError || analysesError;

  useEffect(() => {
    if (!error) {
      return;
    }

    const message = handleApiError(error) || 'Failed to load dashboard data';
    toast.error(message);
  }, [error]);

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'cv_upload':
        return <FileText className="h-4 w-4" />;
      case 'job_match':
        return <Briefcase className="h-4 w-4" />;
      case 'skill_improvement':
        return <TrendingUp className="h-4 w-4" />;
      default:
        return <CheckCircle className="h-4 w-4" />;
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-24 bg-gray-200 rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-6 text-red-800 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
        <h2 className="text-lg font-semibold mb-2">Dashboard unavailable</h2>
        <p className="text-sm">
          We could not connect to the analytics service. Please verify the backend server is running and reload the page.
        </p>
      </div>
    );
  }

  const totalCVs = dashboardData?.overview?.total_cvs ?? (cvAnalyses?.total ?? 0);
  const jobsAnalyzed = dashboardData?.overview?.jobs_analyzed ?? 0;
  const skillsIdentified = dashboardData?.overview?.skills_identified ?? 0;
  const matchScoreAverage = Math.round(dashboardData?.overview?.match_score_avg ?? 0);

  const overview = {
    total_cvs: dashboardData?.overview?.total_cvs ?? totalCVs,
    jobs_analyzed: dashboardData?.overview?.jobs_analyzed ?? jobsAnalyzed,
    skills_identified: dashboardData?.overview?.skills_identified ?? skillsIdentified,
    match_score_avg: dashboardData?.overview?.match_score_avg ?? matchScoreAverage,
    growth_rate: dashboardData?.overview?.growth_rate ?? 0,
  };

  const analyticsActivity = dashboardData?.recent_activities?.map((activity) => ({
    id: activity.id,
    type: activity.type,
    description: activity.description,
    timestamp: activity.timestamp,
    status: activity.status,
    details: activity.status,
  })) ?? [];

  const analysisActivity = cvAnalyses?.analyses?.map((analysis) => ({
    id: analysis.analysis_id,
    type: 'cv_upload',
    description: `Analysis ${analysis.analysis_id.slice(0, 8)} • ${analysis.summary ?? 'CV'}`,
    timestamp: analysis.created_at,
    status: 'completed',
    details: `${analysis.skills.length} skills identified`,
  })) ?? [];

  const recentActivity = [...analyticsActivity, ...analysisActivity]
    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
    .slice(0, 8);

  const skillDistribution = dashboardData?.skill_distribution ?? [];
  const skillProgress = dashboardData?.skill_progress ?? [];
  const skillHighlights = cvAnalyses?.analyses?.[0]?.skills ?? [];

  const radarData = skillProgress
    .map((item) => ({
      skill: item.skill,
      matched: item.current,
      gap: Math.max(item.target - item.current, 0),
    }))
    .filter((entry) => entry.matched > 0 || entry.gap > 0);

  const hasSkillDistribution = skillDistribution.length > 0;
  const hasRadarData = radarData.length > 0;

  return (
    <div className="space-y-6">
      {/* Welcome Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg p-6 text-white"
      >
        <h1 className="text-2xl font-bold mb-2">Welcome back!</h1>
        <p className="text-blue-100">
          Your career development journey continues. Here's what's happening with your profile.
        </p>
      </motion.div>

      {/* Analytics Highlights */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        <motion.div
          whileHover={{ y: -4, scale: 1.02 }}
          className="relative bg-white/70 dark:bg-slate-800/70 backdrop-blur-xl rounded-2xl p-6 border border-white/20 dark:border-slate-700/50 shadow-xl overflow-hidden group"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
          <div className="relative">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl shadow-lg shadow-blue-500/30">
                <FileText className="h-6 w-6 text-white" />
              </div>
              {overview.growth_rate > 0 && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.2, type: 'spring' }}
                  className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-full text-xs font-bold flex items-center space-x-1"
                >
                  <TrendingUp className="h-3 w-3" />
                  <span>+{overview.growth_rate}%</span>
                </motion.div>
              )}
            </div>

            {skillHighlights.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-4">
                {skillHighlights.slice(0, 8).map((skill, idx) => (
                  <span
                    key={`${skill}-${idx}`}
                    className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full text-xs font-semibold"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            )}

            <div className="space-y-1">
              <p className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                {overview.total_cvs.toLocaleString()}
              </p>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">CVs Analyzed</p>
            </div>
          </div>
        </motion.div>

        <motion.div
          whileHover={{ y: -4, scale: 1.02 }}
          className="relative bg-white/70 dark:bg-slate-800/70 backdrop-blur-xl rounded-2xl p-6 border border-white/20 dark:border-slate-700/50 shadow-xl overflow-hidden group"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
          <div className="relative">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl shadow-lg shadow-emerald-500/30">
                <Target className="h-6 w-6 text-white" />
              </div>
              {overview.jobs_analyzed > 0 && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.3, type: 'spring' }}
                  className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-full text-xs font-bold flex items-center space-x-1"
                >
                  <TrendingUp className="h-3 w-3" />
                  <span>+{Math.max(1, Math.round(overview.jobs_analyzed * 0.1))}%</span>
                </motion.div>
              )}
            </div>
            <div className="space-y-1">
              <p className="text-4xl font-bold bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text text-transparent">
                {overview.jobs_analyzed.toLocaleString()}
              </p>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Jobs Analyzed</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Avg Match Score: <span className="font-semibold text-gray-900 dark:text-white">{overview.match_score_avg}%</span></p>
            </div>
          </div>
        </motion.div>

        <motion.div
          whileHover={{ y: -4, scale: 1.02 }}
          className="relative bg-white/70 dark:bg-slate-800/70 backdrop-blur-xl rounded-2xl p-6 border border-white/20 dark:border-slate-700/50 shadow-xl overflow-hidden group"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
          <div className="relative">
            <div className="flex items-center justify-between mb-4">
              <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-600 rounded-xl shadow-lg shadow-purple-500/30">
                <Award className="h-6 w-6 text-white" />
              </div>
              {overview.skills_identified > 0 && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.4, type: 'spring' }}
                  className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-full text-xs font-bold flex items-center space-x-1"
                >
                  <TrendingUp className="h-3 w-3" />
                  <span>+{Math.min(overview.skills_identified, 25)}%</span>
                </motion.div>
              )}
            </div>
            <div className="space-y-1">
              <p className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                {overview.skills_identified.toLocaleString()}
              </p>
              <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Skills Identified</p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Analytics Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-white/70 dark:bg-slate-800/70 backdrop-blur-xl rounded-2xl border border-white/20 dark:border-slate-700/50 shadow-xl p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-lg">
                <Award className="h-5 w-5 text-white" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-gray-900 dark:text-white">Skill Portfolio</h2>
                <p className="text-xs text-gray-500 dark:text-gray-400">Top categories from recent CVs</p>
              </div>
            </div>
          </div>
          <div className="h-80">
            {hasSkillDistribution ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={skillDistribution} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#cbd5f5" />
                  <XAxis dataKey="category" stroke="#64748b" tick={{ fontSize: 12 }} />
                  <YAxis allowDecimals={false} stroke="#64748b" />
                  <RechartsTooltip />
                  <Bar dataKey="count" fill="#6366f1" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-sm text-gray-500 dark:text-gray-400">
                Upload CVs to unlock skill insights.
              </div>
            )}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="bg-white/70 dark:bg-slate-800/70 backdrop-blur-xl rounded-2xl border border-white/20 dark:border-slate-700/50 shadow-xl p-6"
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-lg">
                <Target className="h-5 w-5 text-white" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-gray-900 dark:text-white">Skill Readiness</h2>
                <p className="text-xs text-gray-500 dark:text-gray-400">Matched vs. Missing skills</p>
              </div>
            </div>
          </div>
          <div className="h-80">
            {hasRadarData ? (
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData} outerRadius="70%">
                  <PolarGrid stroke="#cbd5f5" />
                  <PolarAngleAxis dataKey="skill" tick={{ fontSize: 12 }} stroke="#64748b" />
                  <PolarRadiusAxis angle={45} stroke="#94a3b8" tick={{ fontSize: 10 }} />
                  <RechartsTooltip />
                  <Radar name="Matched" dataKey="matched" stroke="#34d399" fill="#34d399" fillOpacity={0.4} />
                  <Radar name="Gaps" dataKey="gap" stroke="#f87171" fill="#f87171" fillOpacity={0.3} />
                  <RechartsLegend />
                </RadarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-sm text-gray-500 dark:text-gray-400">
                Generate CV insights to compare matched vs. missing skills.
              </div>
            )}
          </div>
        </motion.div>
      </div>

      {/* Quick Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700"
      >
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {quickActions.map((action, index) => (
            <motion.button
              key={action.title}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.9 + index * 0.1 }}
              onClick={() => navigate(action.href)}
              className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-blue-300 dark:hover:border-blue-600 hover:shadow-md transition-all group text-left"
            >
              <div className="flex items-center mb-3">
                <div className={`p-2 ${action.color} rounded-lg text-white`}>
                  <action.icon className="h-5 w-5" />
                </div>
                <ArrowRight className="ml-auto h-4 w-4 text-gray-400 group-hover:text-blue-500 transition-colors" />
              </div>
              <h3 className="font-medium text-gray-900 dark:text-white mb-1">{action.title}</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">{action.description}</p>
            </motion.button>
          ))}
        </div>
      </motion.div>

      {/* Recent Activity */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700"
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Recent Activity</h2>
          <Clock className="h-5 w-5 text-gray-400" />
        </div>
        <div className="space-y-4">
          {recentActivity.map((activity, index) => (
            <motion.div
              key={activity.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.9 + index * 0.1 }}
              className="flex items-center space-x-3 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            >
              <div className="flex-shrink-0 p-2 bg-blue-100 dark:bg-blue-900 rounded-lg text-blue-600 dark:text-blue-400">
                {getActivityIcon(activity.type)}
              </div>
              <div className="flex-1">
                <p className="text-sm text-gray-900 dark:text-white">{activity.description}</p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  {formatTimestamp(activity.timestamp)}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </div>
  );
};

export default Dashboard;