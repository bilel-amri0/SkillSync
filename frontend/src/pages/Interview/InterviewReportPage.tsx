import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Award, 
  TrendingUp, 
  TrendingDown, 
  Lightbulb, 
  FileText, 
  ArrowLeft,
  Loader2,
  CheckCircle,
  XCircle
} from 'lucide-react';
import toast from 'react-hot-toast';
import { 
  interviewService, 
  handleInterviewError,
  InterviewReportResponse 
} from '../../services/interviewService';

const InterviewReportPage = () => {
  const { interviewId } = useParams<{ interviewId: string }>();
  const navigate = useNavigate();
  
  const [report, setReport] = useState<InterviewReportResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchReport = async () => {
      if (!interviewId) {
        setError('No interview ID provided');
        setIsLoading(false);
        return;
      }

      try {
        const data = await interviewService.getReport(interviewId);
        setReport(data);
      } catch (err) {
        const errorMsg = handleInterviewError(err);
        setError(errorMsg);
        toast.error(errorMsg);
      } finally {
        setIsLoading(false);
      }
    };

    fetchReport();
  }, [interviewId]);

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-600 dark:text-green-400';
    if (score >= 60) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getScoreBackground = (score: number) => {
    if (score >= 80) return 'bg-green-100 dark:bg-green-900';
    if (score >= 60) return 'bg-yellow-100 dark:bg-yellow-900';
    return 'bg-red-100 dark:bg-red-900';
  };

  if (isLoading) {
    return (
      <div className="max-w-6xl mx-auto py-8">
        <div className="flex flex-col items-center justify-center py-20">
          <Loader2 className="w-12 h-12 text-blue-600 animate-spin mb-4" />
          <p className="text-gray-600 dark:text-gray-300">Loading your interview report...</p>
        </div>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="max-w-6xl mx-auto py-8">
        <div className="bg-red-50 dark:bg-red-900 border border-red-200 dark:border-red-700 rounded-lg p-6">
          <div className="flex items-center mb-4">
            <XCircle className="w-6 h-6 text-red-600 dark:text-red-400 mr-3" />
            <h2 className="text-xl font-bold text-red-900 dark:text-red-100">Error Loading Report</h2>
          </div>
          <p className="text-red-700 dark:text-red-300 mb-4">{error || 'Failed to load interview report'}</p>
          <button
            onClick={() => navigate('/interview')}
            className="flex items-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Interview
          </button>
        </div>
      </div>
    );
  }

  const { analysis, transcript } = report;

  return (
    <div className="max-w-6xl mx-auto py-8 space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6"
      >
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Interview Performance Report
          </h1>
          <button
            onClick={() => navigate('/interview')}
            className="flex items-center px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            New Interview
          </button>
        </div>
        <p className="text-gray-600 dark:text-gray-300">
          Interview ID: {interviewId}
        </p>
      </motion.div>

      {/* Score Card */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
        className={`${getScoreBackground(analysis.overall_score)} rounded-lg shadow-lg p-8`}
      >
        <div className="flex items-center justify-center mb-4">
          <Award className={`w-16 h-16 ${getScoreColor(analysis.overall_score)}`} />
        </div>
        <h2 className="text-center text-4xl font-bold text-gray-900 dark:text-white mb-2">
          {analysis.overall_score.toFixed(0)}%
        </h2>
        <p className="text-center text-gray-700 dark:text-gray-300">
          Overall Performance Score
        </p>
      </motion.div>

      {/* Summary */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6"
      >
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4 flex items-center">
          <FileText className="w-6 h-6 mr-3" />
          Summary
        </h2>
        <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
          {analysis.summary}
        </p>
      </motion.div>

      {/* Strengths and Weaknesses */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Strengths */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6"
        >
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-3 text-green-600" />
            Strengths
          </h2>
          <ul className="space-y-3">
            {analysis.strengths.map((strength, index) => (
              <li key={index} className="flex items-start">
                <CheckCircle className="w-5 h-5 text-green-600 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-gray-700 dark:text-gray-300">{strength}</span>
              </li>
            ))}
          </ul>
        </motion.div>

        {/* Weaknesses */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6"
        >
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center">
            <TrendingDown className="w-5 h-5 mr-3 text-orange-600" />
            Areas for Improvement
          </h2>
          <ul className="space-y-3">
            {analysis.weaknesses.map((weakness, index) => (
              <li key={index} className="flex items-start">
                <XCircle className="w-5 h-5 text-orange-600 mr-3 mt-0.5 flex-shrink-0" />
                <span className="text-gray-700 dark:text-gray-300">{weakness}</span>
              </li>
            ))}
          </ul>
        </motion.div>
      </div>

      {/* Recommendations */}
      {analysis.recommendations && analysis.recommendations.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6"
        >
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4 flex items-center">
            <Lightbulb className="w-6 h-6 mr-3 text-yellow-600" />
            Recommendations
          </h2>
          <ul className="space-y-3">
            {analysis.recommendations.map((recommendation, index) => (
              <li key={index} className="flex items-start">
                <span className="flex-shrink-0 w-6 h-6 bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300 rounded-full flex items-center justify-center text-sm font-medium mr-3">
                  {index + 1}
                </span>
                <span className="text-gray-700 dark:text-gray-300">{recommendation}</span>
              </li>
            ))}
          </ul>
        </motion.div>
      )}

      {/* Transcript */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6"
      >
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
          Interview Transcript
        </h2>
        <div className="space-y-6">
          {transcript.map((item, index) => (
            <div key={index} className="border-l-4 border-blue-500 pl-4 py-2">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  Question {item.question_id}
                </h3>
                <span className="text-xs px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 rounded">
                  {item.category}
                </span>
              </div>
              <p className="text-gray-700 dark:text-gray-300 mb-3">
                {item.question_text}
              </p>
              <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-2">
                  Your Answer:
                </p>
                <p className="text-gray-800 dark:text-gray-200">
                  {item.answer_text}
                </p>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Action Button */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
        className="flex justify-center pb-8"
      >
        <button
          onClick={() => navigate('/interview')}
          className="px-8 py-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors shadow-lg"
        >
          Start New Interview
        </button>
      </motion.div>
    </div>
  );
};

export default InterviewReportPage;
