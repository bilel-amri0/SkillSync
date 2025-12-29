import React, { useState } from 'react';
import { Type, Mic, Briefcase, Target, Clock, ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { startInterview, type StartInterviewRequest } from '../api';
import type { CVAnalysisResponse } from '../api';
import { handleApiError } from '../services/api';

interface NewInterviewPageProps {
  cvData: CVAnalysisResponse | null;
}

interface ModeConfig {
  label: string;
  description: string;
  icon: React.ReactNode;
  features: string[];
  recommended: boolean;
}

const modeConfig: Record<'text' | 'voice', ModeConfig> = {
  text: {
    label: 'Text Interview',
    description: 'Type your answers to AI-generated questions',
    icon: <Type className="h-8 w-8" />,
    features: [
      'Take your time to think',
      'Edit answers before submitting',
      'Review questions multiple times',
      'Best for detailed responses'
    ],
    recommended: true
  },
  voice: {
    label: 'Live Voice Interview',
    description: 'Real-time conversation with AI interviewer',
    icon: <Mic className="h-8 w-8" />,
    features: [
      'Natural conversation flow',
      'Simulates real interview',
      'Immediate feedback',
      'Practice speaking skills'
    ],
    recommended: true
  }
};

export const NewInterviewPage: React.FC<NewInterviewPageProps> = ({ cvData }) => {
  const navigate = useNavigate();
  const [selectedMode, setSelectedMode] = useState<'text' | 'voice'>('text');
  const [jobTitle, setJobTitle] = useState('');
  const [jobDescription, setJobDescription] = useState('');
  const [difficulty, setDifficulty] = useState<'easy' | 'medium' | 'hard'>('medium');
  const [isStarting, setIsStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleStartInterview = async () => {
    if (!cvData) {
      setError('Please upload a CV first');
      return;
    }

    if (!jobTitle.trim()) {
      setError('Please enter a job title');
      return;
    }

    setIsStarting(true);
    setError(null);

    try {
      const payload: StartInterviewRequest = {
        user_id: cvData.analysis_id,
        cv_id: cvData.analysis_id,
        cv_text: cvData.summary || '',
        job_title: jobTitle,
        job_description: jobDescription || undefined,
        difficulty: difficulty,
        skills: cvData.skills || [],
        interview_mode: selectedMode
      };

      const response = await startInterview(payload);
      
      console.log('✅ Interview started:', response);
      
      // Navigate to appropriate interview page based on mode
      if (selectedMode === 'text') {
        navigate(`/interview/text/${response.interview_id}`);
      } else {
        navigate(`/interview/voice/${response.interview_id}`);
      }
    } catch (err) {
      console.error('❌ Failed to start interview:', err);
      setError(handleApiError(err));
    } finally {
      setIsStarting(false);
    }
  };

  if (!cvData) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-yellow-900 dark:text-yellow-100 mb-2">
            CV Required
          </h3>
          <p className="text-yellow-700 dark:text-yellow-300">
            Please upload your CV first to start an interview. The AI will use your skills and experience to generate relevant questions.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-3">
          AI Interview Preparation
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Practice with AI-powered interviews tailored to your target role
        </p>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4">
          <p className="text-red-700 dark:text-red-300">{error}</p>
        </div>
      )}

      {/* Interview Mode Selection */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Choose Interview Mode
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Object.entries(modeConfig).map(([mode, config]) => (
            <button
              key={mode}
              onClick={() => setSelectedMode(mode as 'text' | 'voice')}
              className={`relative p-6 rounded-xl border-2 transition-all text-left ${
                selectedMode === mode
                  ? 'border-blue-600 bg-blue-50 dark:bg-blue-900/20 shadow-lg'
                  : 'border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-700'
              }`}
            >
              {config.recommended && (
                <span className="absolute -top-3 left-4 px-3 py-1 bg-green-600 text-white text-xs font-semibold rounded-full">
                  Recommended
                </span>
              )}
              
              <div className="flex items-start space-x-4 mb-4">
                <div className={`p-3 rounded-lg ${
                  selectedMode === mode ? 'bg-blue-600 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                }`}>
                  {config.icon}
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-1">
                    {config.label}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {config.description}
                  </p>
                </div>
              </div>

              <ul className="space-y-2">
                {config.features.map((feature, idx) => (
                  <li key={idx} className="flex items-center text-sm text-gray-700 dark:text-gray-300">
                    <div className="w-1.5 h-1.5 bg-blue-600 rounded-full mr-2"></div>
                    {feature}
                  </li>
                ))}
              </ul>
            </button>
          ))}
        </div>
      </div>

      {/* Job Details Form */}
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Interview Details
        </h2>
        
        <div className="space-y-4">
          {/* Job Title */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              <Briefcase className="inline h-4 w-4 mr-1" />
              Target Job Title *
            </label>
            <input
              type="text"
              value={jobTitle}
              onChange={(e) => setJobTitle(e.target.value)}
              placeholder="e.g., Senior Frontend Developer"
              className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>

          {/* Job Description */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              <Target className="inline h-4 w-4 mr-1" />
              Job Description (Optional)
            </label>
            <textarea
              value={jobDescription}
              onChange={(e) => setJobDescription(e.target.value)}
              placeholder="Paste the job description to get more targeted questions..."
              rows={4}
              className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white resize-none"
            />
          </div>

          {/* Difficulty Level */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              <Clock className="inline h-4 w-4 mr-1" />
              Difficulty Level
            </label>
            <div className="grid grid-cols-3 gap-3">
              {(['easy', 'medium', 'hard'] as const).map((level) => (
                <button
                  key={level}
                  onClick={() => setDifficulty(level)}
                  className={`px-4 py-3 rounded-lg border-2 font-medium transition-all ${
                    difficulty === level
                      ? 'border-blue-600 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300'
                      : 'border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:border-blue-300'
                  }`}
                >
                  {level.charAt(0).toUpperCase() + level.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* CV Skills Preview */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-xl border border-blue-200 dark:border-blue-800 p-6">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
          Your Skills (from CV)
        </h3>
        <div className="flex flex-wrap gap-2">
          {cvData.skills.slice(0, 10).map((skill, idx) => (
            <span
              key={idx}
              className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded-full text-sm font-medium"
            >
              {skill}
            </span>
          ))}
          {cvData.skills.length > 10 && (
            <span className="px-3 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded-full text-sm">
              +{cvData.skills.length - 10} more
            </span>
          )}
        </div>
      </div>

      {/* Start Button */}
      <div className="flex items-center justify-center">
        <button
          onClick={handleStartInterview}
          disabled={isStarting || !jobTitle.trim()}
          className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center space-x-3 font-semibold text-lg shadow-lg"
        >
          {isStarting ? (
            <>
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              <span>Starting Interview...</span>
            </>
          ) : (
            <>
              <span>Start {modeConfig[selectedMode].label}</span>
              <ArrowRight className="h-5 w-5" />
            </>
          )}
        </button>
      </div>
    </div>
  );
};
