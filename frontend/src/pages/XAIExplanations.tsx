import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Brain, 
  Eye, 
  HelpCircle, 
  TrendingUp, 
  AlertTriangle,
  CheckCircle,
  XCircle,
  ArrowRight,
  Info,
  Target,
  BarChart3,
  Lightbulb,
  Zap
} from 'lucide-react';
import { xaiApi } from '../services/api';
import type { XAIExplanation } from '../types';



const XAIExplanations = () => {
  const [selectedFeature, setSelectedFeature] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'explanations' | 'transparency'>('explanations');

  const { data: xaiData } = useQuery({
    queryKey: ['xai-explanations'],
    queryFn: async () => {
      const response = await xaiApi.getExplanation('job_match', {});
      return response.data;
    },
  });

  const { data: transparencyData } = useQuery({
    queryKey: ['xai-transparency'],
    queryFn: async () => {
      const response = await xaiApi.getTransparency();
      return response.data;
    },
  });

  const getFeatureIcon = (feature: string) => {
    switch (feature) {
      case 'job_match_score':
        return <Target className="h-5 w-5" />;
      case 'skill_recommendation':
        return <Lightbulb className="h-5 w-5" />;
      case 'experience_translation':
        return <Zap className="h-5 w-5" />;
      default:
        return <Brain className="h-5 w-5" />;
    }
  };

  const getFeatureName = (feature: string) => {
    switch (feature) {
      case 'job_match_score':
        return 'Job Match Scoring';
      case 'skill_recommendation':
        return 'Skill Recommendations';
      case 'experience_translation':
        return 'Experience Translation';
      default:
        return feature.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
  };

  const getImpactColor = (impact: number) => {
    if (impact > 0) return 'text-green-600 bg-green-100';
    if (impact < 0) return 'text-red-600 bg-red-100';
    return 'text-gray-600 bg-gray-100';
  };

  const selectedExplanation = xaiData?.find(ex => ex.feature === selectedFeature);

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">XAI Explanations</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Understand how AI makes decisions about your career recommendations
        </p>
      </motion.div>

      {/* Tab Navigation */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-white dark:bg-gray-800 p-1 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 inline-flex"
      >
        <button
          onClick={() => setActiveTab('explanations')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            activeTab === 'explanations'
              ? 'bg-blue-600 text-white'
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
          }`}
        >
          AI Explanations
        </button>
        <button
          onClick={() => setActiveTab('transparency')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            activeTab === 'transparency'
              ? 'bg-blue-600 text-white'
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
          }`}
        >
          Transparency Dashboard
        </button>
      </motion.div>

      <AnimatePresence mode="wait">
        {activeTab === 'explanations' ? (
          <motion.div
            key="explanations"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="space-y-6"
          >
            {/* Explanation List */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Features List */}
              <div className="space-y-4">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Available Explanations
                </h2>
                {xaiData?.map((explanation, index) => (
                  <motion.button
                    key={explanation.feature}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 + index * 0.1 }}
                    onClick={() => setSelectedFeature(explanation.feature)}
                    className={`w-full text-left p-4 rounded-lg border transition-all hover:shadow-md ${
                      selectedFeature === explanation.feature
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600'
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-lg text-blue-600 dark:text-blue-400">
                        {getFeatureIcon(explanation.feature)}
                      </div>
                      <div className="flex-1">
                        <h3 className="font-medium text-gray-900 dark:text-white">
                          {getFeatureName(explanation.feature)}
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
                          {explanation.explanation}
                        </p>
                        <div className="flex items-center mt-2 space-x-4">
                          <span className="text-xs text-gray-500 dark:text-gray-400">
                            Confidence: {explanation.confidence}%
                          </span>
                          <span className="text-xs text-gray-500 dark:text-gray-400">
                            {explanation.factors.length} factors
                          </span>
                        </div>
                      </div>
                      <ArrowRight className="h-5 w-5 text-gray-400" />
                    </div>
                  </motion.button>
                ))}
              </div>

              {/* Selected Explanation Details */}
              <div className="space-y-4">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Detailed Explanation
                </h2>
                {selectedExplanation ? (
                  <motion.div
                    key={selectedFeature}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700"
                  >
                    <div className="flex items-center space-x-3 mb-4">
                      <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-lg text-blue-600 dark:text-blue-400">
                        {getFeatureIcon(selectedExplanation.feature)}
                      </div>
                      <div>
                        <h3 className="font-semibold text-gray-900 dark:text-white">
                          {getFeatureName(selectedExplanation.feature)}
                        </h3>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm text-gray-500 dark:text-gray-400">
                            Confidence: {selectedExplanation.confidence}%
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="mb-6">
                      <h4 className="font-medium text-gray-900 dark:text-white mb-2">Overall Explanation</h4>
                      <p className="text-gray-600 dark:text-gray-400">
                        {selectedExplanation.explanation}
                      </p>
                    </div>

                    <div className="mb-6">
                      <h4 className="font-medium text-gray-900 dark:text-white mb-3">Contributing Factors</h4>
                      <div className="space-y-3">
                        {selectedExplanation.factors.map((factor, index) => (
                          <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg p-3">
                            <div className="flex items-center justify-between mb-2">
                              <h5 className="font-medium text-gray-900 dark:text-white">
                                {factor.name}
                              </h5>
                              <div className="flex items-center space-x-2">
                                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getImpactColor(factor.impact)}`}>
                                  {factor.impact > 0 ? '+' : ''}{factor.impact}%
                                </span>
                                <span className="text-xs text-gray-500 dark:text-gray-400">
                                  {Math.round(factor.weight * 100)}% weight
                                </span>
                              </div>
                            </div>
                            <p className="text-sm text-gray-600 dark:text-gray-400">
                              {factor.description}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="font-medium text-gray-900 dark:text-white mb-2">Data Sources</h4>
                        <ul className="space-y-1">
                          {selectedExplanation.data_sources.map((source, index) => (
                            <li key={index} className="flex items-center text-sm text-gray-600 dark:text-gray-400">
                              <CheckCircle className="h-3 w-3 mr-2 text-green-500" />
                              {source}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <h4 className="font-medium text-gray-900 dark:text-white mb-2">Methodology</h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          {selectedExplanation.methodology}
                        </p>
                      </div>
                    </div>
                  </motion.div>
                ) : (
                  <div className="bg-white dark:bg-gray-800 p-12 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 text-center">
                    <Eye className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                      Select an Explanation
                    </h3>
                    <p className="text-gray-600 dark:text-gray-400">
                      Choose an AI decision from the list to view detailed explanations and factor analysis.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="transparency"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            {/* Transparency Dashboard */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
                <div className="flex items-center">
                  <div className="p-2 bg-blue-100 dark:bg-blue-900 rounded-lg">
                    <BarChart3 className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600 dark:text-gray-400">AI Decisions Today</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">{transparencyData?.ai_decisions_today}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
                <div className="flex items-center">
                  <div className="p-2 bg-green-100 dark:bg-green-900 rounded-lg">
                    <Eye className="h-6 w-6 text-green-600 dark:text-green-400" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Explanations Available</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">{transparencyData?.explanations_available}</p>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
                <div className="flex items-center">
                  <div className="p-2 bg-purple-100 dark:bg-purple-900 rounded-lg">
                    <TrendingUp className="h-6 w-6 text-purple-600 dark:text-purple-400" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg Confidence</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">{transparencyData?.confidence_average}%</p>
                  </div>
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
                <div className="flex items-center">
                  <div className="p-2 bg-orange-100 dark:bg-orange-900 rounded-lg">
                    <Info className="h-6 w-6 text-orange-600 dark:text-orange-400" />
                  </div>
                  <div className="ml-4">
                    <p className="text-sm font-medium text-gray-600 dark:text-gray-400">Data Sources</p>
                    <p className="text-2xl font-bold text-gray-900 dark:text-white">{transparencyData?.data_sources_used}</p>
                  </div>
                </div>
              </div>
            </div>

            {/* AI Ethics and Transparency */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-6">AI Ethics & Transparency</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-medium text-gray-900 dark:text-white mb-3">Our AI Principles</h3>
                  <ul className="space-y-2">
                    <li className="flex items-start">
                      <CheckCircle className="h-4 w-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        <strong>Transparency:</strong> All AI decisions come with explanations
                      </span>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle className="h-4 w-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        <strong>Fairness:</strong> Regular bias testing and correction
                      </span>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle className="h-4 w-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        <strong>Accountability:</strong> Human oversight on all critical decisions
                      </span>
                    </li>
                    <li className="flex items-start">
                      <CheckCircle className="h-4 w-4 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        <strong>Privacy:</strong> Your data is never shared without consent
                      </span>
                    </li>
                  </ul>
                </div>

                <div>
                  <h3 className="font-medium text-gray-900 dark:text-white mb-3">Limitations & Disclaimers</h3>
                  <ul className="space-y-2">
                    <li className="flex items-start">
                      <AlertTriangle className="h-4 w-4 text-yellow-500 mr-2 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        AI recommendations are suggestions, not definitive career advice
                      </span>
                    </li>
                    <li className="flex items-start">
                      <AlertTriangle className="h-4 w-4 text-yellow-500 mr-2 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        Market data may not reflect real-time changes
                      </span>
                    </li>
                    <li className="flex items-start">
                      <AlertTriangle className="h-4 w-4 text-yellow-500 mr-2 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        Personal factors beyond AI analysis should be considered
                      </span>
                    </li>
                    <li className="flex items-start">
                      <AlertTriangle className="h-4 w-4 text-yellow-500 mr-2 mt-0.5 flex-shrink-0" />
                      <span className="text-sm text-gray-600 dark:text-gray-400">
                        Confidence scores indicate AI certainty, not recommendation quality
                      </span>
                    </li>
                  </ul>
                </div>
              </div>

              <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                <div className="flex items-start">
                  <HelpCircle className="h-5 w-5 text-blue-600 dark:text-blue-400 mr-3 mt-0.5" />
                  <div>
                    <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-1">
                      Questions About Our AI?
                    </h4>
                    <p className="text-sm text-blue-700 dark:text-blue-300">
                      Our AI decisions are designed to be explainable and transparent. If you have questions about any recommendation or decision factor, please don't hesitate to reach out for clarification.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Recent Activity */}
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Recent AI Activity</h2>
              <div className="space-y-3">
                {[
                  { time: '2 hours ago', action: 'Generated job match recommendation', confidence: '92%' },
                  { time: '4 hours ago', action: 'Analyzed CV for skill extraction', confidence: '95%' },
                  { time: '1 day ago', action: 'Provided career path suggestions', confidence: '87%' },
                  { time: '2 days ago', action: 'Translated experience description', confidence: '94%' }
                ].map((activity, index) => (
                  <div key={index} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div>
                      <p className="text-sm font-medium text-gray-900 dark:text-white">{activity.action}</p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">{activity.time}</p>
                    </div>
                    <span className="text-sm font-medium text-green-600 dark:text-green-400">
                      {activity.confidence}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default XAIExplanations;