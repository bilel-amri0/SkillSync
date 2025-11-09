import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { 
  Shuffle, 
  Send, 
  Download, 
  Copy, 
  Eye, 
  Settings,
  CheckCircle,
  AlertCircle,
  Lightbulb,
  Target,
  TrendingUp,
  ArrowRight,
  RotateCcw
} from 'lucide-react';
import { experienceApi, handleApiError } from '../services/api';
import toast from 'react-hot-toast';
import type { TranslationResult } from '../types';



const EXPERIENCE_STYLES = [
  { id: 'professional', name: 'Professional', description: 'Formal, achievement-focused with bullet-point structure' },
  { id: 'technical', name: 'Technical', description: 'Precise, skills-focused for engineering roles' },
  { id: 'creative', name: 'Creative', description: 'Engaging, innovation-focused narrative' }
];

const EXPERIENCE_TEMPLATES = [
  'Led a team of 5 developers to build a customer-facing web application...',
  'Developed and maintained RESTful APIs using Node.js and Express...',
  'Implemented automated testing strategies that improved code quality...',
  'Collaborated with designers to create responsive user interfaces...',
  'Optimized database queries that improved application performance by 50%...'
];

const ExperienceTranslator = () => {
  const [originalText, setOriginalText] = useState('');
  const [targetRole, setTargetRole] = useState('');
  const [industry, setIndustry] = useState('');
  const [selectedStyle, setSelectedStyle] = useState('professional');
  const [translationResult, setTranslationResult] = useState<TranslationResult | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  const translateMutation = useMutation({
    mutationFn: async (data: {
      experience_text: string;
      target_role?: string;
      industry?: string;
      style?: string;
    }) => {
      const response = await experienceApi.translate(data);
      return response.data;
    },
    onSuccess: (data) => {
      setTranslationResult(data);
      toast.success('Experience translated successfully!');
    },
    onError: (error) => {
      const errorMessage = handleApiError(error);
      toast.error(`Translation failed: ${errorMessage}`);
    },
  });

  const handleTranslate = () => {
    if (!originalText.trim()) {
      toast.error('Please enter some experience text to translate');
      return;
    }

    translateMutation.mutate({
      experience_text: originalText,
      target_role: targetRole || undefined,
      industry: industry || undefined,
      style: selectedStyle,
    });
  };

  const handleCopyText = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success('Text copied to clipboard!');
  };

  const handleDownload = (format: 'text' | 'markdown' | 'html' | 'json') => {
    if (!translationResult) return;

    let content = '';
    let filename = '';

    switch (format) {
      case 'text':
        content = translationResult.translated_text;
        filename = 'translated_experience.txt';
        break;
      case 'markdown':
        content = `# Translated Experience\n\n**Original:**\n${translationResult.original_text}\n\n**Translated:**\n${translationResult.translated_text}\n\n**Analysis:**\n- Style: ${translationResult.style}\n- Confidence: ${translationResult.confidence_score}%\n- Skills: ${translationResult.analysis.skills_extracted.join(', ')}`;
        filename = 'translated_experience.md';
        break;
      case 'html':
        content = `<h1>Translated Experience</h1><h2>Original</h2><p>${translationResult.original_text}</p><h2>Translated</h2><p>${translationResult.translated_text}</p><h2>Analysis</h2><ul><li>Style: ${translationResult.style}</li><li>Confidence: ${translationResult.confidence_score}%</li><li>Skills: ${translationResult.analysis.skills_extracted.join(', ')}</li></ul>`;
        filename = 'translated_experience.html';
        break;
      case 'json':
        content = JSON.stringify(translationResult, null, 2);
        filename = 'translated_experience.json';
        break;
    }

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    toast.success(`Downloaded as ${filename}`);
  };

  const loadTemplate = (template: string) => {
    setOriginalText(template);
  };

  const resetForm = () => {
    setOriginalText('');
    setTargetRole('');
    setIndustry('');
    setSelectedStyle('professional');
    setTranslationResult(null);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Experience Translator</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Transform your experience descriptions into compelling, professional narratives
        </p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Section */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="space-y-6"
        >
          {/* Original Text Input */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Your Experience</h2>
              <button
                onClick={() => setShowSettings(!showSettings)}
                className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
              >
                <Settings className="h-5 w-5" />
              </button>
            </div>

            <textarea
              value={originalText}
              onChange={(e) => setOriginalText(e.target.value)}
              placeholder="Describe your experience in detail. Include what you did, how you did it, and any measurable outcomes..."
              className="w-full h-64 p-4 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:text-white resize-none"
            />

            {/* Settings Panel */}
            {showSettings && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="mt-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg space-y-4"
              >
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Target Role (Optional)
                  </label>
                  <input
                    type="text"
                    value={targetRole}
                    onChange={(e) => setTargetRole(e.target.value)}
                    placeholder="e.g., Senior Frontend Developer"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-600 dark:text-white"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    Industry (Optional)
                  </label>
                  <input
                    type="text"
                    value={industry}
                    onChange={(e) => setIndustry(e.target.value)}
                    placeholder="e.g., FinTech, Healthcare, E-commerce"
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-600 dark:text-white"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Translation Style
                  </label>
                  <div className="grid grid-cols-1 gap-2">
                    {EXPERIENCE_STYLES.map((style) => (
                      <label key={style.id} className="flex items-center cursor-pointer">
                        <input
                          type="radio"
                          name="style"
                          value={style.id}
                          checked={selectedStyle === style.id}
                          onChange={(e) => setSelectedStyle(e.target.value)}
                          className="mr-3"
                        />
                        <div>
                          <div className="font-medium text-gray-900 dark:text-white">{style.name}</div>
                          <div className="text-sm text-gray-500 dark:text-gray-400">{style.description}</div>
                        </div>
                      </label>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            <div className="mt-4 flex justify-between">
              <button
                onClick={resetForm}
                className="px-4 py-2 text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors flex items-center"
              >
                <RotateCcw className="h-4 w-4 mr-1" />
                Reset
              </button>
              
              <button
                onClick={handleTranslate}
                disabled={translateMutation.isPending || !originalText.trim()}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center"
              >
                {translateMutation.isPending ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Translating...
                  </>
                ) : (
                  <>
                    <Send className="h-4 w-4 mr-2" />
                    Translate
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Templates */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Quick Start Templates</h3>
            <div className="space-y-2">
              {EXPERIENCE_TEMPLATES.map((template, index) => (
                <button
                  key={index}
                  onClick={() => loadTemplate(template)}
                  className="w-full text-left p-3 border border-gray-200 dark:border-gray-600 rounded-lg hover:border-blue-300 dark:hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-all text-sm"
                >
                  {template}
                </button>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Results Section */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-6"
        >
          {translationResult ? (
            <div className="space-y-6">
              {/* Translation Result */}
              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Translated Experience</h2>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => handleCopyText(translationResult.translated_text)}
                      className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                      title="Copy to clipboard"
                    >
                      <Copy className="h-4 w-4" />
                    </button>
                    <div className="relative">
                      <select
                        onChange={(e) => handleDownload(e.target.value as any)}
                        className="px-3 py-1 text-sm border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                        defaultValue=""
                      >
                        <option value="" disabled>Download</option>
                        <option value="text">Text</option>
                        <option value="markdown">Markdown</option>
                        <option value="html">HTML</option>
                        <option value="json">JSON</option>
                      </select>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                    <div className="flex items-center mb-2">
                      <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
                      <span className="font-medium text-green-800 dark:text-green-200">Translation Complete</span>
                    </div>
                    <p className="text-sm text-green-700 dark:text-green-300">
                      Confidence: {translationResult.confidence_score}% • Style: {translationResult.style}
                    </p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Original Text
                    </label>
                    <div className="p-3 bg-gray-50 dark:bg-gray-700 rounded-lg text-sm text-gray-600 dark:text-gray-400">
                      {translationResult.original_text}
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Translated Text
                    </label>
                    <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                      <p className="text-gray-900 dark:text-white leading-relaxed">
                        {translationResult.translated_text}
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Analysis */}
              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Analysis & Insights</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white mb-2">Extracted Skills</h4>
                    <div className="flex flex-wrap gap-1">
                      {translationResult.analysis.skills_extracted.map((skill, index) => (
                        <span
                          key={index}
                          className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs rounded-full"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white mb-2">Action Verbs</h4>
                    <div className="space-y-1">
                      {translationResult.analysis.action_verbs.map((verb, index) => (
                        <div key={index} className="flex items-center text-sm text-gray-600 dark:text-gray-400">
                          <ArrowRight className="h-3 w-3 mr-2" />
                          {verb}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white mb-2">Quantified Achievements</h4>
                    <div className="space-y-1">
                      {translationResult.analysis.quantified_achievements.map((achievement, index) => (
                        <div key={index} className="flex items-center text-sm text-green-600 dark:text-green-400">
                          <TrendingUp className="h-3 w-3 mr-2" />
                          {achievement}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white mb-2">Experience Level</h4>
                    <div className="flex items-center text-sm text-gray-600 dark:text-gray-400">
                      <Target className="h-4 w-4 mr-2" />
                      {translationResult.analysis.experience_level}
                    </div>
                    <div className="mt-2">
                      <span className="text-sm text-gray-500 dark:text-gray-400">Clarity Score:</span>
                      <span className="ml-1 font-medium text-gray-900 dark:text-white">
                        {translationResult.analysis.clarity_score}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Improvements */}
              <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Improvements Made</h3>
                
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white mb-2">Suggestions Applied</h4>
                    <ul className="space-y-1">
                      {translationResult.improvements.suggestions.map((suggestion, index) => (
                        <li key={index} className="flex items-start text-sm text-gray-600 dark:text-gray-400">
                          <Lightbulb className="h-4 w-4 mr-2 mt-0.5 text-yellow-500 flex-shrink-0" />
                          {suggestion}
                        </li>
                      ))}
                    </ul>
                  </div>

                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white mb-2">Added Keywords</h4>
                    <div className="flex flex-wrap gap-1">
                      {translationResult.improvements.added_keywords.map((keyword, index) => (
                        <span
                          key={index}
                          className="px-2 py-1 bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 text-xs rounded-full"
                        >
                          {keyword}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium text-gray-900 dark:text-white mb-2">Enhanced Areas</h4>
                    <div className="grid grid-cols-2 gap-2">
                      {translationResult.improvements.enhanced_areas.map((area, index) => (
                        <div key={index} className="flex items-center text-sm text-gray-600 dark:text-gray-400">
                          <CheckCircle className="h-3 w-3 mr-2 text-green-500" />
                          {area}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-white dark:bg-gray-800 p-12 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 text-center">
              <Shuffle className="h-16 w-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                Ready to Translate
              </h3>
              <p className="text-gray-600 dark:text-gray-400 mb-4">
                Enter your experience description and choose a style to get started with your professional translation.
              </p>
              <div className="flex items-center justify-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                <div className="flex items-center">
                  <CheckCircle className="h-4 w-4 mr-1" />
                  AI-Powered Enhancement
                </div>
                <div className="flex items-center">
                  <CheckCircle className="h-4 w-4 mr-1" />
                  Multiple Export Formats
                </div>
                <div className="flex items-center">
                  <CheckCircle className="h-4 w-4 mr-1" />
                  Skill Analysis
                </div>
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
};

export default ExperienceTranslator;