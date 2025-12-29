import React, { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Send, Loader2, CheckCircle, ArrowLeft, FileText } from 'lucide-react';
import { getNextQuestion, submitAnswer, finishInterview, type InterviewQuestion } from '../api';
import { handleApiError } from '../services/api';

export const LiveInterviewPage: React.FC = () => {
  const { interviewId } = useParams<{ interviewId: string }>();
  const navigate = useNavigate();
  
  const [currentQuestion, setCurrentQuestion] = useState<InterviewQuestion | null>(null);
  const [answer, setAnswer] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [answeredQuestions, setAnsweredQuestions] = useState<Array<{
    question: string;
    answer: string;
    questionId: number;
  }>>([]);

  const handleFinishInterview = useCallback(async () => {
    if (!interviewId) return;

    try {
      const report = await finishInterview(interviewId);
      console.log('✅ Interview finished:', report);
      navigate(`/interview/report/${interviewId}`);
    } catch (err) {
      console.error('❌ Failed to finish interview:', err);
      setError(handleApiError(err));
    }
  }, [interviewId, navigate]);

  const loadNextQuestion = useCallback(async () => {
    if (!interviewId) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await getNextQuestion(interviewId);
      
      if (response.status === 'completed' || !response.next_question) {
        // Interview complete
        await handleFinishInterview();
      } else {
        setCurrentQuestion(response.next_question);
        if (response.progress) {
          setProgress(response.progress);
        }
      }
    } catch (err) {
      console.error('❌ Failed to load next question:', err);
      setError(handleApiError(err));
    } finally {
      setIsLoading(false);
    }
  }, [handleFinishInterview, interviewId]);

  useEffect(() => {
    if (interviewId) {
      loadNextQuestion();
    }
  }, [interviewId, loadNextQuestion]);

  const handleSubmitAnswer = async () => {
    if (!interviewId || !currentQuestion || !answer.trim()) return;

    setIsSubmitting(true);
    setError(null);

    try {
      await submitAnswer({
        interview_id: interviewId,
        question_id: currentQuestion.question_id,
        answer_text: answer
      });

      // Save answered question
      setAnsweredQuestions([...answeredQuestions, {
        question: currentQuestion.question_text,
        answer: answer,
        questionId: currentQuestion.question_id
      }]);

      // Clear answer and load next question
      setAnswer('');
      await loadNextQuestion();
    } catch (err) {
      console.error('❌ Failed to submit answer:', err);
      setError(handleApiError(err));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey && !isSubmitting) {
      handleSubmitAnswer();
    }
  };

  if (isLoading && !currentQuestion) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading interview...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <button
            onClick={() => navigate('/interview')}
            className="flex items-center space-x-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft className="h-5 w-5" />
            <span>Back</span>
          </button>

          {/* Progress */}
          <div className="flex items-center space-x-3">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Question {progress.current} of {progress.total}
            </div>
            <div className="w-32 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-600 to-purple-600 transition-all"
                style={{ width: `${(progress.current / progress.total) * 100}%` }}
              />
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4">
            <p className="text-red-700 dark:text-red-300">{error}</p>
          </div>
        )}

        {/* Current Question */}
        {currentQuestion && (
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-8 shadow-lg">
            <div className="flex items-start space-x-4 mb-6">
              <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                <FileText className="h-6 w-6 text-blue-600 dark:text-blue-400" />
              </div>
              <div className="flex-1">
                <div className="text-sm text-gray-500 dark:text-gray-400 mb-2">
                  {currentQuestion.category}
                </div>
                <h2 className="text-2xl font-semibold text-gray-900 dark:text-white">
                  {currentQuestion.question_text}
                </h2>
              </div>
            </div>

            {/* Answer Input */}
            <div className="space-y-4">
              <textarea
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Type your answer here... (Ctrl+Enter to submit)"
                rows={8}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white resize-none"
                disabled={isSubmitting}
              />

              <div className="flex items-center justify-between">
                <div className="text-sm text-gray-500 dark:text-gray-400">
                  {answer.length} characters
                </div>
                
                <button
                  onClick={handleSubmitAnswer}
                  disabled={isSubmitting || !answer.trim()}
                  className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center space-x-2 font-semibold shadow-lg"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="h-5 w-5 animate-spin" />
                      <span>Submitting...</span>
                    </>
                  ) : (
                    <>
                      <span>Submit Answer</span>
                      <Send className="h-5 w-5" />
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Answered Questions */}
        {answeredQuestions.length > 0 && (
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center">
              <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
              Answered Questions ({answeredQuestions.length})
            </h3>
            <div className="space-y-4">
              {answeredQuestions.map((qa, idx) => (
                <div
                  key={qa.questionId}
                  className="border-l-4 border-green-600 pl-4 py-2 bg-gray-50 dark:bg-gray-700/50 rounded-r"
                >
                  <div className="text-sm font-medium text-gray-900 dark:text-white mb-1">
                    Q{idx + 1}: {qa.question}
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
                    {qa.answer}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tips */}
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-4">
          <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
            💡 Interview Tips
          </h4>
          <ul className="space-y-1 text-sm text-blue-700 dark:text-blue-300">
            <li>• Be specific and provide examples from your experience</li>
            <li>• Use the STAR method (Situation, Task, Action, Result)</li>
            <li>• Take your time to think before answering</li>
            <li>• Press Ctrl+Enter to quickly submit your answer</li>
          </ul>
        </div>
      </div>
    </div>
  );
};
