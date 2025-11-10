import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Mic, 
  MicOff, 
  ArrowRight, 
  FileText, 
  Briefcase, 
  CheckCircle, 
  Loader2,
  MessageSquare
} from 'lucide-react';
import toast from 'react-hot-toast';
import { 
  interviewService, 
  handleInterviewError,
  Question 
} from '../../services/interviewService';

const InterviewPage = () => {
  const navigate = useNavigate();
  
  // Pre-interview state
  const [showPreInterview, setShowPreInterview] = useState(true);
  const [cvText, setCvText] = useState('');
  const [jobDescription, setJobDescription] = useState('');
  const [numQuestions, setNumQuestions] = useState(5);
  
  // Interview state
  const [interviewId, setInterviewId] = useState<string | null>(null);
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [currentAnswer, setCurrentAnswer] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isComplete, setIsComplete] = useState(false);

  const currentQuestion = questions[currentQuestionIndex];

  const handleStartInterview = async () => {
    if (!cvText.trim() || !jobDescription.trim()) {
      toast.error('Please provide both CV and job description');
      return;
    }

    setIsLoading(true);
    try {
      const response = await interviewService.startInterview({
        cv_text: cvText,
        job_description: jobDescription,
        num_questions: numQuestions,
      });

      setInterviewId(response.interview_id);
      setQuestions(response.questions);
      setShowPreInterview(false);
      toast.success('Interview started! Good luck!');
    } catch (error) {
      toast.error(handleInterviewError(error));
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmitAnswer = async () => {
    if (!currentAnswer.trim()) {
      toast.error('Please provide an answer');
      return;
    }

    if (!interviewId || !currentQuestion) {
      return;
    }

    setIsLoading(true);
    try {
      const response = await interviewService.submitAnswer(interviewId, {
        question_id: currentQuestion.question_id,
        answer_text: currentAnswer,
      });

      if (response.is_complete) {
        setIsComplete(true);
        toast.success('Interview completed! Redirecting to report...');
        setTimeout(() => {
          navigate(`/interview-report/${interviewId}`);
        }, 2000);
      } else {
        setCurrentQuestionIndex(currentQuestionIndex + 1);
        setCurrentAnswer('');
        toast.success('Answer submitted!');
      }
    } catch (error) {
      toast.error(handleInterviewError(error));
    } finally {
      setIsLoading(false);
    }
  };

  const toggleRecording = () => {
    // Mock recording functionality
    // In a real implementation, this would use the Web Speech API
    setIsRecording(!isRecording);
    if (!isRecording) {
      toast.success('Recording started (mock)');
    } else {
      toast.success('Recording stopped (mock)');
    }
  };

  if (showPreInterview) {
    return (
      <div className="max-w-4xl mx-auto py-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8"
        >
          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
              Interview Practice
            </h1>
            <p className="text-gray-600 dark:text-gray-300">
              Practice your interview skills with AI-powered questions tailored to your CV and target job
            </p>
          </div>

          <div className="space-y-6">
            {/* CV Input */}
            <div>
              <label className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <FileText className="w-4 h-4 mr-2" />
                Your CV Content
              </label>
              <textarea
                value={cvText}
                onChange={(e) => setCvText(e.target.value)}
                placeholder="Paste your CV content here..."
                className="w-full h-40 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white resize-none"
              />
            </div>

            {/* Job Description Input */}
            <div>
              <label className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <Briefcase className="w-4 h-4 mr-2" />
                Job Description
              </label>
              <textarea
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                placeholder="Paste the job description here..."
                className="w-full h-40 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white resize-none"
              />
            </div>

            {/* Number of Questions */}
            <div>
              <label className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                <MessageSquare className="w-4 h-4 mr-2" />
                Number of Questions
              </label>
              <select
                value={numQuestions}
                onChange={(e) => setNumQuestions(Number(e.target.value))}
                className="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
              >
                <option value={3}>3 Questions</option>
                <option value={5}>5 Questions</option>
                <option value={7}>7 Questions</option>
                <option value={10}>10 Questions</option>
              </select>
            </div>

            {/* Start Button */}
            <button
              onClick={handleStartInterview}
              disabled={isLoading || !cvText.trim() || !jobDescription.trim()}
              className="w-full flex items-center justify-center px-6 py-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                  Starting Interview...
                </>
              ) : (
                <>
                  Start Interview
                  <ArrowRight className="w-5 h-5 ml-2" />
                </>
              )}
            </button>
          </div>
        </motion.div>
      </div>
    );
  }

  if (isComplete) {
    return (
      <div className="max-w-4xl mx-auto py-8">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-12 text-center"
        >
          <CheckCircle className="w-20 h-20 text-green-500 mx-auto mb-6" />
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-4">
            Interview Completed!
          </h2>
          <p className="text-gray-600 dark:text-gray-300 mb-6">
            Redirecting you to your performance report...
          </p>
          <Loader2 className="w-8 h-8 text-blue-600 mx-auto animate-spin" />
        </motion.div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto py-8">
      <AnimatePresence mode="wait">
        <motion.div
          key={currentQuestionIndex}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8"
        >
          {/* Progress */}
          <div className="mb-8">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Question {currentQuestionIndex + 1} of {questions.length}
              </span>
              <span className="text-sm font-medium text-blue-600 dark:text-blue-400">
                {currentQuestion?.category}
              </span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${((currentQuestionIndex + 1) / questions.length) * 100}%` }}
              />
            </div>
          </div>

          {/* Question */}
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              {currentQuestion?.question_text}
            </h2>
          </div>

          {/* Answer Input */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Your Answer
              </label>
              <button
                onClick={toggleRecording}
                className={`flex items-center px-3 py-1 rounded-lg text-sm transition-colors ${
                  isRecording
                    ? 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                    : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
                }`}
              >
                {isRecording ? (
                  <>
                    <MicOff className="w-4 h-4 mr-1" />
                    Stop Recording
                  </>
                ) : (
                  <>
                    <Mic className="w-4 h-4 mr-1" />
                    Start Recording
                  </>
                )}
              </button>
            </div>
            <textarea
              value={currentAnswer}
              onChange={(e) => setCurrentAnswer(e.target.value)}
              placeholder="Type your answer here..."
              className="w-full h-48 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white resize-none"
            />
          </div>

          {/* Submit Button */}
          <button
            onClick={handleSubmitAnswer}
            disabled={isLoading || !currentAnswer.trim()}
            className="w-full flex items-center justify-center px-6 py-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                Submitting...
              </>
            ) : (
              <>
                {currentQuestionIndex === questions.length - 1 ? 'Complete Interview' : 'Next Question'}
                <ArrowRight className="w-5 h-5 ml-2" />
              </>
            )}
          </button>
        </motion.div>
      </AnimatePresence>
    </div>
  );
};

export default InterviewPage;
