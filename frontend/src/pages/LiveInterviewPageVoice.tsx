import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Mic,
  MicOff,
  Keyboard,
  Volume2,
  VolumeX,
  ArrowLeft,
  Activity,
  FileText,
  CheckCircle,
  AlertTriangle
} from 'lucide-react';
import { useAudioStream } from '../hooks/useAudioStream';
import {
  finishInterview,
  getNextQuestion,
  submitAnswer,
  type InterviewQuestion
} from '../api';
import { handleApiError } from '../services/api';

type AnswerHistoryItem = {
  question: string;
  answer: string;
  questionId: number;
};

type BrowserSpeechRecognitionAlternative = { transcript: string };
type BrowserSpeechRecognitionResult = {
  readonly isFinal: boolean;
  0: BrowserSpeechRecognitionAlternative;
};
type BrowserSpeechRecognitionEvent = {
  resultIndex: number;
  results: ArrayLike<BrowserSpeechRecognitionResult>;
};
type BrowserSpeechRecognitionErrorEvent = { error: string };
type BrowserSpeechRecognition = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
  onresult: (event: BrowserSpeechRecognitionEvent) => void;
  onerror: (event: BrowserSpeechRecognitionErrorEvent) => void;
  onend: () => void;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionConstructor = new () => BrowserSpeechRecognition;

type SpeechEnabledWindow = Window & typeof globalThis & {
  webkitSpeechRecognition?: SpeechRecognitionConstructor;
  SpeechRecognition?: SpeechRecognitionConstructor;
};

const getSpeechRecognitionConstructor = (): SpeechRecognitionConstructor | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  const speechWindow = window as SpeechEnabledWindow;
  return speechWindow.SpeechRecognition || speechWindow.webkitSpeechRecognition || null;
};

export const LiveInterviewPageVoice: React.FC = () => {
  const { interviewId } = useParams<{ interviewId: string }>();
  const navigate = useNavigate();

  const [isReady, setIsReady] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentQuestion, setCurrentQuestion] = useState<InterviewQuestion | null>(null);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [answerHistory, setAnswerHistory] = useState<AnswerHistoryItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [voiceDraft, setVoiceDraft] = useState('');
  const [liveTranscript, setLiveTranscript] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);
  const [speechError, setSpeechError] = useState<string | null>(null);

  const recognitionRef = useRef<BrowserSpeechRecognition | null>(null);
  const speechCtorRef = useRef<SpeechRecognitionConstructor | null>(null);

  const audioStream = useAudioStream(interviewId || '', {
    sampleRate: 16000,
    channels: 1,
    encoding: 'pcm_s16le'
  });

  useEffect(() => {
    const ctor = getSpeechRecognitionConstructor();
    speechCtorRef.current = ctor;
    setSpeechSupported(Boolean(ctor));

    return () => {
      recognitionRef.current?.stop();
    };
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      if (audioStream.isConnected) {
        setDuration(prev => prev + 1);
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [audioStream.isConnected]);

  const handleFinishInterview = useCallback(async () => {
    if (!interviewId) return;

    try {
      const report = await finishInterview(interviewId);
      navigate(`/interview/report/${interviewId}`);
      return report;
    } catch (err) {
      console.error('Failed to finish interview:', err);
      setError(handleApiError(err));
    }
  }, [interviewId, navigate]);

  const loadNextQuestion = useCallback(async () => {
    if (!interviewId) return;

    setError(null);
    try {
      const response = await getNextQuestion(interviewId);
      if (response.status === 'completed' || !response.next_question) {
        await handleFinishInterview();
        return;
      }

      setCurrentQuestion(response.next_question);
      if (response.progress) {
        setProgress(response.progress);
      }
      setVoiceDraft('');
      setLiveTranscript('');
    } catch (err) {
      console.error('Failed to fetch next question:', err);
      setError(handleApiError(err));
    }
  }, [interviewId, handleFinishInterview]);

  const startSpeechRecognition = useCallback(() => {
    if (!speechCtorRef.current) {
      setSpeechError('Speech recognition is not supported in this browser.');
      return;
    }

    setSpeechError(null);

    const recognition = new speechCtorRef.current();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.continuous = true;

    recognition.onresult = (event: BrowserSpeechRecognitionEvent) => {
      let interim = '';
      let finalChunk = '';
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const transcript = event.results[i][0].transcript.trim();
        if (event.results[i].isFinal) {
          finalChunk += `${transcript} `;
        } else {
          interim += `${transcript} `;
        }
      }

      if (interim) {
        setLiveTranscript(interim.trim());
      }
      if (finalChunk) {
        setVoiceDraft(prev => (prev ? `${prev} ${finalChunk.trim()}` : finalChunk.trim()));
        setLiveTranscript('');
      }
    };

    recognition.onerror = (event: BrowserSpeechRecognitionErrorEvent) => {
      console.warn('Speech recognition error', event.error);
      setSpeechError(event.error);
    };

    recognition.onend = () => {
      setLiveTranscript('');
    };

    recognition.start();
    recognitionRef.current = recognition;
  }, []);

  const stopSpeechRecognition = useCallback(() => {
    recognitionRef.current?.stop();
    recognitionRef.current = null;
    setLiveTranscript('');
  }, []);

  const handleStartInterview = async () => {
    try {
      await audioStream.connect();
      setIsReady(true);
      setDuration(0);
      await loadNextQuestion();

      setTimeout(() => {
        audioStream.startRecording();
        startSpeechRecognition();
      }, 600);
    } catch (err) {
      console.error('Failed to start voice interview:', err);
      setError('Unable to start the interview. Please try again.');
    }
  };

  const handleToggleMic = () => {
    if (audioStream.isRecording) {
      audioStream.stopRecording();
      stopSpeechRecognition();
    } else {
      audioStream.startRecording();
      startSpeechRecognition();
    }
  };

  const handleSubmitVoiceAnswer = async () => {
    if (!interviewId || !currentQuestion || !voiceDraft.trim()) {
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      await submitAnswer({
        interview_id: interviewId,
        question_id: currentQuestion.question_id,
        answer_text: voiceDraft.trim()
      });

      setAnswerHistory(prev => [
        ...prev,
        {
          question: currentQuestion.question_text,
          answer: voiceDraft.trim(),
          questionId: currentQuestion.question_id
        }
      ]);

      setVoiceDraft('');
      await loadNextQuestion();
    } catch (err) {
      console.error('Failed to submit voice answer:', err);
      setError(handleApiError(err));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleEndInterview = async () => {
    audioStream.disconnect();
    stopSpeechRecognition();
    await handleFinishInterview();
  };

  useEffect(() => {
    if (isReady && !currentQuestion && interviewId) {
      loadNextQuestion();
    }
  }, [isReady, currentQuestion, interviewId, loadNextQuestion]);

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (!isReady) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 flex items-center justify-center p-6">
        <div className="max-w-2xl w-full bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-8 text-center">
          <div className="w-20 h-20 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-6">
            <Mic className="h-10 w-10 text-white" />
          </div>

          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            Live Voice Interview
          </h2>

          <p className="text-gray-600 dark:text-gray-400 mb-6">
            You're about to start a live voice interview with an AI interviewer.
            Make sure your microphone is working and you're in a quiet environment.
          </p>

          {!speechSupported && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6 text-yellow-800">
              Your browser does not expose the Speech Recognition API. You can still upload
              responses manually or switch to text mode.
            </div>
          )}

          {audioStream.error && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-6">
              <p className="text-red-700 dark:text-red-300">{audioStream.error}</p>
            </div>
          )}

          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-6 text-left">
            <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">Quick checklist</h3>
            <ul className="text-sm text-blue-700 dark:text-blue-300 space-y-1">
              <li>✓ Find a quiet place</li>
              <li>✓ Test your microphone and speakers</li>
              <li>✓ Keep a glass of water nearby</li>
              <li>✓ Breathe, think, and speak naturally</li>
            </ul>
          </div>

          <button
            onClick={handleStartInterview}
            className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:from-blue-700 hover:to-purple-700 transition-all font-semibold text-lg shadow-lg"
          >
            Start Voice Interview
          </button>

          <p className="text-sm text-gray-500 dark:text-gray-400 mt-4">
            Prefer typing? <button className="text-blue-600" onClick={() => navigate(`/interview/text/${interviewId}`)}>Switch to text mode</button>
          </p>
        </div>
      </div>
    );
  }

  const progressPercent = progress.total ? (progress.current / progress.total) * 100 : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 p-6">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <button
            onClick={handleEndInterview}
            className="flex items-center space-x-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors"
          >
            <ArrowLeft className="h-5 w-5" />
            <span>End Interview</span>
          </button>

          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Question {progress.current} of {progress.total || '—'}
              <div className="w-40 h-2 bg-gray-200 dark:bg-gray-700 rounded-full mt-1">
                <div
                  className="h-full bg-gradient-to-r from-blue-600 to-purple-600 rounded-full"
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
            </div>

            <div className="flex items-center space-x-2 text-gray-700 dark:text-gray-300">
              <Activity className="h-5 w-5 text-red-600" />
              <span className="font-mono text-lg">{formatDuration(duration)}</span>
            </div>

            <div
              className={`px-3 py-1 rounded-full text-sm font-medium ${
                audioStream.isConnected
                  ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                  : 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
              }`}
            >
              {audioStream.isConnected ? 'Connected' : 'Disconnected'}
            </div>
          </div>
        </div>

        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-4 flex items-center gap-3 text-red-700 dark:text-red-300">
            <AlertTriangle className="h-5 w-5" />
            <p>{error}</p>
          </div>
        )}

        <div className="grid lg:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="text-center mb-6">
              <div
                className={`w-32 h-32 mx-auto mb-4 rounded-full flex items-center justify-center ${
                  audioStream.isSpeaking
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 animate-pulse'
                    : 'bg-gray-200 dark:bg-gray-700'
                }`}
              >
                {audioStream.isSpeaking ? (
                  <Volume2 className="h-16 w-16 text-white" />
                ) : (
                  <VolumeX className="h-16 w-16 text-gray-400" />
                )}
              </div>

              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-1">
                {audioStream.isSpeaking ? 'AI is speaking...' : 'AI is listening...'}
              </h3>
              <p className="text-gray-600 dark:text-gray-400">
                {audioStream.isSpeaking ? 'Please listen to the question' : 'Speak your answer when ready'}
              </p>
            </div>

            <div className="flex justify-center mb-6">
              <button
                onClick={handleToggleMic}
                className={`w-20 h-20 rounded-full flex items-center justify-center transition-all ${
                  audioStream.isRecording
                    ? 'bg-red-600 hover:bg-red-700 shadow-lg shadow-red-600/50'
                    : 'bg-gray-300 dark:bg-gray-600 hover:bg-gray-400 dark:hover:bg-gray-500'
                }`}
              >
                {audioStream.isRecording ? (
                  <MicOff className="h-10 w-10 text-white" />
                ) : (
                  <Mic className="h-10 w-10 text-white" />
                )}
              </button>
            </div>

            <div className="text-center text-sm text-gray-500 dark:text-gray-400">
              {audioStream.isRecording ? '🎤 Recording your answer... Tap to pause' : 'Click the microphone to start speaking'}
            </div>

            {speechError && (
              <p className="mt-4 text-sm text-amber-600 text-center">Speech recognition issue: {speechError}</p>
            )}
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-start gap-3 mb-4">
              <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
                <FileText className="h-6 w-6 text-blue-600 dark:text-blue-300" />
              </div>
              <div>
                <p className="text-sm text-gray-500 dark:text-gray-400">Current question</p>
                <h2 className="text-2xl font-semibold text-gray-900 dark:text-white">
                  {currentQuestion?.question_text || 'Loading next question...'}
                </h2>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-2 capitalize">
                  {currentQuestion?.category || 'general'}
                </p>
              </div>
            </div>

            <div className="bg-gray-50 dark:bg-gray-900/30 rounded-lg p-4">
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-2">Live transcript</p>
              <p className="min-h-[48px] text-gray-800 dark:text-gray-200">
                {liveTranscript || 'Silence detected — start speaking to capture text.'}
              </p>
            </div>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Response composer
              </h3>
              {speechSupported && (
                <span className="text-xs px-2 py-1 rounded-full bg-green-100 text-green-700">
                  Speech-to-text on
                </span>
              )}
            </div>

            <textarea
              value={voiceDraft}
              onChange={(e) => setVoiceDraft(e.target.value)}
              placeholder="Stop recording to edit your transcript before submitting."
              rows={6}
              className="w-full px-4 py-3 border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-900 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-600"
            />

            <div className="flex items-center justify-between mt-4 text-sm text-gray-500 dark:text-gray-400">
              <span>{voiceDraft.length} characters</span>
              <button
                onClick={() => setVoiceDraft('')}
                className="text-blue-600 hover:underline"
                type="button"
              >
                Clear
              </button>
            </div>

            <button
              onClick={handleSubmitVoiceAnswer}
              disabled={isSubmitting || !voiceDraft.trim()}
              className="w-full mt-4 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 font-semibold"
            >
              {isSubmitting ? 'Submitting...' : 'Submit Voice Answer'}
            </button>

            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
              Tip: Pause the microphone to finalize the transcript, edit if needed, then submit to unlock the next question.
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
              <CheckCircle className="h-5 w-5 text-green-600" />
              Answered questions ({answerHistory.length})
            </h3>

            {answerHistory.length === 0 ? (
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Your submitted answers will appear here with quick summaries.
              </p>
            ) : (
              <div className="space-y-4 max-h-72 overflow-y-auto pr-2">
                {answerHistory.map((entry, idx) => (
                  <div key={entry.questionId} className="border-l-4 border-green-600 pl-4">
                    <p className="text-xs uppercase text-gray-500 dark:text-gray-400 mb-1">
                      Question {idx + 1}
                    </p>
                    <h4 className="text-base font-semibold text-gray-900 dark:text-white mb-1">
                      {entry.question}
                    </h4>
                    <p className="text-sm text-gray-600 dark:text-gray-300 line-clamp-3">
                      {entry.answer}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <button
            onClick={() => navigate(`/interview/text/${interviewId}`)}
            className="flex items-center justify-center gap-2 px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600"
          >
            <Keyboard className="h-4 w-4" />
            Switch to Text Mode
          </button>

          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-4 flex-1">
            <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">Voice Interview Tips</h4>
            <ul className="space-y-1 text-sm text-blue-700 dark:text-blue-300">
              <li>• Pause briefly before answering to collect your thoughts.</li>
              <li>• Mention metrics and impact for each story.</li>
              <li>• Ask for clarification if a question is unclear.</li>
              <li>• Keep responses under two minutes for sharper scoring.</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
