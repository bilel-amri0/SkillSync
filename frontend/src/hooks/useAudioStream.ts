import { useState, useEffect, useRef, useCallback } from 'react';

interface AudioStreamConfig {
  sampleRate?: number;
  channels?: number;
  encoding?: string;
}

interface AudioStreamState {
  isConnected: boolean;
  isRecording: boolean;
  isSpeaking: boolean;
  error: string | null;
}

const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) {
    return error.message;
  }

  if (typeof error === 'string') {
    return error;
  }

  return 'Unknown error';
};

export const useAudioStream = (interviewId: string, config?: AudioStreamConfig) => {
  const [state, setState] = useState<AudioStreamState>({
    isConnected: false,
    isRecording: false,
    isSpeaking: false,
    error: null
  });

  const wsRef = useRef<WebSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  const connect = useCallback(async () => {
    try {
      // WebSocket connection to backend
      const apiBase = import.meta.env.VITE_API_URL || window.location.origin;
      const wsBase = apiBase.replace(/^http/i, 'ws');
      const wsUrl = `${wsBase}/api/v2/interviews/live/${interviewId}/ws`;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setState(prev => ({ ...prev, isConnected: true, error: null }));
      };

      ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        setState(prev => ({ ...prev, error: 'Connection failed' }));
      };

      ws.onclose = () => {
        console.log('WebSocket closed');
        setState(prev => ({ ...prev, isConnected: false }));
      };

      ws.onmessage = async (event) => {
        // Handle incoming audio from AI
        if (event.data instanceof Blob) {
          await playAudioChunk(event.data);
        } else {
          // Handle text messages (status updates, etc.)
          const message = JSON.parse(event.data);
          console.log('Received message:', message);
        }
      };

      wsRef.current = ws;
    } catch (err: unknown) {
      console.error('Failed to connect:', err);
      setState(prev => ({ ...prev, error: getErrorMessage(err) }));
    }
  }, [interviewId]);

  const startRecording = useCallback(async () => {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: config?.sampleRate || 16000,
          channelCount: config?.channels || 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      mediaStreamRef.current = stream;

      // Create MediaRecorder
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
          // Send audio chunk to backend
          wsRef.current.send(event.data);
        }
      };

      mediaRecorder.start(100); // Collect data every 100ms
      mediaRecorderRef.current = mediaRecorder;

      setState(prev => ({ ...prev, isRecording: true }));
      console.log('🎤 Recording started');
    } catch (err: unknown) {
      console.error('Failed to start recording:', err);
      setState(prev => ({ ...prev, error: 'Microphone access denied' }));
    }
  }, [config]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }

    setState(prev => ({ ...prev, isRecording: false }));
    console.log('🎤 Recording stopped');
  }, []);

  const playAudioChunk = async (audioBlob: Blob) => {
    try {
      setState(prev => ({ ...prev, isSpeaking: true }));

      // Create audio context if not exists
      if (!audioContextRef.current) {
        audioContextRef.current = new AudioContext();
      }

      const arrayBuffer = await audioBlob.arrayBuffer();
      const audioBuffer = await audioContextRef.current.decodeAudioData(arrayBuffer);

      const source = audioContextRef.current.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContextRef.current.destination);

      source.onended = () => {
        setState(prev => ({ ...prev, isSpeaking: false }));
      };

      source.start();
    } catch (err) {
      console.error('Failed to play audio:', err);
      setState(prev => ({ ...prev, isSpeaking: false }));
    }
  };

  const disconnect = useCallback(() => {
    stopRecording();

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    setState({
      isConnected: false,
      isRecording: false,
      isSpeaking: false,
      error: null
    });
  }, [stopRecording]);

  const sendMessage = useCallback((message: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, [disconnect]);

  return {
    ...state,
    connect,
    disconnect,
    startRecording,
    stopRecording,
    sendMessage
  };
};
