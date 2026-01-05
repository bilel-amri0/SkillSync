import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import { useState, useEffect } from 'react';
import type { CVAnalysisResponse } from './api';
import { loadLatestAnalysis, saveLatestAnalysis, saveLatestGuidance, clearCareerSnapshots } from './services/careerStore';
import type { CareerGuidanceResponse } from './types/careerGuidance';
import { authService, isAuthenticated as checkAuth, clearAuthData } from './services/authService';

// Layout Components
import Layout from './components/Layout/Layout';
import ProtectedRoute from './components/Auth/ProtectedRoute';

// Page Components
import Dashboard from './pages/Dashboard';
import { CVAnalysisPage } from './pages/CVAnalysisPage';
import JobMatching from './pages/JobMatching';
import { MLCareerGuidancePage } from './pages/MLCareerGuidancePage';
import { NewInterviewPage } from './pages/NewInterviewPage';
import { LiveInterviewPage } from './pages/LiveInterviewPage';
import { LiveInterviewPageVoice } from './pages/LiveInterviewPageVoice';
import Login from './pages/Auth/Login';
import Register from './pages/Auth/Register';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      refetchOnWindowFocus: false,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
});

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    return checkAuth();
  });
  const [latestCvAnalysis, setLatestCvAnalysis] = useState<CVAnalysisResponse | null>(() => loadLatestAnalysis());
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);

  // Validate session on mount
  useEffect(() => {
    const validateSession = async () => {
      if (checkAuth()) {
        const isValid = await authService.validateSession();
        if (!isValid) {
          setIsAuthenticated(false);
          clearAuthData();
        }
      }
      setIsCheckingAuth(false);
    };
    validateSession();
  }, []);

  const handleCvAnalyzed = (analysis: CVAnalysisResponse) => {
    setLatestCvAnalysis(analysis);
    saveLatestAnalysis(analysis);
  };

  const handleGuidanceComplete = (guidance: CareerGuidanceResponse | null) => {
    if (guidance) {
      saveLatestGuidance(guidance);
    }
  };

  const handleLogin = () => {
    setIsAuthenticated(true);
  };

  const handleLogout = async () => {
    await authService.logout();
    setIsAuthenticated(false);
    setLatestCvAnalysis(null);
    clearCareerSnapshots();
  };

  // Show loading while checking auth
  if (isCheckingAuth) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
          <Routes>
            {/* Auth Routes */}
            <Route 
              path="/login" 
              element={
                isAuthenticated ? 
                  <Navigate to="/dashboard" replace /> : 
                  <Login onLogin={handleLogin} />
              } 
            />
            <Route 
              path="/register" 
              element={
                isAuthenticated ? 
                  <Navigate to="/dashboard" replace /> : 
                  <Register onRegister={handleLogin} />
              } 
            />

            {/* Protected Routes */}
            <Route 
              path="/" 
              element={
                <ProtectedRoute isAuthenticated={isAuthenticated}>
                  <Layout onLogout={handleLogout} />
                </ProtectedRoute>
              }
            >
              <Route index element={<Navigate to="/dashboard" replace />} />
              <Route path="dashboard" element={<Dashboard />} />
              <Route path="cv-analysis" element={<CVAnalysisPage />} />
              <Route path="job-matching" element={<JobMatching />} />
              <Route 
                path="career-guidance" 
                element={<MLCareerGuidancePage onCvAnalyzed={handleCvAnalyzed} onGuidanceComplete={handleGuidanceComplete} />} 
              />
              <Route 
                path="interview" 
                element={<NewInterviewPage cvData={latestCvAnalysis} />} 
              />
              <Route path="interview/text/:interviewId" element={<LiveInterviewPage />} />
              <Route path="interview/voice/:interviewId" element={<LiveInterviewPageVoice />} />
            </Route>

            {/* Fallback */}
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </div>

        {/* Toast Notifications */}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
            success: {
              duration: 3000,
              iconTheme: {
                primary: '#10B981',
                secondary: '#fff',
              },
            },
            error: {
              duration: 5000,
              iconTheme: {
                primary: '#EF4444',
                secondary: '#fff',
              },
            },
          }}
        />
      </Router>
    </QueryClientProvider>
  );
}

export default App;