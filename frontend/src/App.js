import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './contexts/ThemeContext';
import { AuthProvider } from './contexts/AuthContext';

// Lazy load components for better performance
const Landing = lazy(() => import('./components/ResponsiveLandingEnhanced'));
const Dashboard = lazy(() => import('./components/AdvancedDashboard'));
const LiveDetection = lazy(() => import('./components/EnhancedLiveDetection'));
const UploadDetection = lazy(() => import('./components/EnhancedUploadDetection'));
const RegisterForm = lazy(() => import('./components/EnhancedSignupForm'));
const LoginForm = lazy(() => import('./components/EnhancedStylishLoginForm'));
const ForgotPassword = lazy(() => import('./components/ForgotPassword'));
const ResetPassword = lazy(() => import('./components/ResetPassword'));

function App() {
  return (
    <AuthProvider>
      <ThemeProvider>
        <Router>
          <Suspense fallback={<div>Loading...</div>}>
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/login" element={<LoginForm />} />
              <Route path="/forgot-password" element={<ForgotPassword />} />
              <Route path="/reset-password" element={<ResetPassword />} />
              <Route path="/create-account" element={<RegisterForm />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/live" element={<LiveDetection />} />
              <Route path="/upload" element={<UploadDetection />} />
            </Routes>
          </Suspense>
        </Router>
      </ThemeProvider>
    </AuthProvider>
  );
}

export default App;
