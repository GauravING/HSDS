import React, { useState, useEffect } from 'react';
import { useNavigate, Link, useLocation } from 'react-router-dom';
import { useTheme } from '../contexts/ThemeContext';
import { useAuth } from '../contexts/AuthContext';
import '../styles/EnhancedStylishLoginForm.css';
// 1. Import FiMail here
import { FiMail, FiLock, FiEye, FiEyeOff, FiCheck } from 'react-icons/fi';
import { motion, AnimatePresence } from 'framer-motion';

const EnhancedStylishLoginForm = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  
  const [showPassword, setShowPassword] = useState(false);
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);
  const [showSuccess, setShowSuccess] = useState(false);

  const { isDarkMode } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const auth = useAuth();
  const contextLogin = auth && auth.login ? auth.login : null;

  const fromSignup = location.state?.fromSignup || false;
  const signupEmail = location.state?.email || '';

  useEffect(() => {
    if (fromSignup) {
      setShowSuccess(true);
      setFormData(prev => ({ ...prev, email: signupEmail }));
      setTimeout(() => setShowSuccess(false), 3000);
    }
  }, [fromSignup, signupEmail]);

  const validateForm = () => {
    const newErrors = {};
    if (!formData.email) {
      newErrors.email = "Email is required";
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = "Email address is invalid";
    }
    if (!formData.password) {
      newErrors.password = "Password is required";
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleBlur = (e) => {
    const { name } = e.target;
    setTouched(prev => ({ ...prev, [name]: true }));
    validateForm();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setTouched({ email: true, password: true });
    const isValid = validateForm();
    if (!isValid) return;
    setIsLoading(true);
    try {
      const { buildUrl } = await import('../utils/api');
      const res = await fetch(buildUrl('/auth/login'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: formData.email, password: formData.password }),
      });

      const data = await res.json().catch(() => ({}));
      if (res.status === 200) {
        const token = data.access_token || null;
        if (token) {
          // Prefer context storage
          if (contextLogin) contextLogin(token);
          else localStorage.setItem('access_token', token);
        }
        navigate('/dashboard');
      } else {
        const msg = data.msg || 'Login failed';
        setErrors({ form: msg });
      }
    } catch (err) {
      setErrors({ form: 'Login failed. Please try again.' });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`enhanced-login-page ${isDarkMode ? 'dark-mode' : ''}`}>
      <motion.div 
        className="login-container"
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ type: "spring", stiffness: 100, damping: 20 }}
      >
        <AnimatePresence>
          {showSuccess && (
            <motion.div
              className="success-message"
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <FiCheck />
              <span>Account created successfully! Please log in.</span>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="login-form-container">
          <div className="login-header">
            <img src="police-officer.png" alt="Police Officer" className="brand-icon" />
            <h1 className="login-header-title">Log in to your account</h1>
            <p className="login-header-subtitle">Welcome back! Please enter your details.</p>
          </div>

          <form className="login-form" onSubmit={handleSubmit} noValidate>
            <div className="input-group">
              {/* 2. Add the icon span here */}
              <span className="icon"><FiMail /></span>
              <input
                type="email"
                name="email"
                placeholder="Email Address"
                value={formData.email}
                onChange={handleChange}
                onBlur={handleBlur}
                className={touched.email && errors.email ? 'input-error' : ''}
              />
              {touched.email && errors.email && <div className="error-message">{errors.email}</div>}
            </div>

            <div className="input-group">
              <span className="icon"><FiLock /></span>
              <input
                type={showPassword ? 'text' : 'password'}
                name="password"
                placeholder="Password"
                value={formData.password}
                onChange={handleChange}
                onBlur={handleBlur}
                className={touched.password && errors.password ? 'input-error' : ''}
              />
              <span 
                className="toggle-password" 
                onClick={() => setShowPassword(prev => !prev)}
              >
                {showPassword ? <FiEyeOff /> : <FiEye />}
              </span>
              {touched.password && errors.password && <div className="error-message">{errors.password}</div>}
            </div>

            <div className="checkbox-group">
              <label className="remember-me">
                <input 
                  type="checkbox" 
                  checked={rememberMe} 
                  onChange={(e) => setRememberMe(e.target.checked)}
                />
                Remember me
              </label>
              <div className="forgot-password">
                <Link to="/forgot-password">Forgot password?</Link>
              </div>
            </div>

            <button type="submit" className="submit-button" disabled={isLoading}>
              {isLoading ? 'Logging In...' : 'Log In'}
            </button>
          </form>

          {errors.form && <div className="form-error" style={{ color: 'red', marginTop: '12px' }}>{errors.form}</div>}

          <p className="signup-link">
            Don't have an account? <Link to="/create-account">Sign up here</Link>
          </p>
        </div>
      </motion.div>
    </div>
  );
};

export default EnhancedStylishLoginForm;