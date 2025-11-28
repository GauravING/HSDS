import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import '../styles/EnhancedStylishLoginForm.css'; // reuse styles for form

const ForgotPassword = () => {
  const [email, setEmail] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const validateEmail = (email) => {
    return /\S+@\S+\.\S+/.test(email);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setError('');
    if (!email) {
      setError('Email is required');
      return;
    }
    if (!validateEmail(email)) {
      setError('Email address is invalid');
      return;
    }
    // Simulate sending reset email
    setSubmitted(true);
    // In a real app, this would send an email with a reset link
    // For demo, we'll show a link to reset password page
  };

  return (
    <div className="enhanced-login-page">
      <div className="login-container">
        <div className="login-form-container">
          <h1 className="login-header-title">Reset Your Password</h1>
          {!submitted ? (
            <form className="login-form" onSubmit={handleSubmit} noValidate>
              <div className="input-group">
                <input
                  type="email"
                  name="email"
                  placeholder="Enter your email address"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className={error ? 'input-error' : ''}
                />
                {error && <div className="error-message">{error}</div>}
              </div>
              <button type="submit" className="submit-button">
                Send Reset Link
              </button>
            </form>
          ) : (
            <div className="success-message">
              A password reset link has been sent to your email.
              <br />
              <Link to="/reset-password" style={{ color: '#007bff', textDecoration: 'underline' }}>
                Click here to reset your password
              </Link>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ForgotPassword;
