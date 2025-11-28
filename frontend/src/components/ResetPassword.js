import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/EnhancedStylishLoginForm.css'; // reuse styles for form

const ResetPassword = () => {
  const [formData, setFormData] = useState({
    newPassword: '',
    confirmPassword: ''
  });
  const [errors, setErrors] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const navigate = useNavigate();

  const validateForm = () => {
    const newErrors = {};
    if (!formData.newPassword) {
      newErrors.newPassword = 'New password is required';
    } else if (formData.newPassword.length < 6) {
      newErrors.newPassword = 'Password must be at least 6 characters';
    }
    if (!formData.confirmPassword) {
      newErrors.confirmPassword = 'Confirm password is required';
    } else if (formData.newPassword !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match';
    }
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (validateForm()) {
      // Simulate password reset
      setSubmitted(true);
      setTimeout(() => {
        navigate('/login');
      }, 3000);
    }
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
                  type="password"
                  name="newPassword"
                  placeholder="New Password"
                  value={formData.newPassword}
                  onChange={handleChange}
                  className={errors.newPassword ? 'input-error' : ''}
                />
                {errors.newPassword && <div className="error-message">{errors.newPassword}</div>}
              </div>
              <div className="input-group">
                <input
                  type="password"
                  name="confirmPassword"
                  placeholder="Confirm New Password"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                  className={errors.confirmPassword ? 'input-error' : ''}
                />
                {errors.confirmPassword && <div className="error-message">{errors.confirmPassword}</div>}
              </div>
              <button type="submit" className="submit-button">
                Reset Password
              </button>
            </form>
          ) : (
            <div className="success-message">
              Your password has been reset successfully. Redirecting to login...
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ResetPassword;
