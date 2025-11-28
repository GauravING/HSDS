import React, { useState } from "react";
import "./Modal.css";

const SignupForm = ({ onClose }) => {
  const [formData, setFormData] = useState({ name: "", email: "", password: "" });
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    setError("");
    setMessage("");

    if (formData.password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }

    try {
      const { buildUrl } = await import('../utils/api');

      // Check allowed_emails whitelist (if backend provides it)
      try {
        const allowResp = await fetch(buildUrl('/auth/allowed_emails'));
        const allowJson = await allowResp.json().catch(() => ({}));
        const list = allowJson.allowed_emails || [];
        if (Array.isArray(list) && list.length > 0) {
          const emailLower = (formData.email || '').trim().toLowerCase();
          if (!list.map((s) => s.toLowerCase()).includes(emailLower)) {
            setError('Email not allowed to sign up. Use an approved email.');
            return;
          }
        }
      } catch (err) {
        // ignore whitelist check failures and proceed with signup attempt
      }

      const username = formData.email.split('@')[0];
      const payload = {
        username,
        email: formData.email,
        password: formData.password,
        full_name: formData.name,
      };
      const res = await fetch(buildUrl('/auth/signup'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      if (res.status === 201) {
        setMessage('✅ Account created successfully. Please login.');
        setTimeout(() => onClose(), 1200);
      } else {
        setError(data.msg || 'Signup failed');
      }
    } catch (err) {
      setError('Signup failed. Please try again.');
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <h2>Signup</h2>
        <form onSubmit={handleSignup}>
          <label>Name</label>
          <input
            type="text"
            name="name"
            placeholder="Name"
            value={formData.name}
            onChange={handleChange}
            required
          />
          <label>Email</label>
          <input
            type="email"
            name="email"
            placeholder="Email"
            value={formData.email}
            onChange={handleChange}
            required
          />
          <label>Password</label>
          <input
            type="password"
            name="password"
            placeholder="Password"
            value={formData.password}
            onChange={handleChange}
            required
          />
          <button type="submit">Create Account</button>
        </form>

        {error && <p className="error-msg">{error}</p>}
        {message && <p className="success-msg">{message}</p>}

        <button className="close-btn" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
};

export default SignupForm;
