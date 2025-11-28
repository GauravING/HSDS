import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useTheme } from "../contexts/ThemeContext";
import { motion, AnimatePresence } from "framer-motion";
import { Shield, BarChart2, Smartphone, Lock, Menu, X } from "react-feather";

import "../styles/ResponsiveLandingEnhanced.css";

const ResponsiveLandingEnhanced = () => {
  const [scrollY, setScrollY] = useState(0);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  useTheme();
  const navigate = useNavigate();

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const features = [
    {
      icon: <Shield size={32} />,
      title: "Real-time Detection",
      description:
        "Advanced AI algorithms for instant helmet and seatbelt detection with 99.9% accuracy.",
    },
    {
      icon: <BarChart2 size={32} />,
      title: "Analytics Dashboard",
      description:
        "Comprehensive insights and reporting for safety compliance monitoring.",
    },
    {
      icon: <Smartphone size={32} />,
      title: "Mobile Responsive",
      description:
        "Seamless experience across all devices with responsive design.",
    },
    {
      icon: <Lock size={32} />,
      title: "Secure & Reliable",
      description: "Enterprise-grade security with 99.9% uptime guarantee.",
    },
  ];

  // stats removed (unused) to satisfy ESLint; keep feature content minimal here

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 50, scale: 0.95 },
    show: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: { duration: 0.6, ease: "easeOut" },
    },
  };

  return (
    <div className="responsive-landing-enhanced dark-mode-advanced">
      {/* Enhanced Navigation */}
      <nav className={`navbar ${scrollY > 50 ? "scrolled" : ""}`}>
        <div className="nav-container">
          <motion.div
            className="nav-logo"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <span className="logo-text">SafetyAI</span>
            <span className="logo-subtitle">Advanced</span>
          </motion.div>

          <div className="nav-menu-desktop">
            <a href="#features" className="nav-link">
              Features
            </a>
            <a href="#SafetyCheck" className="nav-link">
              Safety Check
            </a>
            <a href="#contact" className="nav-link">
              Contact
            </a>
          </div>

          <div className="nav-actions">
            <motion.button
              className="btn-login"
              onClick={() => navigate("/login")}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Login
            </motion.button>
            <motion.button
              className="btn-signup"
              onClick={() => navigate("/create-account")}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Sign Up
            </motion.button>
          </div>

          <div className="hamburger" onClick={() => setIsMenuOpen(!isMenuOpen)}>
            <AnimatePresence mode="wait">
              <motion.span
                key={isMenuOpen ? "open" : "closed"}
                initial={{ opacity: 0, rotate: 0 }}
                animate={{ opacity: 1, rotate: isMenuOpen ? 180 : 0 }}
                exit={{ opacity: 0, rotate: -180 }}
                transition={{ duration: 0.3 }}
              >
                {isMenuOpen ? <X /> : <Menu />}
              </motion.span>
            </AnimatePresence>
          </div>
        </div>
      </nav>

      {/* Mobile Menu Overlay */}
      <AnimatePresence>
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0, x: "100%" }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: "100%" }}
            transition={{ duration: 0.5, ease: "easeInOut" }}
            className="mobile-menu-overlay"
          >
            <a href="#features" onClick={() => setIsMenuOpen(false)}>
              Features
            </a>
            <a href="#stats" onClick={() => setIsMenuOpen(false)}>
              Statistics
            </a>
            <a href="#contact" onClick={() => setIsMenuOpen(false)}>
              Contact
            </a>
            <div className="mobile-actions">
              <button className="btn-login" onClick={() => navigate("/login")}>
                Login
              </button>
              <button
                className="btn-signup"
                onClick={() => navigate("/create-account")}
              >
                Sign Up
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Enhanced Hero Section */}
      <section className="hero">
        <div className="hero-container">
          <motion.div
            className="hero-content"
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          >
            <div className="hero-badge-container">
              <span className="hero-badge">AI-Powered Safety</span>
            </div>
            <h1 className="hero-title">
              <span className="gradient-text">Revolutionize Safety</span>
              <br />
              <span className="highlight-text">with AI Detection</span>
            </h1>
            <p className="hero-subtitle">
              Advanced computer vision technology that automatically detects
              helmet and seatbelt usage in real-time, ensuring compliance and
              saving lives.
            </p>
            <motion.button
              className="btn-signup"
              onClick={() => navigate("/create-account")}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Start Free Trial →
            </motion.button>
          </motion.div>

          <motion.div
            className="hero-visual"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8, ease: "easeOut", delay: 0.3 }}
          >
            <div className="image-container">
              <img
                src="Gemini_Generated_Image_865flr865flr865f.png"
                alt="AI-powered safety detection on a car"
                className="hero-image"
              />
            </div>
          </motion.div>
        </div>
      </section>

      {/* Enhanced Features Section */}
      <section id="features" className="features">
        <div className="container">
          <div className="section-header">
            <h2 className="section-title">Powerful Features</h2>
            <p className="section-subtitle">
              Discover how our AI-powered system enhances safety compliance.
            </p>
          </div>
          <motion.div
            className="features-grid"
            variants={containerVariants}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, amount: 0.3 }}
          >
            {features.map((feature, index) => (
              <motion.div
                key={index}
                className="feature-card"
                variants={itemVariants}
              >
                <div className="feature-icon-wrapper">
                  <span className="feature-icon">{feature.icon}</span>
                </div>
                <h3 className="feature-title">{feature.title}</h3>
                <p className="feature-description">{feature.description}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Enhanced CTA Section */}
      <section className="cta" id="SafetyCheck">
        <div className="aurora-bg">
          <div className="aurora-dot aurora-dot-1"></div>
          <div className="aurora-dot aurora-dot-2"></div>
        </div>
        <div className="heading-for-landing-page">
          Detect Safety Compliance Instantly
        </div>
        <div className="subtitle-for-landing-page">
          Live detection with your camera or batch processing via uploads.
        </div>
        <div className="landing-page-description">
          <div className="landing-page-for-healmet">
            <div className="healmet-details">
              <div className="landing-page-Heading">
                <span id="landing-page-title">Helmet Usage</span>
              </div>

              <div className="landing-page-image-and-description">
                <div className="image-for-landing-page">
                  <img
                    src="helmet.png"
                    alt="AI-powered safety detection on a car"
                  />
                </div>
                <div className="description-for-landing-page">
                  <ul>
                    <li>
                      Identifies whether two-wheeler riders are wearing helmets
                      correctly.
                    </li>
                    <li>Detects improper or missing helmets.</li>
                    <li>Helps enforce safety compliance for riders.</li>
                  </ul>
                </div>
              </div>

              <div className="landing-page-bottom-div">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
          <div className="landing-page-for-healmet">
            <div className="healmet-details">
              <div className="landing-page-Heading">
                <span id="landing-page-title">Seatbelt Usage</span>
              </div>

              <div className="landing-page-image-and-description">
                <div className="image-for-landing-page">
                  <img
                    src="/seatbelt.jpg"
                    alt="Seatbelt detection example"
                  />
                </div>
                <div className="description-for-landing-page">
                  <ul>
                    <li>
                      Detects if four-wheeler drivers and passengers are
                      properly wearing seatbelts.
                    </li>
                    <li>Identifies unfastened or missing seatbelts.</li>
                    <li>Promotes road safety for vehicle occupants.</li>
                  </ul>
                </div>
              </div>

              <div className="landing-page-bottom-div">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Enhanced Footer */}
      <footer id="contact" className="footer">
        <div className="container">
          <div className="footer-content">
            <div className="footer-brand">
              <h3 className="footer-logo">SafetyAI</h3>
              <p className="footer-tagline">
                Advanced AI-powered safety compliance monitoring.
              </p>
            </div>
            <div className="footer-contact-new">
              <h4>Contact</h4>
              <p>Email: info@safetyai.com</p>
              <p>Phone: +1 (555) 123-4567</p>
            </div>
          </div>
          <div className="footer-bottom">
            <p>
              &copy; 2024 SafetyAI. Final Year Engineering Project. All rights
              reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default ResponsiveLandingEnhanced;
