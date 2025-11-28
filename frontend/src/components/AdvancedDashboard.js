import React from 'react';
import { useNavigate } from "react-router-dom";
import { useTheme } from "../contexts/ThemeContext"; // Assuming you have this context for colors
import { FiPlayCircle, FiUpload } from 'react-icons/fi';
import { motion } from "framer-motion";
import '../styles/AdvancedDashboard.css';

const AdvancedDashboard = () => {
  const navigate = useNavigate();
  // A fallback for colors if useTheme is not available
  const { colors } = useTheme() || { colors: { primary: '#667eea', secondary: '#764ba2', textSecondary: '#64748b' } };

  // Framer Motion variants for animations
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.2 }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1, transition: { duration: 0.5 } }
  };

  return (
    <div className="advanced-dashboard">
      {/* Aurora Background Effect */}
      <div className="aurora-bg">
        <div className="aurora-dot aurora-dot-1"></div>
        <div className="aurora-dot aurora-dot-2"></div>
      </div>
      
      <nav className="dashboard-nav">
        <h1 className="dashboard-title">Safety<span className="title-accent">AI</span></h1>
        <div className="nav-buttons">
          <motion.button className="nav-button active" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            Dashboard
          </motion.button>
          <motion.button onClick={() => navigate("/login")} className="nav-button" whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            Log out
          </motion.button>
        </div>
      </nav>

      <main className="dashboard-main">
        <motion.div
          className="action-grid"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          <motion.div className="action-card" variants={itemVariants}>
            <h2>Live Detection</h2>
            <p>Use your camera for real-time helmet and seatbelt detection</p>
            <motion.button onClick={() => navigate("/live")} className="action-button primary" whileHover={{ scale: 1.05, y: -5 }} whileTap={{ scale: 0.95 }}>
              <FiPlayCircle size={20} />
              Start Live Detection
            </motion.button>
          </motion.div>

          <motion.div className="action-card" variants={itemVariants}>
            <h2>Upload Detection</h2>
            <p>Upload images or videos for batch processing and analysis</p>
            <motion.button onClick={() => navigate("/upload")} className="action-button secondary" whileHover={{ scale: 1.05, y: -5 }} whileTap={{ scale: 0.95 }}>
              <FiUpload size={20} />
              Upload Files
            </motion.button>
          </motion.div>
        </motion.div>
      </main>
    </div>
  );
};

export default AdvancedDashboard;