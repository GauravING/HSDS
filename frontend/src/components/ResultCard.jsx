// ResultCard.jsx
import React from "react";
import "./ResultCard.css";
import { FaTimes } from "react-icons/fa";

const ResultCard = ({ image, results, onClose }) => {
  return (
    <div className="result-card glass">
      <button className="close-btn" onClick={onClose}>
        <FaTimes />
      </button>
      <h2>Detection Result</h2>
      <img src={image} alt="Preview" className="preview-img" />
      <div className="result-details">
        <p>🪖 Helmet Detected: <strong>{results.helmet}</strong></p>
        <p>🧷 Seatbelt Detected: <strong>{results.seatbelt}</strong></p>
      </div>
    </div>
  );
};

export default ResultCard;
