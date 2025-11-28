import React from "react";
import { useNavigate } from "react-router-dom";
import "../style.css";

const Dashboard = () => {
  const navigate = useNavigate();

  return (
    <div className="dashboard-container">
      <nav className="dashboard-navbar">
        <h2>
          Helmet <span>&</span> Seatbelt Detection
        </h2>
        <div className="nav-buttons">
          <button onClick={() => navigate("/")}>Home</button> {/* ✅ NEW */}
          <button onClick={() => navigate("/live")}>Live Detection</button>
          <button onClick={() => navigate("/upload")}>Upload</button>
        </div>
      </nav>

      <div className="dashboard-content">
        <h1>Ensuring Safety with AI</h1>
        <p>
          Our system uses real-time AI to detect helmets and seatbelts on road users.
          Use your webcam or upload media.
        </p>
        <div className="card-container">
          <div className="card">
            <h2>Live Detection</h2>
            <p>Use your camera for real-time detection</p>
            <button onClick={() => navigate("/live")}>Get Started</button>
          </div>
          <div className="card">
            <h2>Upload Detection</h2>
            <p>Check an image or video upload</p>
            <button onClick={() => navigate("/upload")}>Get Started</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
