import React, { useState, useEffect } from "react";
import "./Landing.css";
import Lottie from "lottie-react";
import helmetAnimation from "../assets/helmet.json";
import LoginForm from "./LoginForm";
import SignupForm from "./SignupForm";
import Particles from "react-tsparticles";

const quotes = [
  "Safety first — always wear a helmet.",
  "Seatbelts save lives.",
  "Drive safe, stay safe.",
  "Your family is waiting — be careful.",
];

const Landing = () => {
  const [quoteIndex, setQuoteIndex] = useState(0);
  const [showLogin, setShowLogin] = useState(false);
  const [showSignup, setShowSignup] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setQuoteIndex((prev) => (prev + 1) % quotes.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const openLogin = () => {
    setShowLogin(true);
    setShowSignup(false);
  };

  const openSignup = () => {
    setShowSignup(true);
    setShowLogin(false);
  };

  const closeModals = () => {
    setShowLogin(false);
    setShowSignup(false);
  };

  return (
    <div className="landing-container">
      {/* Particle Background */}
      <Particles
        options={{
          background: { color: "#0f2027" },
          particles: {
            number: { value: 60 },
            size: { value: 3 },
            move: { speed: 1 },
            opacity: { value: 0.6 },
            links: { enable: true, color: "#00ffcc" },
          },
        }}
        className="particles"
      />

      {/* Main Content */}
      <div className="content fade-up">
        <div className="icon-container tilt">
          <Lottie animationData={helmetAnimation} loop={true} className="helmet-icon" />
        </div>
        <h1 className="project-title glow-text">
          Helmet & Seatbelt Detection System
        </h1>
        <p className="quote">{quotes[quoteIndex]}</p>

        <div className="button-container fade-up-delayed">
          <button className="btn login-btn" onClick={openLogin}>
            Login
          </button>
          <button className="btn signup-btn" onClick={openSignup}>
            Signup
          </button>
        </div>
      </div>

      {/* Modals */}
      {showLogin && <LoginForm onClose={closeModals} />}
      {showSignup && <SignupForm onClose={closeModals} />}
    </div>
  );
};

export default Landing;
