import React, { useEffect, useRef, useState } from "react";

const LiveDetection = () => {
  const videoRef = useRef(null);
  const [streamStarted, setStreamStarted] = useState(false);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 1280,
          height: 720,
          facingMode: "user", // Force front camera
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play(); // Ensure it starts
      }

      setStreamStarted(true);
    } catch (err) {
      console.error("Camera error:", err);
      alert("Camera access denied or unavailable.");
    }
  };

  useEffect(() => {
    return () => {
      const videoElement = videoRef.current;
      if (videoElement && videoElement.srcObject) {
        videoElement.srcObject.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #0f0f0f, #1a1a1a)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: "20px",
      }}
    >
      <div
        style={{
          background: "rgba(255, 255, 255, 0.05)",
          padding: "40px",
          borderRadius: "16px",
          textAlign: "center",
          color: "white",
          width: "90%",
          maxWidth: "650px",
          boxShadow: "0 0 30px rgba(0, 191, 255, 0.4)",
        }}
      >
        <h2 style={{ marginBottom: "15px" }}>🚀 Live Detection</h2>
        <p style={{ marginBottom: "25px", color: "#ccc" }}>
          Use your webcam to perform real-time helmet & seatbelt detection.
        </p>

        {!streamStarted && (
          <button
            onClick={startCamera}
            style={{
              padding: "14px 28px",
              background: "linear-gradient(90deg, #00bfff, #1e90ff)",
              border: "none",
              borderRadius: "8px",
              color: "#fff",
              fontWeight: "bold",
              cursor: "pointer",
              fontSize: "1rem",
            }}
          >
            🎥 Start Live Detection
          </button>
        )}

        {streamStarted && (
          <div style={{ marginTop: "20px" }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{
                borderRadius: "12px",
                maxHeight: "450px",
                width: "100%",
                border: "3px solid rgba(0, 191, 255, 0.5)",
                boxShadow: "0 0 20px rgba(0, 191, 255, 0.7)",
                background: "black", // Fallback if no video
              }}
            />
            <div
              style={{
                marginTop: "10px",
                background: "rgba(0, 191, 255, 0.7)",
                padding: "6px 12px",
                borderRadius: "8px",
                color: "white",
                fontSize: "0.9rem",
                fontWeight: "bold",
              }}
            >
              Live Camera Active
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default LiveDetection;
