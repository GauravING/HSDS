import React, { useEffect, useRef, useState } from "react";
import { useTheme } from "../contexts/ThemeContext";
import LoadingSpinner from "./LoadingSpinner";
import { buildUrl } from '../utils/api';
import { useAuth } from '../contexts/AuthContext';
import '../styles/EnhancedLiveDetection.css'; // Import the new CSS file

const EnhancedLiveDetection = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const { isDarkMode } = useTheme();
  const [streamStarted, setStreamStarted] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [detectionResults, setDetectionResults] = useState([]);
  const [resultsRaw, setResultsRaw] = useState([]);
  const [error, setError] = useState(null);
  const auth = useAuth();

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: 1280,
          height: 720,
          facingMode: "user",
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setStreamStarted(true);
      setError(null);
    } catch (err) {
      console.error("Camera error:", err);
      setError("Camera access denied or unavailable.");

    }
  };

  // Clean up camera on unmount
  useEffect(() => {
    return () => {
      try {
        stopCamera();
      } catch (e) {
        // ignore
      }
    };
  }, []);

  const startDetection = async () => {
    // Capture current video frame and POST to backend /detect/debug to run inference
    if (!videoRef.current) return;
    setIsDetecting(true);
    try {
      const canvas = canvasRef.current;
      const video = videoRef.current;
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert canvas to blob
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.9));
      if (!blob) throw new Error('Failed to capture frame');

      const form = new FormData();
      form.append('file', blob, 'frame.jpg');
      form.append('debug', 'true');

  // Use authFetch if available so JWT is attached (optional)
  const fetcher = auth && auth.authFetch ? auth.authFetch : fetch;

      const res = await fetcher(buildUrl('/detect/debug'), {
        method: 'POST',
        body: form,
      });

      if (!res.ok) {
        const txt = await res.text().catch(() => '');
        throw new Error(`Detection failed: ${res.status} ${txt}`);
      }

      const data = await res.json().catch(() => ({}));
      // Backend returns { file, results }
      const results = data.results || [];
      // Map results into front-end shape expected by this component
      const mapped = results.map(r => ({
        type: (r.violation_type || r.vehicle_type || 'unknown').toString(),
        confidence: Number((r.confidence_score || (r.metadata || {}).helmet_confidence || 0) || 0),
        status: r.is_violation ? 'not detected' : 'detected',
        raw: r,
      }));

      setDetectionResults(mapped);
  setResultsRaw(results);
    } catch (err) {
      console.error('Live detection error', err);
      setError(String(err));
    } finally {
      setIsDetecting(false);
    }
  };

  const stopCamera = () => {
    const stream = videoRef.current?.srcObject;
    if (stream) {
      const tracks = stream.getTracks();
      tracks.forEach(track => track.stop());
    }
    setStreamStarted(false);
    setIsDetecting(false);
    setDetectionResults([]);
  };

  return (
    <div className={`live-detection-page ${isDarkMode ? 'dark-mode' : ''}`}>
      <div className="detection-container">
        <h1>Live Compliance Detection</h1>
        <p>
          Start your camera to begin real-time detection of helmets and seatbelts.
        </p>

        {error && <div className="error-message">{error}</div>}

        {!streamStarted ? (
          <div className="start-prompt">
            <button
              className="start-camera-btn"
              onClick={startCamera}
            >
              Start Camera
            </button>
          </div>
        ) : (
          <div className="detection-active">
            <div className="video-area">
              <video ref={videoRef} className="video-stream" autoPlay playsInline muted />
              <canvas ref={canvasRef} className="detection-canvas" />
            </div>

            <div className="detection-controls">
              <button
                className={`control-button ${isDetecting ? 'loading' : 'start'}`}
                onClick={startDetection}
                disabled={isDetecting}
              >
                {isDetecting ? <LoadingSpinner size="small" /> : 'Start Detection'}
              </button>
              <button className="control-button stop" onClick={stopCamera} disabled={isDetecting}>
                Stop Camera
              </button>
            </div>

            {resultsRaw && resultsRaw.length > 0 && (
              <div className="results-card">
                <h3>Detection Results</h3>
                {(resultsRaw || []).map((r, index) => (
                  <div key={index} className="detection-item" style={{ textAlign: 'left' }}>
                    <div><strong>Vehicle:</strong> {r.vehicle_type || 'unknown'}</div>
                    <div><strong>Violation:</strong> {r.violation_type || 'none'}</div>
                    <div><strong>Number plate:</strong> {r.number_plate || 'N/A'}</div>
                    <div><strong>Timestamp:</strong> {r.timestamp || 'N/A'}</div>
                    <div><strong>Confidence:</strong> {typeof r.confidence_score !== 'undefined' ? String(r.confidence_score) : (typeof r.confidence !== 'undefined' ? String(r.confidence) : 'N/A')}</div>
                    {r.metadata && (
                      <div style={{ marginTop: 6 }}>
                        <strong>Metadata:</strong>
                        <pre style={{ whiteSpace: 'pre-wrap', margin: 6, background: '#111', color: '#dcdcdc', padding: 8, borderRadius: 6 }}>{JSON.stringify(r.metadata, null, 2)}</pre>
                      </div>
                    )}
                  </div>
                ))}
                <div style={{ marginTop: 10 }}>
                  <button
                    className="save-button"
                    onClick={async () => {
                      if (!resultsRaw || resultsRaw.length === 0) return;
                      try {
                        const { buildUrl } = await import('../utils/api');
                        const fetcher = auth && auth.authFetch ? auth.authFetch : fetch;
                        const res = await fetcher(buildUrl('/violations/save'), {
                          method: 'POST',
                          headers: { 'Content-Type': 'application/json' },
                          body: JSON.stringify({ violations: resultsRaw }),
                        });
                        const j = await res.json().catch(() => ({}));
                        if (!res.ok) {
                          setError(`Save failed: ${res.status} ${j.msg || JSON.stringify(j)}`);
                        } else {
                          // show a simple success message
                          setError(`Saved: ${JSON.stringify(j)}`);
                        }
                      } catch (err) {
                        setError(String(err));
                      }
                    }}
                  >
                    Save violations to DB
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default EnhancedLiveDetection;