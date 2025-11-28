// UploadSection.jsx
import React, { useRef, useState } from "react";
import { FaUpload, FaCamera } from "react-icons/fa";
import Webcam from "react-webcam";
import "./UploadSection.css";

const UploadSection = ({ onUpload }) => {
  const webcamRef = useRef(null);
  const [showWebcam, setShowWebcam] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) onUpload(file);
  };

  const captureWebcamImage = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    fetch(imageSrc)
      .then(res => res.blob())
      .then(blob => {
        const file = new File([blob], "webcam-capture.jpg", { type: "image/jpeg" });
        onUpload(file);
      });
  };

  return (
    <div className="upload-section">
      <div className="upload-buttons">
        <label htmlFor="file-upload" className="upload-btn">
          <FaUpload /> Upload File
        </label>
        <input
          id="file-upload"
          type="file"
          accept="image/*,video/*"
          hidden
          onChange={handleFileChange}
        />

        <button className="upload-btn" onClick={() => setShowWebcam(!showWebcam)}>
          <FaCamera /> {showWebcam ? "Close Camera" : "Use Webcam"}
        </button>
      </div>

      {showWebcam && (
        <div className="webcam-box">
          <Webcam
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            width="100%"
            height={240}
          />
          <button className="capture-btn" onClick={captureWebcamImage}>
            Capture & Detect
          </button>
        </div>
      )}
    </div>
  );
};

export default UploadSection;
