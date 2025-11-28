import React, { useState } from "react";

const UploadDetection = () => {
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = () => {
    if (file) {
      alert(`Uploaded file: ${file.name}`);
      // TODO: Add backend upload API here
    } else {
      alert("Please select a file first.");
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "url('/pexels-pixabay-210182.jpg') no-repeat center center",
        backgroundSize: "cover",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: "20px",
      }}
    >
      <div
        style={{
          background: "rgba(0, 0, 0, 0.75)",
          padding: "40px",
          borderRadius: "12px",
          textAlign: "center",
          color: "white",
          width: "90%",
          maxWidth: "500px",
          boxShadow: "0 0 20px rgba(255, 255, 255, 0.1)",
        }}
      >
        <h2 style={{ marginBottom: "10px" }}>Upload Detection</h2>
        <p>Select an image or video to perform helmet & seatbelt detection.</p>

        <input
          type="file"
          accept="image/*,video/*"
          onChange={handleFileChange}
          style={{
            margin: "20px 0",
            padding: "10px",
            borderRadius: "6px",
            width: "100%",
          }}
        />

        <button
          onClick={handleUpload}
          style={{
            padding: "10px 20px",
            backgroundColor: "#00bfff",
            color: "#fff",
            border: "none",
            borderRadius: "6px",
            fontWeight: "bold",
            cursor: "pointer",
          }}
        >
          Upload
        </button>

        {file && (
          <div style={{ marginTop: "20px" }}>
            <h4>Preview:</h4>
            {file.type.startsWith("image/") ? (
              <img
                src={URL.createObjectURL(file)}
                alt="preview"
                style={{ width: "100%", borderRadius: "8px", marginTop: "10px" }}
              />
            ) : (
              <video width="100%" controls style={{ borderRadius: "8px", marginTop: "10px" }}>
                <source src={URL.createObjectURL(file)} type={file.type} />
              </video>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadDetection;
