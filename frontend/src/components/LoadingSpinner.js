import React from 'react';

const LoadingSpinner = ({ size = 'medium', message = 'Loading...' }) => {
  const sizeMap = {
    small: '20px',
    medium: '40px',
    large: '60px'
  };

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '20px'
    }}>
      <div style={{
        width: sizeMap[size],
        height: sizeMap[size],
        border: '4px solid rgba(0, 191, 255, 0.3)',
        borderTop: '4px solid #00bfff',
        borderRadius: '50%',
        animation: 'spin 1s linear infinite'
      }}></div>
      {message && (
        <p style={{
          marginTop: '10px',
          color: '#00bfff',
          fontSize: '14px'
        }}>
          {message}
        </p>
      )}
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default LoadingSpinner;
