import React, { createContext, useContext } from 'react';

const ThemeContext = createContext();

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

export const ThemeProvider = ({ children }) => {
  const theme = {
    isDarkMode: false,
    toggleTheme: () => {},
    colors: {
      primary: '#007bff',
      secondary: '#dc3545',
      background: '#ffffff',
      surface: '#f8f9fa',
      text: '#000000',
      textSecondary: '#666666',
      border: '#dee2e6'
    }
  };

  return (
    <ThemeContext.Provider value={theme}>
      {children}
    </ThemeContext.Provider>
  );
};
