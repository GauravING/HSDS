import React, { createContext, useContext, useEffect, useState } from 'react';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [accessToken, setAccessToken] = useState(() => {
    try {
      return localStorage.getItem('access_token');
    } catch (e) {
      return null;
    }
  });

  useEffect(() => {
    try {
      if (accessToken) localStorage.setItem('access_token', accessToken);
      else localStorage.removeItem('access_token');
    } catch (e) {
      // ignore localStorage issues
    }
  }, [accessToken]);

  const login = (token) => setAccessToken(token);
  const logout = () => setAccessToken(null);

  const authFetch = async (input, init = {}) => {
    const headers = new Headers(init.headers || {});
    if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`);
    const merged = { ...init, headers };
    return fetch(input, merged);
  };

  return (
    <AuthContext.Provider value={{ accessToken, login, logout, authFetch }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  return useContext(AuthContext);
};

export default AuthContext;
