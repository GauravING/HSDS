// Small helper to resolve API base URL and build full endpoint URLs.
// It prefers Vite's VITE_API, then CRA's REACT_APP_API, otherwise uses relative paths
// so the CRA dev proxy will forward requests to backend during development.
const trimTrailingSlash = (url) => String(url).replace(/\/$/, '');

const getRuntimeOverride = () => {
  if (typeof window === 'undefined') return null;
  const override = window.__HSD_API_BASE__ || null;
  return override ? trimTrailingSlash(override) : null;
};

const detectWindowBase = () => {
  if (typeof window === 'undefined') return '';

  // Allow embedding apps to override via global so deployments can set it before bundle loads.
  if (window.__HSD_API_BASE__) {
    return trimTrailingSlash(window.__HSD_API_BASE__);
  }

  const loc = window.location || {};
  const protocol = loc.protocol;

  // If we're served from file:// (common when opening dist/index.html directly), relative
  // URLs break with "Failed to fetch". In that case point to the default backend.
  if (protocol === 'file:') {
    return 'http://localhost:8000';
  }

  const port = loc.port;
  // During CRA/Vite dev servers (ports 3000/5173) we rely on the dev proxy, so return ''
  // to keep requests relative and let the proxy forward to Flask.
  if (port === '3000' || port === '5173') {
    return '';
  }

  // When running from a production host (e.g., Netlify, S3, or nginx) there's usually no
  // proxy for /detect. Default to the backend dev URL unless explicitly overridden via env.
  return 'http://localhost:8000';
};

const getApiBase = () => {
  try {
    const vite = typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API;
    if (vite) return trimTrailingSlash(vite);
  } catch (e) {
    // ignore
  }

  // For CRA (react-scripts) the env vars are available on `process.env` at build time.
  // Use a safe typeof check so this doesn't throw in environments where `process` is not defined.
  const cra = (typeof process !== 'undefined' && process.env && process.env.REACT_APP_API) || null;
  if (cra) return trimTrailingSlash(cra);

  return detectWindowBase();
};

const API_BASE = getApiBase();

export const resolveApiBase = () => getRuntimeOverride() || API_BASE;

export const setRuntimeApiBase = (value) => {
  if (typeof window === 'undefined') return;
  if (!value) {
    delete window.__HSD_API_BASE__;
  } else {
    window.__HSD_API_BASE__ = trimTrailingSlash(value);
  }
};

export const buildUrl = (path) => {
  const base = resolveApiBase();
  if (!path) return base || '/';
  if (base) return `${base}${path.startsWith('/') ? '' : '/'}${path}`.replace(/([^:])\/\//g, '$1/');
  return path;
};

export default API_BASE;
