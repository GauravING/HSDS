import React, { useEffect, useRef, useState } from "react";
import { FiUploadCloud } from 'react-icons/fi';
import { useTheme } from "../contexts/ThemeContext";
import { useAuth } from '../contexts/AuthContext';
import LoadingSpinner from "./LoadingSpinner";
import '../styles/EnhancedUploadDetection.css'; // Import the new CSS file
import defaultApiBase, { buildUrl, resolveApiBase, setRuntimeApiBase } from '../utils/api';

const prettyPrintJson = (text) => {
  if (!text) return '';
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch (e) {
    return text;
  }
};

const parseErrorMessage = (payload) => {
  if (!payload) return 'Request failed';
  try {
    const parsed = JSON.parse(payload);
    return parsed.error || parsed.detail || parsed.msg || 'Request failed';
  } catch (e) {
    return payload;
  }
};

const ensureAbsoluteUrl = (path) => {
  if (!path) return null;
  if (/^https?:\/\//i.test(path)) {
    return path;
  }
  return buildUrl(path);
};

const formatTimestamp = (value) => {
  if (!value) return 'N/A';
  const date = new Date(value);
  if (!Number.isNaN(date.getTime())) {
    return date.toLocaleString();
  }
  return value;
};

const formatConfidence = (value) => {
  if (value === null || value === undefined) return '—';
  const num = Number(value);
  if (Number.isNaN(num)) return value;
  const normalized = num <= 1 ? num * 100 : num;
  return `${normalized.toFixed(1)}%`;
};

/* Legacy implementation retained temporarily for reference.
const EnhancedUploadDetection = () => {
  const { isDarkMode } = useTheme();
  const auth = useAuth();
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState(null);
  const [resultsRaw, setResultsRaw] = useState([]);
  const [annotatedUrl, setAnnotatedUrl] = useState(null);
  const [responseJson, setResponseJson] = useState(null);
  const [useRawVideoDebug, setUseRawVideoDebug] = useState(false);
  const [maxFrames, setMaxFrames] = useState(30);
  const [frameSkip, setFrameSkip] = useState(5);
  const [dragOver, setDragOver] = useState(false);
  const [apiBase, setApiBase] = useState(() => resolveApiBase() || '');
  const fileInputRef = useRef(null);
  return (
    <div className={`upload-page-container ${isDarkMode ? 'dark-mode' : ''}`}>
      <div className="upload-container">
        <h1>Upload for Detection</h1>
        <p>
          Upload an image or video file to check for helmet and seatbelt compliance. (Max: 2 GB)
        </p>

        <div className="api-base-config" style={{ marginBottom: 16, textAlign: 'left' }}>
          <label style={{ fontWeight: 600 }}>Backend API base URL</label>
          <div style={{ display: 'flex', gap: 8, marginTop: 6 }}>
            <input
              type="text"
              value={apiBase}
              onChange={(e) => handleApiBaseChange(e.target.value)}
              placeholder="http://localhost:8000"
              style={{ flex: 1 }}
            />
            <button type="button" onClick={resetApiBase} className="small-button">
              Reset
            </button>
          </div>
          <small>Requests will be sent to: {apiBase ? `${apiBase}` : '(relative via dev proxy)'}</small>
        </div>

        <div className="endpoint-toggle">
          <label>
            <input
              type="checkbox"
              checked={useRawVideoDebug}
              onChange={() => setUseRawVideoDebug((prev) => !prev)}
            />
            Use /detect/raw_video_debug (debug, unauthenticated)
          </label>
          {useRawVideoDebug && (
            <div className="debug-options">
              <label>
                Frame skip
                <input
                  type="number"
                  min="1"
                  value={frameSkip}
                  onChange={(e) => setFrameSkip(Number(e.target.value) || 5)}
                  style={{ marginLeft: 8, width: 80 }}
                />
              </label>
              <label style={{ marginLeft: 16 }}>
                Max frames
                <input
                  type="number"
                  min="1"
                  value={maxFrames}
                  onChange={(e) => setMaxFrames(Number(e.target.value) || 30)}
                  style={{ marginLeft: 8, width: 80 }}
                />
              </label>
            </div>
          )}
        </div>

        {!file ? (
          <div
            className={`drag-drop-area ${dragOver ? 'drag-over' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={triggerFileInput}
          >
            <FiUploadCloud className="upload-icon" />
            <p>Drag & drop a file here</p>
            <p>or</p>
            <p className="browse-link">Browse files</p>
          </div>
        ) : (
          <div className="file-preview">
            {file.type.startsWith('image/') ? (
              <img src={preview} alt="File preview" className="preview-image" />
            ) : (
              <video src={preview} controls className="preview-video" />
            )}
            <p className="file-name">{file.name}</p>
          </div>
        )}

        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="image/*,video/*"
          style={{ display: 'none' }}
        />
        
        <button 
          onClick={handleUpload} 
          disabled={!file || uploading}
          className={`upload-button ${uploading ? 'loading' : ''}`}
        >
          {uploading ? <LoadingSpinner size="small" /> : 'Start Detection'}
        </button>
        
        {results && results.error && (
          <div className="results-card">
            <h3>Error</h3>
            <p className="error-text">{results.error}</p>
          </div>
        )}

        {results && !results.error && (
          <div className="results-card">
            <h3>Detection Results</h3>
            <p className="results-file-info">
              File: {results.filename} ({results.type})
            </p>

            {annotatedUrl && (
              <div style={{ margin: '12px 0' }}>
                <h4>Annotated Result</h4>
                <img src={annotatedUrl} alt="Annotated result" style={{ maxWidth: '100%', borderRadius: 8 }} />
              </div>
            )}

            {(results.results || []).length === 0 ? (
              <p>No detections returned by the backend.</p>
            ) : (
              <div className="results-table-wrapper">
                <table className="results-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Frame</th>
                      <th>Vehicle Type</th>
                      <th>Violation</th>
                      <th>Number Plate</th>
                      <th>Timestamp</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(results.results || []).map((r, index) => (
                      <tr key={`${r.frame_index ?? index}-${index}`}>
                        <td>{index + 1}</td>
                        <td>{typeof r.frame_index === 'number' ? r.frame_index : '–'}</td>
                        <td>{r.vehicle_type || 'Unknown'}</td>
                        <td>{r.violation || r.violation_type || 'None'}</td>
                        <td>{r.number_plate_text || r.number_plate || 'N/A'}</td>
                        <td>{formatTimestamp(r.timestamp)}</td>
                        <td>{formatConfidence(r.confidence_score ?? r.vehicle_confidence ?? r.confidence)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {(results.results || []).some((r) => r.metadata) && (
                  <details className="metadata-details">
                    <summary>Raw metadata (JSON)</summary>
                    <pre>{JSON.stringify(results.results, null, 2)}</pre>
                  </details>
                )}
              </div>
            )}

            <p className="processing-time-info">
              Processing time: {results.processingTime ? `${results.processingTime} sec` : 'N/A'}
            </p>
            {results && !results.error && (
              <div style={{ marginTop: 10 }}>
                <button onClick={saveViolations} disabled={uploading} className="save-button">
                  {uploading ? 'Saving…' : 'Save violations to DB'}
                </button>
                {results.saved_summary && (
                  <pre style={{ marginTop: 8, background: '#f3f3f3', padding: 8 }}>{JSON.stringify(results.saved_summary, null, 2)}</pre>
                )}
              </div>
            )}
          </div>
        )}

        {responseJson && (
          <div className="results-card" style={{ marginTop: 16 }}>
            <h3>Raw Response JSON</h3>
            <pre style={{ maxHeight: 300, overflow: 'auto' }}>{responseJson}</pre>
          </div>
        )}
      </div>
    </div>
  );
};
            <label className="toggle-field">
              <input
                type="checkbox"
                checked={useRawVideoDebug}
                onChange={() => setUseRawVideoDebug((prev) => !prev)}
              />
              <div>
                <strong>Debug endpoint</strong>
                <p>Use /detect/raw_video_debug (no auth, raw summaries)</p>
              </div>
            </label>
            {useRawVideoDebug && (
              <div className="debug-controls">
                <label className="field">
                  <span>Frame skip</span>
                  <input
                    type="number"
                    min="1"
                    value={frameSkip}
                    onChange={(e) => setFrameSkip(Number(e.target.value) || 5)}
                  />
                </label>
                <label className="field">
                  <span>Max frames</span>
                  <input
                    type="number"
                    min="1"
                    value={maxFrames}
                    onChange={(e) => setMaxFrames(Number(e.target.value) || 30)}
                  />
                </label>
              </div>
            )}
          </div>
        </section>

        <section className="panel-card preview-panel">
          <h2 className="panel-title">Upload</h2>
          {!file ? (
            <div
              className={`drag-drop-area ${dragOver ? 'drag-over' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={triggerFileInput}
            >
              <FiUploadCloud className="upload-icon" />
              <p>Drag & drop a file here</p>
              <p>or</p>
              <p className="browse-link">Browse files</p>
            </div>
          ) : (
            <div className="file-preview fancy-shadow">
              {file.type.startsWith('image/') ? (
                <img src={preview} alt="File preview" className="preview-image" />
              ) : (
                <video src={preview} controls className="preview-video" />
              )}
              <div className="file-pill">{file.name}</div>
            </div>
          )}

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="image/*,video/*"
            style={{ display: 'none' }}
          />

          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            className={`upload-button ${uploading ? 'loading' : ''}`}
          >
            {uploading ? <LoadingSpinner size="small" /> : 'Start Detection'}
          </button>
        </section>
      </div>

      <div className="results-stack">
        {results && results.error && (
          <div className="results-card error-card">
            <h3>Error</h3>
            <p className="error-text">{results.error}</p>
          </div>
        )}

        {results && !results.error && (
          <div className="results-card">
            <div className="results-header">
              <div>
                <p className="eyebrow">Detection Results</p>
                <h3>{results.filename}</h3>
                <p className="results-file-info">{results.type} • {(results.results || []).length} entities</p>
              </div>
              <div className="stats-grid">
                <div className="stat-pill">
                  <span>Processing time</span>
                  <strong>{results.processingTime ? `${results.processingTime}` : 'N/A'}</strong>
                </div>
                <div className="stat-pill">
                  <span>Violations</span>
                  <strong>{(results.results || []).filter((r) => r.violation || r.violation_type).length}</strong>
                </div>
              </div>
            </div>

            {annotatedUrl && (
              <div className="annotated-preview">
                <img src={annotatedUrl} alt="Annotated result" />
              </div>
            )}

            {(results.results || []).length === 0 ? (
              <p>No detections returned by the backend.</p>
            ) : (
              <div className="results-table-wrapper">
                <table className="results-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Frame</th>
                      <th>Vehicle Type</th>
                      <th>Violation</th>
                      <th>Number Plate</th>
                      <th>Timestamp</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(results.results || []).map((r, index) => (
                      <tr key={`${r.frame_index ?? index}-${index}`}>
                        <td>{index + 1}</td>
                        <td>{typeof r.frame_index === 'number' ? r.frame_index : '–'}</td>
                        <td>{r.vehicle_type || 'Unknown'}</td>
                        <td>{r.violation || r.violation_type || 'None'}</td>
                        <td>{r.number_plate_text || r.number_plate || 'N/A'}</td>
                        <td>{formatTimestamp(r.timestamp)}</td>
                        <td>{formatConfidence(r.confidence_score ?? r.vehicle_confidence ?? r.confidence)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {(results.results || []).some((r) => r.metadata) && (
                  <details className="metadata-details">
                    <summary>Raw metadata (JSON)</summary>
                    <pre>{JSON.stringify(results.results, null, 2)}</pre>
                  </details>
                )}
              </div>
            )}

            <div className="results-actions">
              <button onClick={saveViolations} disabled={uploading} className="save-button">
                {uploading ? 'Saving…' : 'Save violations to DB'}
              </button>
              {results.saved_summary && (
                <pre>{JSON.stringify(results.saved_summary, null, 2)}</pre>
              )}
            </div>
          </div>
        )}

        {responseJson && (
          <div className="results-card">
            <h3>Raw Response JSON</h3>
            <pre style={{ maxHeight: 300, overflow: 'auto' }}>{responseJson}</pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default EnhancedUploadDetection;
*/

const EnhancedUploadDetection = () => {
  const { isDarkMode } = useTheme();
  const auth = useAuth();

  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [results, setResults] = useState(null);
  const [annotatedUrl, setAnnotatedUrl] = useState(null);
  const [responseJson, setResponseJson] = useState(null);
  const [useRawVideoDebug, setUseRawVideoDebug] = useState(false);
  const [maxFrames, setMaxFrames] = useState('30');
  const [frameSkip, setFrameSkip] = useState('5');
  const [dragOver, setDragOver] = useState(false);
  const [apiBase, setApiBase] = useState(() => resolveApiBase() || '');

  const fileInputRef = useRef(null);

  useEffect(() => () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
  }, [previewUrl]);

  const resetResultState = () => {
    setResults(null);
    setAnnotatedUrl(null);
    setResponseJson(null);
  };

  const setPreviewForFile = (selectedFile) => {
    setPreviewUrl((current) => {
      if (current) {
        URL.revokeObjectURL(current);
      }
      if (!selectedFile) {
        return null;
      }
      return URL.createObjectURL(selectedFile);
    });
  };

  const handleFileChange = (event) => {
    const selected = event.target.files && event.target.files[0];
    setFile(selected || null);
    setPreviewForFile(selected || null);
    if (selected) {
      resetResultState();
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    const dropped = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (!dropped) return;
    setFile(dropped);
    setPreviewForFile(dropped);
    resetResultState();
  };

  const triggerFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const clearSelection = () => {
    setFile(null);
    setPreviewForFile(null);
    resetResultState();
  };

  const handleApiBaseChange = (value) => {
    setApiBase(value);
    setRuntimeApiBase(value || null);
  };

  const resetApiBase = () => {
    const fallback = defaultApiBase || '';
    setApiBase(fallback);
    setRuntimeApiBase(fallback || null);
  };

  const normalizeResultsPayload = (data, debugMode) => {
    const rows = Array.isArray(data?.results)
      ? data.results
      : Array.isArray(data?.summaries)
        ? data.summaries
        : [];
    return {
      filename: data?.file || file?.name || 'Uploaded file',
      type: data?.type || file?.type || (file?.name ? file.name.split('.').pop() : 'upload'),
      processingTime: data?.processing_time || data?.processingTime || (debugMode ? 'Debug run' : null),
      results: rows,
      saved_summary: data?.saved_summary,
    };
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setResults(null);
    setAnnotatedUrl(null);
    setResponseJson(null);

    try {
      const formData = new FormData();
      formData.append('file', file);
      if (frameSkip) formData.append('frame_skip', frameSkip);
      if (maxFrames) formData.append('max_frames', maxFrames);
      if (!useRawVideoDebug) {
        formData.append('debug', 'true');
      }

      const endpoint = useRawVideoDebug ? '/detect/raw_video_debug' : '/detect/upload';
      const url = buildUrl(endpoint);
      const requestInit = { method: 'POST', body: formData };
      const fetcher = useRawVideoDebug ? fetch : auth?.authFetch || fetch;
      const response = await fetcher(url, requestInit);
      const text = await response.text();
      setResponseJson(prettyPrintJson(text));

      if (!response.ok) {
        throw new Error(parseErrorMessage(text));
      }

      const data = text ? JSON.parse(text) : {};
      setResults(normalizeResultsPayload(data, useRawVideoDebug));
      const annotated = data?.annotated_image || data?.annotatedUrl;
      setAnnotatedUrl(ensureAbsoluteUrl(annotated));
    } catch (error) {
      setResults({ error: error.message || 'Upload failed' });
    } finally {
      setUploading(false);
    }
  };

  const saveViolations = async () => {
    if (!results || !Array.isArray(results.results) || results.results.length === 0) return;

    setSaving(true);
    try {
      const url = buildUrl('/violations/save');
      const payload = JSON.stringify({ violations: results.results });
      const init = {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: payload,
      };
      const fetcher = auth?.authFetch || fetch;
      const response = await fetcher(url, init);
      const text = await response.text();

      setResponseJson((prev) => {
        const pretty = prettyPrintJson(text);
        if (!prev) return pretty;
        return `${prev}\n\n--- Save Response ---\n${pretty}`;
      });

      if (!response.ok) {
        throw new Error(parseErrorMessage(text));
      }

      const data = text ? JSON.parse(text) : {};
      setResults((prev) => (prev ? { ...prev, saved_summary: data } : prev));
    } catch (error) {
      setResults((prev) => (prev ? { ...prev, saved_summary: { error: error.message || 'Failed to save' } } : prev));
    } finally {
      setSaving(false);
    }
  };

  const hasDetections = Array.isArray(results?.results) && results.results.length > 0;

  return (
    <div className={`upload-page-container ${isDarkMode ? 'dark-mode' : ''}`}>
      <div className="page-header">
        <div>
          <p className="eyebrow">Helmet & Seatbelt Monitoring</p>
          <h1>Upload Detection</h1>
          <p className="subtitle">Send an image or video for compliance review and receive detailed violations, timestamps, and OCR plates.</p>
        </div>
        <div className="header-action">
          <button type="button" className="ghost-button" onClick={resetApiBase}>
            Reset API Base
          </button>
        </div>
      </div>

      <div className="vertical-stack">
        <section className="panel-card control-panel">
          <h2 className="panel-title">Upload configuration</h2>
          <div className="controls-grid">
            <div className="field">
              <label>Backend API base URL</label>
              <div className="api-base-row">
                <input
                  type="text"
                  value={apiBase}
                  onChange={(e) => handleApiBaseChange(e.target.value)}
                  placeholder="http://localhost:8000"
                />
                <button type="button" className="small-button" onClick={resetApiBase}>
                  Default
                </button>
              </div>
              <small className="field-hint">Requests will be sent to: {apiBase || '(relative via dev proxy)'}</small>
            </div>

            <label className="toggle-field">
              <input
                type="checkbox"
                checked={useRawVideoDebug}
                onChange={() => setUseRawVideoDebug((prev) => !prev)}
              />
              <div>
                <strong>Use debug endpoint</strong>
                <p>Route uploads through /detect/raw_video_debug (no auth required).</p>
              </div>
            </label>

            {useRawVideoDebug && (
              <div className="debug-controls">
                <label className="field">
                  <span>Frame skip</span>
                  <input
                    type="number"
                    min="1"
                    value={frameSkip}
                    onChange={(e) => setFrameSkip(e.target.value)}
                  />
                </label>
                <label className="field">
                  <span>Max frames</span>
                  <input
                    type="number"
                    min="1"
                    value={maxFrames}
                    onChange={(e) => setMaxFrames(e.target.value)}
                  />
                </label>
              </div>
            )}
          </div>

          <div className="divider" />

          <div className="upload-panel-body">
            {!file ? (
              <div
                className={`drag-drop-area ${dragOver ? 'drag-over' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={triggerFileInput}
              >
                <FiUploadCloud className="upload-icon" />
                <p>Drag & drop a file here</p>
                <p>or</p>
                <p className="browse-link">Browse files</p>
              </div>
            ) : (
              <div className="file-preview fancy-shadow">
                {file.type.startsWith('image/') ? (
                  <img src={previewUrl} alt="File preview" className="preview-image" />
                ) : (
                  <video src={previewUrl} controls className="preview-video" />
                )}
                <div className="file-pill">{file.name}</div>
              </div>
            )}

            <input
              type="file"
              ref={fileInputRef}
              accept="image/*,video/*"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />

            <div className="upload-actions">
              <button
                onClick={handleUpload}
                disabled={!file || uploading}
                className={`upload-button ${uploading ? 'loading' : ''}`}
              >
                {uploading ? <LoadingSpinner size="small" /> : 'Start detection'}
              </button>
              {file && (
                <button type="button" className="ghost-button" onClick={clearSelection}>
                  Remove file
                </button>
              )}
            </div>
          </div>
        </section>

        <section className="results-stack">
          {results?.error && (
            <div className="results-card error-card">
              <h3>Error</h3>
              <p className="error-text">{results.error}</p>
            </div>
          )}

          {!results?.error && results && (
            <div className="results-card">
              <div className="results-header">
                <div>
                  <p className="eyebrow">Detection results</p>
                  <h3>{results.filename}</h3>
                  <p className="results-file-info">
                    {results.type} • {results.results?.length || 0} entries
                  </p>
                </div>
                <div className="stats-grid">
                  <div className="stat-pill">
                    <span>Processing time</span>
                    <strong>{results.processingTime || '—'}</strong>
                  </div>
                  <div className="stat-pill">
                    <span>Violations</span>
                    <strong>{(results.results || []).filter((r) => r.violation || r.violation_type || r.is_violation).length}</strong>
                  </div>
                </div>
              </div>

              {annotatedUrl && (
                <div className="annotated-preview">
                  <img src={annotatedUrl} alt="Annotated result" />
                </div>
              )}

              {!hasDetections ? (
                <p>No detections returned by the backend.</p>
              ) : (
                <div className="results-table-wrapper">
                  <table className="results-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>Frame</th>
                        <th>Vehicle Type</th>
                        <th>Violation</th>
                        <th>Number Plate</th>
                        <th>Timestamp</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.results.map((entry, index) => (
                        <tr key={`${entry.frame_index ?? index}-${index}`}>
                          <td>{index + 1}</td>
                          <td>{typeof entry.frame_index === 'number' ? entry.frame_index : '–'}</td>
                          <td>{entry.vehicle_type || 'Unknown'}</td>
                          <td>{entry.violation || entry.violation_type || 'None'}</td>
                          <td>{entry.number_plate_text || entry.number_plate || 'N/A'}</td>
                          <td>{formatTimestamp(entry.timestamp)}</td>
                          <td>{formatConfidence(entry.confidence_score ?? entry.vehicle_confidence ?? entry.confidence)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {results.results.some((row) => row.metadata) && (
                    <details className="metadata-details">
                      <summary>Raw metadata (JSON)</summary>
                      <pre>{JSON.stringify(results.results, null, 2)}</pre>
                    </details>
                  )}
                </div>
              )}

              <div className="results-actions">
                <button onClick={saveViolations} disabled={!hasDetections || saving} className="save-button">
                  {saving ? 'Saving…' : 'Save violations to DB'}
                </button>
                {results.saved_summary && (
                  <pre>{JSON.stringify(results.saved_summary, null, 2)}</pre>
                )}
              </div>
            </div>
          )}

          {responseJson && (
            <div className="results-card raw-response-card">
              <h3>Raw response JSON</h3>
              <pre style={{ maxHeight: 320, overflow: 'auto' }}>{responseJson}</pre>
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

export default EnhancedUploadDetection;