# Helmet & Seatbelt Detection System

> Computer-vision platform that ingests images or videos, detects helmet/seatbelt violations, persists results, and surfaces them through a polished React dashboard.

## Table of Contents
1. [Project Highlights](#project-highlights)
2. [System Overview](#system-overview)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [Repository Structure](#repository-structure)
6. [Prerequisites](#prerequisites)
7. [Environment Variables](#environment-variables)
8. [Quick Start](#quick-start)
9. [Core Workflows](#core-workflows)
10. [API Reference](#api-reference)
11. [Model Weights & Data](#model-weights--data)
12. [Testing](#testing)
13. [Deployment Notes](#deployment-notes)
14. [Troubleshooting](#troubleshooting)
15. [Contributing](#contributing)

## Project Highlights
- **Hierarchical inference pipeline** (vehicle → person/head → helmet & seatbelt → ANPR) orchestrated in `backend/app/services/inference.py`.
- **Authenticated uploads** with JWT plus a debug-friendly unauthenticated endpoint for rapid iteration.
- **Rich React UI** featuring drag-and-drop uploads, API base overrides, tabular results with timestamps, manual violation persistence, and raw JSON views.
- **Database-backed violation history** exposed through dedicated routes to review, filter, and update status.
- **Developer tooling** (PowerShell scripts, DB sanity checks, raw model debugging) to keep large vision experiments manageable.

## System Overview
The platform is split into a Flask backend (model execution, persistence, auth) and a Create-React-App frontend (UX for uploads, dashboards, auth flows). Uploads land on `/detect/upload`, are inspected by the inference service, and optionally logged in the `violations` table. The frontend can also route requests to `/detect/raw_video_debug` to bypass auth and view raw summaries.

## Architecture
```
┌──────────────┐      ┌────────────────┐      ┌────────────────────┐
│ React UI     │──▶──▶│ Flask Gateway  │──▶──▶│ Inference Pipeline │
│ (uploads,    │      │ (/detect, auth)│      │  (YOLO, Torch)     │
│ dashboards)  │◀───◀─│ REST responses │◀───◀─│                    │
└──────────────┘      └────────────────┘      └────────────────────┘
         │                         │                      │
         │                         ▼                      ▼
         │                 SQLAlchemy / DB         Annotated media
         ▼
 Raw JSON / Manual save
```

## Tech Stack
| Layer      | Technologies |
|------------|--------------|
| Frontend   | React 19, react-scripts, react-router, Context API, custom CSS modules |
| Backend    | Flask 2.3, Flask-JWT-Extended, SQLAlchemy, OpenCV, Ultralytics YOLOv8, Torch |
| Database   | MySQL (default) – configurable via SQLALCHEMY URI |
| Tooling    | PowerShell/Python helpers, npm scripts, pip virtualenvs |

## Repository Structure
```
backend/
  app.py, config.py, requirements.txt
  app/
    auth/, detect/, violations/
    db/ (models & CRUD)
    services/inference.py     # hierarchical detection pipeline
    models/*.pt               # model weights (large, keep out of git if needed)
frontend/
  package.json, src/
    components/EnhancedUploadDetection.js
    contexts/AuthContext.js, ThemeContext.js
    styles/EnhancedUploadDetection.css
```

## Prerequisites
- **Python 3.10+** with virtualenv support
- **Node.js 18+** (npm 9+)
- **MySQL 8.x** database (or adjust SQLALCHEMY URI for another engine)
- **Model weights** placed in `backend/app/models/`
- Optional: `ffmpeg` on PATH for video transcoding fallbacks

## Environment Variables
Create `.env` files locally (never commit secrets). Templates:

### `backend/.env.example`
```
FLASK_ENV=development
SECRET_KEY=replace-me
JWT_SECRET_KEY=replace-me
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=helmet_user
MYSQL_PASSWORD=helmet_pass
MYSQL_DATABASE=helmet_db
SQLALCHEMY_DATABASE_URI=mysql+mysqlconnector://helmet_user:helmet_pass@localhost:3306/helmet_db
UPLOAD_FOLDER=app/static/uploads
MAX_CONTENT_LENGTH=2147483648   # 2 GB
DEVICE=cpu                      # or cuda:0
```

### `frontend/.env.local.example`
```
REACT_APP_API=http://localhost:8000
REACT_APP_DEFAULT_EMAIL=demo@example.com
REACT_APP_DEFAULT_PASSWORD=changeme
```

Set `window.__HSD_API_BASE__` at runtime (e.g., via `<script>` before bundle) to override the backend URL without rebuilds.

## Quick Start
1. **Backend setup**
   ```powershell
   cd backend
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env   # edit credentials
   python app.py
   ```
2. **Frontend setup**
   ```powershell
   cd frontend
   npm install
   npm start
   ```
3. Visit `http://localhost:3000` (CRA dev server) and log in or use the upload screen.

## Core Workflows
- **Upload & Detect**: `EnhancedUploadDetection` supports drag/drop, frame throttling, annotated previews, and manual DB persistence.
- **Debug Video**: Toggle “Use debug endpoint” to route to `/detect/raw_video_debug` (no auth) and inspect raw summaries.
- **Auto-save**: Backend accepts `auto_save=true` (form/query) to log violations server-side without manual confirmation.
- **Annotated Assets**: Image uploads generate annotated versions saved under `app/static/uploads/results/`.

## API Reference
| Method | Endpoint                  | Auth | Description |
|--------|---------------------------|------|-------------|
| POST   | `/auth/signup`            | No   | Register user (checks allowed email list) |
| POST   | `/auth/login`             | No   | Issue JWT access token |
| GET    | `/auth/profile`           | Yes  | Fetch current user profile |
| POST   | `/detect/upload`          | Yes  | Upload image/video, run full inference, optional auto-save |
| POST   | `/detect/raw_video_debug` | No   | Debug helper returning summaries + raw model outputs |
| POST   | `/detect/debug`           | No   | Image-only debug with raw outputs |
| GET    | `/violations/`            | Yes  | List violations (admin vs user scope) |
| POST   | `/violations/save`        | Yes  | Persist violations from frontend payload |
| GET    | `/api/health`             | No   | Service & database liveness check |

## Model Weights & Data
- Place `.pt` weights inside `backend/app/models`. Large files should stay out of GitHub; provide download links or use Git LFS.
- Upload folders (`app/static/uploads/violations` and `app/static/uploads/results`) should remain empty in git. Use `.gitkeep` if directory presence is required.

## Testing
- **Backend**: add tests around `app/services/inference.py` or route handlers and run via `pytest` (not yet configured—consider adding soon).
- **Frontend**: CRA test runner.
  ```powershell
  cd frontend
  npm test -- --runInBand --watchAll=false
  ```
  Resolve failing suites before release.

## Deployment Notes
- Configure production web server (Gunicorn/Uvicorn + reverse proxy) to serve Flask on port 8000+ with HTTPS termination.
- Set environment variables for database, secrets, and storage paths on the target platform.
- Build frontend with `npm run build` and serve the static bundle (Nginx, S3, Netlify) or integrate with Flask static hosting.
- For containers, bake model weights into an artifact store or mount them at runtime.

## Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `413 Request Entity Too Large` | Upload exceeded server limit | Increase `MAX_CONTENT_LENGTH` and proxy limits |
| `db_ok: false` on `/api/health` | Wrong env vars / DB unreachable | Confirm `.env`, test via `python tools/test_db_conn.py` |
| Blank frontend after fetch | Backend URL mismatch | Update API base input or set `REACT_APP_API` / `window.__HSD_API_BASE__` |
| Video processing fails | Codec unsupported by OpenCV | Install `ffmpeg` so backend can transcode |

## Contributing
1. Fork and clone the repo.
2. Create a feature branch (`git checkout -b feature/my-change`).
3. Follow formatting conventions (Prettier/ESLint for frontend, Black/flake8 optional for backend).
4. Include tests or screenshots for UI changes.
5. Open a pull request describing the motivation and verification steps.

---
For more backend internals, read `backend/backend_info.txt`. For UI customization or API wiring, explore `frontend/src/components/EnhancedUploadDetection.js` and related styles.
