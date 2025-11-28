# Frontend → Backend Endpoint Map

This file lists the frontend files that call the backend, which backend route they call, HTTP method, payload shape, whether auth is used, and the expected response shape.

Notes
- The frontend resolves the API base using `frontend/src/utils/api.js`'s `buildUrl(path)` function. It uses `REACT_APP_API` (CRA) or `VITE_API` (Vite) if set; otherwise it returns the path unchanged so the dev proxy or same-origin will be used.
- The `AuthContext` (`frontend/src/contexts/AuthContext.js`) exposes `authFetch(input, init)` which automatically attaches `Authorization: Bearer <token>` when a token is stored.

## Endpoints referenced by frontend

| Frontend file | Backend endpoint (relative) | Method | Payload | Auth? | Expected response |
|---|---:|:---:|---|:---:|---|
| `src/components/UploadCard.tsx` | `/detect/upload` | POST | FormData: { file: File } | Optional (uses `authFetch`) | JSON array of detections or detailed `{ file, results, processing_time }` |
| `src/components/EnhancedUploadDetection.js` | `/detect/upload` | POST | FormData: { file: File, debug: 'true' (opt) } | Optional | `{ file: string, results: [ ...violations ], processing_time }` |
| `src/components/EnhancedLiveDetection.js` | `/detect/debug` | POST | FormData: { file: Blob, debug: 'true' } | Optional | `{ file, results: [ ... ] }` (debug raw outputs; no DB writes) |
| `src/components/UploadCard.tsx` (constant) | `/detect/upload` | POST | FormData | Optional | Detection[] (component expects an array) |
| `src/components/LoginForm.js` | `/auth/login` | POST | JSON: { email, password } | No | 200 + `{ access_token: string, ... }` on success; error JSON otherwise |
| `src/components/EnhancedStylishLoginForm.js` | `/auth/login` | POST | JSON: { email, password } | No | same as above |
| `src/components/SignupForm.js` | `/auth/allowed_emails` | GET | — | No | `{ allowed_emails: [ ... ] }` (optional whitelist) |
| `src/components/SignupForm.js` | `/auth/signup` | POST | JSON: { username, email, password, full_name } | No | 201 on success; error JSON otherwise |
| `src/components/EnhancedSignupForm.js` | `/auth/allowed_emails` | GET | — | No | `{ allowed_emails: [ ... ] }` |
| `src/components/EnhancedSignupForm.js` | `/auth/signup` | POST | JSON: { username, email, password, full_name } | No | 201 on success |

## Where to change the backend base URL

- `frontend/src/utils/api.js` — `buildUrl(path)` will prepend the base if `REACT_APP_API` (CRA) or `VITE_API` (Vite) is set. To point the frontend to your backend at `http://localhost:8000`, either:
  - set the environment variable before starting the dev server (PowerShell):

```powershell
$env:REACT_APP_API='http://localhost:8000'
npm start
```

  - or create a `.env` file in the frontend root (or `updated_frontend`) with:

```
REACT_APP_API=http://localhost:8000
```

## Auth helper (adds Authorization header)
- `frontend/src/contexts/AuthContext.js` exposes `authFetch(input, init)` which wraps fetch and sets `Authorization: Bearer <token>` when a token exists. Many components use `const fetcher = auth && auth.authFetch ? auth.authFetch : fetch;` before calling `buildUrl(...)`.

## Backend route locations (for reference)
- `backend/app/detect/routes.py` — blueprint `url_prefix='/detect'` implements `/detect/upload` and `/detect/debug`.
- `backend/app/auth/routes.py` — contains `/auth/login`, `/auth/signup`, `/auth/allowed_emails`.
- `backend/app/violations/routes.py` — blueprint `url_prefix='/violations'` (used by admin/violation listing UI if present).

## Quick checklist to verify connectivity
1. Ensure backend is running (default: `http://localhost:8000`).
2. Set `REACT_APP_API` (or `VITE_API`) to backend URL and restart frontend dev server.
3. Use the app UI (Upload or Live Detection) to POST to `/detect/debug` or `/detect/upload` and inspect the JSON response and server logs.

If you'd like, I can also:
- generate a single-file `frontend/ENDPOINTS.json` mapping for programmatic use,
- add a small health-check page to the frontend that pings these endpoints and displays status,
- or update `.env.development` in `updated_frontend` to point to `http://localhost:8000` for you.

---
Created by the code-assistant on behalf of the developer to document frontend→backend calls.
