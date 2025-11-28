# app/__init__.py
from flask import Flask, request
from sqlalchemy import text
# Ensure .env is loaded early for the app factory (defensive)
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parents[1] / '.env')

from config import Config
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from app.db import db_init
from app.db import db
from app.db.models import AllowedEmail
from pathlib import Path
import os


def create_app():
    app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))
    app.config.from_object(Config)
    # Configure CORS to allow the frontend dev server to send Authorization
    # headers and cookies during development. When credentials are used the
    # Access-Control-Allow-Origin header must be the specific origin (not '*').
    # Read allowed frontend origins from env (comma-separated) for flexibility.
    # Include common dev ports (3000 for CRA, 5173 for Vite) by default
    frontend_origins_env = os.environ.get("FRONTEND_ORIGINS", "")
    if frontend_origins_env:
        origins = [o.strip() for o in frontend_origins_env.split(",") if o.strip()]
    else:
        origins = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]

    # When the frontend is opened directly from the filesystem (file://index.html)
    # browsers send Origin: null and block responses unless the backend explicitly
    # whitelists "null". Allow this in debug builds or when ALLOW_FILE_ORIGINS=true
    # so users can double-click the build output and still hit the API.
    # Default to allowing "null" origins since many users open the built frontend
    # directly from the filesystem. Set ALLOW_FILE_ORIGINS=false to disable.
    allow_file_origins = os.environ.get("ALLOW_FILE_ORIGINS", "true").lower()
    if allow_file_origins == "true" or (allow_file_origins == "auto" and app.debug):
        if "null" not in origins:
            origins.append("null")

    allow_all = any(o == "*" for o in origins)
    cors_kwargs = {
        "resources": {r"/*": {"origins": origins if not allow_all else "*"}},
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type", "Authorization"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    }
    # If the user explicitly allows '*', disable credentials so Flask-CORS can
    # emit Access-Control-Allow-Origin: * (Authorization header still works).
    if allow_all:
        cors_kwargs["supports_credentials"] = False
    else:
        cors_kwargs["supports_credentials"] = True

    CORS(app, **cors_kwargs)

    # Initialize DB
    db_init(app)

    # JWT setup
    jwt = JWTManager(app)

    # Ensure upload folder exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Optional request logging to help debug CORS/preflight and upload issues
    if app.debug:
        @app.before_request
        def _log_request():
            try:
                app.logger.debug(
                    "Incoming request: %s %s from %s headers=%s",
                    request.method,
                    request.path,
                    request.remote_addr,
                    dict(request.headers),
                )
            except Exception:
                pass

        # Print masked DB URI for easier debugging (don't expose password)
        try:
            uri = app.config.get('SQLALCHEMY_DATABASE_URI')
            if uri:
                masked = uri
                try:
                    # mask password between first ':' after scheme and the '@'
                    if '//' in uri and '@' in uri:
                        prefix, rest = uri.split('//', 1)
                        userinfo, hostpart = rest.split('@', 1)
                        if ':' in userinfo:
                            user, _pw = userinfo.split(':', 1)
                            masked = f"{prefix}//{user}:***@{hostpart}"
                except Exception:
                    masked = uri
                app.logger.debug('SQLALCHEMY_DATABASE_URI=%s', masked)
        except Exception:
            pass

    # --- Register Blueprints ---
    from app.auth.routes import auth_bp
    from app.detect.routes import detect_bp
    from app.violations.routes import violations_bp  # if this file exists

    # Register blueprints without adding an extra url_prefix here because
    # each blueprint already defines its own `url_prefix` (e.g. '/auth', '/detect').
    # This keeps routes like '/auth/signup', '/detect/upload', '/violations' which
    # match the frontend expectations (frontend calls '/auth/*', '/detect/*').
    app.register_blueprint(auth_bp)
    app.register_blueprint(detect_bp)
    app.register_blueprint(violations_bp)

    @app.route("/api/health", methods=["GET"])
    def health():
        # Basic health-check: confirm app is up and DB is reachable
        db_ok = False
        allowed_count = None
        exc_text = None
        try:
            # simple lightweight query
            with app.app_context():
                # SELECT 1 equivalent via SQLAlchemy (use text() per SQLAlchemy 2.0)
                db.session.execute(text("SELECT 1"))
                db_ok = True
                try:
                    allowed_count = db.session.query(AllowedEmail).count()
                except Exception:
                    allowed_count = None
        except Exception as e:
            db_ok = False
            try:
                # include a short exception message in debug to help troubleshooting
                if app.debug:
                    exc_text = str(e)
            except Exception:
                exc_text = None

        resp = {"status": "ok", "db_ok": db_ok, "allowed_emails_count": allowed_count}
        if exc_text:
            resp["db_error"] = exc_text
        return resp, 200

    return app
