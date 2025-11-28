# app/auth/routes.py
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import (
    create_access_token,
    jwt_required,
    get_jwt_identity,
)
from datetime import datetime, timedelta
from app.db import db
from app.db import crud
from app.db.models import AllowedEmail
from app.utils.security import hash_password, check_password

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")

# Access token expires in 24 hours
ACCESS_EXPIRES = timedelta(hours=24)


@auth_bp.route("/signup", methods=["POST"])
def signup():
    """
    Signup endpoint.
    Request JSON:
      {
        "username": "...",
        "email": "...",
        "password": "...",
        "full_name": "..." (optional)
      }
    Only allows emails present in allowed_emails table.
    """
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    full_name = data.get("full_name", "").strip()

    if not username or not email or not password:
        return jsonify({"msg": "username, email, and password are required"}), 400

    try:
        # Check if email is in allowed list
        if not crud.is_email_allowed(db.session, email):
            return jsonify({"msg": "Email not allowed to sign up"}), 403

        # Check if already registered
        if crud.get_user_by_email(db.session, email):
            return jsonify({"msg": "Email already registered"}), 409

        # Hash password and create user
        hashed_pw = hash_password(password)
        user = crud.create_user(
            db.session,
            {
                "username": username,
                "email": email,
                "hashed_password": hashed_pw,
                "full_name": full_name,
            },
        )

        return jsonify({"msg": "User registered successfully", "user_id": user.id}), 201

    except Exception as e:
        current_app.logger.exception("Error during signup: %s", e)
        db.session.rollback()
        return jsonify({"msg": "Server error during signup"}), 500


# Provide a lightweight GET handler so browser navigations or dev checks don't return 405
@auth_bp.route("/signup", methods=["GET"])
def signup_get():
    """Informational GET for development: tells callers how to POST to signup.

    This prevents browser navs resulting in 405 Method Not Allowed during development.
    """
    # Keep the response simple and machine-friendly so frontend dev code can inspect it
    return (
        jsonify(
            {
                "msg": "POST to /auth/signup with JSON {username,email,password,full_name}.",
                "example": {"username": "alice", "email": "alice@example.com", "password": "secret"},
            }
        ),
        200,
    )


@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Login endpoint.
    Request JSON:
      { "email": "...", "password": "..." }
    Returns:
      { "access_token": "<JWT>", "expires_in": <seconds> }
    """
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"msg": "email and password are required"}), 400

    try:
        # Only allow login for emails present in the allowed_emails table
        if not crud.is_email_allowed(db.session, email):
            return jsonify({"msg": "Email not allowed to login"}), 403

        user = crud.get_user_by_email(db.session, email)
        # Debug logging to help diagnose credential issues (safe in dev)
        try:
            current_app.logger.debug("Login attempt for email=%s, user_found=%s", email, bool(user))
            if user:
                current_app.logger.debug("User id=%s, has_hashed_password=%s", user.id, bool(user.hashed_password))
        except Exception:
            pass

        if not user or not check_password(password, user.hashed_password):
            current_app.logger.warning("Invalid credentials for email=%s", email)
            return jsonify({"msg": "Invalid credentials"}), 401

        if not user.is_active:
            return jsonify({"msg": "User is inactive"}), 403

        # Update last_login
        user.last_login = datetime.utcnow()
        db.session.commit()

        # Build token
        identity = {"id": user.id, "email": user.email, "is_admin": user.is_admin}
        access_token = create_access_token(identity=identity, expires_delta=ACCESS_EXPIRES)

        return (
            jsonify(
                {
                    "access_token": access_token,
                    "expires_in": int(ACCESS_EXPIRES.total_seconds()),
                    "user": {
                        "id": user.id,
                        "username": user.username,
                        "email": user.email,
                        "is_admin": user.is_admin,
                    },
                }
            ),
            200,
        )

    except Exception as e:
        current_app.logger.exception("Error during login: %s", e)
        db.session.rollback()
        return jsonify({"msg": "Server error during login"}), 500


# Provide a lightweight GET handler so browser navigations or dev checks don't return 405
@auth_bp.route("/login", methods=["GET"])
def login_get():
    """Informational GET for development: tells callers how to POST to login.

    This prevents simple browser navigations resulting in 405 Method Not Allowed during development.
    """
    return (
        jsonify(
            {
                "msg": "POST to /auth/login with JSON {email, password}.",
                "example": {"email": "alice@example.com", "password": "secret"},
            }
        ),
        200,
    )


@auth_bp.route("/profile", methods=["GET"])
@jwt_required()
def profile():
    """
    Return authenticated user's profile.
    """
    identity = get_jwt_identity()
    if not identity:
        return jsonify({"msg": "Not authenticated"}), 401

    try:
        user = crud.get_user_by_email(db.session, identity.get("email"))
        if not user:
            return jsonify({"msg": "User not found"}), 404

        return jsonify(
            {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "is_admin": user.is_admin,
                "profile_image": user.profile_image,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "last_login": user.last_login.isoformat() if user.last_login else None,
            }
        )

    except Exception as e:
        current_app.logger.exception("Error loading profile: %s", e)
        return jsonify({"msg": "Server error loading profile"}), 500


@auth_bp.route("/allowed_emails", methods=["GET"])
def allowed_emails():
    """Return the current list of allowed emails (strings).

    Useful for frontend signup UI during development. Returns an empty list on error.
    """
    try:
        # Use the CRUD helper if available
        try:
            from app.db import crud as _crud
            emails = _crud.list_allowed_emails(db.session)
        except Exception:
            # fallback: query directly
            emails = [r.email for r in db.session.query(AllowedEmail).all()]
        return jsonify({"allowed_emails": emails}), 200
    except Exception as e:
        current_app.logger.exception("Error fetching allowed_emails: %s", e)
        return jsonify({"allowed_emails": []}), 200


# ------------------ Dev helpers ------------------
@auth_bp.route("/dev-reset-password", methods=["POST"])  # dev-only
def dev_reset_password():
    """
    Dev-only endpoint to reset a user's password to a known value.
    Only available when Flask app.debug is True.

    Request JSON: { "email": "...", "new_password": "..." }
    """
    if not current_app.debug:
        return jsonify({"msg": "Not allowed"}), 403

    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    new_password = data.get("new_password")
    if not email or not new_password:
        return jsonify({"msg": "email and new_password required"}), 400

    try:
        user = crud.get_user_by_email(db.session, email)
        if not user:
            return jsonify({"msg": "User not found"}), 404
        hashed = hash_password(new_password)
        user.hashed_password = hashed
        db.session.add(user)
        db.session.commit()
        return jsonify({"msg": "Password reset"}), 200
    except Exception as e:
        current_app.logger.exception("Error in dev_reset_password: %s", e)
        db.session.rollback()
        return jsonify({"msg": "Server error"}), 500

