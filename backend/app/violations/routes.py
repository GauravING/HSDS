# app/violations/routes.py
from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from app.db import db, crud
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

violations_bp = Blueprint("violations", __name__, url_prefix="/violations")


def _parse_timestamp_for_db(val):
    """Normalize various timestamp inputs into a naive UTC datetime for DB.

    Accepts: datetime, ISO strings, RFC-2822 strings (e.g. 'Thu, 27 Nov 2025 10:18:08 GMT')
    Returns: datetime (naive, UTC) or None
    """
    if val is None:
        return datetime.utcnow()
    if isinstance(val, datetime):
        # convert aware -> naive UTC
        if val.tzinfo is not None:
            try:
                return val.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:
                return val.replace(tzinfo=None)
        return val
    if isinstance(val, str):
        # try ISO first
        try:
            return datetime.fromisoformat(val)
        except Exception:
            pass
        # try RFC-2822 parser
        try:
            dt = parsedate_to_datetime(val)
            if dt is None:
                return None
            if dt.tzinfo is not None:
                return dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except Exception:
            try:
                # last resort: let datetime parse common format
                return datetime.strptime(val, "%a, %d %b %Y %H:%M:%S %Z")
            except Exception:
                return None
    return None

@violations_bp.route("/", methods=["GET"])
@jwt_required()
def list_violations():
    """
    GET /api/violations?start_date=2025-01-01&end_date=2025-02-01&limit=50
    Returns a list of recent violations (optionally filtered by date).
    Admins see all; normal users see their own.
    """
    identity = get_jwt_identity()
    if not identity:
        return jsonify({"msg": "Not authenticated"}), 401

    user_id = identity.get("id")
    is_admin = identity.get("is_admin", False)

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    limit = int(request.args.get("limit", 50))

    start_dt = None
    end_dt = None
    try:
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        if end_date:
            end_dt = datetime.fromisoformat(end_date)
    except Exception:
        return jsonify({"msg": "Invalid date format"}), 400

    try:
        violations = crud.list_violations(db.session, start=start_dt, end=end_dt, limit=limit)
        if not is_admin:
            # Filter only user’s violations if you’re tracking user_id in your inference pipeline
            violations = [v for v in violations if v.get("user_id") == user_id]
        return jsonify(violations)
    except Exception as e:
        current_app.logger.exception("Error listing violations: %s", e)
        return jsonify({"msg": "Server error"}), 500


@violations_bp.route("/<int:violation_id>", methods=["GET"])
@jwt_required()
def get_violation(violation_id):
    """
    Get details of a single violation by ID.
    """
    try:
        violation = crud.get_violation_by_id(db.session, violation_id)
        if not violation:
            return jsonify({"msg": "Not found"}), 404
        return jsonify(violation)
    except Exception as e:
        current_app.logger.exception("Error getting violation: %s", e)
        return jsonify({"msg": "Server error"}), 500


@violations_bp.route("/<int:violation_id>", methods=["PATCH"])
@jwt_required()
def update_violation(violation_id):
    """
    PATCH /api/violations/<id>
    Allows marking a violation as 'reviewed' or updating notes.
    JSON body example: { "reviewed": true, "notes": "Checked by admin" }
    Only admins can update.
    """
    identity = get_jwt_identity()
    if not identity or not identity.get("is_admin"):
        return jsonify({"msg": "Admin access required"}), 403

    data = request.get_json() or {}
    try:
        updated = crud.update_violation(db.session, violation_id, data)
        if not updated:
            return jsonify({"msg": "Violation not found"}), 404
        return jsonify({"msg": "Violation updated"})
    except Exception as e:
        current_app.logger.exception("Error updating violation: %s", e)
        return jsonify({"msg": "Server error"}), 500



@violations_bp.route("/save", methods=["POST"])
@jwt_required()
def save_violations():
    """
    POST /violations/save
    Body: { "violations": [ { ... } ] }
    Persists provided violation dicts into the violations table.
    Returns a summary of saved ids and any errors.
    """
    identity = get_jwt_identity()
    if not identity:
        return jsonify({"msg": "Not authenticated"}), 401

    data = request.get_json() or {}
    violations = data.get("violations") or []
    if not isinstance(violations, list) or len(violations) == 0:
        return jsonify({"msg": "No violations provided"}), 400

    saved = []
    errors = []
    for idx, v in enumerate(violations):
        try:
            # Only persist if it's marked as violation or if caller forces it
            if not v.get("is_violation") and not v.get("force_save"):
                errors.append({"index": idx, "msg": "Not a violation; skipped"})
                continue

            obj = crud.log_violation(
                db.session,
                number_plate=v.get("number_plate"),
                violation_type=v.get("violation_type"),
                vehicle_type=v.get("vehicle_type"),
                confidence_score=v.get("confidence_score"),
                location=v.get("location"),
                image_path=v.get("image_path"),
                video_clip_path=v.get("video_clip_path"),
                status=v.get("status", "Pending Review"),
                # Normalize timestamp: accept datetime or common string formats
                timestamp=_parse_timestamp_for_db(v.get("timestamp")),
            )
            saved.append({"index": idx, "violation_id": getattr(obj, "violation_id", None)})
        except Exception as e:
            current_app.logger.exception("Error saving violation: %s", e)
            errors.append({"index": idx, "msg": str(e)})

    return jsonify({"saved": saved, "errors": errors}), 200
