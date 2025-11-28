# app/db/crud.py
"""
CRUD helpers used by inference and routes.
Expecting a SQLAlchemy Session instance as the first arg.
"""
from app.db.models import Violation, User, AllowedEmail
from sqlalchemy.orm import Session
from sqlalchemy import inspect
from sqlalchemy.exc import ProgrammingError
from datetime import datetime
from decimal import Decimal
import logging
from types import SimpleNamespace
from sqlalchemy import select

logger = logging.getLogger("db.crud")


def create_violation(db: Session, vdata: dict):
    """
    vdata expected keys:
      - number_plate (optional)
      - violation_type (required)
      - vehicle_type
      - confidence_score (float)
      - timestamp (datetime) optional
      - location, image_path, video_clip_path, status
      - extra (dict) optional
    Returns the created Violation ORM object.
    """
    try:
        # allow vdata to include plate_image_path and metadata
        v = Violation(
            number_plate = vdata.get("number_plate"),
            violation_type = vdata.get("violation_type"),
            vehicle_type = vdata.get("vehicle_type"),
            confidence_score = Decimal(str(vdata.get("confidence_score", 0.0))) if vdata.get("confidence_score") is not None else None,
            timestamp = vdata.get("timestamp", datetime.utcnow()),
            location = vdata.get("location"),
            image_path = vdata.get("image_path"),
            plate_image_path = vdata.get("plate_image_path"),
            metadata_json = (vdata.get("metadata") if isinstance(vdata.get("metadata"), str) else __import__("json").dumps(vdata.get("metadata")) ) if vdata.get("metadata") is not None else None,
            video_clip_path = vdata.get("video_clip_path"),
            status = vdata.get("status", "Pending Review")
        )
        db.add(v)
        db.commit()
        db.refresh(v)
        return v
    except Exception as e:
        # log full context to help debugging schema/constraint issues
        try:
            logger.exception("create_violation failed for vdata=%s: %s", vdata, e)
        except Exception:
            logger.exception("create_violation failed and vdata logging failed: %s", e)
        db.rollback()
        raise


def log_violation(db: Session, number_plate=None, violation_type=None, vehicle_type=None, confidence_score=None, location=None, image_path=None, video_clip_path=None, status="Pending Review", timestamp=None):
    """
    Convenience wrapper used by routes when inference returns simple dicts.
    Mirrors create_violation but accepts explicit named args.
    """
    try:
        v = Violation(
            number_plate=number_plate,
            violation_type=violation_type,
            vehicle_type=vehicle_type,
            confidence_score=Decimal(str(confidence_score)) if confidence_score is not None else None,
            timestamp=timestamp or datetime.utcnow(),
            location=location,
            image_path=image_path,
            video_clip_path=video_clip_path,
            status=status,
        )
        db.add(v)
        db.commit()
        db.refresh(v)
        return v
    except Exception as e:
        # Some deployments may have a schema that differs from the model
        # (e.g. missing plate_image_path or metadata_json). Attempt a
        # safe insert using table reflection that only writes existing
        # columns, then return the created row. If that also fails, re-raise.
        try:
            db.rollback()
        except Exception:
            pass

        try:
            # reflect existing columns in the violations table using the
            # Flask-SQLAlchemy engine (sa_db.get_engine()) so we don't rely
            # on the provided session.bind which can be None in some setups.
            from app.db import db as sa_db
            from app.db import get_session as _get_session

            try:
                engine = None
                try:
                    engine = sa_db.get_engine()
                except Exception:
                    engine = getattr(sa_db, "engine", None)

                if engine is None:
                    raise RuntimeError("No engine available for reflection")

                insp = inspect(engine)
            except Exception as ie:
                logger.exception("Failed to obtain engine for reflection: %s", ie)
                raise

            cols_info = {c['name']: c for c in insp.get_columns('violations')}

            insert_data = {
                'number_plate': number_plate,
                'violation_type': violation_type,
                'vehicle_type': vehicle_type,
                'confidence_score': Decimal(str(confidence_score)) if confidence_score is not None else None,
                'timestamp': timestamp or datetime.utcnow(),
                'location': location,
                'image_path': image_path,
                'video_clip_path': video_clip_path,
                'status': status,
            }

            # filter to only columns that exist in the DB and respect NOT NULL
            filtered = {}
            for k, v in insert_data.items():
                if k not in cols_info:
                    continue
                col = cols_info[k]
                nullable = col.get('nullable', True)
                # If the DB column is NOT NULL but our value is None, supply a safe default
                if v is None and not nullable:
                    # derive a safe default based on column type
                    t = str(col.get('type', '')).lower()
                    if 'char' in t or 'text' in t or 'varchar' in t:
                        fv = ''
                    elif 'int' in t or 'numeric' in t or 'decimal' in t or 'float' in t:
                        fv = 0
                    elif 'date' in t or 'time' in t:
                        fv = datetime.utcnow()
                    else:
                        # fallback to empty string for unknown types
                        fv = ''
                    filtered[k] = fv
                else:
                    filtered[k] = v

            # Use a fresh engine connection and transaction for the core insert
            conn = engine.connect()
            trans = conn.begin()
            try:
                stmt = Violation.__table__.insert().values(**filtered)
                res = conn.execute(stmt)
                trans.commit()
            except Exception:
                try:
                    trans.rollback()
                except Exception:
                    pass
                raise
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

            # Attempt to discover the PK of the inserted row
            pk = None
            try:
                if hasattr(res, 'inserted_primary_key') and res.inserted_primary_key:
                    pk = res.inserted_primary_key[0]
                elif hasattr(res, 'lastrowid'):
                    pk = res.lastrowid
            except Exception:
                pk = None

            # If we have a PK, try to read the inserted row via Core select
            if pk:
                try:
                    conn2 = engine.connect()
                    try:
                        # select only the columns that actually exist in the DB
                        cols_to_select = [Violation.__table__.c[c] for c in cols_info.keys() if c in Violation.__table__.c]
                        # ensure primary key is selected
                        if 'violation_id' in Violation.__table__.c and Violation.__table__.c['violation_id'] not in cols_to_select:
                            cols_to_select.insert(0, Violation.__table__.c['violation_id'])

                        stmt = select(*cols_to_select).where(Violation.__table__.c.violation_id == pk)
                        res = conn2.execute(stmt)
                        row = res.fetchone()
                        if row:
                            data = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
                            # return a lightweight object with attributes used by caller
                            return SimpleNamespace(**data)
                    finally:
                        try:
                            conn2.close()
                        except Exception:
                            pass
                except Exception:
                    # fall through to last-resort heuristics below
                    pass

            # As a last resort, try to find a recent row by matching timestamp and image_path using Core
            try:
                conn3 = engine.connect()
                try:
                    available_cols = [c for c in cols_info.keys() if c in Violation.__table__.c]
                    sel_cols = [Violation.__table__.c[c] for c in available_cols]
                    stmt = select(*sel_cols).order_by(Violation.__table__.c.violation_id.desc()).limit(1)
                    if image_path is not None:
                        stmt = select(*sel_cols).where(Violation.__table__.c.image_path == image_path).order_by(Violation.__table__.c.violation_id.desc()).limit(1)
                    res = conn3.execute(stmt)
                    row = res.fetchone()
                    if row:
                        data = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
                        return SimpleNamespace(**data)
                finally:
                    try:
                        conn3.close()
                    except Exception:
                        pass
            except Exception:
                return None
        except ProgrammingError:
            # re-raise DB programming errors for visibility
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise

def get_violation_by_id(db: Session, violation_id: int):
    return db.query(Violation).filter(Violation.violation_id == violation_id).first()

def list_violations(db: Session, page: int = 1, per_page: int = 20):
    # SQLAlchemy pagination simple implementation
    offset = (page - 1) * per_page
    q = db.query(Violation).order_by(Violation.timestamp.desc()).offset(offset).limit(per_page).all()
    return q

def create_user(db: Session, user_data: dict):
    u = User(
        username = user_data["username"],
        email = user_data["email"],
        hashed_password = user_data["hashed_password"],
        full_name = user_data.get("full_name"),
        is_active = user_data.get("is_active", True),
        is_admin = user_data.get("is_admin", False),
        profile_image = user_data.get("profile_image")
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


def get_user_by_email(db: Session, email: str):
    """Return a User ORM object by email or None."""
    return db.query(User).filter(User.email == email).first()

    

def is_email_allowed(db: Session, email: str):
    return db.query(AllowedEmail).filter(AllowedEmail.email == email).first() is not None


def list_allowed_emails(db: Session):
    """Return a list of allowed email strings."""
    try:
        rows = db.query(AllowedEmail).all()
        return [r.email for r in rows]
    except Exception:
        return []

def list_violations_time_range(db: Session, start=None, end=None, limit=50):
    query = db.query(Violation)
    if start:
        query = query.filter(Violation.timestamp >= start)
    if end:
        query = query.filter(Violation.timestamp <= end)
    query = query.order_by(Violation.timestamp.desc()).limit(limit)
    return [v for v in query.all()]

def get_violation_by_id_dict(db: Session, violation_id: int):
    v = db.query(Violation).filter(Violation.violation_id == violation_id).first()
    return v

def update_violation(db: Session, violation_id: int, data: dict):
    v = db.query(Violation).filter(Violation.violation_id == violation_id).first()
    if not v:
        return None
    if "status" in data:
        v.status = data["status"]
    if "notes" in data:
        # keep notes as an optional free-text column; create if model updated later
        try:
            v.notes = data["notes"]
        except Exception:
            pass
    db.add(v)
    db.commit()
    db.refresh(v)
    return v
