# app/db/__init__.py
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import scoped_session, sessionmaker

# Initialize SQLAlchemy instance (used everywhere)
db = SQLAlchemy()


def db_init(app):
    """
    Initialize the SQLAlchemy instance with the Flask app.
    This should be called from create_app() in app/__init__.py.
    """
    db.init_app(app)

    with app.app_context():
        # Import models only *after* db is set up to avoid circular imports
        from app.db import models  # noqa: F401
        db.create_all()


def get_session():
    """
    Returns a scoped SQLAlchemy session bound to the Flask app's engine.
    Used by services like inference.
    """
    engine = db.get_engine()
    SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
    return SessionLocal()
