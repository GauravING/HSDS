# app/db/models.py
from datetime import datetime
from app.db import db
from sqlalchemy import Enum

class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    hashed_password = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(100))
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    profile_image = db.Column(db.String(255))

    def __repr__(self):
        return f"<User {self.username}>"


class Violation(db.Model):
    __tablename__ = "violations"

    violation_id = db.Column(db.Integer, primary_key=True)
    number_plate = db.Column(db.String(20))
    violation_type = db.Column(Enum("No Helmet", "No Seatbelt", name="violation_type_enum"))
    vehicle_type = db.Column(db.String(50))
    confidence_score = db.Column(db.Numeric(5, 4))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    location = db.Column(db.String(255))
    image_path = db.Column(db.String(255))
    plate_image_path = db.Column(db.String(255))
    # avoid using the reserved attribute name `metadata` on Declarative models
    metadata_json = db.Column(db.Text)
    video_clip_path = db.Column(db.String(255))
    status = db.Column(Enum("Pending Review", "Processed", "Dismissed", name="violation_status_enum"))

    def __repr__(self):
        return f"<Violation {self.violation_id} - {self.violation_type}>"


class AllowedEmail(db.Model):
    __tablename__ = "allowed_emails"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)

    def __repr__(self):
        return f"<AllowedEmail {self.email}>"
