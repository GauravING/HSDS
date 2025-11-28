#!/usr/bin/env python
# Print recent violations from DB for debugging
from app import create_app
from app.db import db
from app.db.crud import list_violations_time_range

app = create_app()

with app.app_context():
    vs = list_violations_time_range(db.session, limit=20)
    import json
    out = []
    for v in vs:
        out.append({
            "violation_id": getattr(v, "violation_id", None),
            "number_plate": getattr(v, "number_plate", None),
            "violation_type": getattr(v, "violation_type", None),
            "vehicle_type": getattr(v, "vehicle_type", None),
            "confidence_score": (str(getattr(v, "confidence_score", None)) if getattr(v, "confidence_score", None) is not None else None),
            "timestamp": (getattr(v, "timestamp").isoformat() if getattr(v, "timestamp", None) is not None else None),
            "image_path": getattr(v, "image_path", None),
            "status": getattr(v, "status", None),
        })
    print(json.dumps(out, indent=2))
