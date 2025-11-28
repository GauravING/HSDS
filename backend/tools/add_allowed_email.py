"""
Run from backend folder with the project's virtualenv active:

python tools/add_allowed_email.py someone@example.com

This will add the provided email to the allowed_emails table (if not already present).
"""
import sys
from app import create_app
from app.db import db
from app.db.models import AllowedEmail

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python tools/add_allowed_email.py alice@example.com')
        sys.exit(1)
    email = sys.argv[1].strip().lower()
    app = create_app()
    with app.app_context():
        existing = db.session.query(AllowedEmail).filter(AllowedEmail.email == email).first()
        if existing:
            print(f'{email} already present (id={existing.id})')
            sys.exit(0)
        a = AllowedEmail(email=email)
        db.session.add(a)
        db.session.commit()
        print(f'Added allowed email: {email} (id={a.id})')
