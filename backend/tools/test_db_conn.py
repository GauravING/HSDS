"""
Test DB connectivity using the project's config.
Usage:
  cd backend
  python tools/test_db_conn.py

This script will:
 - print the SQLALCHEMY_DATABASE_URI used by the app
 - attempt to connect via SQLAlchemy and run SELECT 1
 - attempt to count rows in allowed_emails (if the table exists)
 - print full exception traceback on failure
"""
import traceback
import sys
import os
from pathlib import Path

# Ensure project root (backend/) is on sys.path so 'app' package can be imported
# This makes the script work whether invoked from backend/ or backend/tools/
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app
from config import Config

from sqlalchemy import create_engine, text


def main():
    print("Using Config from backend/config.py")
    print(f"MYSQL_HOST={Config.MYSQL_HOST}")
    print(f"MYSQL_PORT={Config.MYSQL_PORT}")
    print(f"MYSQL_USER={Config.MYSQL_USER}")
    hidden_pw = '***' if Config.MYSQL_PASSWORD else '(empty)'
    print(f"MYSQL_PASSWORD={hidden_pw}")
    print(f"MYSQL_DATABASE={Config.MYSQL_DATABASE}")
    print(f"SQLALCHEMY_DATABASE_URI={Config.SQLALCHEMY_DATABASE_URI}")

    print('\nAttempting SQLAlchemy engine connect...')
    try:
        engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, pool_pre_ping=True)
        with engine.connect() as conn:
            r = conn.execute(text('SELECT 1')).fetchone()
            print('SELECT 1 ->', r)
            # Try to count allowed_emails if present
            try:
                r2 = conn.execute(text('SELECT COUNT(*) as c FROM allowed_emails')).fetchone()
                print('allowed_emails count ->', r2[0])
            except Exception as e:
                print('Could not query allowed_emails (maybe table missing):', e)
        print('\nDB connection SUCCESS')
    except Exception as e:
        print('\nDB connection FAILED')
        traceback.print_exc()


if __name__ == '__main__':
    main()
