import sys
from pathlib import Path

# Ensure project root (backend/) is on sys.path so imports work regardless of CWD
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
print("SQLALCHEMY_DATABASE_URI=", Config.SQLALCHEMY_DATABASE_URI)