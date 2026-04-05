"""
conftest.py — makes the project root available on sys.path so that
`import src.*` works when pytest is invoked from any directory.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
