"""
Configuration file for integration tests.
This file runs before all tests and fixes the Python path.
"""

import sys
from pathlib import Path

# Add the project root to Python path so all tests can import from src
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added to Python path: {project_root}")

print(f"Python path: {sys.path[:3]}...") 