"""Run the tests from the repo root (python -m pytest); the scripts under test
are top-level modules that read data/ and best_params.json relative to it."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
