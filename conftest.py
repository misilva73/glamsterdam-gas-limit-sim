"""Makes the repo root importable so a bare `pytest` resolves `data`/`sim`/`analysis`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
