import sys
from pathlib import Path

# Ensure `src.*` imports work from any test location
sys.path.insert(0, str(Path(__file__).parent))
