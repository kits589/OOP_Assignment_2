"""Put the repository root on ``sys.path`` so ``import src...`` works.

Lets ``pytest`` be run from the repository root without installing anything.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
