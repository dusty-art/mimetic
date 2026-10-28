"""Mimetic: muscle-driven talking faces."""
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
# Keep downloaded model weights inside the project so disk use is easy to audit.
os.environ.setdefault("HF_HOME", str(_ROOT / ".cache" / "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
