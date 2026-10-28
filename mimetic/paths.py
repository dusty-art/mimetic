from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EXPERIMENTS = ROOT / "experiments"
CHECKPOINTS = ROOT / "checkpoints"
