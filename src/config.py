"""プロジェクト内のパスと共通設定。"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT_DIR / "input"
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
OUTPUT_DIR = ROOT_DIR / "output"
RUNS_DIR = OUTPUT_DIR / "runs"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
FIGURE_DIR = OUTPUT_DIR / "figures"
EXPERIMENTS_LOG = OUTPUT_DIR / "experiments.csv"
MODEL_DIR = ROOT_DIR / "models"
SEED = 42
