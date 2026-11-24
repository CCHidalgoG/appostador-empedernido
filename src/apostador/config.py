import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
DUCKDB_PATH = DATA_DIR / "apostador.duckdb"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
BACKTESTS_DIR = OUTPUTS_DIR / "backtests"

APIFOOTBALL_RAW_DIR = RAW_DIR / "apifootball"
IMPECT_RAW_DIR = RAW_DIR / "impect"
ODDS_RAW_DIR = RAW_DIR / "theoddsapi"
SOCCERDATA_RAW_DIR = RAW_DIR / "soccerdata"

os.environ["SOCCERDATA_DIR"] = str(SOCCERDATA_RAW_DIR)


def ensure_runtime_directories():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    SOCCERDATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

