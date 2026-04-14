import os
from pathlib import Path

# Project root directory (assumes config.py is in scripts/ and we go one level up)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data_interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data_processed"
PBI_DIR = DATA_PROCESSED_DIR / "powerbi"

# Outputs directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR
TABLES_DIR = OUTPUTS_DIR / "tables"
PYTHON_DIR = PROJECT_ROOT / "python"

# Ensure directories exist
for d in [DATA_INTERIM_DIR, DATA_PROCESSED_DIR, PBI_DIR, CHARTS_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR, PYTHON_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)
