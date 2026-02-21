"""
Configuration for Infant Growth ML+WHO Hybrid Prediction System.
"""
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"

for d in [DATA_DIR, RAW_DIR, MODELS_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Server ────────────────────────────────────────────────────
PORT = int(os.environ.get("PORT", 8000))
HOST = os.environ.get("HOST", "0.0.0.0")
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# ── Auth (optional) ───────────────────────────────────────────
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "false").lower() == "true"
AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "changeme")
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")

# ── NHANES Configuration ─────────────────────────────────────
NHANES_CYCLES = [
    '1999-2000', '2001-2002', '2003-2004', '2005-2006', '2007-2008',
    '2009-2010', '2011-2012', '2013-2014', '2015-2016', '2017-2018'
]

CYCLE_URL_MAP = {
    '1999-2000': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/1999/DataFiles',
    '2001-2002': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2001/DataFiles',
    '2003-2004': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2003/DataFiles',
    '2005-2006': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2005/DataFiles',
    '2007-2008': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2007/DataFiles',
    '2009-2010': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2009/DataFiles',
    '2011-2012': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles',
    '2013-2014': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles',
    '2015-2016': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles',
    '2017-2018': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles',
}

# File code suffixes by cycle
DEMO_SUFFIX = {
    '1999-2000': 'DEMO', '2001-2002': 'DEMO_B', '2003-2004': 'DEMO_C',
    '2005-2006': 'DEMO_D', '2007-2008': 'DEMO_E', '2009-2010': 'DEMO_F',
    '2011-2012': 'DEMO_G', '2013-2014': 'DEMO_H', '2015-2016': 'DEMO_I',
    '2017-2018': 'DEMO_J',
}

BMX_SUFFIX = {
    '1999-2000': 'BMX', '2001-2002': 'BMX_B', '2003-2004': 'BMX_C',
    '2005-2006': 'BMX_D', '2007-2008': 'BMX_E', '2009-2010': 'BMX_F',
    '2011-2012': 'BMX_G', '2013-2014': 'BMX_H', '2015-2016': 'BMX_I',
    '2017-2018': 'BMX_J',
}

# ── ML Training ───────────────────────────────────────────────
MODEL1_FEATURES = [
    'age_months', 'sex_female', 'birth_weight_kg',
    'income_poverty_ratio', 'family_size',
    'race_2', 'race_3', 'race_4', 'race_5'
]

MODEL3_FEATURES = MODEL1_FEATURES + [
    'length_cm', 'head_circ_cm', 'arm_circ_cm', 'arm_length_cm'
]

MAX_AGE_MONTHS = 36
DOWNLOAD_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY = 2
