"""
Data seeding and ETL pipeline to ingest raw loan applications and financial ratios into MySQL.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pandas as pd
from src.config import RAW_DATA_PATH
from src.database import db_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def seed_database_from_csv(csv_path: str = None, sample_size: int = 1000):
    """
    Populates database with loan applicants from CSV.
    """
    path = csv_path or str(RAW_DATA_PATH)
    logger.info(f"Loading applicants from {path} (sample size: {sample_size})...")
    df = pd.read_csv(path, nrows=sample_size)

    conn = db_manager.conn
    cursor = conn.cursor()

    logger.info(f"Ingesting {len(df)} records into 'applicants' table ({db_manager.db_type.upper()})...")
    inserted = 0

    applicant_cols = [
        "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY", "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "AMT_GOODS_PRICE", "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED",
        "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "OWN_CAR_AGE", "REGION_RATING_CLIENT"
    ]
    present_cols = [c for c in applicant_cols if c in df.columns]

    for _, row in df[present_cols].iterrows():
        try:
            placeholders = ", ".join(["%s" if db_manager.db_type == "mysql" else "?"] * len(present_cols))
            col_names = ", ".join([c.lower() for c in present_cols])
            query = f"INSERT OR REPLACE INTO applicants ({col_names}) VALUES ({placeholders})"
            if db_manager.db_type == "mysql":
                query = f"REPLACE INTO applicants ({col_names}) VALUES ({placeholders})"

            vals = tuple(None if pd.isna(val) else val for val in row.values)
            cursor.execute(query, vals)
            inserted += 1
        except Exception as e:
            logger.debug(f"Row skip: {e}")

    conn.commit()
    logger.info(f"Successfully ingested {inserted} records into database!")

    kpis = db_manager.get_portfolio_kpis()
    logger.info(f"Current Database Portfolio Stats: {kpis}")


if __name__ == "__main__":
    seed_database_from_csv()
