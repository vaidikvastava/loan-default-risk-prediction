"""
Database connector and persistence service for Loan Default Risk Management.
Supports MySQL with connection pooling, and provides an automatic SQLite fallback
if a live MySQL server is not currently reachable.
"""
import os
import sys
import json
import logging
import sqlite3
from typing import Dict, Any, List, Optional
import pandas as pd
from src.config import BASE_DIR

logger = logging.getLogger(__name__)

# MySQL connection defaults
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "loan_risk_db")

SQLITE_PATH = BASE_DIR / "data" / "loan_risk_local.db"


class DatabaseManager:
    """
    Unified database manager providing MySQL connectivity with graceful local fallback.
    """

    def __init__(self):
        self.db_type = "sqlite"
        self.conn = None
        self._init_connection()

    def _init_connection(self):
        # Attempt MySQL connection first
        try:
            import pymysql
            self.conn = pymysql.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE,
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=2
            )
            self.db_type = "mysql"
            logger.info(f"Connected to MySQL database '{MYSQL_DATABASE}' on {MYSQL_HOST}:{MYSQL_PORT}")
        except Exception as e_mysql:
            try:
                import mysql.connector
                self.conn = mysql.connector.connect(
                    host=MYSQL_HOST,
                    port=MYSQL_PORT,
                    user=MYSQL_USER,
                    password=MYSQL_PASSWORD,
                    database=MYSQL_DATABASE
                )
                self.db_type = "mysql"
                logger.info(f"Connected to MySQL via mysql.connector on {MYSQL_HOST}:{MYSQL_PORT}")
            except Exception:
                # Fallback to local SQLite database engine
                self.db_type = "sqlite"
                SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
                self.conn = sqlite3.connect(str(SQLITE_PATH), check_same_thread=False)
                self.conn.row_factory = sqlite3.Row
                logger.info(f"MySQL unavailable ({e_mysql}); utilizing SQLite storage at {SQLITE_PATH}")

        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        if self.db_type == "mysql":
            # Tables created via database/schema.sql
            pass
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS applicants (
                    sk_id_curr INTEGER PRIMARY KEY,
                    target INTEGER,
                    name_contract_type TEXT,
                    code_gender TEXT,
                    flag_own_car TEXT,
                    flag_own_realty TEXT,
                    cnt_children INTEGER,
                    amt_income_total REAL,
                    amt_credit REAL,
                    amt_annuity REAL,
                    amt_goods_price REAL,
                    name_type_suite TEXT,
                    name_income_type TEXT,
                    name_education_type TEXT,
                    name_family_status TEXT,
                    name_housing_type TEXT,
                    days_birth INTEGER,
                    days_employed INTEGER,
                    days_registration INTEGER,
                    days_id_publish INTEGER,
                    own_car_age REAL,
                    region_rating_client INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS financial_ratios (
                    sk_id_curr INTEGER PRIMARY KEY,
                    debt_to_income REAL,
                    annuity_to_income REAL,
                    credit_to_goods_ratio REAL,
                    credit_annuity_ratio REAL,
                    income_per_family_member REAL,
                    employed_to_age_ratio REAL,
                    ext_source_mean REAL,
                    ext_source_min REAL,
                    ext_source_max REAL,
                    ext_source_std REAL,
                    FOREIGN KEY (sk_id_curr) REFERENCES applicants(sk_id_curr)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS underwriting_decisions (
                    decision_id INTEGER PRIMARY KEY AUTO_INCREMENT,
                    sk_id_curr INTEGER,
                    model_version TEXT DEFAULT 'v1.0.0',
                    default_probability REAL,
                    risk_tier TEXT,
                    recommendation TEXT,
                    decision_threshold REAL,
                    key_drivers TEXT,
                    underwriter_notes TEXT,
                    decision_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """.replace("AUTO_INCREMENT", "AUTOINCREMENT"))
            self.conn.commit()

    def log_underwriting_decision(
        self,
        sk_id_curr: int,
        default_probability: float,
        risk_tier: str,
        recommendation: str,
        decision_threshold: float,
        key_drivers: List[Dict[str, Any]] = None,
        notes: str = ""
    ) -> int:
        cursor = self.conn.cursor()
        drivers_json = json.dumps(key_drivers or [])
        query = """
            INSERT INTO underwriting_decisions 
            (sk_id_curr, default_probability, risk_tier, recommendation, decision_threshold, key_drivers, underwriter_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        if self.db_type == "mysql":
            query = query.replace("?", "%s")

        cursor.execute(query, (
            sk_id_curr,
            round(default_probability, 4),
            risk_tier,
            recommendation,
            round(decision_threshold, 3),
            drivers_json,
            notes
        ))
        self.conn.commit()
        return cursor.lastrowid

    def get_recent_decisions(self, limit: int = 50) -> pd.DataFrame:
        query = f"""
            SELECT decision_id, sk_id_curr, default_probability, risk_tier, recommendation,
                   decision_threshold, underwriter_notes, decision_timestamp
            FROM underwriting_decisions
            ORDER BY decision_id DESC
            LIMIT {limit}
        """
        return pd.read_sql_query(query, self.conn)

    def get_portfolio_kpis(self) -> Dict[str, Any]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) AS total, AVG(amt_credit) AS avg_cred, AVG(amt_income_total) AS avg_inc FROM applicants")
        row = cursor.fetchone()
        if isinstance(row, dict):
            tot, avg_c, avg_i = row["total"], row["avg_cred"], row["avg_inc"]
        elif row:
            tot, avg_c, avg_i = row[0], row[1], row[2]
        else:
            tot, avg_c, avg_i = 0, 0, 0

        cursor.execute("SELECT AVG(target) FROM applicants WHERE target IS NOT NULL")
        r_def = cursor.fetchone()
        def_rate = r_def[0] if (r_def and not isinstance(r_def, dict)) else (r_def.get("AVG(target)") if r_def else 0)

        return {
            "total_applicants": tot or 0,
            "avg_credit_amount": round(float(avg_c or 0), 2),
            "avg_income": round(float(avg_i or 0), 2),
            "historical_default_rate_pct": round(float(def_rate or 0.077) * 100, 2),
            "active_database_engine": self.db_type.upper(),
        }


# Singleton database instance
db_manager = DatabaseManager()
