"""Load the raw supply-chain CSV into MySQL (admin only).

Config via env vars (or a .env file): DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME.
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

load_dotenv()

CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "dynamic_supply_chain_logistics_dataset.csv"
TABLE = "supply_chain_legacy"

legacy_mapping = {
    'timestamp': 'TS_UTC',
    'vehicle_gps_latitude': 'V_LAT',
    'vehicle_gps_longitude': 'V_LON',
    'iot_temperature': 'IOT_TEMP_VAL_C',
    'cargo_condition_status': 'CGO_COND_CD',
    'risk_classification': 'RISK_CLS_TXT',
    'delay_probability': 'DELAY_PROB_DEC',
    'port_congestion_level': 'PRT_CNG_LVL',
    'route_risk_level': 'RT_RSK_IDX'
}


def _url(database=None):
    return URL.create(
        "mysql+pymysql",
        username=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        database=database,
    )


def require_admin(conn):
    """Only a MySQL admin (global privileges incl. CREATE USER, with grant option) may ingest."""
    grants = [row[0] for row in conn.execute(text("SHOW GRANTS"))]
    if not any(
        " ON *.* " in g and "WITH GRANT OPTION" in g and ("ALL PRIVILEGES" in g or "CREATE USER" in g)
        for g in grants
    ):
        user = conn.execute(text("SELECT CURRENT_USER()")).scalar()
        raise PermissionError(f"User '{user}' is not an admin; ingestion denied.")


def ingest():
    engine = create_engine(_url())
    db_name = os.getenv("DB_NAME", "supply_chain")

    with engine.begin() as conn:
        require_admin(conn)
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{db_name}`"))

    df = pd.read_csv(CSV_PATH, usecols=list(legacy_mapping)).rename(columns=legacy_mapping)
    df["TS_UTC"] = pd.to_datetime(df["TS_UTC"])

    engine = create_engine(_url(db_name))
    df.to_sql(TABLE, engine, if_exists="replace", index=False, chunksize=5000)
    print(f"Loaded {len(df)} rows into {db_name}.{TABLE}")


if __name__ == "__main__":
    ingest()
