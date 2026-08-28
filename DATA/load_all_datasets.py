"""
Project 1: Customer Churn Prediction & LTV Engine
Loads all 9 pipeline CSVs into PostgreSQL as-is -- one table per file,
with columns/types inferred directly from each CSV by pandas.

Assumes the 'telco_churn' database already exists but is empty (0 tables).
This script creates the tables itself -- no schema.sql needed.

Usage:
    python src/db/load_all_datasets.py --data-dir data/

Connection is read from environment variables:
    DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME (defaults shown in get_engine())
"""

import argparse
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text

# filename -> table name
FILES_TO_TABLES = {
    "telco_churn.csv":                       "telco_churn",
    "telco_churn_cleaned.csv":               "telco_churn_cleaned",
    "telco_churn_eda.csv":                   "telco_churn_eda",
    "telco_churn_feature_engineered.csv":    "telco_churn_feature_engineered",
    "customer_churn_predictions.csv":        "customer_churn_predictions",
    "customer_ltv_segments.csv":             "customer_ltv_segments",
    "priority_retention_customers.csv":      "priority_retention_customers",
    "feature_importance.csv":                "feature_importance",
    "shap_feature_importance.csv":           "shap_feature_importance",
}

# telco_churn.csv has blank strings in TotalCharges for tenure=0 customers --
# force it to text on read so pandas/to_sql doesn't choke trying to infer a
# numeric column that has blanks in it.
STRING_OVERRIDES = {
    "telco_churn.csv": {"TotalCharges": str},
}


def get_engine():
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "")
    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ.get("DB_NAME", "telco_churn")
    if not password:
        print("WARNING: DB_PASSWORD is not set.", file=sys.stderr)
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Folder containing the 9 CSVs")
    args = parser.parse_args()

    engine = get_engine()
    with engine.connect():
        print("PostgreSQL connection successful.\n")

    missing = []
    for filename in FILES_TO_TABLES:
        if not os.path.exists(os.path.join(args.data_dir, filename)):
            missing.append(filename)
    if missing:
        print("ERROR: these files are missing from --data-dir:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    for filename, table in FILES_TO_TABLES.items():
        path = os.path.join(args.data_dir, filename)
        dtype_overrides = STRING_OVERRIDES.get(filename)
        df = pd.read_csv(path, dtype=dtype_overrides)

        print(f"Loading {filename} -> table '{table}' ({len(df)} rows, {len(df.columns)} cols) ...")
        df.to_sql(table, engine, if_exists="replace", index=False)

        with engine.connect() as conn:
            db_count = conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()

        status = "OK" if db_count == len(df) else "MISMATCH"
        print(f"  -> {status}: source {len(df)} rows, DB now has {db_count} rows.\n")

    print("Done. shap_summary.png is an image, not tabular data -- it isn't loaded; "
          "keep it in your repo/docs folder for the report instead.")


if __name__ == "__main__":
    main()
