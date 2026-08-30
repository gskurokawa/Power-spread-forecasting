"""
One-time loader: push the modelling table (and predictions) into a cloud
Postgres database (e.g. Neon). Reads the connection string from DATABASE_URL.

Easiest: put it in a .env file in this folder (which .gitignore excludes):

    DATABASE_URL=postgresql://USER:PASS@HOST/DB?sslmode=require

then just run:  python load_to_postgres.py
(Alternatively set it in the shell for one session with $env:DATABASE_URL=... )

Requires:  python -m pip install sqlalchemy psycopg2-binary pandas python-dotenv
"""
import os
import pandas as pd
from sqlalchemy import create_engine

# load a local .env if present (never committed); harmless if python-dotenv absent
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

url = os.environ.get("DATABASE_URL")
if not url:
    raise SystemExit("No DATABASE_URL found — put it in a .env file or set it in the shell.")

engine = create_engine(url, pool_pre_ping=True)

# modelling table — the CSV index is the hourly timestamp
mt = pd.read_csv("modelling_table.csv", index_col=0)
mt.index.name = "timestamp"
mt.to_sql("modelling_table", engine, if_exists="replace",
          index=True, index_label="timestamp", chunksize=2000, method="multi")
print(f"modelling_table: {len(mt):,} rows loaded")

# predictions — for the dashboard charts
if os.path.exists("predictions.csv"):
    pr = pd.read_csv("predictions.csv")
    pr.to_sql("predictions", engine, if_exists="replace",
              index=False, chunksize=2000, method="multi")
    print(f"predictions:     {len(pr):,} rows loaded")

print("done — data is now in cloud Postgres.")
