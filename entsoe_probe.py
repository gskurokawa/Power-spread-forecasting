#%%
# entsoe_probe.py — one direct ENTSO-E query, show the REAL error
import os, traceback
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass
import pandas as pd
from entsoe import EntsoePandasClient

key = os.environ.get("ENTSOE_API_KEY")
print("API key present:", bool(key), "| length:", len(key or ""))

client = EntsoePandasClient(api_key=key, retry_count=3, retry_delay=0, timeout=30)
now = pd.Timestamp.now(tz="Europe/Brussels")
start, end = now.floor("D") - pd.Timedelta(days=1), now.floor("D") + pd.Timedelta(days=1)
print("querying DE_LU day-ahead prices", start, "->", end)
try:
    s = client.query_day_ahead_prices("DE_LU", start=start, end=end)
    print("SUCCESS — got", len(s), "rows:\n", s.tail())
except Exception as e:
    print("FAILED:", type(e).__name__, "-", str(e)[:300])
    resp = getattr(e, "response", None)
    if resp is not None:
        print("HTTP status:", resp.status_code)
        print("body (first 500):", resp.text[:500])
    traceback.print_exc()