import pandas as pd
import zipfile
from pathlib import Path

zip_files = sorted(Path("dataset").glob("*.zip"))
dfs = []
for z in zip_files:
    with zipfile.ZipFile(z, "r") as archive:
        for name in archive.namelist():
            if name.endswith(".csv"):
                with archive.open(name) as f:
                    df = pd.read_csv(f)
                    df.columns = [
                        "open_time", "open", "high", "low", "close", "volume",
                        "close_time", "quote_asset_volume", "number_of_trades",
                        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                    ]
                    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df["timestamp"] = pd.to_datetime(df["open_time"], unit="us", utc=True)
for col in ["open", "high", "low", "close", "volume"]:
    df[col] = pd.to_numeric(df[col])
df = df.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")
df_5m = df.resample("300s").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}).dropna()
df_5m.to_csv("ethusdt_5m_1year.csv")
print(df_5m.shape)