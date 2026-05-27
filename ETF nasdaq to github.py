import shutil
import traceback
import time
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

tickers = [
    "ITA", "PPA", "SHLD", "XAR", "EUAD", "ARKX",
    "DFEN", "MISL", "UFO", "FITE", "NATO", "JEDI",
    "WAR", "DE", "DFNS", "ASIA", "SPDV"
]

etf_tickers = {
    "ITA", "PPA", "SHLD", "XAR", "EUAD", "ARKX",
    "DFEN", "MISL", "UFO", "FITE", "NATO", "JEDI",
    "WAR", "ASIA", "SPDV"
}
stock_tickers = {"DE", "DFNS"}

HK_TZ = ZoneInfo("Asia/Hong_Kong")

BASE_DIR = Path(__file__).resolve().parent
TEMP_DOWNLOAD_DIR = BASE_DIR / "downloads"
DATA_DIR = BASE_DIR / "data"
DEBUG_DIR = BASE_DIR / "debug"

print(f"Repository base directory: {BASE_DIR}")
print(f"Download directory: {TEMP_DOWNLOAD_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Debug directory: {DEBUG_DIR}")

TEMP_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,text/plain,*/*",
    "Origin": "https://www.nasdaq.com",
})

def get_current_date_str():
    return datetime.now(HK_TZ).strftime("%Y%m%d")

def clear_download_dir(folder: Path):
    for f in folder.glob("*"):
        if f.is_file():
            f.unlink()

def save_debug_response(ticker: str, label: str, content: str):
    stamp = datetime.now(HK_TZ).strftime("%Y%m%d_%H%M%S")
    path = DEBUG_DIR / f"{ticker}_{label}_{stamp}.txt"
    path.write_text(content, encoding="utf-8")
    print(f"Saved debug response: {path}", flush=True)

def get_assetclass(ticker: str):
    return "stocks" if ticker in stock_tickers else "etf"

def extract_rows(payload):
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    trades_table = data.get("tradesTable")
    if not isinstance(trades_table, dict):
        return []
    rows = trades_table.get("rows", [])
    return rows if isinstance(rows, list) else []

def download_nasdaq_csv(ticker: str, download_dir: Path):
    clear_download_dir(download_dir)

    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start_date = "2010-01-01"
    assetclass = get_assetclass(ticker)

    url = (
        f"https://api.nasdaq.com/api/quote/{ticker}/historical"
        f"?assetclass={assetclass}&fromdate={start_date}&limit=9999&todate={end_date}"
    )

    headers = {
        "Referer": (
            f"https://www.nasdaq.com/market-activity/"
            f"{'stocks' if assetclass == 'stocks' else 'etf'}/{ticker.lower()}/historical"
        )
    }

    print(f"Downloading historical data for {ticker}: {url}", flush=True)
    response = session.get(url, headers=headers, timeout=60)
    print(f"HTTP status for {ticker}: {response.status_code}", flush=True)

    if response.status_code != 200:
        save_debug_response(ticker, "http_error", response.text[:5000])
        response.raise_for_status()

    try:
        payload = response.json()
    except Exception:
        save_debug_response(ticker, "invalid_json", response.text[:5000])
        raise ValueError(f"Invalid JSON response for {ticker}")

    rows = extract_rows(payload)

    if not rows:
        save_debug_response(ticker, "empty_rows", response.text[:5000])
        raise ValueError(f"No historical rows returned for {ticker}")

    df = pd.DataFrame(rows)
    file_path = download_dir / f"{ticker}_raw.csv"
    df.to_csv(file_path, index=False)
    print(f"Saved raw CSV for {ticker}: {file_path}", flush=True)
    return file_path

def get_output_paths(ticker: str):
    current_date = get_current_date_str()
    ticker_dir = DATA_DIR / f"{ticker} Database"
    ticker_dir.mkdir(parents=True, exist_ok=True)

    daily_file_path = ticker_dir / f"{ticker}_hist_till{current_date}.csv"
    master_file_path = ticker_dir / f"{ticker}_MasterData.csv"
    return ticker_dir, daily_file_path, master_file_path

def standardize_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close/Last",
        "volume": "Volume"
    }
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    if "Date" not in df.columns or "Close/Last" not in df.columns:
        save_debug_response(ticker, "missing_columns", str(df.columns.tolist()))
        raise ValueError(f"Missing Date or Close/Last for {ticker}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    for col in [c for c in ["Open", "High", "Low", "Close/Last", "Volume"] if c in df.columns]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(r"[,$]", "", regex=True),
            errors="coerce"
        )

    df["Close"] = df["Close/Last"]
    df["daily_return"] = df["Close"].pct_change()
    df["MA_20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA_50"] = df["Close"].rolling(50, min_periods=1).mean()

    return df

def save_with_display_date(df: pd.DataFrame, path: Path):
    out = df.copy()
    out["Date"] = out["Date"].dt.strftime("%d/%m/%Y")
    out.to_csv(path, index=False)

def process_ticker(ticker: str):
    print("=" * 80)
    print(f"Processing {ticker}...")

    ticker_dir, daily_file_path, master_file_path = get_output_paths(ticker)

    latest_file = download_nasdaq_csv(ticker, TEMP_DOWNLOAD_DIR)

    if daily_file_path.exists():
        daily_file_path.unlink()

    shutil.move(str(latest_file), str(daily_file_path))
    print(f"Downloaded and renamed to: {daily_file_path}")

    daily_df = pd.read_csv(daily_file_path)
    daily_df = standardize_dataframe(daily_df, ticker)
    save_with_display_date(daily_df, daily_file_path)

    if master_file_path.exists():
        master_df = pd.read_csv(master_file_path)
        master_df.columns = [c.strip() for c in master_df.columns]
        master_df["Date"] = pd.to_datetime(master_df["Date"], format="%d/%m/%Y", errors="coerce")
        master_df = master_df.dropna(subset=["Date"])
    else:
        master_df = pd.DataFrame(columns=daily_df.columns)

    updated_master_df = pd.concat([master_df, daily_df], ignore_index=True)
    updated_master_df = (
        updated_master_df
        .dropna(subset=["Date"])
        .sort_values("Date")
        .drop_duplicates(subset="Date", keep="last")
        .reset_index(drop=True)
    )

    save_with_display_date(updated_master_df, master_file_path)
    print(f"Saved master file: {master_file_path}")

def main():
    print("Starting ETF download/update job...")
    for ticker in tickers:
        try:
            process_ticker(ticker)
            time.sleep(1)
        except Exception as e:
            print(f"Failed for {ticker}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
