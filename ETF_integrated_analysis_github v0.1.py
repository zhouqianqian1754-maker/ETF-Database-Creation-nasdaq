import os
import time
from io import StringIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy.stats import skew, kurtosis

tickers_etf = ["ITA", "PPA", "SHLD", "XAR", "EUAD", "ARKX",
               "DFEN", "MISL", "UFO", "FITE", "NATO", "JEDI",
               "WAR", "ASIA", "SPDV"]

tickers_stocks = ["DE", "DFNS"]
tickers = tickers_etf + tickers_stocks

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_FOLDER = OUTPUT_DIR / f"ETF_365D_Analysis_{RUN_TIMESTAMP}"
RUN_FOLDER.mkdir(parents=True, exist_ok=True)

LOCAL_MASTER_REPO = BASE_DIR / "ETF-Database-Creation-nasdaq"
RAW_BASE = "https://raw.githubusercontent.com/zhouqianqian1754-maker/ETF-Database-Creation-nasdaq/main/data"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nasdaq.com/",
})

def max_drawdown(price_series):
    s = pd.to_numeric(price_series, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan
    running_max = s.cummax()
    dd = (s / running_max - 1.0) * 100
    return dd.min()

def to_numeric_safe(series_or_value):
    if isinstance(series_or_value, pd.Series):
        return pd.to_numeric(
            series_or_value.astype(str).str.replace(r"[,$%]", "", regex=True),
            errors="coerce"
        )
    return pd.to_numeric(
        str(series_or_value).replace(",", "").replace("$", "").replace("%", ""),
        errors="coerce"
    )

def get_value_from_summary(summary, candidate_keys):
    for key in candidate_keys:
        val = summary.get(key)
        if isinstance(val, dict):
            value = val.get("value")
            if value not in [None, "", "N/A", "--"]:
                return value
        elif val not in [None, "", "N/A", "--"]:
            return val
    return np.nan

def get_master_df(ticker, close_col="Close/Last"):
    local_file = LOCAL_MASTER_REPO / "data" / f"{ticker} Database" / f"{ticker}_MasterData.csv"
    raw_url = f"{RAW_BASE}/{ticker}%20Database/{ticker}_MasterData.csv"

    try:
        if local_file.exists():
            df = pd.read_csv(local_file)
        else:
            r = session.get(raw_url, timeout=30)
            if r.status_code == 404:
                print(f"{ticker}: MasterData not found locally or on GitHub raw, skipping historical analysis")
                return None
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))

        df.columns = [c.strip() for c in df.columns]
        if "Date" not in df.columns or close_col not in df.columns:
            return None

        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df[close_col] = pd.to_numeric(
            df[close_col].astype(str).str.replace(r"[,$]", "", regex=True),
            errors="coerce"
        )

        if "Volume" in df.columns:
            df["Volume"] = pd.to_numeric(
                df["Volume"].astype(str).str.replace(r"[,$]", "", regex=True),
                errors="coerce"
            )

        df = df.dropna(subset=["Date", close_col]).sort_values("Date").reset_index(drop=True)
        return df

    except Exception as e:
        print(f"{ticker}: failed loading MasterData -> {e}")
        return None

def latest_momentum_12_1_from_master_df(df, close_col="Close/Last"):
    if df is None or len(df) < 252:
        return np.nan
    recent = df[close_col].iloc[-21]
    past = df[close_col].iloc[-252]
    if pd.isna(recent) or pd.isna(past) or past == 0:
        return np.nan
    return (recent - past) / past

def fetch_nasdaq_summary(ticker, assetclass):
    url = f"https://api.nasdaq.com/api/quote/{ticker.lower()}/summary?assetclass={assetclass}"
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        j = r.json()
        summary = (j.get("data") or {}).get("summaryData") or {}
        if not summary:
            return None

        return {
            "Ticker": ticker,
            "Share Volume": to_numeric_safe(get_value_from_summary(summary, ["ShareVolume"])),
            "Previous Close": to_numeric_safe(get_value_from_summary(summary, ["PreviousClose"])),
            "Market Cap": to_numeric_safe(get_value_from_summary(summary, ["MarketCap"])),
            "Weighted Alpha": to_numeric_safe(get_value_from_summary(summary, ["WeightedAlpha"])),
            "Beta": to_numeric_safe(get_value_from_summary(summary, ["Beta"])),
            "Standard Deviation": to_numeric_safe(get_value_from_summary(summary, ["StandardDeviation", "StdDev"])),
            "Assets Under Management (,000)": to_numeric_safe(get_value_from_summary(summary, ["AssetsUnderManagement", "AssetsUnderManagement000", "AUM", "NetAssets"])),
            "Expense Ratio": to_numeric_safe(get_value_from_summary(summary, ["ExpenseRatio", "NetExpenseRatio"]))
        }
    except Exception as e:
        print(f"{ticker}: Nasdaq fetch failed -> {e}")
        return {"Ticker": ticker}

def fetch_yahoo_summary(ticker):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="2y", auto_adjust=False)
        fast = getattr(tk, "fast_info", {}) or {}
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            info = {}

        hist_close = hist["Close"].dropna() if "Close" in hist.columns else pd.Series(dtype=float)

        return {
            "Ticker": ticker,
            "Share Volume (yf)": fast.get("lastVolume", np.nan) if isinstance(fast, dict) else np.nan,
            "Previous Close (yf)": fast.get("previousClose", np.nan) if isinstance(fast, dict) else np.nan,
            "52 Week High (yf)": hist_close.tail(252).max() if len(hist_close) else info.get("fiftyTwoWeekHigh", np.nan),
            "52 Week Low (yf)": hist_close.tail(252).min() if len(hist_close) else info.get("fiftyTwoWeekLow", np.nan),
            "Market Cap (yf)": info.get("marketCap", np.nan),
            "PE_ratio (yf)": info.get("trailingPE", np.nan),
            "Beta (yf)": info.get("beta", np.nan)
        }
    except Exception as e:
        print(f"{ticker}: Yahoo fetch failed -> {e}")
        return {"Ticker": ticker}

def score_column(series, bigger=True):
    series = pd.to_numeric(series, errors="coerce")
    valid_mask = series.notna()
    n = int(valid_mask.sum())
    scores = pd.Series(np.nan, index=series.index, dtype="float64")

    if n == 0:
        return scores
    if n == 1:
        scores.loc[valid_mask] = 1.0
        return scores

    ranks = series[valid_mask].rank(method="min", ascending=not bigger)
    scores.loc[valid_mask] = (n - ranks) / (n - 1)
    return scores

def run_analysis_for_group(tickers, weights, output_suffix):
    analysis_rows, summary_rows, yf_rows = [], [], []

    current_date_display = datetime.now().strftime("%d/%m/%Y")
    current_date_file = datetime.now().strftime("%Y%m%d")
    current_ts = pd.Timestamp.today().normalize()
    start_52w = current_ts - pd.DateOffset(weeks=52)
    start_104w = current_ts - pd.DateOffset(weeks=104)

    for ticker in tickers:
        print(f"Processing {ticker} ...")
        df = get_master_df(ticker)

        if df is not None and not df.empty:
            df_52w = df[df["Date"] >= start_52w]
            df_104w = df[df["Date"] >= start_104w]

            high_52w = df_52w["Close/Last"].max() if not df_52w.empty else np.nan
            low_52w = df_52w["Close/Last"].min() if not df_52w.empty else np.nan
            high_104w = df_104w["Close/Last"].max() if not df_104w.empty else np.nan
            low_104w = df_104w["Close/Last"].min() if not df_104w.empty else np.nan

            avg_vol_50d = df["Volume"].tail(50).mean() if "Volume" in df.columns else np.nan
            avg_vol_20d = df["Volume"].tail(20).mean() if "Volume" in df.columns else np.nan
            avg_vol_65d = df["Volume"].tail(65).mean() if "Volume" in df.columns else np.nan
            momentum_12_1_val = latest_momentum_12_1_from_master_df(df)

            df_365 = df.tail(365).copy()
            df_365["daily_return"] = df_365["Close/Last"].pct_change()
            ret = df_365["daily_return"].dropna()

            if len(ret) > 1:
                analysis_rows.append({
                    "Ticker": ticker,
                    "Rows_used": len(df_365),
                    "Avg_daily_return": ret.mean(),
                    "Std_daily_return": ret.std(ddof=1),
                    "Skewness": skew(ret, bias=False),
                    "Kurtosis": kurtosis(ret, bias=False, fisher=True),
                    "Max_drawdown": max_drawdown(df_365["Close/Last"]),
                    "52 Week High": high_52w,
                    "52 Week Low": low_52w,
                    "104 Week High": high_104w,
                    "104 Week Low": low_104w,
                    "50 Day Avg Daily Volume": avg_vol_50d,
                    "Average Daily Volume 20 Days": avg_vol_20d,
                    "Average Daily Volume 65 Days": avg_vol_65d,
                    "momentum_12_1": momentum_12_1_val,
                    "Current_Date": current_date_display
                })

        assetclass = "etf" if ticker in tickers_etf else "stocks"
        summary_rows.append(fetch_nasdaq_summary(ticker, assetclass))
        yf_rows.append(fetch_yahoo_summary(ticker))
        time.sleep(1.5)

    analysis_df = pd.DataFrame(analysis_rows)
    summary_df = pd.DataFrame(summary_rows)
    yf_df = pd.DataFrame(yf_rows)

    final_df = pd.DataFrame()
    if not summary_df.empty and not analysis_df.empty:
        final_df = pd.merge(summary_df, analysis_df, on="Ticker", how="outer", suffixes=("_api", "_hist"))
    elif not summary_df.empty:
        final_df = summary_df.copy()
    elif not analysis_df.empty:
        final_df = analysis_df.copy()

    if not yf_df.empty:
        final_df = pd.merge(final_df, yf_df, on="Ticker", how="left") if not final_df.empty else yf_df.copy()

    score_detail_df = pd.DataFrame()

    if not final_df.empty:
        if "Current_Date_api" in final_df.columns and "Current_Date_hist" in final_df.columns:
            final_df["Current_Date"] = final_df["Current_Date_hist"].combine_first(final_df["Current_Date_api"])
            final_df = final_df.drop(columns=["Current_Date_api", "Current_Date_hist"])
        elif "Current_Date_hist" in final_df.columns:
            final_df = final_df.rename(columns={"Current_Date_hist": "Current_Date"})
        elif "Current_Date_api" in final_df.columns:
            final_df = final_df.rename(columns={"Current_Date_api": "Current_Date"})

        text_cols = {"Ticker", "Current_Date"}
        for col in final_df.columns:
            if col not in text_cols:
                final_df[col] = pd.to_numeric(final_df[col], errors="coerce")

        if "Max_drawdown" in final_df.columns:
            final_df["Abs(Max_drawdown)"] = final_df["Max_drawdown"].abs()
            final_df = final_df.drop(columns=["Max_drawdown"])

        if all(col in final_df.columns for col in ["Previous Close", "52 Week Low", "52 Week High"]):
            denom_52w = final_df["52 Week High"] - final_df["52 Week Low"]
            final_df["Price_frac_52W"] = (final_df["Previous Close"] - final_df["52 Week Low"]) / denom_52w.replace(0, np.nan)

        if all(col in final_df.columns for col in ["Previous Close", "104 Week Low", "104 Week High"]):
            denom_104w = final_df["104 Week High"] - final_df["104 Week Low"]
            final_df["Price_frac_104W"] = (final_df["Previous Close"] - final_df["104 Week Low"]) / denom_104w.replace(0, np.nan)

        bigger_is_better = [
            "Share Volume (yf)", "Market Cap", "Weighted Alpha", "Assets Under Management (,000)",
            "Avg_daily_return", "Skewness", "50 Day Avg Daily Volume", "Average Daily Volume 20 Days",
            "Average Daily Volume 65 Days", "PE_ratio (yf)", "momentum_12_1"
        ]
        smaller_is_better = [
            "Beta", "Standard Deviation", "Expense Ratio", "Std_daily_return",
            "Kurtosis", "Abs(Max_drawdown)", "Price_frac_52W", "Price_frac_104W"
        ]

        score_detail_df = final_df[["Ticker"]].copy()

        for col in bigger_is_better:
            if col in final_df.columns:
                score_detail_df[f"{col}_score"] = score_column(final_df[col], bigger=True)

        for col in smaller_is_better:
            if col in final_df.columns:
                score_detail_df[f"{col}_score"] = score_column(final_df[col], bigger=False)

        weighted_sum = pd.Series(0.0, index=score_detail_df.index)
        weight_used = pd.Series(0.0, index=score_detail_df.index)

        for col, wt in weights.items():
            if col in score_detail_df.columns:
                valid = score_detail_df[col].notna()
                weighted_sum.loc[valid] += score_detail_df.loc[valid, col] * wt
                weight_used.loc[valid] += wt

        score_detail_df["Score"] = np.where(weight_used > 0, weighted_sum / weight_used, np.nan)
        final_df = pd.merge(final_df, score_detail_df, on="Ticker", how="left")

    final_output = RUN_FOLDER / f"ETF_365D_Analysis_{current_date_file}_{output_suffix}.csv"
    scoring_output = RUN_FOLDER / f"ETF_365D_Scoring_{current_date_file}_{output_suffix}.csv"

    if not final_df.empty:
        final_df.to_csv(final_output, index=False)
    if not score_detail_df.empty:
        score_detail_df.to_csv(scoring_output, index=False)

    return final_df, score_detail_df

weights_con = {
    "Share Volume (yf)_score": 10,
    "Market Cap_score": 2,
    "Weighted Alpha_score": 0,
    "Assets Under Management (,000)_score": 10,
    "Avg_daily_return_score": 0,
    "Skewness_score": 0,
    "50 Day Avg Daily Volume_score": 4,
    "Average Daily Volume 20 Days_score": 6,
    "Average Daily Volume 65 Days_score": 6,
    "PE_ratio (yf)_score": 0,
    "momentum_12_1_score": 0,
    "Beta_score": 12,
    "Standard Deviation_score": 10,
    "Expense Ratio_score": 18,
    "Std_daily_return_score": 0,
    "Kurtosis_score": 0,
    "Abs(Max_drawdown)_score": 16,
    "Price_frac_52W_score": 3,
    "Price_frac_104W_score": 3
}

weights_grow = {
    "momentum_12_1_score": 20,
    "Price_frac_52W_score": 18,
    "Price_frac_104W_score": 15,
    "Weighted Alpha_score": 10,
    "Avg_daily_return_score": 8,
    "Assets Under Management (,000)_score": 8,
    "Share Volume (yf)_score": 7,
    "Average Daily Volume 20 Days_score": 5,
    "Average Daily Volume 65 Days_score": 5,
    "50 Day Avg Daily Volume_score": 4,
    "Market Cap_score": 4,
    "Beta_score": 2,
    "Standard Deviation_score": 2,
    "Abs(Max_drawdown)_score": 1,
    "Expense Ratio_score": 1,
    "Std_daily_return_score": 1,
    "PE_ratio (yf)_score": 1,
    "Skewness_score": 0,
    "Kurtosis_score": 0
}

run_analysis_for_group(tickers, weights_con, "con")
run_analysis_for_group(tickers, weights_grow, "grow")