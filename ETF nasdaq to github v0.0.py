import os
import time
import shutil
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.chrome.service import Service


tickers = [
    "ITA", "PPA", "SHLD", "XAR", "EUAD", "ARKX",
    "DFEN", "MISL", "UFO", "FITE", "NATO", "JEDI",
    "WAR", "DE", "DFNS", "ASIA", "SPDV"
]

BASE_DIR = Path(__file__).resolve().parent
TEMP_DOWNLOAD_DIR = BASE_DIR / "downloads"
DATA_DIR = BASE_DIR / "data"
DEBUG_DIR = BASE_DIR / "debug"

print(f"Repository base directory: {BASE_DIR}")
print(f"Download directory: {TEMP_DOWNLOAD_DIR}")
print(f"Data directory: {DATA_DIR}")

TEMP_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def get_current_date_str():
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def build_driver(download_dir: Path):
    print("Entering build_driver()...")

    chrome_bin = os.environ.get("CHROME_BIN")
    chromedriver_path = os.environ.get("CHROMEDRIVER_PATH")

    if not chrome_bin:
        raise RuntimeError("CHROME_BIN is not set")
    if not chromedriver_path:
        raise RuntimeError("CHROMEDRIVER_PATH is not set")

    chrome_options = Options()
    chrome_options.binary_location = chrome_bin
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")

    prefs = {
        "download.default_directory": str(download_dir.resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "profile.default_content_settings.popups": 0
    }
    chrome_options.add_experimental_option("prefs", prefs)

    print(f"Using Chrome binary: {chrome_bin}")
    print(f"Using ChromeDriver path: {chromedriver_path}")

    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(60)

    driver.execute_cdp_cmd(
        "Page.setDownloadBehavior",
        {
            "behavior": "allow",
            "downloadPath": str(download_dir.resolve())
        }
    )

    print("Chrome driver created successfully.")
    return driver


def clear_download_dir(folder: Path):
    for f in folder.glob("*"):
        if f.is_file():
            f.unlink()


def accept_cookies_if_present(driver, timeout=8):
    print("Checking cookie popup...")
    wait = WebDriverWait(driver, timeout)
    try:
        try:
            iframe = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "iframe[src*='onetrust']"))
            )
            driver.switch_to.frame(iframe)
            print("Cookie iframe found.")
        except TimeoutException:
            print("No cookie iframe found.")

        btn = wait.until(
            EC.element_to_be_clickable((By.ID, "onetrust-accept-btn-handler"))
        )
        driver.execute_script("arguments[0].click();", btn)
        print("Cookie banner accepted.")
    except Exception:
        print("No cookie accept action needed.")
    finally:
        driver.switch_to.default_content()


def wait_for_download_complete(folder: Path, timeout=120, pattern="*.csv"):
    print(f"Waiting for download in {folder}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        csv_files = list(folder.glob(pattern))
        crdownload_files = list(folder.glob("*.crdownload"))

        if csv_files and not crdownload_files:
            latest_file = max(csv_files, key=lambda f: f.stat().st_ctime)
            if latest_file.is_file():
                print(f"Download completed: {latest_file}")
                return latest_file

        time.sleep(1)

    raise TimeoutError("Download did not complete within timeout.")


def safe_click(driver, element):
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", element)
    time.sleep(1)
    try:
        element.click()
    except Exception:
        driver.execute_script("arguments[0].click();", element)


def save_debug_artifacts(driver, ticker, label):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    html_path = DEBUG_DIR / f"{ticker}_{label}_{stamp}.html"
    png_path = DEBUG_DIR / f"{ticker}_{label}_{stamp}.png"

    try:
        html_path.write_text(driver.page_source, encoding="utf-8")
        driver.save_screenshot(str(png_path))
        print(f"Saved debug HTML: {html_path}")
        print(f"Saved debug screenshot: {png_path}")
    except Exception as e:
        print(f"Could not save debug artifacts: {e}")


def click_max_if_present(driver, wait):
    max_selectors = [
        "//button[normalize-space()='MAX']",
        "//button[contains(., 'MAX')]",
        "//*[self::button or self::span][normalize-space()='MAX']",
        "//*[contains(@class,'historical') or contains(@class,'time') or contains(@class,'range')]//button[contains(., 'MAX')]"
    ]

    for xpath in max_selectors:
        try:
            print(f"Trying MAX selector: {xpath}")
            elem = wait.until(EC.presence_of_element_located((By.XPATH, xpath)))
            safe_click(driver, elem)
            print("MAX clicked successfully.")
            time.sleep(3)
            return True
        except Exception:
            continue

    print("MAX button not found or not clickable. Continuing without MAX.")
    return False


def click_download_button(driver, wait):
    download_selectors = [
        "//span[contains(., 'Download historical data')]/ancestor::button",
        "//button[contains(., 'Download historical data')]",
        "//button[contains(., 'Download Data')]",
        "//button[contains(., 'Download')]",
        "//*[@aria-label='Download historical data']"
    ]

    for xpath in download_selectors:
        try:
            print(f"Trying download selector: {xpath}")
            elem = wait.until(EC.presence_of_element_located((By.XPATH, xpath)))
            safe_click(driver, elem)
            print("Download button clicked.")
            return True
        except Exception:
            continue

    return False


def download_nasdaq_csv(ticker: str, download_dir: Path):
    clear_download_dir(download_dir)
    driver = build_driver(download_dir)

    try:
        url = f"https://www.nasdaq.com/market-activity/etf/{ticker.lower()}/historical"
        print(f"Opening URL for {ticker}: {url}")
        driver.get(url)
        print("Page loaded.")
        time.sleep(5)

        accept_cookies_if_present(driver)

        wait = WebDriverWait(driver, 30)

        try:
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        except TimeoutException:
            save_debug_artifacts(driver, ticker, "body_timeout")
            raise

        try:
            print("Waiting for page text indicating historical data...")
            wait.until(
                lambda d: "Historical Data" in d.page_source or "Download historical data" in d.page_source
            )
            print("Historical section text detected.")
        except TimeoutException:
            print("Historical section text not detected; continuing.")
            save_debug_artifacts(driver, ticker, "historical_text_missing")

        click_max_if_present(driver, wait)

        if not click_download_button(driver, wait):
            save_debug_artifacts(driver, ticker, "download_button_missing")
            raise TimeoutException("Could not find clickable download button.")

        latest_file = wait_for_download_complete(download_dir, timeout=120, pattern="*.csv")
        print(f"Downloaded raw file for {ticker}: {latest_file}")
        return latest_file

    except Exception:
        save_debug_artifacts(driver, ticker, "download_failure")
        raise

    finally:
        print("Closing driver...")
        driver.quit()


def kama(closes, n=10, fast_period=2, slow_period=30):
    fast_SC = 2.0 / (fast_period + 1)
    slow_SC = 2.0 / (slow_period + 1)

    kama_vals = np.full(len(closes), np.nan)
    if len(closes) < n:
        return kama_vals

    kama_vals[n - 1] = closes[:n].mean()

    for i in range(n, len(closes)):
        change = abs(closes[i] - closes[i - n])
        volatility = sum(abs(closes[i - j + 1] - closes[i - j]) for j in range(1, n + 1))
        ER = 0.0 if volatility == 0.0 else change / volatility
        SC = (ER * (fast_SC - slow_SC) + slow_SC) ** 2
        kama_vals[i] = kama_vals[i - 1] + SC * (closes[i] - kama_vals[i - 1])

    return kama_vals


def momentum_12_1(prices, skip_days=21, lookback_days=252):
    prices = pd.Series(prices).astype(float)
    mom = np.full(len(prices), np.nan)
    start_idx = lookback_days - 1

    for i in range(start_idx, len(prices)):
        recent_idx = i - skip_days
        past_idx = i - lookback_days + 1

        if recent_idx < 0 or past_idx < 0:
            continue

        recent = prices.iloc[recent_idx]
        past = prices.iloc[past_idx]

        if pd.notna(recent) and pd.notna(past) and past != 0:
            mom[i] = (recent - past) / past

    return mom


def calc_CCI(df, n=20, col_close="Close"):
    df = df.copy()
    df["TP"] = (df["High"] + df["Low"] + df[col_close]) / 3
    df["MA_TP"] = df["TP"].rolling(n).mean()
    df["Deviation"] = abs(df["TP"] - df["MA_TP"])
    df["MD"] = df["Deviation"].rolling(n).mean()
    df["CCI"] = (df["TP"] - df["MA_TP"]) / (df["MD"] * 0.015)
    return df[["Date", "TP", "MA_TP", "MD", "CCI"]]


def calc_williams_r(df, n=14, col_close="Close"):
    df = df.copy()
    df["HH"] = df["High"].rolling(n).max()
    df["LL"] = df["Low"].rolling(n).min()
    df["Williams_R"] = ((df["HH"] - df[col_close]) / (df["HH"] - df["LL"])) * -100
    return df[["Date", "Williams_R"]]


def calc_stochastic(df, n=9, k_period=3, d_period=3, col_close="Close"):
    df = df.copy()
    df["L_n"] = df["Low"].rolling(n).min()
    df["H_n"] = df["High"].rolling(n).max()
    df["RSV"] = (df[col_close] - df["L_n"]) / (df["H_n"] - df["L_n"]) * 100

    df["K"] = 50.0
    df["D"] = 50.0

    for i in range(n, len(df)):
        df.loc[df.index[i], "K"] = (2 / 3 * df["K"].iloc[i - 1] + 1 / 3 * df["RSV"].iloc[i])
        df.loc[df.index[i], "D"] = (2 / 3 * df["D"].iloc[i - 1] + 1 / 3 * df["K"].iloc[i])

    df["J"] = 3 * df["K"] - 2 * df["D"]
    return df[["Date", "K", "D", "J"]]


def calc_ATR(df, n=14, col_close="Close"):
    df = df.copy()
    df["HL"] = df["High"] - df["Low"]
    df["HC"] = abs(df["High"] - df[col_close].shift(1))
    df["LC"] = abs(df["Low"] - df[col_close].shift(1))
    df["TR"] = df[["HL", "HC", "LC"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(n).mean()
    return df[["Date", "TR", "ATR"]]


def calc_historical_volatility(df, n=20, col_close="Close"):
    df = df.copy()
    df["log_return"] = np.log(df[col_close] / df[col_close].shift(1))
    df["HV_daily"] = df["log_return"].rolling(n).std()
    df["HV_annual"] = df["HV_daily"] * np.sqrt(252) * 100
    return df[["Date", "log_return", "HV_daily", "HV_annual"]]


def calc_OBV(df, col_close="Close", col_volume="Volume"):
    df = df.copy()
    df["OBV"] = 0.0

    for i in range(1, len(df)):
        if df[col_close].iloc[i] > df[col_close].iloc[i - 1]:
            df.loc[df.index[i], "OBV"] = df["OBV"].iloc[i - 1] + df[col_volume].iloc[i]
        elif df[col_close].iloc[i] < df[col_close].iloc[i - 1]:
            df.loc[df.index[i], "OBV"] = df["OBV"].iloc[i - 1] - df[col_volume].iloc[i]
        else:
            df.loc[df.index[i], "OBV"] = df["OBV"].iloc[i - 1]

    return df[["Date", "OBV"]]


def calc_MFI(df, n=14, col_close="Close", col_volume="Volume"):
    df = df.copy()
    df["TP_mfi"] = (df["High"] + df["Low"] + df[col_close]) / 3
    df["MF"] = df["TP_mfi"] * df[col_volume]
    df["TP_change"] = df["TP_mfi"] > df["TP_mfi"].shift(1)
    df["Positive_MF"] = np.where(df["TP_change"], df["MF"], 0)
    df["Negative_MF"] = np.where(~df["TP_change"], df["MF"], 0)
    df["PMF_sum"] = df["Positive_MF"].rolling(n).sum()
    df["NMF_sum"] = df["Negative_MF"].rolling(n).sum()
    df["MR"] = df["PMF_sum"] / df["NMF_sum"]
    df["MFI"] = 100 - (100 / (1 + df["MR"]))
    return df[["Date", "MFI"]]


def calc_sharpe_ratio(df, col_close="Close", rf_annual=0.03, window=252):
    df = df.copy()

    if "daily_return" not in df.columns:
        df["daily_return"] = df[col_close].pct_change()

    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    df["excess_ret"] = df["daily_return"] - rf_daily

    rolling_mean = df["excess_ret"].rolling(window).mean()
    rolling_std = df["excess_ret"].rolling(window).std(ddof=1)

    df["Sharpe_Ratio"] = np.where(
        rolling_std != 0,
        (rolling_mean / rolling_std) * np.sqrt(252),
        np.nan
    )

    return df[["Date", "excess_ret", "Sharpe_Ratio"]]


def calc_cagr(df, col_close="Close", rolling_window=252):
    df = df.copy()
    df["CAGR_Full"] = np.nan
    df["CAGR_252D"] = np.nan

    if len(df) < 2:
        return df[["Date", "CAGR_Full", "CAGR_252D"]]

    start_price = df[col_close].iloc[0]
    start_date = pd.to_datetime(df["Date"].iloc[0])

    for i in range(1, len(df)):
        end_price = df[col_close].iloc[i]
        end_date = pd.to_datetime(df["Date"].iloc[i])
        days = (end_date - start_date).days

        if days > 0 and start_price > 0 and end_price > 0:
            df.loc[df.index[i], "CAGR_Full"] = (end_price / start_price) ** (365.25 / days) - 1

    for i in range(rolling_window, len(df)):
        start_price_roll = df[col_close].iloc[i - rolling_window]
        end_price_roll = df[col_close].iloc[i]
        start_date_roll = pd.to_datetime(df["Date"].iloc[i - rolling_window])
        end_date_roll = pd.to_datetime(df["Date"].iloc[i])
        days_roll = (end_date_roll - start_date_roll).days

        if days_roll > 0 and start_price_roll > 0 and end_price_roll > 0:
            df.loc[df.index[i], "CAGR_252D"] = (end_price_roll / start_price_roll) ** (365.25 / days_roll) - 1

    return df[["Date", "CAGR_Full", "CAGR_252D"]]


def create_labels(df, col_close="Close"):
    df = df.copy()
    df["next_close"] = df[col_close].shift(-1)
    df["signal"] = (df["next_close"] > df[col_close]).astype(int)
    df = df.dropna(subset=["next_close"])
    return df


def process_ticker(ticker: str):
    print("=" * 80)
    print(f"Processing {ticker}...")

    ticker_dir = DATA_DIR / f"{ticker} Database"
    ticker_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ticker folder: {ticker_dir}")
    print(f"Ticker folder exists: {ticker_dir.exists()}")

    latest_file = download_nasdaq_csv(ticker, TEMP_DOWNLOAD_DIR)

    current_date = get_current_date_str()
    new_file_path = ticker_dir / f"{ticker}_hist_till{current_date}.csv"

    if new_file_path.exists():
        new_file_path.unlink()

    shutil.move(str(latest_file), str(new_file_path))
    print(f"Downloaded and renamed to: {new_file_path}")
    print(f"Saved download exists: {new_file_path.exists()}")

    df = pd.read_csv(new_file_path)
    df.columns = [c.strip() for c in df.columns]

    if "Close/Last" not in df.columns:
        print(f"Could not find 'Close/Last' column for {ticker}, columns: {df.columns.tolist()}")
        return

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date", ascending=True).reset_index(drop=True)

    numeric_cols = [c for c in ["Open", "High", "Low", "Close/Last", "Volume"] if c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace("[,$]", "", regex=True), errors="coerce")

    df["Close"] = df["Close/Last"]
    close_col = "Close"

    df["daily_return"] = df[close_col].pct_change()
    df["daily_return_pct"] = df["daily_return"] * 100
    df["MA_5"] = df[close_col].rolling(window=5, min_periods=1).mean()
    df["MA_20"] = df[close_col].rolling(window=20, min_periods=1).mean()
    df["MA_50"] = df[close_col].rolling(window=50, min_periods=1).mean()
    df["MA_200"] = df[close_col].rolling(window=200, min_periods=1).mean()
    df["log_ret_1"] = np.log(df[close_col]).diff(1)
    df["roc_5"] = df[close_col].pct_change(5)
    df["roc_20"] = df[close_col].pct_change(20)

    for window in [20, 50, 100]:
        df[f"sma_{window}"] = df[close_col].rolling(window).mean()

    delta = df[close_col].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    df["EMA_12"] = df[close_col].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df[close_col].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]

    df["KAMA"] = kama(df[close_col].values)
    df["momentum_12_1"] = momentum_12_1(df[close_col], skip_days=21, lookback_days=252)

    if set(["High", "Low", "Close"]).issubset(df.columns):
        cci = calc_CCI(df, n=20, col_close="Close")
        df = df.merge(cci[["Date", "TP", "MA_TP", "MD", "CCI"]], on="Date", how="left")

        wr = calc_williams_r(df, n=14, col_close="Close")
        df = df.merge(wr[["Date", "Williams_R"]], on="Date", how="left")

        stoch = calc_stochastic(df, n=9, k_period=3, d_period=3, col_close="Close")
        df = df.merge(stoch[["Date", "K", "D", "J"]], on="Date", how="left")

        atr = calc_ATR(df, n=14, col_close="Close")
        df = df.merge(atr[["Date", "TR", "ATR"]], on="Date", how="left")

    hv = calc_historical_volatility(df, n=20, col_close="Close")
    df = df.merge(hv[["Date", "log_return", "HV_daily", "HV_annual"]], on="Date", how="left")

    sharpe = calc_sharpe_ratio(df, col_close="Close", rf_annual=0.03, window=252)
    df = df.merge(sharpe[["Date", "excess_ret", "Sharpe_Ratio"]], on="Date", how="left")

    cagr_df = calc_cagr(df, col_close="Close", rolling_window=252)
    df = df.merge(cagr_df[["Date", "CAGR_Full", "CAGR_252D"]], on="Date", how="left")

    if set(["Close", "Volume"]).issubset(df.columns):
        obv = calc_OBV(df, col_close="Close", col_volume="Volume")
        df = df.merge(obv[["Date", "OBV"]], on="Date", how="left")

    if set(["High", "Low", "Close", "Volume"]).issubset(df.columns):
        mfi = calc_MFI(df, n=14, col_close="Close", col_volume="Volume")
        df = df.merge(mfi[["Date", "MFI"]], on="Date", how="left")

    prices = df[close_col].astype(float)
    returns = np.log(prices / prices.shift(1))
    df["vol_6m"] = returns.rolling(126).std()

    rolling = df[close_col].rolling(window=20, min_periods=20)
    df["BB_mid"] = rolling.mean()
    df["BB_upper"] = df["BB_mid"] + 2 * rolling.std(ddof=0)
    df["BB_lower"] = df["BB_mid"] - 2 * rolling.std(ddof=0)

    df = create_labels(df, col_close=close_col)
    df = df.sort_values("Date", ascending=True).reset_index(drop=True)
    df["Date"] = df["Date"].dt.strftime("%d/%m/%Y")

    df.to_csv(new_file_path, index=False)
    print(f"Saved indicators into: {new_file_path}")

    master_path = ticker_dir / f"{ticker}_MasterData.csv"
    hist_files = [f for f in ticker_dir.glob("*.csv") if f.name.startswith(f"{ticker}_hist_till")]

    if not hist_files:
        print(f"No hist files for {ticker}")
        return

    def get_date_from_filename(f):
        date_str = f.name.split("_hist_till")[1].split(".")[0]
        return datetime.strptime(date_str, "%Y%m%d")

    hist_files = sorted(hist_files, key=get_date_from_filename)

    df_list = []
    for f in hist_files:
        temp_df = pd.read_csv(f)
        temp_df.columns = [c.strip() for c in temp_df.columns]

        if "Date" not in temp_df.columns:
            print(f"Skipping {f}: no Date column")
            continue

        temp_df["Date"] = pd.to_datetime(temp_df["Date"], format="%d/%m/%Y", errors="coerce").dt.normalize()
        temp_df = temp_df.dropna(subset=["Date"])
        df_list.append(temp_df)

    if not df_list:
        print(f"No valid hist files for {ticker}")
        return

    master_df = pd.concat(df_list, ignore_index=True)
    master_df = master_df.drop_duplicates(subset="Date", keep="last")
    master_df = master_df.sort_values(by="Date", ascending=True).reset_index(drop=True)

    master_df.to_csv(master_path, index=False, date_format="%d/%m/%Y")
    print(f"Saved master file: {master_path}")
    print(
        f"Saved {master_path}: rows={len(master_df)}, "
        f"oldest={master_df['Date'].iloc[0].date()}, "
        f"newest={master_df['Date'].iloc[-1].date()}, "
        f"ascending={master_df['Date'].is_monotonic_increasing}"
    )


def main():
    print("Starting ETF download/update job...")
    for ticker in tickers:
        try:
            process_ticker(ticker)
        except Exception as e:
            print(f"Failed for {ticker}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()