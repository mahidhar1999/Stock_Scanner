import pandas as pd
import yfinance as yf
import numpy as np
import requests
import time

# ---------------------------
# LOAD SYMBOL LIST
# ---------------------------
print("Loading symbols...")
df = pd.read_csv("stocks.csv")
symbols = (df["SYMBOL"] + ".NS").tolist()
total = len(symbols)
print("Total symbols:", total)

# ---------------------------
# NSE SESSION (FAST)
# ---------------------------
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.nseindia.com/"
})

def get_fundamentals_fast(symbol):
    """
    Extremely fast fundamentals:
    - Sector from NSE API
    - Shares Outstanding → compute MarketCap manually
    """
    short = symbol.replace(".NS", "")

    try:
        url = f"https://www.nseindia.com/api/quote-equity?symbol={short}"
        data = session.get(url).json()

        sector = data["industryInfo"].get("industry")
        shares = data["securityInfo"].get("issuedSize")  # shares outstanding

        return sector, shares

    except:
        return None, None

# ---------------------------
# NIFTY INDEX
# ---------------------------
print("Downloading Nifty...")
nifty = yf.download("^NSEI", period="300d", auto_adjust=True)
nifty_close = nifty[[col for col in nifty.columns if "Close" in col][0]]

# ---------------------------
# PROCESS BATCHES
# ---------------------------
results = []
batch_size = 300

def process_batch(batch):
    print(f"\nDownloading price batch ({len(batch)})...")

    data = yf.download(
        batch,
        period="300d",
        group_by="ticker",
        auto_adjust=True,
        threads=True
    )

    for sym in batch:
        print("Processing:", sym)

        try:
            stock = data[sym].copy()
            if stock.empty:
                print("⚠ No price data:", sym)
                continue

            close = stock[[c for c in stock.columns if "Close" in c][0]]
            stock["Close"] = close

            # EMAs
            stock["EMA5"] = close.ewm(span=5).mean()
            stock["EMA20"] = close.ewm(span=20).mean()
            stock["EMA50"] = close.ewm(span=50).mean()
            stock["EMA200"] = close.ewm(span=200).mean()

            # RS Calculation
            nifty_aligned = nifty_close.asof(stock.index)
            arr = close.to_numpy()
            nifty_arr = nifty_aligned.to_numpy()

            shift = 65
            arr_shift = np.concatenate([np.full(shift, np.nan), arr[:-shift]])
            nifty_shift = np.concatenate([np.full(shift, np.nan), nifty_arr[:-shift]])

            rs = (arr / arr_shift) / (nifty_arr / nifty_shift) - 1
            stock["RS"] = rs

            last = stock.iloc[-1]

            # FAST FUNDAMENTALS
            mcap, sector = get_fundamentals_fast(sym)

            sector, shares = get_fundamentals_fast(sym)
            if shares:
                mcap = last["Close"] * shares
            else:
                mcap = None


            results.append({
                "Symbol": sym.replace(".NS", ""),
                "Close": last["Close"],
                "RS": last["RS"],
                "EMA5": last["EMA5"],
                "EMA20": last["EMA20"],
                "EMA50": last["EMA50"],
                "EMA200": last["EMA200"],
                "MarketCap": mcap,
                "Sector": sector
            })

        except Exception as e:
            print("Error processing", sym, "→", e)


# ---------------------------
# RUN ALL BATCHES
# ---------------------------
for i in range(0, total, batch_size):
    batch = symbols[i:i + batch_size]
    process_batch(batch)
    time.sleep(1)

# ---------------------------
# SAVE CSVs
# ---------------------------
df_all = pd.DataFrame(results)
df_all.to_csv("all_stocks_raw.csv", index=False)

df_filtered = df_all[
    (df_all["RS"] > 0) &
    (df_all["EMA5"] > df_all["EMA20"]) &
    (df_all["EMA20"] > df_all["EMA50"]) &
    (df_all["EMA50"] > df_all["EMA200"])
]

df_filtered.to_csv("all_stocks_filtered.csv", index=False)

print("\nSaved all CSVs successfully!")
