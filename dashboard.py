import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import math
from time import sleep

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(page_title="RS Screener Dashboard (Multi-Timeframe)", layout="wide")
st.title("📊 RS Screener — Multi-Timeframe (Daily / Weekly / Monthly)")
st.caption("Daily RS = 65 bars | Weekly RS = 13 weeks | Monthly RS = 3 months (TradingView-exact alignment)")

# -----------------------------------------------------
# LOAD CSVs (cached)
# -----------------------------------------------------
@st.cache_data(ttl=86400)
def load_csv_data():
    raw = pd.read_csv("all_stocks_raw.csv")
    filt = pd.read_csv("all_stocks_filtered.csv")
    return raw, filt

raw_df, filtered_df = load_csv_data()

# Sidebar filters
st.sidebar.header("Filters")
min_price = st.sidebar.number_input("Minimum Price", value=50)
min_rs = st.sidebar.number_input("Minimum RS (daily)", value=0.0)

filtered = raw_df[
    (raw_df["Close"] > min_price) &
    (raw_df["RS"] > min_rs)
]

with st.expander("Filtered Stocks (After Conditions)"):
    st.dataframe(filtered, use_container_width=True)

with st.expander("Strong Trend Stocks (5 > 20 > 50 > 200)"):
    st.dataframe(filtered_df, use_container_width=True)

# -----------------------------------------------------
# Utility: safe ticker -> yahoo format
# -----------------------------------------------------
def to_yf(ticker):
    if ticker.endswith(".NS"):
        return ticker
    return ticker + ".NS"

# -----------------------------------------------------
# Batch download helper (cached)
# -----------------------------------------------------
@st.cache_data(ttl=1800)
def batch_download(tickers, period="24mo"):
    """
    Download multiple tickers in one batch. Returns DataFrame (multiindex) as returned by yfinance.
    """
    # ensure ticker list non-empty
    if not tickers:
        return pd.DataFrame()
    # call yfinance
    data = yf.download(tickers, period=period, interval="1d", auto_adjust=False, threads=True, group_by='ticker')
    return data

# -----------------------------------------------------
# Compute RS for one pair of series (array-like), given lookback in bars
# -----------------------------------------------------
def compute_rs_from_series(stock_close, bench_close, lookback):
    # stock_close and bench_close must be aligned same index length
    rs = (stock_close / stock_close.shift(lookback)) / (bench_close / bench_close.shift(lookback)) - 1
    rs = rs.replace([np.inf, -np.inf], np.nan)
    return rs

# -----------------------------------------------------
# Compute multi-timeframe RS across symbol_list (batched)
# -----------------------------------------------------
@st.cache_data(ttl=1800)
def compute_multi_rs(symbol_list, batch_size=40):
    """
    Returns DataFrame with columns:
    Symbol, RS_Daily (latest), RS_Weekly (latest), RS_Month (latest)
    """
    results = []
    symbols_yf = [to_yf(s) for s in symbol_list]

    # download NIFTY raw (same period)
    nifty = yf.download("^NSEI", period="24mo", interval="1d", auto_adjust=False)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    nifty = nifty.reset_index()[["Date", "Close"]].rename(columns={"Close":"NIFTY"})

    # process in batches
    for i in range(0, len(symbols_yf), batch_size):
        batch = symbols_yf[i:i+batch_size]
        data = batch_download(batch, period="24mo")
        # if single ticker, yfinance returns DataFrame with columns
        if not isinstance(data.columns, pd.MultiIndex):
            # single ticker DataFrame
            # find which ticker
            if len(batch) == 1:
                t = batch[0]
                df = data.copy().reset_index()[["Date", "Close"]].rename(columns={"Close":"Close"})
                # align asof
                merged = pd.merge_asof(df.sort_values("Date"), nifty.sort_values("Date"), on="Date", direction="backward")
                merged["NIFTY"] = merged["NIFTY"].fillna(method="ffill")
                # daily RS (65)
                rs_daily_series = compute_rs_from_series(merged["Close"], merged["NIFTY"], 65)
                rs_daily = rs_daily_series.dropna().iloc[-1] if not rs_daily_series.dropna().empty else np.nan
                # weekly / monthly resample
                s_close = df.set_index("Date")["Close"]
                s_week = s_close.resample('W-FRI').last().dropna()
                n_week = nifty.set_index("Date")["NIFTY"].resample('W-FRI').last().dropna()
                if len(s_week) > 20 and len(n_week) > 20:
                    # align weekly
                    merged_week = pd.merge_asof(s_week.reset_index().sort_values("Date"), n_week.reset_index().sort_values("Date"), on="Date", direction="backward")
                    rs_w = compute_rs_from_series(merged_week["Close"], merged_week["NIFTY"], 13)
                    rs_week = rs_w.dropna().iloc[-1] if not rs_w.dropna().empty else np.nan
                else:
                    rs_week = np.nan
                s_month = s_close.resample('M').last().dropna()
                n_month = nifty.set_index("Date")["NIFTY"].resample('M').last().dropna()
                if len(s_month) > 6 and len(n_month) > 6:
                    merged_month = pd.merge_asof(s_month.reset_index().sort_values("Date"), n_month.reset_index().sort_values("Date"), on="Date", direction="backward")
                    rs_m = compute_rs_from_series(merged_month["Close"], merged_month["NIFTY"], 3)
                    rs_month = rs_m.dropna().iloc[-1] if not rs_m.dropna().empty else np.nan
                else:
                    rs_month = np.nan
                results.append({
                    "Symbol": t.replace(".NS",""),
                    "RS_Daily": rs_daily,
                    "RS_Weekly": rs_week,
                    "RS_Month": rs_month
                })
            continue

        # multi-ticker batch
        # data.columns is MultiIndex (ticker, Price)
        tickers_in_data = sorted({c[0] for c in data.columns})
        for t in tickers_in_data:
            try:
                df = data[t].copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.reset_index()[["Date","Close"]]
                # align with nifty by merge_asof
                merged = pd.merge_asof(df.sort_values("Date"), nifty.sort_values("Date"), on="Date", direction="backward")
                merged["NIFTY"] = merged["NIFTY"].fillna(method="ffill")
                # daily RS
                rs_daily_series = compute_rs_from_series(merged["Close"], merged["NIFTY"], 65)
                rs_daily = rs_daily_series.dropna().iloc[-1] if not rs_daily_series.dropna().empty else np.nan

                # weekly RS
                s_close = df.set_index("Date")["Close"]
                s_week = s_close.resample('W-FRI').last().dropna()
                n_week = nifty.set_index("Date")["NIFTY"].resample('W-FRI').last().dropna()
                if len(s_week) > 20 and len(n_week) > 20:
                    merged_week = pd.merge_asof(s_week.reset_index().sort_values("Date"), n_week.reset_index().sort_values("Date"), on="Date", direction="backward")
                    rs_w = compute_rs_from_series(merged_week["Close"], merged_week["NIFTY"], 13)
                    rs_week = rs_w.dropna().iloc[-1] if not rs_w.dropna().empty else np.nan
                else:
                    rs_week = np.nan

                # monthly RS
                s_month = s_close.resample('M').last().dropna()
                n_month = nifty.set_index("Date")["NIFTY"].resample('M').last().dropna()
                if len(s_month) > 6 and len(n_month) > 6:
                    merged_month = pd.merge_asof(s_month.reset_index().sort_values("Date"), n_month.reset_index().sort_values("Date"), on="Date", direction="backward")
                    rs_m = compute_rs_from_series(merged_month["Close"], merged_month["NIFTY"], 3)
                    rs_month = rs_m.dropna().iloc[-1] if not rs_m.dropna().empty else np.nan
                else:
                    rs_month = np.nan

                results.append({
                    "Symbol": t.replace(".NS",""),
                    "RS_Daily": rs_daily,
                    "RS_Weekly": rs_week,
                    "RS_Month": rs_month
                })
            except Exception as e:
                # if one ticker fails, continue
                results.append({
                    "Symbol": t.replace(".NS",""),
                    "RS_Daily": np.nan,
                    "RS_Weekly": np.nan,
                    "RS_Month": np.nan
                })

    return pd.DataFrame(results)

# -----------------------------------------------------
# UI: compute button and progress
# -----------------------------------------------------
st.subheader("Multi-Timeframe RS Scanner")
colA, colB = st.columns([3,1])
with colA:
    st.write("Compute Daily/Weekly/Monthly RS for filtered symbols and create a composite Trend Score.")
with colB:
    compute_btn = st.button("Compute Multi-TF RS")

rs_df = None
if compute_btn:
    symbol_list = filtered_df["Symbol"].tolist()
    if not symbol_list:
        st.warning("No symbols in filtered list.")
    else:
        with st.spinner("Downloading & computing RS (this may take a short while)..."):
            # compute multi RS (cached)
            rs_df = compute_multi_rs(symbol_list, batch_size=40)

            # compute percentiles for each RS (we use rank pct)
            for col in ["RS_Daily","RS_Weekly","RS_Month"]:
                rs_df[col + "_pct"] = rs_df[col].rank(method="average", pct=True).fillna(0)

            # composite score: weights 0.5 month, 0.3 week, 0.2 daily
            rs_df["Score"] = (0.5 * rs_df["RS_Month_pct"] + 0.3 * rs_df["RS_Weekly_pct"] + 0.2 * rs_df["RS_Daily_pct"]) * 100
            rs_df["Score"] = rs_df["Score"].round(2)

            # sort by Score desc
            rs_df.sort_values("Score", ascending=False, inplace=True)
            rs_df.reset_index(drop=True, inplace=True)

            st.success("Multi-TF RS computed.")
            st.dataframe(rs_df[["Symbol","RS_Daily","RS_Weekly","RS_Month","Score"]].head(200), use_container_width=True)

            # cache latest table to session for chart viewer
            st.session_state["multi_rs_table"] = rs_df

# If table already in session, allow quick access
if "multi_rs_table" in st.session_state and rs_df is None:
    rs_df = st.session_state["multi_rs_table"]
    st.info("Using previously computed Multi-TF RS (cached in session).")

# -----------------------------------------------------
# Selected Stock Details (Multi-TF RS) WITH NAVIGATION
# -----------------------------------------------------
st.subheader("📈 Selected Stock Details (Multi-TF RS)")

symbol_list_chart = filtered_df["Symbol"].tolist()

if not symbol_list_chart:
    st.warning("Filtered symbol list empty.")
else:

    # Maintain index in session state
    if "stock_index" not in st.session_state:
        st.session_state.stock_index = 0

    # ---- DROPDOWN (no key, avoids overwriting session state) ----
    stock_symbol = st.selectbox(
        "Pick a stock for details:",
        symbol_list_chart,
        index=st.session_state.stock_index
    )

    # ---- NAVIGATION BUTTONS ----
    col_prev, col_empty, col_next = st.columns([1, 6, 1])

    with col_prev:
        if st.button("⬅ Previous"):
            st.session_state.stock_index = max(0, st.session_state.stock_index - 1)
            st.rerun()

    with col_next:
        if st.button("Next ➡"):
            st.session_state.stock_index = min(len(symbol_list_chart) - 1, st.session_state.stock_index + 1)
            st.rerun()

    # ---- SYNC dropdown change ----
    synced_symbol = symbol_list_chart[st.session_state.stock_index]
    if stock_symbol != synced_symbol:
        st.session_state.stock_index = symbol_list_chart.index(stock_symbol)

    # -----------------------------------------------------
    # Load stock price history
    # -----------------------------------------------------
    @st.cache_data(ttl=1800)
    def load_stock_raw(ticker):
        d = yf.download(ticker, period="24mo", interval="1d", auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        return d.reset_index()

    ticker_full = to_yf(stock_symbol)
    stock_df = load_stock_raw(ticker_full)

    # -----------------------------------------------------
    # Price Chart
    # -----------------------------------------------------
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=stock_df['Date'],
        open=stock_df['Open'], high=stock_df['High'],
        low=stock_df['Low'], close=stock_df['Close'],
        increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ))

    fig.update_layout(
        title=f"{stock_symbol} — Price (24 Months)",
        height=450,
        xaxis_rangeslider_visible=True
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # RS Values Display
    # -----------------------------------------------------
    if rs_df is not None:
        row = rs_df[rs_df["Symbol"] == stock_symbol]

        if not row.empty:
            r = row.iloc[0]

            st.metric("Composite Score", f"{r['Score']:.2f}")

            c1, c2, c3 = st.columns(3)
            c1.metric("RS Daily (65)", f"{r['RS_Daily']:.2f}" if not pd.isna(r["RS_Daily"]) else "N/A")
            c2.metric("RS Weekly (13)", f"{r['RS_Weekly']:.2f}" if not pd.isna(r["RS_Weekly"]) else "N/A")
            c3.metric("RS Monthly (3)", f"{r['RS_Month']:.2f}" if not pd.isna(r["RS_Month"]) else "N/A")

        else:
            st.info("RS not computed for this stock yet.")
    else:
        st.info("Compute Multi-TF RS above to view RS details.")
# -----------------------------------------------------
# FUNDAMENTAL DATA (YFinance)
# -----------------------------------------------------
st.subheader("📘 Fundamental Snapshot")

# ---------- Formatting helper ----------
def format_market_cap(value):
    if value is None or value != value:
        return "N/A"

    # International format
    if value >= 1_000_000_000_000:
        intl = f"{value/1_000_000_000_000:.2f}T"
    elif value >= 1_000_000_000:
        intl = f"{value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        intl = f"{value/1_000_000:.2f}M"
    else:
        intl = f"{value:,}"

    # Indian Format (Crores)
    cr = value / 10_000_000
    if cr >= 100_000:
        indi = f"{cr/100_000:.2f} Lakh Cr"
    else:
        indi = f"{cr:.2f} Cr"

    return f"{intl}  ({indi})"


@st.cache_data(ttl=86400)
def load_fundamentals(ticker):
    """Fetch and clean yfinance fundamentals."""
    try:
        return yf.Ticker(ticker).info
    except:
        return {}


info = load_fundamentals(ticker_full)

if not info:
    st.warning("No fundamentals available for this stock.")
else:
    colA, colB, colC = st.columns(3)

    # -------------------------------------------------
    # COLUMN A — VALUATION
    # -------------------------------------------------
    mc = info.get("marketCap")
    colA.metric("Market Cap", format_market_cap(mc))

    pe = info.get("trailingPE")
    colA.metric("P/E Ratio (TTM)", f"{pe:.2f}" if pe else "N/A")

    ps = info.get("priceToSalesTrailing12Months")
    colA.metric("Price-to-Sales", f"{ps:.2f}" if ps else "N/A")

    # -------------------------------------------------
    # COLUMN B — PRICE & EARNINGS
    # -------------------------------------------------
    high_52 = info.get("fiftyTwoWeekHigh")
    low_52 = info.get("fiftyTwoWeekLow")
    eps = info.get("trailingEps")

    colB.metric("52 Week High", f"{high_52:.2f}" if high_52 else "N/A")
    colB.metric("52 Week Low", f"{low_52:.2f}" if low_52 else "N/A")
    colB.metric("EPS (TTM)", f"{eps:.2f}" if eps else "N/A")

    # -------------------------------------------------
    # COLUMN C — PROFITABILITY & GROWTH
    # -------------------------------------------------
    roe = info.get("returnOnEquity")
    pm = info.get("profitMargins")
    rev = info.get("revenueGrowth")

    colC.metric("ROE %", f"{roe*100:.2f}%" if roe else "N/A")
    colC.metric("Profit Margin %", f"{pm*100:.2f}%" if pm else "N/A")
    colC.metric("Revenue Growth %", f"{rev*100:.2f}%" if rev else "N/A")

    # -------------------------------------------------
    # COMPANY PROFILE
    # -------------------------------------------------
    st.write("### 📂 Company Profile")
    st.write(f"**Sector:** {info.get('sector', 'N/A')}")
    st.write(f"**Industry:** {info.get('industry', 'N/A')}")
    st.write(f"**Website:** {info.get('website', 'N/A')}")
