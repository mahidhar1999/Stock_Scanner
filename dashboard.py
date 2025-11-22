import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="RS Screener Dashboard", layout="wide")
st.title("📊 RS Screener — Multi-Timeframe Strength Analyzer")
st.caption("Daily RS = 65 bars | Weekly RS = 13 weeks | Monthly RS = 3 months (TradingView-exact)")


# =====================================================
# LOAD CSV DATA (cached)
# =====================================================
@st.cache_data(ttl=86400)
def load_csv():
    raw = pd.read_csv("all_stocks_raw.csv")
    filt = pd.read_csv("all_stocks_filtered.csv")
    return raw, filt


raw_df, filtered_df = load_csv()

# Convert MarketCap to Crores
raw_df["MarketCapCr"] = raw_df["MarketCap"] / 1e7



# =====================================================
# SIDEBAR FILTERS
# =====================================================
st.sidebar.header("Filter Stocks")

min_rs = st.sidebar.number_input("Min RS (Daily)", value=0.0)
min_mcap = st.sidebar.number_input("Min MarketCap (Cr)", value=0.0)
sector_filter = st.sidebar.selectbox(
    "Sector Filter",
    ["All"] + sorted(raw_df["Sector"].dropna().unique().tolist())
)



# =====================================================
# APPLY FILTER LOGIC
# =====================================================
filtered = raw_df[
    (raw_df["RS"] > min_rs) &
    (raw_df["MarketCapCr"] > min_mcap)
]

if sector_filter != "All":
    filtered = filtered[filtered["Sector"] == sector_filter]

filtered_display = filtered[
    ["Symbol", "Close", "MarketCapCr", "RS", 
     "EMA5", "EMA20", "EMA50", "EMA200", "Sector"]
]

# Add serial numbers
filtered_display = filtered_display.reset_index(drop=True)
filtered_display.index += 1
filtered_display.index.name = "#"



# =====================================================
# DISPLAY — FILTERED STOCKS
# =====================================================
with st.expander("📌 Filtered Stocks (After Conditions)"):
    st.dataframe(filtered_display, use_container_width=True)



# =====================================================
# SECTOR RS RANKING
# =====================================================
st.subheader("📈 Sector Strength Ranking (Market-Cap Weighted RS)")

# Filter only bullish stocks
df_bull = raw_df[raw_df["RS"] > 0].copy()

# Weight = MarketCapCr
df_bull["Weight"] = df_bull["MarketCapCr"]

sector_rs = df_bull.groupby("Sector").apply(
    lambda x: (x["RS"] * x["Weight"]).sum() / x["Weight"].sum()
).reset_index(name="Weighted_RS")

# Sort in descending order
sector_rs = sector_rs.sort_values("Weighted_RS", ascending=False)

# Assign proper sequential numbering
sector_rs = sector_rs.reset_index(drop=True)
sector_rs.index = sector_rs.index + 1
sector_rs.index.name = "#"

st.dataframe(sector_rs, use_container_width=True, hide_index=False)


# =====================================================
# TICKER UTILITY
# =====================================================
def to_yf(ticker):
    return ticker if ticker.endswith(".NS") else ticker + ".NS"



# =====================================================
# RS CALCULATION HELPERS
# =====================================================
def compute_rs(series_stock, series_bench, lookback):
    rs = (series_stock / series_stock.shift(lookback)) / \
         (series_bench / series_bench.shift(lookback)) - 1
    return rs.replace([np.inf, -np.inf], np.nan)


@st.cache_data(ttl=1800)
def batch_download(tickers, period="24mo"):
    if not tickers:
        return pd.DataFrame()
    return yf.download(
        tickers, period=period, interval="1d",
        auto_adjust=False, threads=True, group_by='ticker'
    )



# =====================================================
# COMPUTE MULTI-TIMEFRAME RS
# =====================================================
@st.cache_data(ttl=1800)
def compute_multi_rs(symbols, batch_size=40):

    results = []
    symbols_yf = [to_yf(s) for s in symbols]

    # Download NIFTY
    nifty = yf.download("^NSEI", period="24mo")
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    nifty = nifty.reset_index()[["Date", "Close"]].rename(columns={"Close": "NIFTY"})

    # Process in batches
    for i in range(0, len(symbols_yf), batch_size):

        batch = symbols_yf[i:i+batch_size]
        data = batch_download(batch)

        tickers = sorted({c[0] for c in data.columns})

        for t in tickers:
            try:
                df = data[t].copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.reset_index()[["Date", "Close"]]

                merged = pd.merge_asof(df.sort_values("Date"),
                                       nifty.sort_values("Date"),
                                       on="Date", direction="backward")

                # DAILY RS (65)
                rs_daily = compute_rs(merged["Close"], merged["NIFTY"], 65)
                rs_daily_last = rs_daily.dropna().iloc[-1]

                # WEEKLY RS
                s_w = df.set_index("Date")["Close"].resample("W-FRI").last()
                n_w = nifty.set_index("Date")["NIFTY"].resample("W-FRI").last()
                m_w = pd.merge_asof(s_w.reset_index(), n_w.reset_index(), on="Date")
                rs_week = compute_rs(m_w["Close"], m_w["NIFTY"], 13).dropna().iloc[-1]

                # MONTHLY RS
                s_m = df.set_index("Date")["Close"].resample("M").last()
                n_m = nifty.set_index("Date")["NIFTY"].resample("M").last()
                m_m = pd.merge_asof(s_m.reset_index(), n_m.reset_index(), on="Date")
                rs_month = compute_rs(m_m["Close"], m_m["NIFTY"], 3).dropna().iloc[-1]

                results.append({
                    "Symbol": t.replace(".NS", ""),
                    "RS_Daily": rs_daily_last,
                    "RS_Weekly": rs_week,
                    "RS_Month": rs_month
                })

            except:
                results.append({
                    "Symbol": t.replace(".NS", ""),
                    "RS_Daily": np.nan,
                    "RS_Weekly": np.nan,
                    "RS_Month": np.nan
                })

    return pd.DataFrame(results)



# =====================================================
# COMPUTE BUTTON — MULTI-TF RS
# =====================================================
st.subheader("🧪 Multi-Timeframe RS Scanner")

compute_btn = st.button("🔍 Compute Multi-TF RS")
rs_df = None

if compute_btn:

    symbol_list = filtered_df["Symbol"].tolist()

    with st.spinner("Computing RS…"):
        rs_df = compute_multi_rs(symbol_list)

        # Percentile weights
        for col in ["RS_Daily", "RS_Weekly", "RS_Month"]:
            rs_df[col+"_pct"] = rs_df[col].rank(pct=True)

        rs_df["Score"] = (
            0.5 * rs_df["RS_Month_pct"] +
            0.3 * rs_df["RS_Weekly_pct"] +
            0.2 * rs_df["RS_Daily_pct"]
        ) * 100

        # Merge Close + MarketCapCr from raw_df
        rs_df = rs_df.merge(
            raw_df[["Symbol", "Close", "MarketCapCr"]],
            on="Symbol",
            how="left"
        )

        # Sort by score
        rs_df = rs_df.sort_values("Score", ascending=False)

        # 🔥 Add ranking number
        rs_df = rs_df.reset_index(drop=True)
        rs_df.index = rs_df.index + 1
        rs_df.index.name = "#"

        # Show table
        st.dataframe(
            rs_df[[
                "Symbol", "Close", "MarketCapCr",
                "RS_Daily", "RS_Weekly", "RS_Month",
                "Score"
            ]],
            use_container_width=True
        )

        st.session_state["rs_table"] = rs_df


# Load cached RS table
if "rs_table" in st.session_state and rs_df is None:
    rs_df = st.session_state["rs_table"]
    st.info("Using cached RS table.")

    st.dataframe(
        rs_df[[
            "Symbol", "Close", "MarketCapCr",
            "RS_Daily", "RS_Weekly", "RS_Month",
            "Score"
        ]],
        use_container_width=True
    )


# =====================================================
# SELECTED STOCK SECTION
# =====================================================
st.subheader("📌 Stock Details + Chart")

symbol_list = filtered_df["Symbol"].tolist()

if symbol_list:

    # Sync index
    if "idx" not in st.session_state:
        st.session_state.idx = 0

    stock_symbol = st.selectbox(
        "Choose a stock",
        symbol_list,
        index=st.session_state.idx
    )

    # NAVIGATION BUTTONS
    col_p, col_n = st.columns([1,1])

    with col_p:
        if st.button("⬅ Previous"):
            st.session_state.idx = max(st.session_state.idx - 1, 0)
            st.rerun()

    with col_n:
        if st.button("Next ➡"):
            st.session_state.idx = min(st.session_state.idx + 1, len(symbol_list)-1)
            st.rerun()

    # LOAD PRICE DATA
    @st.cache_data(ttl=1800)
    def load_price(ticker):
        df = yf.download(to_yf(ticker), period="24mo")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.reset_index()

    df_price = load_price(stock_symbol)

    # PRICE CHART
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_price['Date'],
        open=df_price['Open'], high=df_price['High'],
        low=df_price['Low'], close=df_price['Close'],
        increasing_line_color="#26A69A",
        decreasing_line_color="#EF5350"
    ))
    fig.update_layout(height=450, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    # MULTI RS METRICS
    if rs_df is not None:
        row = rs_df[rs_df["Symbol"] == stock_symbol].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", f"{row['Score']:.2f}")
        c2.metric("RS Daily", f"{row['RS_Daily']:.2f}")
        c3.metric("RS Weekly", f"{row['RS_Weekly']:.2f}")
        c4.metric("RS Monthly", f"{row['RS_Month']:.2f}")



# =====================================================
# FUNDAMENTAL SNAPSHOT (for selected stock)
# =====================================================
st.subheader("📘 Fundamental Snapshot")

def format_mcap(v):
    return f"{v/1e7:.2f} Cr" if pd.notna(v) else "N/A"

info = yf.Ticker(to_yf(stock_symbol)).info

c1, c2, c3 = st.columns(3)

c1.metric("Market Cap", format_mcap(info.get("marketCap")))
c1.metric("P/E Ratio", info.get("trailingPE", "N/A"))

c2.metric("52W High", info.get("fiftyTwoWeekHigh", "N/A"))
c2.metric("52W Low", info.get("fiftyTwoWeekLow", "N/A"))

c3.metric("ROE %", f"{info.get('returnOnEquity',0)*100:.2f}%")
c3.metric("Profit Margin %", f"{info.get('profitMargins',0)*100:.2f}%")

st.write("**Sector:**", info.get("sector", "N/A"))
st.write("**Industry:**", info.get("industry", "N/A"))
st.write("**Website:**", info.get("website", "N/A"))
