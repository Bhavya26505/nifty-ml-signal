import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
from google import genai
import warnings
warnings.filterwarnings('ignore')

print("=" * 65)
print("  NIFTY 50 ML PREDICTION — FULL GLOBAL + MACRO MODEL")
print("=" * 65)

# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAD COMPLETE DATASET
# ══════════════════════════════════════════════════════════════
print("\n[1/7] Loading complete dataset...")

df_base = pd.read_excel("NIFTY_Complete_Dataset.xlsx")
df_base.columns = [c.strip() for c in df_base.columns]
df_base['Date'] = pd.to_datetime(df_base['Date'])
df_base = df_base.sort_values('Date').reset_index(drop=True)
print(f"    Base dataset : {len(df_base)} rows | {df_base.columns.tolist()[:7]}")

# ══════════════════════════════════════════════════════════════
# STEP 2 — FETCH FRESH LIVE DATA FROM YFINANCE
# ══════════════════════════════════════════════════════════════
print("\n[2/7] Fetching fresh live data...")

def fetch_yf(symbol, period="400d", col_name=None):
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False)
        df.columns = df.columns.get_level_values(0)
        df = df[['Open','High','Low','Close','Volume']].dropna()
        df.index = pd.to_datetime(df.index)
        if col_name:
            df = df[['Close']].rename(columns={'Close': col_name})
        return df
    except Exception as e:
        print(f"    Warning: could not fetch {symbol}: {e}")
        return pd.DataFrame()

# NIFTY
nifty_live = fetch_yf("^NSEI", "400d")
# Global indices
sp500_live  = fetch_yf("^GSPC",   "400d", "SP500_Close_live")
nasdaq_live = fetch_yf("^NDX",    "400d", "Nasdaq_Close_live")
crude_live  = fetch_yf("CL=F",    "400d", "Crude_Close_live")
usdinr_live = fetch_yf("USDINR=X","400d", "USDINR_Close_live")
vix_live    = fetch_yf("^VIX",    "400d", "VIX_global_live")

# Merge fresh data on top of base dataset
if len(nifty_live) > 0:
    nifty_live_df = nifty_live.reset_index().rename(columns={'index':'Date','Date':'Date'})
    nifty_live_df.columns = ['Date'] + [c for c in nifty_live_df.columns if c != 'Date']

    # Rename NIFTY columns to match base
    nifty_live_df = nifty_live.reset_index()
    nifty_live_df.columns = ['Date','Open','High','Low','Close','Volume']
    nifty_live_df['Shares Traded'] = nifty_live_df['Volume']
    nifty_live_df['Turnover (₹ Cr)'] = 0

    # Add global data to live rows
    for live_df, col in [
        (sp500_live,  'SP500_Close'),
        (nasdaq_live, 'Nasdaq_Close'),
        (crude_live,  'Crude_Close'),
        (usdinr_live, 'USDINR_Close'),
    ]:
        if len(live_df) > 0:
            live_df2 = live_df.reset_index()
            live_df2.columns = ['Date', col]
            nifty_live_df = nifty_live_df.merge(live_df2, on='Date', how='left')
            nifty_live_df[col] = nifty_live_df[col].ffill()

    # Add VIX (use India proxy columns)
    if len(vix_live) > 0:
        vix_df = vix_live.reset_index()
        vix_df.columns = ['Date','VIX_Close']
        nifty_live_df = nifty_live_df.merge(vix_df, on='Date', how='left')
        nifty_live_df['VIX_Close'] = nifty_live_df['VIX_Close'].ffill()

    # Combine base + live, remove duplicates (keep live version)
    df_combined = pd.concat([df_base, nifty_live_df], sort=False)
    df_combined = df_combined.drop_duplicates('Date', keep='last')
    df_combined = df_combined.sort_values('Date').reset_index(drop=True)

    # Forward fill macro annual columns for new rows
    macro_cols = [c for c in df_base.columns if c.startswith('macro_')]
    for col in macro_cols:
        if col in df_combined.columns:
            df_combined[col] = df_combined[col].ffill()

    print(f"    Combined     : {len(df_combined)} rows (base {len(df_base)} + {len(df_combined)-len(df_base)} new live rows)")
else:
    df_combined = df_base.copy()
    print("    Using base dataset only (yfinance unavailable)")

# Current price info
close_price      = float(df_combined['Close'].iloc[-1])
open_price       = float(df_combined['Open'].iloc[-1])
high_price       = float(df_combined['High'].iloc[-1])
low_price        = float(df_combined['Low'].iloc[-1])
prev_close       = float(df_combined['Close'].iloc[-2])
daily_change     = round(close_price - prev_close, 2)
daily_change_pct = round((daily_change / prev_close) * 100, 2)
week_52_high     = float(df_combined['Close'].tail(252).max())
week_52_low      = float(df_combined['Close'].tail(252).min())

print(f"    Latest close : ₹{close_price:,.2f} ({daily_change_pct:+.2f}%)")
print(f"    Latest date  : {df_combined['Date'].iloc[-1].date()}")

# ══════════════════════════════════════════════════════════════
# STEP 3 — ENGINEER ALL FEATURES
# ══════════════════════════════════════════════════════════════
print("\n[3/7] Engineering features...")

df = df_combined.copy()

# ── NIFTY technical features ─────────────────────────────────
df['Daily_Return']    = df['Close'].pct_change() * 100
df['Log_Return']      = np.log(df['Close'] / df['Close'].shift(1)) * 100
df['HL_Range']        = (df['High'] - df['Low']) / df['Close'] * 100
df['OC_Range']        = (df['Close'] - df['Open']) / df['Open'] * 100
df['Gap_Open']        = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) * 100

for w in [5, 10, 20, 50, 200]:
    df[f'MA_{w}'] = df['Close'].rolling(w).mean()

df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

# RSI
delta = df['Close'].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df['RSI_14'] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))

# MACD
df['MACD']        = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

# Bollinger Bands
ma20  = df['Close'].rolling(20).mean()
std20 = df['Close'].rolling(20).std()
df['BB_Upper']  = ma20 + 2 * std20
df['BB_Lower']  = ma20 - 2 * std20
df['BB_PctB']   = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
df['BB_Width']  = (df['BB_Upper'] - df['BB_Lower']) / ma20 * 100

# Volatility & momentum
df['Volatility_10']  = df['Daily_Return'].rolling(10).std()
df['Volatility_20']  = df['Daily_Return'].rolling(20).std()
df['Volatility_60']  = df['Daily_Return'].rolling(60).std()
df['ATR']            = (df['High'] - df['Low']).rolling(14).mean()
df['Volume_MA20']    = df['Close'].rolling(20).mean()
df['Volume_Ratio']   = df['Daily_Return'].rolling(5).std() / (df['Volatility_20'] + 1e-10)
df['Mom_5']          = df['Close'] / df['Close'].shift(5)  - 1
df['Mom_10']         = df['Close'] / df['Close'].shift(10) - 1
df['Mom_20']         = df['Close'] / df['Close'].shift(20) - 1
df['Mom_60']         = df['Close'] / df['Close'].shift(60) - 1

# Return lags
for lag in [1, 2, 3, 5, 10, 20]:
    df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)

# Market regime
df['Bull_Regime']    = (df['MA_50'] > df['MA_200']).astype(int)
df['Above_MA20']     = (df['Close'] > df['MA_20']).astype(int)
df['Above_MA50']     = (df['Close'] > df['MA_50']).astype(int)
df['Pct_From_52W_H'] = (df['Close'] - df['Close'].rolling(252).max()) / df['Close'].rolling(252).max() * 100
df['Pct_From_52W_L'] = (df['Close'] - df['Close'].rolling(252).min()) / df['Close'].rolling(252).min() * 100

# Calendar
df['Month']      = df['Date'].dt.month
df['DayOfWeek']  = df['Date'].dt.dayofweek
df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
df['IsMonthEnd'] = (df['Date'].dt.is_month_end).astype(int)

# ── India VIX features ───────────────────────────────────────
if 'VIX_Close' in df.columns:
    df['VIX_Level']      = df['VIX_Close']
    df['VIX_Change']     = df['VIX_Close'].pct_change() * 100
    df['VIX_MA10']       = df['VIX_Close'].rolling(10).mean()
    df['VIX_Regime']     = (df['VIX_Close'] > 20).astype(int)
    df['VIX_Spike']      = (df['VIX_Change'] > 15).astype(int)
    df['VIX_Relief']     = (df['VIX_Change'] < -10).astype(int)

# ── Global market features (PREVIOUS day — no lookahead) ─────
# SP500
if 'SP500_Close' in df.columns:
    df['SP500_Return_1d']  = df['SP500_Close'].pct_change().shift(1) * 100
    df['SP500_Return_5d']  = df['SP500_Close'].pct_change(5).shift(1) * 100
    df['SP500_MA20']       = df['SP500_Close'].rolling(20).mean().shift(1)
    df['SP500_Regime']     = (df['SP500_Close'].shift(1) > df['SP500_MA20']).astype(int)

# NASDAQ
if 'Nasdaq_Close' in df.columns:
    df['Nasdaq_Return_1d'] = df['Nasdaq_Close'].pct_change().shift(1) * 100
    df['Nasdaq_Return_5d'] = df['Nasdaq_Close'].pct_change(5).shift(1) * 100

# Crude Oil
if 'Crude_Close' in df.columns:
    df['Crude_Return_1d']  = df['Crude_Close'].pct_change().shift(1) * 100
    df['Crude_Return_5d']  = df['Crude_Close'].pct_change(5).shift(1) * 100
    df['Crude_MA20']       = df['Crude_Close'].rolling(20).mean().shift(1)
    df['Crude_Regime']     = (df['Crude_Close'].shift(1) > df['Crude_MA20']).astype(int)

# USD/INR
if 'USDINR_Close' in df.columns:
    df['USDINR_Return_1d'] = df['USDINR_Close'].pct_change().shift(1) * 100
    df['USDINR_MA20']      = df['USDINR_Close'].rolling(20).mean().shift(1)
    df['INR_Weakening']    = (df['USDINR_Close'].shift(1) > df['USDINR_MA20']).astype(int)

# Rolling correlation NIFTY vs SP500 (20-day)
if 'SP500_Close' in df.columns:
    nifty_ret  = df['Close'].pct_change()
    sp500_ret  = df['SP500_Close'].pct_change()
    df['Corr_NIFTY_SP500_20'] = nifty_ret.rolling(20).corr(sp500_ret)

# ── Annual macro features ─────────────────────────────────────
macro_cols = [c for c in df.columns if c.startswith('macro_')]
print(f"    Macro cols   : {len(macro_cols)} — {macro_cols}")

# Fill any remaining macro nulls
for col in macro_cols:
    df[col] = df[col].ffill().bfill()

# Clean dataset
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['Close', 'RSI_14', 'MACD_Hist', 'BB_PctB'])

print(f"    After feature engineering: {len(df)} rows")

# ══════════════════════════════════════════════════════════════
# STEP 4 — DEFINE FEATURE COLUMNS
# ══════════════════════════════════════════════════════════════

# Technical features
tech_features = [
    # Price action
    'Log_Return', 'HL_Range', 'OC_Range', 'Gap_Open',
    # Momentum
    'RSI_14', 'BB_PctB', 'BB_Width', 'MACD_Hist',
    'Mom_5', 'Mom_10', 'Mom_20', 'Mom_60',
    # Volatility
    'Volatility_10', 'Volatility_20', 'Volatility_60', 'ATR',
    # Return lags
    'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3',
    'Return_Lag_5', 'Return_Lag_10',
    # Regime
    'Bull_Regime', 'Above_MA20', 'Above_MA50',
    'Pct_From_52W_H', 'Pct_From_52W_L',
    # Calendar
    'Month', 'DayOfWeek', 'IsMonthEnd',
]

# VIX features
vix_features = []
for col in ['VIX_Level','VIX_Change','VIX_Regime','VIX_Spike','VIX_Relief']:
    if col in df.columns and df[col].notna().sum() > 1000:
        vix_features.append(col)

# Global features
global_features = []
for col in ['SP500_Return_1d','SP500_Return_5d','SP500_Regime',
            'Nasdaq_Return_1d','Nasdaq_Return_5d',
            'Crude_Return_1d','Crude_Return_5d','Crude_Regime',
            'USDINR_Return_1d','INR_Weakening',
            'Corr_NIFTY_SP500_20']:
    if col in df.columns and df[col].notna().sum() > 1000:
        global_features.append(col)

all_features = tech_features + vix_features + global_features + macro_cols

# Remove features with too many nulls
valid_features = []
for f in all_features:
    if f in df.columns:
        null_pct = df[f].isnull().mean()
        if null_pct < 0.3:
            valid_features.append(f)
        else:
            print(f"    Dropped {f} ({null_pct:.1%} null)")

print(f"    Tech features   : {len(tech_features)}")
print(f"    VIX features    : {len(vix_features)}")
print(f"    Global features : {len(global_features)}")
print(f"    Macro features  : {len(macro_cols)}")
print(f"    Total valid     : {len(valid_features)}")

# ══════════════════════════════════════════════════════════════
# STEP 5 — TRAIN MODELS
# ══════════════════════════════════════════════════════════════
print("\n[5/7] Training models...")

# Targets
df['Target_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df['Target_Close']     = df['Close'].shift(-1)
df['Target_High']      = df['High'].shift(-1)
df['Target_Low']       = df['Low'].shift(-1)

df_model = df.dropna(subset=['Target_Direction','Target_Close'] + valid_features)

X       = df_model[valid_features].values[:-1]
y_dir   = df_model['Target_Direction'].values[:-1]
y_close = df_model['Target_Close'].values[:-1]
y_high  = df_model['Target_High'].values[:-1]
y_low   = df_model['Target_Low'].values[:-1]

# Handle any remaining nans/infs in X
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

sc = StandardScaler()
X_sc = sc.fit_transform(X)

# ── Direction classifier ──────────────────────────────────────
clf = xgb.XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.75,
    min_child_weight=3, gamma=0.1,
    eval_metric='logloss', random_state=42, verbosity=0
)
clf.fit(X_sc, y_dir)

# Walk-forward accuracy check — last 252 days
split = max(len(X) - 252, int(len(X) * 0.85))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y_dir[:split], y_dir[split:]
sc_check = StandardScaler()
clf_check = xgb.XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.75,
    eval_metric='logloss', random_state=42, verbosity=0
)
clf_check.fit(sc_check.fit_transform(X_tr), y_tr)
recent_acc = accuracy_score(y_te, clf_check.predict(sc_check.transform(X_te)))
print(f"    Direction model  — recent 1Y accuracy: {recent_acc*100:.2f}%")

# ── Price regressors ─────────────────────────────────────────
reg_close = xgb.XGBRegressor(
    n_estimators=400, max_depth=5, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.75,
    random_state=42, verbosity=0
)
reg_high = xgb.XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.75,
    random_state=42, verbosity=0
)
reg_low = xgb.XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.75,
    random_state=42, verbosity=0
)
reg_close.fit(X_sc, y_close)
reg_high.fit(X_sc, y_high)
reg_low.fit(X_sc, y_low)
print(f"    All 4 models trained on {len(X):,} samples | {len(valid_features)} features")

# ── Feature importance (top 10) ──────────────────────────────
importances = clf.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]
print(f"    Top 10 features by importance:")
for i in top_idx:
    print(f"      {valid_features[i]:<35} {importances[i]:.4f}")

# ══════════════════════════════════════════════════════════════
# STEP 6 — PREDICT TOMORROW
# ══════════════════════════════════════════════════════════════
print("\n[6/7] Generating tomorrow's prediction...")

latest_row = df[valid_features].iloc[[-1]]
latest_X   = np.nan_to_num(latest_row.values, nan=0.0, posinf=0.0, neginf=0.0)
latest_sc  = sc.transform(latest_X)

prob_up    = float(clf.predict_proba(latest_sc)[0][1])
prob_down  = round(1 - prob_up, 4)
prob_up    = round(prob_up, 4)
pred_close = round(float(reg_close.predict(latest_sc)[0]), 2)
pred_high  = round(float(reg_high.predict(latest_sc)[0]), 2)
pred_low   = round(float(reg_low.predict(latest_sc)[0]), 2)
pred_change     = round(pred_close - close_price, 2)
pred_change_pct = round((pred_change / close_price) * 100, 2)

# Signal with confidence bands
if prob_up > 0.60:
    signal = "LONG";  action = "BUY / HOLD — Nifty ETF (NIFTYBEES)"; hist_acc = 60.3; confidence = "STRONG"
elif prob_up < 0.40:
    signal = "SHORT"; action = "SHORT Nifty Futures / Exit longs";    hist_acc = 55.4; confidence = "STRONG"
elif prob_up > 0.55:
    signal = "LONG";  action = "MILD BUY — small position only";       hist_acc = 54.1; confidence = "MILD"
elif prob_up < 0.45:
    signal = "SHORT"; action = "MILD SHORT — small position only";     hist_acc = 47.5; confidence = "MILD"
else:
    signal = "FLAT";  action = "Park in Liquid Fund / FD (~6% p.a.)"; hist_acc = 50.0; confidence = "NEUTRAL"

# Key indicators for output
rsi       = round(float(df['RSI_14'].iloc[-1]), 2)
bb_pctb   = round(float(df['BB_PctB'].iloc[-1]), 3)
macd_hist = round(float(df['MACD_Hist'].iloc[-1]), 2)
vol_20    = round(float(df['Volatility_20'].iloc[-1]), 2)
mom_5     = round(float(df['Mom_5'].iloc[-1]) * 100, 2)
mom_20    = round(float(df['Mom_20'].iloc[-1]) * 100, 2)
bull      = bool(df['Bull_Regime'].iloc[-1])
ma50      = round(float(df['MA_50'].iloc[-1]), 2)
ma200     = round(float(df['MA_200'].iloc[-1]), 2)
bb_upper  = round(float(df['BB_Upper'].iloc[-1]), 2)
bb_lower  = round(float(df['BB_Lower'].iloc[-1]), 2)
atr       = round(float(df['ATR'].iloc[-1]), 2)
vix_val   = round(float(df['VIX_Level'].iloc[-1]), 2) if 'VIX_Level' in df.columns else None
vix_chg   = round(float(df['VIX_Change'].iloc[-1]), 2) if 'VIX_Change' in df.columns else None
sp500_ret = round(float(df['SP500_Return_1d'].iloc[-1]), 2) if 'SP500_Return_1d' in df.columns else None
ndx_ret   = round(float(df['Nasdaq_Return_1d'].iloc[-1]), 2) if 'Nasdaq_Return_1d' in df.columns else None
crude_ret = round(float(df['Crude_Return_1d'].iloc[-1]), 2) if 'Crude_Return_1d' in df.columns else None
usdinr_ret= round(float(df['USDINR_Return_1d'].iloc[-1]), 2) if 'USDINR_Return_1d' in df.columns else None

rsi_note  = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
macd_note = "BULLISH"  if macd_hist > 0 else "BEARISH"
regime    = "BULL"     if bull else "BEAR"

# Next trading day
next_day = date.today() + timedelta(days=1)
if next_day.weekday() == 5: next_day += timedelta(days=2)
elif next_day.weekday() == 6: next_day += timedelta(days=1)
next_day_str = next_day.strftime("%d %b %Y (%A)")

print(f"    Signal      : {signal} ({confidence})")
print(f"    P(UP)       : {prob_up*100:.1f}%")
print(f"    Pred Close  : ₹{pred_close:,.2f} ({pred_change_pct:+.2f}%)")
print(f"    Pred Range  : ₹{pred_low:,.2f} — ₹{pred_high:,.2f}")
print(f"    VIX         : {vix_val} ({vix_chg:+.1f}%)" if vix_val else "    VIX: N/A")
print(f"    SP500 (prev): {sp500_ret:+.2f}%" if sp500_ret is not None else "    SP500: N/A")
print(f"    Nasdaq (prev): {ndx_ret:+.2f}%" if ndx_ret is not None else "    Nasdaq: N/A")

# ══════════════════════════════════════════════════════════════
# STEP 7 — NEWS + GEMINI ANALYSIS
# ══════════════════════════════════════════════════════════════
print("\n[7/7] Gemini sentiment analysis...")

news_context   = "No news available."
news_sentiment = "UNKNOWN"
news_count     = 0
top_headlines  = []

if os.path.exists("news.json"):
    try:
        with open("news.json", "r") as f:
            news_data = json.load(f)
        today_str = date.today().strftime("%Y-%m-%d")
        headlines_today     = news_data.get("today_headlines", [])
        headlines_yesterday = news_data.get("yesterday_headlines", [])
        headlines_week      = news_data.get("week_headlines", [])
        news_count          = news_data.get("total_headlines", 0)
        news_sentiment      = news_data.get("sentiment_score", "MIXED")
        top_headlines       = headlines_today + headlines_yesterday

        today_text = "\n".join([
            f"  [{h.get('age','today')}] {h['title']} ({h.get('source','')})"
            for h in headlines_today[:10]
        ])
        yesterday_text = "\n".join([
            f"  [{h.get('age','yesterday')}] {h['title']} ({h.get('source','')})"
            for h in headlines_yesterday[:8]
        ])
        week_text = "\n".join([
            f"  [{h.get('age','')}] {h['title']}"
            for h in headlines_week[:5]
        ])
        news_context = f"""Market news context (overall sentiment: {news_sentiment}):

TODAY'S NEWS ({len(headlines_today)} headlines):
{today_text if today_text else '  No headlines yet for today'}

YESTERDAY'S NEWS ({len(headlines_yesterday)} headlines):
{yesterday_text if yesterday_text else '  No headlines from yesterday'}

THIS WEEK ({len(headlines_week)} headlines):
{week_text if week_text else '  No additional weekly headlines'}"""
        print(f"    {news_count} headlines — {news_sentiment}")
    except Exception as e:
        print(f"    news.json error: {e}")

gemini_analysis    = "Sentiment analysis not available."
gemini_summary     = ""
gemini_bull_points = []
gemini_bear_points = []

try:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        client = genai.Client(api_key=api_key)

        # Build global context string
        global_ctx = ""
        if sp500_ret is not None:
            global_ctx += f"\n- S&P 500 (previous session): {sp500_ret:+.2f}%"
        if ndx_ret is not None:
            global_ctx += f"\n- NASDAQ 100 (previous session): {ndx_ret:+.2f}%"
        if crude_ret is not None:
            global_ctx += f"\n- Crude Oil WTI (previous session): {crude_ret:+.2f}%"
        if usdinr_ret is not None:
            global_ctx += f"\n- USD/INR change: {usdinr_ret:+.2f}%"
        if vix_val is not None:
            global_ctx += f"\n- India VIX: {vix_val:.2f} (change: {vix_chg:+.1f}%, {'FEAR SPIKE' if vix_val > 20 else 'elevated' if vix_val > 16 else 'calm'})"

        macro = {}
        for col in macro_cols:
            if col in df.columns:
                macro[col] = round(float(df[col].iloc[-1]), 2)

        prompt = f"""You are a professional Indian equity market analyst.
Analyse today's NIFTY 50 data and give prediction for tomorrow.

TODAY'S NIFTY 50:
- Close: ₹{close_price:,.2f} | Change: {daily_change_pct:+.2f}%
- 52W High: ₹{week_52_high:,.2f} | 52W Low: ₹{week_52_low:,.2f}

TECHNICAL INDICATORS:
- RSI (14): {rsi} — {rsi_note}
- BB %B: {bb_pctb} ({'oversold zone' if bb_pctb < 0.2 else 'overbought zone' if bb_pctb > 0.8 else 'mid range'})
- MACD Histogram: {macd_hist} — {macd_note}
- ATR: {atr} | Volatility 20D: {vol_20}%
- 5D Momentum: {mom_5:+.2f}% | 20D Momentum: {mom_20:+.2f}%
- MA50: ₹{ma50:,.2f} | MA200: ₹{ma200:,.2f}
- Market Regime: {regime} (MA50 {'>' if bull else '<'} MA200)

GLOBAL MARKET SIGNALS (PREVIOUS SESSION):{global_ctx if global_ctx else ' Data unavailable'}

INDIA MACRO CONTEXT:
- RBI Repo Rate: {macro.get('macro_repo_rate', 'N/A')}%
- GDP Growth: {macro.get('macro_gdp_growth', 'N/A')}%
- CPI Inflation: {macro.get('macro_cpi', 'N/A')}%
- Fiscal Deficit: {macro.get('macro_fiscal_deficit', 'N/A')}% of GDP
- Bank Credit Growth: {macro.get('macro_credit_growth', 'N/A')}%
- G-Sec 10Y Yield: {macro.get('macro_gsec_10y', 'N/A')}%
- IIP Growth: {macro.get('macro_iip_growth', 'N/A')}%
- USD/INR (annual): {macro.get('macro_usdinr_ye', macro.get('macro_cad_gdp', 'N/A'))}

ML MODEL — TOMORROW'S PREDICTION ({next_day_str}):
- Signal: {signal} ({confidence} confidence)
- P(UP tomorrow): {prob_up*100:.1f}% | P(DOWN): {prob_down*100:.1f}%
- Predicted Close: ₹{pred_close:,.2f} ({pred_change_pct:+.2f}%)
- Predicted Range: ₹{pred_low:,.2f} — ₹{pred_high:,.2f}
- Historical accuracy at this confidence: {hist_acc}%
- Model trained on: {len(X):,} days | {len(valid_features)} features

{news_context}

Respond in EXACTLY this format:

SUMMARY: [2 sentences — today's technical + global picture driving the ML signal]

BULL_CASE: [1 sentence — strongest reason market could go up tomorrow]

BEAR_CASE: [1 sentence — biggest risk that could push market down tomorrow]

ACTION: [{action}]

DISCLAIMER: Algorithmic analysis for educational purposes only. Not SEBI-registered investment advice."""

        import time
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=prompt
                )
                break
            except Exception as retry_err:
                if '429' in str(retry_err) and attempt < 2:
                    print(f"    Rate limit — waiting 30s (attempt {attempt+1})")
                    time.sleep(30)
                else:
                    raise retry_err

        raw = response.text.strip()
        gemini_analysis = raw

        for line in raw.split('\n'):
            line = line.strip()
            if line.startswith('SUMMARY:'):
                gemini_summary = line.replace('SUMMARY:','').strip()
            elif line.startswith('BULL_CASE:'):
                gemini_bull_points = [line.replace('BULL_CASE:','').strip()]
            elif line.startswith('BEAR_CASE:'):
                gemini_bear_points = [line.replace('BEAR_CASE:','').strip()]

        print("    Gemini analysis complete")
    else:
        gemini_analysis = "GEMINI_API_KEY not set in GitHub secrets."
        print("    Skipped — no API key")

except Exception as e:
    gemini_analysis = f"Analysis failed: {str(e)[:120]}"
    print(f"    Gemini error: {e}")

# ══════════════════════════════════════════════════════════════
# SAVE signal.json
# ══════════════════════════════════════════════════════════════
today_label = date.today().strftime("%d %b %Y")
gen_at      = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

output = {
    "date":             today_label,
    "generated_at":     gen_at,
    "next_trading_day": next_day_str,
    "market": {
        "close":        close_price,
        "open":         open_price,
        "high":         high_price,
        "low":          low_price,
        "prev_close":   prev_close,
        "change":       daily_change,
        "change_pct":   daily_change_pct,
        "week_52_high": week_52_high,
        "week_52_low":  week_52_low
    },
    "prediction": {
        "signal":          signal,
        "confidence":      confidence,
        "action":          action,
        "prob_up":         round(prob_up * 100, 1),
        "prob_down":       round(prob_down * 100, 1),
        "hist_accuracy":   hist_acc,
        "pred_close":      pred_close,
        "pred_high":       pred_high,
        "pred_low":        pred_low,
        "pred_change":     pred_change,
        "pred_change_pct": pred_change_pct
    },
    "indicators": {
        "rsi_14":        rsi,
        "rsi_note":      rsi_note,
        "bb_pctb":       bb_pctb,
        "macd_hist":     macd_hist,
        "macd_note":     macd_note,
        "volatility_20": vol_20,
        "momentum_5d":   mom_5,
        "momentum_20d":  mom_20,
        "atr":           atr,
        "bull_regime":   bull,
        "regime":        regime,
        "ma_50":         ma50,
        "ma_200":        ma200,
        "bb_upper":      bb_upper,
        "bb_lower":      bb_lower,
        "vix":           vix_val,
        "vix_change_pct": vix_chg
    },
    "global_signals": {
        "sp500_return_1d":  sp500_ret,
        "nasdaq_return_1d": ndx_ret,
        "crude_return_1d":  crude_ret,
        "usdinr_return_1d": usdinr_ret,
        "vix_level":        vix_val,
        "vix_change_pct":   vix_chg
    },
    "macro": {k: round(float(df[k].iloc[-1]),2) for k in macro_cols if k in df.columns},
    "news": {
        "count":     news_count,
        "sentiment": news_sentiment,
        "headlines": top_headlines
    },
    "analysis": {
        "full":        gemini_analysis,
        "summary":     gemini_summary,
        "bull_points": gemini_bull_points,
        "bear_points": gemini_bear_points
    },
    "model_info": {
        "training_days":      len(X),
        "features_used":      len(valid_features),
        "tech_features":      len(tech_features),
        "vix_features":       len(vix_features),
        "global_features":    len(global_features),
        "macro_features":     len(macro_cols),
        "recent_1y_accuracy": round(recent_acc * 100, 2)
    }
}

with open("signal.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("=" * 65)
print(f"  DATE        : {today_label}")
print(f"  FOR         : {next_day_str}")
print(f"  SIGNAL      : {signal} ({confidence})")
print(f"  P(UP)       : {prob_up*100:.1f}%")
print(f"  PRED CLOSE  : ₹{pred_close:,.2f} ({pred_change_pct:+.2f}%)")
print(f"  PRED RANGE  : ₹{pred_low:,.2f} — ₹{pred_high:,.2f}")
print(f"  VIX         : {vix_val}")
print(f"  SP500 (prev): {sp500_ret}")
print(f"  FEATURES    : {len(valid_features)} total")
print(f"  TRAINED ON  : {len(X):,} days")
print(f"  RECENT ACC  : {recent_acc*100:.2f}%")
print("=" * 65)
print("Done.")
