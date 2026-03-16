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

print("=" * 60)
print("  NIFTY 50 ML PREDICTION SYSTEM — FULL MODEL")
print("=" * 60)

# ══════════════════════════════════════════════════════════════
# STEP 1 — LOAD 25-YEAR HISTORICAL DATA
# ══════════════════════════════════════════════════════════════
print("\n[1/7] Loading 25-year historical data...")

df_hist = pd.read_excel(
    "NIFTY50_combined_25years.xlsx",
    parse_dates=['Date']
)
df_hist = df_hist.sort_values('Date').reset_index(drop=True)
df_hist.set_index('Date', inplace=True)
df_hist.rename(columns={
    'Shares Traded': 'Volume',
    'Turnover (₹ Cr)': 'Turnover'
}, inplace=True)
df_hist = df_hist[['Open','High','Low','Close','Volume']].dropna()
print(f"    Loaded {len(df_hist)} days — {df_hist.index[0].date()} to {df_hist.index[-1].date()}")

# ══════════════════════════════════════════════════════════════
# STEP 2 — FETCH FRESH DATA FROM YFINANCE
# ══════════════════════════════════════════════════════════════
print("\n[2/7] Fetching fresh live data from Yahoo Finance...")

try:
    live = yf.download("^NSEI", period="400d", interval="1d", progress=False)
    live.columns = live.columns.get_level_values(0)
    live = live[['Open','High','Low','Close','Volume']].dropna()

    # Merge: historical base + fresh live rows, remove duplicates
    df_combined = pd.concat([df_hist, live])
    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
    df_combined = df_combined.sort_index()
    print(f"    Combined dataset: {len(df_combined)} days")
    print(f"    Latest date     : {df_combined.index[-1].date()}")
except Exception as e:
    print(f"    yfinance failed ({e}) — using historical data only")
    df_combined = df_hist.copy()

# Latest price info
close_price      = float(df_combined['Close'].iloc[-1])
open_price       = float(df_combined['Open'].iloc[-1])
high_price       = float(df_combined['High'].iloc[-1])
low_price        = float(df_combined['Low'].iloc[-1])
prev_close       = float(df_combined['Close'].iloc[-2])
daily_change     = round(close_price - prev_close, 2)
daily_change_pct = round((daily_change / prev_close) * 100, 2)
week_52_high     = float(df_combined['Close'].tail(252).max())
week_52_low      = float(df_combined['Close'].tail(252).min())

print(f"    Close  : ₹{close_price:,.2f}")
print(f"    Change : {daily_change:+.2f} ({daily_change_pct:+.2f}%)")

# ══════════════════════════════════════════════════════════════
# STEP 3 — LOAD MACRO FACTORS
# ══════════════════════════════════════════════════════════════
print("\n[3/7] Loading macro factors...")

def parse_macro_sheet(sheet_df):
    d = sheet_df.copy()
    d.columns = ['Indicator'] + list(d.iloc[0, 1:].values)
    d = d.iloc[1:].reset_index(drop=True)
    d = d[d['Indicator'].notna()].set_index('Indicator').T
    d.index = pd.to_numeric(d.index, errors='coerce')
    d = d.dropna(how='all')
    d.index = d.index.astype(int)
    return d.apply(pd.to_numeric, errors='coerce')

macro_dict = {}
try:
    macro_sheets = pd.read_excel(
        "India_Macro_Factors_2000_2025.xlsx",
        sheet_name=None
    )
    macro_gdp  = parse_macro_sheet(macro_sheets['GDP & Growth'])
    macro_mon  = parse_macro_sheet(macro_sheets['Monetary Policy'])
    macro_ext  = parse_macro_sheet(macro_sheets['External Sector'])
    macro_bank = parse_macro_sheet(macro_sheets['Banking & Credit'])
    macro_ind  = parse_macro_sheet(macro_sheets['Industry & Production'])

    macro_all = pd.concat([macro_gdp, macro_mon, macro_ext, macro_bank, macro_ind], axis=1)
    macro_all = macro_all.loc[:, ~macro_all.columns.duplicated()]

    # Get latest available year's macro values
    latest_year = macro_all.index.max()
    latest_macro = macro_all.loc[latest_year]

    # Map key macro indicators — use exact column names from your file
    repo_col  = [c for c in macro_all.columns if 'Repo' in str(c) and 'Reverse' not in str(c)]
    gdp_col   = [c for c in macro_all.columns if 'Real GDP' in str(c) or 'GDP Growth' in str(c)]
    inr_col   = [c for c in macro_all.columns if 'USD' in str(c) and 'INR' in str(c)]
    cpi_col   = [c for c in macro_all.columns if 'CPI' in str(c) or 'Inflation' in str(c)]
    credit_col= [c for c in macro_all.columns if 'Credit Growth' in str(c) or 'Bank Credit' in str(c)]
    iip_col   = [c for c in macro_all.columns if 'IIP' in str(c)]
    fiscal_col= [c for c in macro_all.columns if 'Fiscal Deficit' in str(c)]
    cad_col   = [c for c in macro_all.columns if 'Current Account' in str(c)]

    macro_dict = {
        'macro_repo_rate'    : float(latest_macro[repo_col[0]])   if repo_col   else 6.25,
        'macro_gdp_growth'   : float(latest_macro[gdp_col[0]])    if gdp_col    else 6.5,
        'macro_usd_inr'      : float(latest_macro[inr_col[0]])    if inr_col    else 85.0,
        'macro_cpi'          : float(latest_macro[cpi_col[0]])    if cpi_col    else 4.5,
        'macro_credit_growth': float(latest_macro[credit_col[0]]) if credit_col else 12.0,
        'macro_iip_growth'   : float(latest_macro[iip_col[0]])    if iip_col    else 5.0,
        'macro_fiscal_deficit': float(latest_macro[fiscal_col[0]])if fiscal_col else 5.0,
        'macro_cad'          : float(latest_macro[cad_col[0]])    if cad_col    else -1.5,
    }
    # Fill NaN with sensible defaults
    macro_dict = {k: (v if not np.isnan(v) else 0.0) for k, v in macro_dict.items()}
    print(f"    Macro data loaded — year {latest_year}")
    for k, v in macro_dict.items():
        print(f"    {k:<28}: {v}")

except Exception as e:
    print(f"    Macro load failed ({e}) — using defaults")
    macro_dict = {
        'macro_repo_rate'    : 6.25,
        'macro_gdp_growth'   : 6.5,
        'macro_usd_inr'      : 85.0,
        'macro_cpi'          : 4.5,
        'macro_credit_growth': 12.0,
        'macro_iip_growth'   : 5.0,
        'macro_fiscal_deficit': 5.0,
        'macro_cad'          : -1.5,
    }

# ══════════════════════════════════════════════════════════════
# STEP 4 — ENGINEER ALL FEATURES
# ══════════════════════════════════════════════════════════════
print("\n[4/7] Engineering features...")

df = df_combined.copy()
df['Daily_Return']  = df['Close'].pct_change() * 100
df['Log_Return']    = np.log(df['Close'] / df['Close'].shift(1)) * 100

for w in [5, 10, 20, 50, 200]:
    df[f'MA_{w}'] = df['Close'].rolling(w).mean()

df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

delta = df['Close'].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
df['RSI_14'] = 100 - (100 / (1 + gain / loss))

df['MACD']        = df['EMA_12'] - df['EMA_26']
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

ma20  = df['Close'].rolling(20).mean()
std20 = df['Close'].rolling(20).std()
df['BB_Upper'] = ma20 + 2 * std20
df['BB_Lower'] = ma20 - 2 * std20
df['BB_PctB']  = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

df['Volatility_20'] = df['Daily_Return'].rolling(20).std()
df['Volatility_60'] = df['Daily_Return'].rolling(60).std()
df['ATR']           = (df['High'] - df['Low']).rolling(14).mean()
df['Volume_MA20']   = df['Volume'].rolling(20).mean()
df['Volume_Ratio']  = df['Volume'] / df['Volume_MA20']
df['Mom_5']         = df['Close'] / df['Close'].shift(5)  - 1
df['Mom_20']        = df['Close'] / df['Close'].shift(20) - 1

for lag in [1, 2, 3, 5, 10, 20]:
    df[f'Return_Lag_{lag}'] = df['Daily_Return'].shift(lag)

df['Bull_Regime'] = (df['MA_50'] > df['MA_200']).astype(int)
df['Month']       = df.index.month
df['DayOfWeek']   = df.index.dayofweek

# Add macro features as constant columns for current year
for k, v in macro_dict.items():
    df[k] = v

df = df.dropna()
print(f"    Features ready — {len(df)} trading days")

# ══════════════════════════════════════════════════════════════
# STEP 5 — TRAIN TWO MODELS
# ══════════════════════════════════════════════════════════════
print("\n[5/7] Training models...")

tech_features = [
    'Log_Return', 'RSI_14', 'BB_PctB', 'MACD_Hist',
    'Volume_Ratio', 'Volatility_20', 'Mom_5', 'Mom_20',
    'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_3',
    'Return_Lag_5', 'Month', 'DayOfWeek', 'Bull_Regime',
    'ATR', 'Volatility_60', 'MA_50', 'MA_200',
]
macro_features = list(macro_dict.keys())
all_features   = tech_features + macro_features

# ── Model A: Direction Classifier (next day up/down) ─────────
df['Target_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# ── Model B: Price Regressor (next day close) ────────────────
df['Target_Close'] = df['Close'].shift(-1)
df['Target_High']  = df['High'].shift(-1)
df['Target_Low']   = df['Low'].shift(-1)

df_model = df.dropna(subset=['Target_Direction', 'Target_Close'])

X = df_model[all_features].values[:-1]
y_dir   = df_model['Target_Direction'].values[:-1]
y_close = df_model['Target_Close'].values[:-1]
y_high  = df_model['Target_High'].values[:-1]
y_low   = df_model['Target_Low'].values[:-1]

sc = StandardScaler()
X_sc = sc.fit_transform(X)

# Train classifier
clf = xgb.XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='logloss', random_state=42, verbosity=0
)
clf.fit(X_sc, y_dir)

# Quick accuracy check on last 252 days (1 year)
split = max(len(X) - 252, int(len(X) * 0.8))
clf_check = xgb.XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='logloss', random_state=42, verbosity=0
)
clf_check.fit(sc.transform(X[:split]), y_dir[:split])
check_acc = accuracy_score(y_dir[split:], clf_check.predict(sc.transform(X[split:])))
print(f"    Direction model — recent 1Y accuracy: {check_acc*100:.2f}%")

# Train close price regressor
reg_close = xgb.XGBRegressor(
    n_estimators=400, max_depth=5, learning_rate=0.04,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0
)
reg_close.fit(X_sc, y_close)

# Train high regressor
reg_high = xgb.XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0
)
reg_high.fit(X_sc, y_high)

# Train low regressor
reg_low = xgb.XGBRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0
)
reg_low.fit(X_sc, y_low)

print(f"    All 4 models trained ({len(X):,} samples)")

# ══════════════════════════════════════════════════════════════
# STEP 6 — GENERATE TOMORROW'S PREDICTION
# ══════════════════════════════════════════════════════════════
print("\n[6/7] Generating tomorrow's prediction...")

latest    = df[all_features].iloc[[-1]]
latest_sc = sc.transform(latest)

# Direction
prob_up   = float(clf.predict_proba(latest_sc)[0][1])
prob_down = round(1 - prob_up, 4)
prob_up_r = round(prob_up, 4)

# Price predictions
pred_close = round(float(reg_close.predict(latest_sc)[0]), 2)
pred_high  = round(float(reg_high.predict(latest_sc)[0]), 2)
pred_low   = round(float(reg_low.predict(latest_sc)[0]), 2)

# Price change prediction
pred_change     = round(pred_close - close_price, 2)
pred_change_pct = round((pred_change / close_price) * 100, 2)

# Confidence band logic (from your validated walk-forward results)
if prob_up > 0.60:
    signal        = "LONG"
    action        = "BUY / HOLD — Nifty ETF (NIFTYBEES)"
    hist_accuracy = 60.3
    confidence    = "STRONG"
elif prob_up < 0.40:
    signal        = "SHORT"
    action        = "SHORT Nifty Futures / Exit longs"
    hist_accuracy = 55.4
    confidence    = "STRONG"
elif prob_up > 0.55:
    signal        = "LONG"
    action        = "MILD BUY — small position only"
    hist_accuracy = 54.1
    confidence    = "MILD"
elif prob_up < 0.45:
    signal        = "SHORT"
    action        = "MILD SHORT — small position only"
    hist_accuracy = 47.5
    confidence    = "MILD"
else:
    signal        = "FLAT"
    action        = "Park in Liquid Fund / FD (~6% p.a.)"
    hist_accuracy = 50.0
    confidence    = "NEUTRAL"

# Key indicators
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
vol_ratio = round(float(df['Volume_Ratio'].iloc[-1]), 2)
atr       = round(float(df['ATR'].iloc[-1]), 2)

rsi_note  = "OVERSOLD"   if rsi < 30  else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
macd_note = "BULLISH"    if macd_hist > 0 else "BEARISH"
regime    = "BULL"       if bull else "BEAR"

# Next trading day
next_day = date.today() + timedelta(days=1)
if next_day.weekday() == 5:
    next_day += timedelta(days=2)
elif next_day.weekday() == 6:
    next_day += timedelta(days=1)
next_day_str = next_day.strftime("%d %b %Y (%A)")

print(f"    Signal        : {signal} ({confidence})")
print(f"    P(UP)         : {prob_up_r*100:.1f}%")
print(f"    Pred Close    : ₹{pred_close:,.2f} ({pred_change_pct:+.2f}%)")
print(f"    Pred Range    : ₹{pred_low:,.2f} — ₹{pred_high:,.2f}")
print(f"    For           : {next_day_str}")

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
        if news_data.get("date") == today_str:
            headlines      = news_data.get("headlines", [])
            news_count     = len(headlines)
            news_sentiment = news_data.get("sentiment_score", "MIXED")
            top_headlines  = headlines[-15:]
            headline_text  = "\n".join([
                f"- [{h.get('fetched_at','')[:16]}] {h['title']}"
                for h in top_headlines
            ])
            news_context = (
                f"Today's market news ({news_count} headlines, "
                f"overall sentiment: {news_sentiment}):\n{headline_text}"
            )
            print(f"    {news_count} headlines loaded — {news_sentiment}")
        else:
            print("    news.json is from different date")
    except Exception as e:
        print(f"    news.json read error: {e}")

gemini_analysis    = "Sentiment analysis not available."
gemini_bull_points = []
gemini_bear_points = []
gemini_summary     = ""

try:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        client = genai.Client(api_key=api_key)


        prompt = f"""You are a professional Indian equity market analyst.
Analyse today's NIFTY 50 data and give prediction for tomorrow.

TODAY'S MARKET:
- Close: ₹{close_price:,.2f} | Change: {daily_change_pct:+.2f}%
- 52W High: ₹{week_52_high:,.2f} | 52W Low: ₹{week_52_low:,.2f}

TECHNICAL INDICATORS:
- RSI (14): {rsi} — {rsi_note}
- BB %B: {bb_pctb} ({'oversold zone' if bb_pctb < 0.2 else 'overbought zone' if bb_pctb > 0.8 else 'mid range'})
- MACD Histogram: {macd_hist} — {macd_note}
- ATR: {atr} (daily volatility measure)
- 5D Momentum: {mom_5:+.2f}% | 20D Momentum: {mom_20:+.2f}%
- Volume Ratio: {vol_ratio}x vs 20D average
- Volatility 20D: {vol_20}%
- MA50: ₹{ma50:,.2f} | MA200: ₹{ma200:,.2f}
- BB Upper: ₹{bb_upper:,.2f} | BB Lower: ₹{bb_lower:,.2f}
- Market Regime: {regime} (MA50 {'>' if bull else '<'} MA200)

INDIA MACRO FACTORS (latest):
- RBI Repo Rate: {macro_dict['macro_repo_rate']}%
- GDP Growth: {macro_dict['macro_gdp_growth']}%
- CPI Inflation: {macro_dict['macro_cpi']}%
- USD/INR: {macro_dict['macro_usd_inr']}
- Bank Credit Growth: {macro_dict['macro_credit_growth']}%
- IIP Growth: {macro_dict['macro_iip_growth']}%
- Fiscal Deficit: {macro_dict['macro_fiscal_deficit']}% of GDP

ML MODEL PREDICTION FOR TOMORROW ({next_day_str}):
- Signal: {signal} ({confidence} confidence)
- P(UP tomorrow): {prob_up_r*100:.1f}%
- P(DOWN tomorrow): {prob_down*100:.1f}%
- Predicted Close: ₹{pred_close:,.2f} ({pred_change_pct:+.2f}%)
- Predicted Range: ₹{pred_low:,.2f} — ₹{pred_high:,.2f}
- Historical accuracy at this confidence: {hist_accuracy}%
- Model trained on: 25 years of NIFTY data + India macro factors

{news_context}

Respond in EXACTLY this format:

SUMMARY: [2 sentences — today's technical picture and what is driving the ML signal]

BULL_CASE: [1 sentence — strongest reason market could go up tomorrow]

BEAR_CASE: [1 sentence — biggest risk that could push market down tomorrow]

ACTION: [Recommended action based on signal]

DISCLAIMER: Algorithmic analysis for educational purposes only. Not SEBI-registered investment advice."""

        response       = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        raw_response   = response.text.strip()
        gemini_analysis = raw_response

        for line in raw_response.split('\n'):
            line = line.strip()
            if line.startswith('SUMMARY:'):
                gemini_summary = line.replace('SUMMARY:', '').strip()
            elif line.startswith('BULL_CASE:'):
                gemini_bull_points = [line.replace('BULL_CASE:', '').strip()]
            elif line.startswith('BEAR_CASE:'):
                gemini_bear_points = [line.replace('BEAR_CASE:', '').strip()]

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
    "date":              today_label,
    "generated_at":      gen_at,
    "next_trading_day":  next_day_str,
    "market": {
        "close":       close_price,
        "open":        open_price,
        "high":        high_price,
        "low":         low_price,
        "prev_close":  prev_close,
        "change":      daily_change,
        "change_pct":  daily_change_pct,
        "week_52_high": week_52_high,
        "week_52_low":  week_52_low
    },
    "prediction": {
        "signal":          signal,
        "confidence":      confidence,
        "action":          action,
        "prob_up":         round(prob_up_r * 100, 1),
        "prob_down":       round(prob_down * 100, 1),
        "hist_accuracy":   hist_accuracy,
        "pred_close":      pred_close,
        "pred_high":       pred_high,
        "pred_low":        pred_low,
        "pred_change":     pred_change,
        "pred_change_pct": pred_change_pct
    },
    "indicators": {
        "rsi_14":         rsi,
        "rsi_note":       rsi_note,
        "bb_pctb":        bb_pctb,
        "macd_hist":      macd_hist,
        "macd_note":      macd_note,
        "volatility_20":  vol_20,
        "momentum_5d":    mom_5,
        "momentum_20d":   mom_20,
        "volume_ratio":   vol_ratio,
        "atr":            atr,
        "bull_regime":    bull,
        "regime":         regime,
        "ma_50":          ma50,
        "ma_200":         ma200,
        "bb_upper":       bb_upper,
        "bb_lower":       bb_lower
    },
    "macro": macro_dict,
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
        "training_days":    len(X),
        "features_used":    len(all_features),
        "tech_features":    len(tech_features),
        "macro_features":   len(macro_features),
        "recent_1y_accuracy": round(check_acc * 100, 2)
    }
}

with open("signal.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("=" * 60)
print(f"  DATE      : {today_label}")
print(f"  FOR       : {next_day_str}")
print(f"  SIGNAL    : {signal} ({confidence})")
print(f"  P(UP)     : {prob_up_r*100:.1f}%")
print(f"  PRED CLOSE: ₹{pred_close:,.2f} ({pred_change_pct:+.2f}%)")
print(f"  PRED RANGE: ₹{pred_low:,.2f} — ₹{pred_high:,.2f}")
print(f"  NEWS      : {news_count} headlines ({news_sentiment})")
print(f"  FEATURES  : {len(all_features)} ({len(tech_features)} tech + {len(macro_features)} macro)")
print(f"  TRAINED ON: {len(X):,} days")
print("=" * 60)
print("Done.")
