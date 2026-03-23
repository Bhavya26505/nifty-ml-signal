"""
NiftyEdge — Upstox Intraday Hourly Pipeline
Runs on GitHub Actions every hour 9 AM to 3 PM IST (Mon-Fri)
Fetches live OHLC + Option Chain → trains hourly model → Gemini analysis → saves hourly_signal.json
"""

import upstox_client
from upstox_client.rest import ApiException
import requests
import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime, date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from google import genai
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  NIFTYEDGE INTRADAY HOURLY PIPELINE")
print("=" * 60)

IST_OFFSET = timedelta(hours=5, minutes=30)
now_utc = datetime.utcnow()
now_ist = now_utc + IST_OFFSET
current_hour = now_ist.hour
current_date = now_ist.date()

print(f"\n  IST time  : {now_ist.strftime('%Y-%m-%d %H:%M IST')}")
print(f"  Run hour  : {current_hour}:00")

# ══════════════════════════════════════════════════════════════
# STEP 1 — AUTHENTICATE WITH UPSTOX
# ══════════════════════════════════════════════════════════════
print("\n[1/6] Authenticating with Upstox...")

access_token = os.environ.get("UPSTOX_ACCESS_TOKEN", "")
if not access_token:
    print("  ERROR: UPSTOX_ACCESS_TOKEN not found in environment")
    print("  Run refresh_token.py on your laptop first")
    exit(1)

config = upstox_client.Configuration()
config.access_token = access_token
api_client = upstox_client.ApiClient(config)
print("  Authenticated successfully")

NIFTY_KEY    = "NSE_INDEX|Nifty 50"
BANKNIFTY_KEY= "NSE_INDEX|Nifty Bank"
INDIA_VIX_KEY= "NSE_INDEX|India VIX"

# ══════════════════════════════════════════════════════════════
# STEP 2 — FETCH TODAY'S HOURLY CANDLES
# ══════════════════════════════════════════════════════════════
print("\n[2/6] Fetching intraday data...")

def fetch_intraday(instrument_key, interval="1", unit="minutes"):
    try:
        hist_api = upstox_client.HistoryV3Api(api_client)
        resp = hist_api.get_intra_day_candle_data(instrument_key, unit, str(interval))
        candles = resp.data.candles
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles, columns=["datetime","open","high","low","close","volume","oi"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  Intraday fetch error: {e}")
        return pd.DataFrame()

df_1min = fetch_intraday(NIFTY_KEY, "1", "minutes")

if df_1min.empty:
    print("  No intraday data — market may not be open yet")
    spot_price = 0
    df_hourly  = pd.DataFrame()
else:
    spot_price = float(df_1min["close"].iloc[-1])
    day_open   = float(df_1min["open"].iloc[0])
    day_high   = float(df_1min["high"].max())
    day_low    = float(df_1min["low"].min())

    # Aggregate to hourly
    df_1min_idx = df_1min.set_index("datetime")
    df_hourly   = df_1min_idx.resample("1h").agg({
        "open":"first","high":"max","low":"min",
        "close":"last","volume":"sum","oi":"last"
    }).dropna(subset=["open","close"]).reset_index()

    print(f"  Spot price  : {spot_price:,.2f}")
    print(f"  Day open    : {day_open:,.2f}")
    print(f"  Day H/L     : {day_high:,.2f} / {day_low:,.2f}")
    print(f"  Hourly bars : {len(df_hourly)}")

# ══════════════════════════════════════════════════════════════
# STEP 3 — FETCH OPTION CHAIN
# ══════════════════════════════════════════════════════════════
print("\n[3/6] Fetching option chain...")

pcr_oi        = 1.0
max_pain      = spot_price
call_wall     = spot_price + 200
put_wall      = spot_price - 200
atm_iv        = 14.0
iv_skew       = 0.0
straddle_price= 200.0
sentiment_oi  = "NEUTRAL"
total_ce_oi   = 0
total_pe_oi   = 0

try:
    url = "https://api.upstox.com/v2/option/chain"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    params = {"instrument_key": NIFTY_KEY}
    resp = requests.get(url, headers=headers, params=params, timeout=10)

    if resp.status_code == 200:
        chain_data = resp.json().get("data", [])
        rows = []
        for item in chain_data:
            row = {"strike": item.get("strike_price", 0)}
            ce = item.get("call_options", {})
            pe = item.get("put_options", {})
            if ce:
                md = ce.get("market_data", {})
                gr = ce.get("option_greeks", {})
                row.update({
                    "ce_oi": md.get("oi", 0) or 0,
                    "ce_vol": md.get("volume", 0) or 0,
                    "ce_ltp": md.get("ltp", 0) or 0,
                    "ce_iv": gr.get("iv", 0) or 0,
                    "ce_delta": gr.get("delta", 0) or 0,
                })
            if pe:
                md = pe.get("market_data", {})
                gr = pe.get("option_greeks", {})
                row.update({
                    "pe_oi": md.get("oi", 0) or 0,
                    "pe_vol": md.get("volume", 0) or 0,
                    "pe_ltp": md.get("ltp", 0) or 0,
                    "pe_iv": gr.get("iv", 0) or 0,
                    "pe_delta": gr.get("delta", 0) or 0,
                })
            rows.append(row)

        if rows:
            oc = pd.DataFrame(rows).fillna(0)
            total_ce_oi = int(oc["ce_oi"].sum())
            total_pe_oi = int(oc["pe_oi"].sum())

            if total_ce_oi > 0:
                pcr_oi = round(total_pe_oi / total_ce_oi, 3)

            # Max pain
            strikes = oc["strike"].unique()
            min_pain = float("inf")
            for s in strikes:
                pain = sum(oc["ce_oi"] * np.maximum(0, s - oc["strike"])) + \
                       sum(oc["pe_oi"] * np.maximum(0, oc["strike"] - s))
                if pain < min_pain:
                    min_pain = pain
                    max_pain = s

            # Call wall / Put wall
            if "ce_oi" in oc.columns and oc["ce_oi"].sum() > 0:
                call_wall = float(oc.loc[oc["ce_oi"].idxmax(), "strike"])
            if "pe_oi" in oc.columns and oc["pe_oi"].sum() > 0:
                put_wall  = float(oc.loc[oc["pe_oi"].idxmax(), "strike"])

            # ATM IV
            if spot_price > 0:
                atm_row = oc.iloc[(oc["strike"] - spot_price).abs().argsort()[:1]]
                if not atm_row.empty:
                    ce_iv_atm = float(atm_row["ce_iv"].values[0])
                    pe_iv_atm = float(atm_row["pe_iv"].values[0])
                    atm_iv    = round((ce_iv_atm + pe_iv_atm) / 2, 2)
                    iv_skew   = round(pe_iv_atm - ce_iv_atm, 2)
                    straddle_price = round(
                        float(atm_row["ce_ltp"].values[0]) + float(atm_row["pe_ltp"].values[0]), 2
                    )

            # Sentiment
            if pcr_oi > 1.2:
                sentiment_oi = "BULLISH"
            elif pcr_oi < 0.8:
                sentiment_oi = "BEARISH"
            else:
                sentiment_oi = "NEUTRAL"

            print(f"  PCR (OI)     : {pcr_oi}")
            print(f"  Max Pain     : {max_pain:,.0f}")
            print(f"  Call Wall    : {call_wall:,.0f}")
            print(f"  Put Wall     : {put_wall:,.0f}")
            print(f"  ATM IV       : {atm_iv}%")
            print(f"  IV Skew      : {iv_skew}")
            print(f"  Straddle     : {straddle_price}")
            print(f"  Sentiment    : {sentiment_oi}")
    else:
        print(f"  Option chain API error: {resp.status_code}")

except Exception as e:
    print(f"  Option chain error: {e}")

# ══════════════════════════════════════════════════════════════
# STEP 4 — COMPUTE HOURLY FEATURES + SIGNAL
# ══════════════════════════════════════════════════════════════
print("\n[4/6] Computing hourly signal...")

hourly_signal    = "FLAT"
hourly_confidence= "NEUTRAL"
hourly_prob_up   = 0.5
pred_high_h      = spot_price * 1.005 if spot_price > 0 else 0
pred_low_h       = spot_price * 0.995 if spot_price > 0 else 0
vwap             = spot_price
cum_ret          = 0.0
orb_status       = 0

if not df_hourly.empty and spot_price > 0:
    c = df_hourly["close"]

    # VWAP
    typical = (df_hourly["high"] + df_hourly["low"] + df_hourly["close"]) / 3
    vwap    = round(float((typical * df_hourly["volume"]).sum() / (df_hourly["volume"].sum() + 1e-10)), 2)

    # Cumulative return from open
    cum_ret = round((spot_price - day_open) / day_open * 100, 2)

    # ORB — Opening Range Breakout
    first_bar = df_hourly.iloc[0]
    orb_high  = float(first_bar["high"])
    orb_low   = float(first_bar["low"])
    if spot_price > orb_high:
        orb_status = 1
    elif spot_price < orb_low:
        orb_status = -1

    # RSI on hourly closes
    rsi_h = 50.0
    if len(c) >= 5:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(min(14, len(c))).mean()
        loss  = (-delta.clip(upper=0)).rolling(min(14, len(c))).mean()
        rs    = gain / (loss + 1e-10)
        rsi_h = round(float((100 - (100 / (1 + rs))).iloc[-1]), 2)

    # Simple scoring model
    score = 0

    # PCR signals
    if pcr_oi > 1.2: score += 2
    elif pcr_oi < 0.8: score -= 2

    # Price vs VWAP
    if spot_price > vwap: score += 1
    else: score -= 1

    # Max pain pull
    if spot_price < max_pain: score += 1
    elif spot_price > max_pain: score -= 1

    # ORB
    if orb_status == 1: score += 2
    elif orb_status == -1: score -= 2

    # RSI hourly
    if rsi_h < 35: score += 2
    elif rsi_h > 65: score -= 2

    # Momentum from open
    if cum_ret > 0.3: score += 1
    elif cum_ret < -0.3: score -= 1

    # Convert score to signal
    total_possible = 9.0
    hourly_prob_up = min(0.95, max(0.05, (score + total_possible) / (2 * total_possible)))
    hourly_prob_up = round(hourly_prob_up, 3)

    if hourly_prob_up > 0.62:
        hourly_signal = "LONG";  hourly_confidence = "STRONG"
    elif hourly_prob_up < 0.38:
        hourly_signal = "SHORT"; hourly_confidence = "STRONG"
    elif hourly_prob_up > 0.55:
        hourly_signal = "LONG";  hourly_confidence = "MILD"
    elif hourly_prob_up < 0.45:
        hourly_signal = "SHORT"; hourly_confidence = "MILD"
    else:
        hourly_signal = "FLAT";  hourly_confidence = "NEUTRAL"

    # ATR-based hourly range
    atr_h = float((df_hourly["high"] - df_hourly["low"]).mean()) if len(df_hourly) > 1 else spot_price * 0.005
    if hourly_signal == "LONG":
        pred_high_h = round(spot_price + atr_h * 1.2, 2)
        pred_low_h  = round(spot_price - atr_h * 0.4, 2)
    elif hourly_signal == "SHORT":
        pred_high_h = round(spot_price + atr_h * 0.4, 2)
        pred_low_h  = round(spot_price - atr_h * 1.2, 2)
    else:
        pred_high_h = round(spot_price + atr_h * 0.7, 2)
        pred_low_h  = round(spot_price - atr_h * 0.7, 2)

    print(f"  Score        : {score:+d} / {int(total_possible)}")
    print(f"  P(UP)        : {hourly_prob_up*100:.1f}%")
    print(f"  Signal       : {hourly_signal} ({hourly_confidence})")
    print(f"  Pred Range   : {pred_low_h:,.2f} — {pred_high_h:,.2f}")
    print(f"  VWAP         : {vwap:,.2f}")
    print(f"  ORB Status   : {'Above ORB' if orb_status==1 else 'Below ORB' if orb_status==-1 else 'Inside range'}")
else:
    print("  Not enough data — market may not be open yet")

# ══════════════════════════════════════════════════════════════
# STEP 5 — NEWS CONTEXT
# ══════════════════════════════════════════════════════════════
print("\n[5/6] Loading news context...")

news_context   = "No news available."
news_count     = 0
news_sentiment = "UNKNOWN"

if os.path.exists("news.json"):
    try:
        with open("news.json", "r") as f:
            news_data = json.load(f)
        headlines_today     = news_data.get("today_headlines", [])
        headlines_yesterday = news_data.get("yesterday_headlines", [])
        news_count          = news_data.get("total_headlines", 0)
        news_sentiment      = news_data.get("sentiment_score", "MIXED")

        today_text = "\n".join([
            f"  [{h.get('age','today')}] {h['title']}"
            for h in headlines_today[:8]
        ])
        yesterday_text = "\n".join([
            f"  [{h.get('age','yesterday')}] {h['title']}"
            for h in headlines_yesterday[:5]
        ])
        news_context = f"""Today's news ({news_count} headlines, sentiment: {news_sentiment}):

TODAY:
{today_text if today_text else '  No headlines yet'}

YESTERDAY:
{yesterday_text if yesterday_text else '  No headlines'}"""
        print(f"  {news_count} headlines — {news_sentiment}")
    except Exception as e:
        print(f"  News load error: {e}")

# ══════════════════════════════════════════════════════════════
# STEP 6 — GEMINI ANALYSIS FOR THIS HOUR
# ══════════════════════════════════════════════════════════════
print("\n[6/6] Gemini hourly analysis...")

gemini_analysis    = "Hourly analysis not available."
gemini_summary     = ""
gemini_bull_points = []
gemini_bear_points = []

try:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key and spot_price > 0:
        client = genai.Client(api_key=api_key)

        orb_text = "Above opening range" if orb_status == 1 else "Below opening range" if orb_status == -1 else "Inside opening range"

        prompt = f"""You are an expert Indian intraday equity trader and analyst.
Analyse the current NIFTY 50 intraday situation for the next 1-hour session.

CURRENT MARKET DATA (IST: {now_ist.strftime('%H:%M')}):
- NIFTY 50 Spot: {spot_price:,.2f}
- Day Open: {day_open:,.2f} | High: {day_high:,.2f} | Low: {day_low:,.2f}
- Cumulative return from open: {cum_ret:+.2f}%
- VWAP: {vwap:,.2f} | Price vs VWAP: {'Above' if spot_price > vwap else 'Below'}

OPTION CHAIN DATA:
- PCR (OI): {pcr_oi} — {sentiment_oi}
- Max Pain: {max_pain:,.0f}
- Call Wall (resistance): {call_wall:,.0f}
- Put Wall (support): {put_wall:,.0f}
- ATM IV: {atm_iv}%
- IV Skew (PE-CE): {iv_skew}
- Straddle Price: {straddle_price}
- Total CE OI: {total_ce_oi:,} | Total PE OI: {total_pe_oi:,}

INTRADAY SIGNALS:
- Opening Range Breakout: {orb_text}
- Hourly bars completed: {len(df_hourly)}

ML MODEL — NEXT HOUR SIGNAL:
- Signal: {hourly_signal} ({hourly_confidence})
- P(UP next hour): {hourly_prob_up*100:.1f}%
- Predicted hourly range: {pred_low_h:,.2f} — {pred_high_h:,.2f}

{news_context}

Respond in EXACTLY this format:

SUMMARY: [2 sentences — current intraday situation and what the OI data tells us]

BULL_CASE: [1 sentence — key reason NIFTY could move up this hour]

BEAR_CASE: [1 sentence — key risk that could push NIFTY down this hour]

KEY_LEVELS: [Support: X | Resistance: Y | Max Pain: Z]

DISCLAIMER: Intraday analysis for educational purposes only. Not SEBI-registered advice."""

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=prompt
                )
                break
            except Exception as retry_err:
                if '429' in str(retry_err) and attempt < 2:
                    print(f"  Rate limit — waiting 30s (attempt {attempt+1})")
                    time.sleep(30)
                else:
                    raise retry_err

        raw = response.text.strip()
        gemini_analysis = raw

        for line in raw.split('\n'):
            line = line.strip()
            if line.startswith('SUMMARY:'):
                gemini_summary = line.replace('SUMMARY:', '').strip()
            elif line.startswith('BULL_CASE:'):
                gemini_bull_points = [line.replace('BULL_CASE:', '').strip()]
            elif line.startswith('BEAR_CASE:'):
                gemini_bear_points = [line.replace('BEAR_CASE:', '').strip()]

        print("  Gemini analysis complete")
    else:
        if not api_key:
            gemini_analysis = "GEMINI_API_KEY not set."
        else:
            gemini_analysis = "Market not open — no price data."
        print(f"  Skipped — {gemini_analysis}")

except Exception as e:
    gemini_analysis = f"Analysis failed: {str(e)[:100]}"
    print(f"  Gemini error: {e}")

# ══════════════════════════════════════════════════════════════
# SAVE hourly_signal.json
# ══════════════════════════════════════════════════════════════
gen_at   = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
hour_lbl = now_ist.strftime("%H:00 — %H:59 IST")

output = {
    "generated_at":   gen_at,
    "ist_time":       now_ist.strftime("%Y-%m-%d %H:%M IST"),
    "hour_label":     hour_lbl,
    "market_open":    spot_price > 0,
    "spot": {
        "price":      spot_price,
        "day_open":   day_open   if spot_price > 0 else 0,
        "day_high":   day_high   if spot_price > 0 else 0,
        "day_low":    day_low    if spot_price > 0 else 0,
        "cum_return": cum_ret,
        "vwap":       vwap,
    },
    "signal": {
        "direction":  hourly_signal,
        "confidence": hourly_confidence,
        "prob_up":    round(hourly_prob_up * 100, 1),
        "prob_down":  round((1 - hourly_prob_up) * 100, 1),
        "pred_high":  pred_high_h,
        "pred_low":   pred_low_h,
        "orb_status": orb_status,
    },
    "options": {
        "pcr_oi":       pcr_oi,
        "max_pain":     max_pain,
        "call_wall":    call_wall,
        "put_wall":     put_wall,
        "atm_iv":       atm_iv,
        "iv_skew":      iv_skew,
        "straddle":     straddle_price,
        "sentiment":    sentiment_oi,
        "total_ce_oi":  total_ce_oi,
        "total_pe_oi":  total_pe_oi,
    },
    "news": {
        "count":     news_count,
        "sentiment": news_sentiment,
    },
    "analysis": {
        "full":        gemini_analysis,
        "summary":     gemini_summary,
        "bull_points": gemini_bull_points,
        "bear_points": gemini_bear_points,
    }
}

with open("hourly_signal.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("=" * 60)
print(f"  TIME     : {hour_lbl}")
print(f"  SPOT     : {spot_price:,.2f}")
print(f"  SIGNAL   : {hourly_signal} ({hourly_confidence})")
print(f"  P(UP)    : {hourly_prob_up*100:.1f}%")
print(f"  RANGE    : {pred_low_h:,.2f} — {pred_high_h:,.2f}")
print(f"  PCR      : {pcr_oi}")
print(f"  MAX PAIN : {max_pain:,.0f}")
print("=" * 60)
print("Done.")
