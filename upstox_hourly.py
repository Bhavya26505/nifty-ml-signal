"""
NiftyEdge — Upstox Intraday Hourly Pipeline
Runs on GitHub Actions every hour 9 AM to 3 PM IST (Mon-Fri)

CHANGES FROM ORIGINAL:
  1. Extended Token instead of Access Token (no daily refresh needed)
  2. GEX (Gamma Exposure) computed from option chain
  3. IV Skew 25-delta computed from option chain
  4. Both GEX + Skew fed into scoring model (more accurate signals)
  5. Both GEX + Skew written to hourly_signal.json (shown on dashboard)
  6. Gemini prompt updated with GEX + Skew context
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
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from google import genai
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  NIFTYEDGE INTRADAY HOURLY PIPELINE")
print("=" * 60)

IST_OFFSET   = timedelta(hours=5, minutes=30)
now_utc      = datetime.utcnow()
now_ist      = now_utc + IST_OFFSET
current_hour = now_ist.hour
current_date = now_ist.date()

print(f"\n  IST time  : {now_ist.strftime('%Y-%m-%d %H:%M IST')}")
print(f"  Run hour  : {current_hour}:00")

# ══════════════════════════════════════════════════════════════
# STEP 1 — AUTHENTICATE WITH UPSTOX
# ══════════════════════════════════════════════════════════════
print("\n[1/7] Authenticating with Upstox...")

# ── CHANGE: use UPSTOX_EXTENDED_TOKEN instead of UPSTOX_ACCESS_TOKEN
# Extended Token lasts 1 year — no daily refresh needed.
# Get it from: account.upstox.com/developer/apps → your app → "Get Extended Token"
# Save it as GitHub Secret: UPSTOX_EXTENDED_TOKEN
access_token = os.environ.get("UPSTOX_EXTENDED_TOKEN", "")
if not access_token:
    # fallback to old access token if extended token not set yet
    access_token = os.environ.get("UPSTOX_ACCESS_TOKEN", "")
if not access_token:
    print("  ERROR: Neither UPSTOX_EXTENDED_TOKEN nor UPSTOX_ACCESS_TOKEN found")
    print("  Get Extended Token from account.upstox.com/developer/apps")
    print("  Save as GitHub Secret: UPSTOX_EXTENDED_TOKEN")
    exit(1)

config             = upstox_client.Configuration()
config.access_token= access_token
api_client         = upstox_client.ApiClient(config)
print("  Authenticated successfully")

NIFTY_KEY     = "NSE_INDEX|Nifty 50"
BANKNIFTY_KEY = "NSE_INDEX|Nifty Bank"
INDIA_VIX_KEY = "NSE_INDEX|India VIX"

# ══════════════════════════════════════════════════════════════
# STEP 2 — FETCH TODAY'S HOURLY CANDLES  (unchanged)
# ══════════════════════════════════════════════════════════════
print("\n[2/7] Fetching intraday data...")

def fetch_intraday(instrument_key, interval="1", unit="minutes"):
    try:
        hist_api = upstox_client.HistoryV3Api(api_client)
        resp     = hist_api.get_intra_day_candle_data(instrument_key, unit, str(interval))
        candles  = resp.data.candles
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles, columns=["datetime","open","high","low","close","volume","oi"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  Intraday fetch error: {e}")
        return pd.DataFrame()

df_1min   = fetch_intraday(NIFTY_KEY, "1", "minutes")
spot_price= 0
day_open  = 0
day_high  = 0
day_low   = 0
df_hourly = pd.DataFrame()

if df_1min.empty:
    print("  No intraday data — market may not be open yet")
else:
    spot_price = float(df_1min["close"].iloc[-1])
    day_open   = float(df_1min["open"].iloc[0])
    day_high   = float(df_1min["high"].max())
    day_low    = float(df_1min["low"].min())

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
# STEP 3 — BLACK-SCHOLES HELPERS  (NEW — needed for GEX + Skew)
# ══════════════════════════════════════════════════════════════

CONTRACT_SIZE = 75   # Nifty lot size (updated 2024)
RISK_FREE     = 0.065

def bs_price(S, K, T, r, sigma, opt_type):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def calc_iv(mkt_price, S, K, T, r, opt_type):
    if mkt_price <= 0.5 or T <= 0:
        return np.nan
    try:
        return brentq(
            lambda sig: bs_price(S, K, T, r, sig, opt_type) - mkt_price,
            1e-4, 10.0, maxiter=100
        )
    except Exception:
        return np.nan

def calc_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or np.isnan(sigma):
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))

def get_dte_years():
    """Days to nearest Thursday expiry as fraction of year."""
    now           = now_ist
    days_to_thurs = (3 - now.weekday()) % 7
    if days_to_thurs == 0 and now.hour >= 15:
        days_to_thurs = 7
    expiry_dt = now + timedelta(days=days_to_thurs)
    expiry_dt = expiry_dt.replace(hour=15, minute=30, second=0)
    dte_secs  = (expiry_dt - now).total_seconds()
    return max(dte_secs / (365 * 24 * 3600), 0.001)

# ══════════════════════════════════════════════════════════════
# STEP 4 — FETCH OPTION CHAIN + COMPUTE GEX + SKEW  (UPGRADED)
# ══════════════════════════════════════════════════════════════
print("\n[3/7] Fetching option chain + computing GEX & Skew...")

# Defaults (used if fetch fails)
pcr_oi         = 1.0
max_pain       = spot_price
call_wall      = spot_price + 200 if spot_price > 0 else 0
put_wall       = spot_price - 200 if spot_price > 0 else 0
atm_iv         = 14.0
iv_skew        = 0.0
straddle_price = 200.0
sentiment_oi   = "NEUTRAL"
total_ce_oi    = 0
total_pe_oi    = 0

# NEW defaults for GEX + Skew
gex_total      = 0.0
gex_zero_gamma = spot_price
gex_regime     = "UNKNOWN"
gex_strength   = 0.0
skew_25d       = 0.0
pc_iv_ratio    = 1.0
iv_slope       = 0.0
skew_bias      = "NEUTRAL"

try:
    url     = "https://api.upstox.com/v2/option/chain"
    headers = {
        "Accept":        "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    params = {"instrument_key": NIFTY_KEY}
    resp   = requests.get(url, headers=headers, params=params, timeout=10)

    if resp.status_code == 200:
        chain_data = resp.json().get("data", [])
        rows = []

        T = get_dte_years()  # time to expiry in years

        for item in chain_data:
            strike = item.get("strike_price", 0)
            if not strike:
                continue
            row = {"strike": float(strike)}

            ce = item.get("call_options", {})
            pe = item.get("put_options", {})

            if ce:
                md = ce.get("market_data",    {})
                gr = ce.get("option_greeks",  {})
                ce_ltp = float(md.get("ltp", 0) or 0)
                ce_oi  = int(  md.get("oi",  0) or 0)
                # Use Upstox IV if available, else compute ourselves
                ce_iv_raw = float(gr.get("iv", 0) or 0) / 100.0  # Upstox gives IV in %
                if ce_iv_raw <= 0 and ce_ltp > 0 and spot_price > 0:
                    ce_iv_raw = calc_iv(ce_ltp, spot_price, strike, T, RISK_FREE, "CE") or 0
                ce_gamma = calc_gamma(spot_price, strike, T, RISK_FREE, ce_iv_raw) if ce_iv_raw > 0 else 0
                row.update({
                    "ce_oi":    ce_oi,
                    "ce_vol":   int(md.get("volume", 0) or 0),
                    "ce_ltp":   ce_ltp,
                    "ce_iv":    ce_iv_raw,
                    "ce_gamma": ce_gamma,
                    "ce_delta": float(gr.get("delta", 0) or 0),
                    "ce_chg_oi":int(md.get("change_oi", 0) or 0),
                })

            if pe:
                md = pe.get("market_data",    {})
                gr = pe.get("option_greeks",  {})
                pe_ltp = float(md.get("ltp", 0) or 0)
                pe_oi  = int(  md.get("oi",  0) or 0)
                pe_iv_raw = float(gr.get("iv", 0) or 0) / 100.0
                if pe_iv_raw <= 0 and pe_ltp > 0 and spot_price > 0:
                    pe_iv_raw = calc_iv(pe_ltp, spot_price, strike, T, RISK_FREE, "PE") or 0
                pe_gamma = calc_gamma(spot_price, strike, T, RISK_FREE, pe_iv_raw) if pe_iv_raw > 0 else 0
                row.update({
                    "pe_oi":    pe_oi,
                    "pe_vol":   int(md.get("volume", 0) or 0),
                    "pe_ltp":   pe_ltp,
                    "pe_iv":    pe_iv_raw,
                    "pe_gamma": pe_gamma,
                    "pe_delta": float(gr.get("delta", 0) or 0),
                    "pe_chg_oi":int(md.get("change_oi", 0) or 0),
                })

            rows.append(row)

        if rows:
            oc = pd.DataFrame(rows).fillna(0)

            # ── Basic OI metrics (same as before) ─────────────
            total_ce_oi = int(oc["ce_oi"].sum())
            total_pe_oi = int(oc["pe_oi"].sum())
            if total_ce_oi > 0:
                pcr_oi = round(total_pe_oi / total_ce_oi, 3)

            # Max pain
            strikes_list = sorted(oc["strike"].unique())
            min_pain = float("inf")
            for s in strikes_list:
                pain = (oc["ce_oi"] * np.maximum(0, s - oc["strike"])).sum() + \
                       (oc["pe_oi"] * np.maximum(0, oc["strike"] - s)).sum()
                if pain < min_pain:
                    min_pain = pain
                    max_pain = s

            if oc["ce_oi"].sum() > 0:
                call_wall = float(oc.loc[oc["ce_oi"].idxmax(), "strike"])
            if oc["pe_oi"].sum() > 0:
                put_wall  = float(oc.loc[oc["pe_oi"].idxmax(), "strike"])

            # ATM metrics
            if spot_price > 0:
                atm_idx = (oc["strike"] - spot_price).abs().argsort().iloc[0]
                atm_row = oc.iloc[atm_idx]
                ce_iv_atm = float(atm_row.get("ce_iv", 0))
                pe_iv_atm = float(atm_row.get("pe_iv", 0))
                if ce_iv_atm > 0 and pe_iv_atm > 0:
                    atm_iv        = round((ce_iv_atm + pe_iv_atm) / 2 * 100, 2)  # back to %
                    iv_skew       = round((pe_iv_atm - ce_iv_atm) * 100, 2)       # in %
                straddle_price    = round(float(atm_row.get("ce_ltp", 0)) + float(atm_row.get("pe_ltp", 0)), 2)

            sentiment_oi = ("BULLISH" if pcr_oi > 1.2
                            else "BEARISH" if pcr_oi < 0.8
                            else "NEUTRAL")

            # ── NEW: GEX calculation ───────────────────────────
            # GEX = Gamma × OI × ContractSize × Spot² × 0.01
            # CE = +GEX (dealers long gamma), PE = -GEX (dealers short gamma)
            if spot_price > 0:
                oc["ce_gex"] =  oc["ce_gamma"] * oc["ce_oi"] * CONTRACT_SIZE * (spot_price**2) * 0.01
                oc["pe_gex"] = -oc["pe_gamma"] * oc["pe_oi"] * CONTRACT_SIZE * (spot_price**2) * 0.01
                oc["net_gex"] = oc["ce_gex"] + oc["pe_gex"]

                gex_total = float(oc["net_gex"].sum())

                # Zero-gamma level: strike where cumulative GEX flips sign
                gex_by_strike = oc.groupby("strike")["net_gex"].sum().sort_index()
                cumgex = gex_by_strike.cumsum()
                positive_flips = cumgex[cumgex > 0]
                if not positive_flips.empty:
                    gex_zero_gamma = float(positive_flips.index[0])
                else:
                    gex_zero_gamma = spot_price

                gex_regime   = "PIN"   if gex_total > 0 else "TREND"
                # Strength: ratio of total GEX to ATM GEX
                atm_strike_rounded = round(spot_price / 50) * 50
                atm_gex = abs(gex_by_strike.get(atm_strike_rounded, 1e-9))
                gex_strength = round(abs(gex_total) / (atm_gex + 1e-9), 2)

            # ── NEW: IV Skew 25-delta ──────────────────────────
            # 25-delta ≈ ATM ± 200 points for Nifty
            if spot_price > 0:
                atm_rounded  = round(spot_price / 50) * 50
                otm_put_stk  = atm_rounded - 200
                otm_call_stk = atm_rounded + 200

                otm_put_row  = oc[oc["strike"] == otm_put_stk]
                otm_call_row = oc[oc["strike"] == otm_call_stk]

                if not otm_put_row.empty and not otm_call_row.empty:
                    otm_pe_iv  = float(otm_put_row["pe_iv"].values[0])
                    otm_ce_iv  = float(otm_call_row["ce_iv"].values[0])
                    if otm_pe_iv > 0 and otm_ce_iv > 0:
                        skew_25d = round(otm_pe_iv - otm_ce_iv, 4)  # in decimal

                # PC IV ratio: avg put IV / avg call IV
                avg_pe_iv = oc[oc["pe_iv"] > 0]["pe_iv"].mean()
                avg_ce_iv = oc[oc["ce_iv"] > 0]["ce_iv"].mean()
                if avg_ce_iv > 0:
                    pc_iv_ratio = round(float(avg_pe_iv / avg_ce_iv), 3)

                # IV slope: near-ATM vs far OTM
                near_iv = oc[abs(oc["strike"] - atm_rounded) <= 100][["ce_iv","pe_iv"]].values.flatten()
                far_iv  = oc[abs(oc["strike"] - atm_rounded) >= 400][["ce_iv","pe_iv"]].values.flatten()
                near_iv = near_iv[near_iv > 0]
                far_iv  = far_iv[far_iv > 0]
                if len(near_iv) > 0 and len(far_iv) > 0:
                    iv_slope = round(float(far_iv.mean() - near_iv.mean()), 4)

                skew_bias = ("BEARISH" if skew_25d > 0.02
                             else "BULLISH" if skew_25d < -0.02
                             else "NEUTRAL")

            print(f"  PCR (OI)     : {pcr_oi}")
            print(f"  Max Pain     : {max_pain:,.0f}")
            print(f"  Call Wall    : {call_wall:,.0f}")
            print(f"  Put Wall     : {put_wall:,.0f}")
            print(f"  ATM IV       : {atm_iv}%")
            print(f"  IV Skew 25d  : {skew_25d:.4f} ({skew_bias})")   # NEW
            print(f"  PC IV Ratio  : {pc_iv_ratio}")                   # NEW
            print(f"  GEX Total    : {gex_total:,.0f} ({gex_regime})") # NEW
            print(f"  Zero Gamma   : {gex_zero_gamma:,.0f}")           # NEW
            print(f"  GEX Strength : {gex_strength}")                  # NEW
            print(f"  Straddle     : {straddle_price}")
            print(f"  Sentiment    : {sentiment_oi}")
    else:
        print(f"  Option chain API error: {resp.status_code}")
        if resp.status_code == 401:
            print("  Token expired — update UPSTOX_EXTENDED_TOKEN in GitHub Secrets")

except Exception as e:
    print(f"  Option chain error: {e}")

# ══════════════════════════════════════════════════════════════
# STEP 5 — COMPUTE HOURLY FEATURES + SIGNAL  (UPGRADED SCORING)
# ══════════════════════════════════════════════════════════════
print("\n[4/7] Computing hourly signal...")

hourly_signal     = "FLAT"
hourly_confidence = "NEUTRAL"
hourly_prob_up    = 0.5
pred_high_h       = spot_price * 1.005 if spot_price > 0 else 0
pred_low_h        = spot_price * 0.995 if spot_price > 0 else 0
vwap              = spot_price
cum_ret           = 0.0
orb_status        = 0
rsi_h             = 50.0

if not df_hourly.empty and spot_price > 0:
    c = df_hourly["close"]

    # VWAP
    typical = (df_hourly["high"] + df_hourly["low"] + df_hourly["close"]) / 3
    if df_hourly["volume"].sum() > 0:
        vwap = round(float((typical * df_hourly["volume"]).sum() / df_hourly["volume"].sum()), 2)
    else:
        vwap = round(float(typical.mean()), 2)

    # Cumulative return from open
    cum_ret = round((spot_price - day_open) / day_open * 100, 2)

    # ORB
    first_bar = df_hourly.iloc[0]
    orb_high  = float(first_bar["high"])
    orb_low   = float(first_bar["low"])
    if spot_price > orb_high:
        orb_status = 1
    elif spot_price < orb_low:
        orb_status = -1

    # RSI on hourly closes
    if len(c) >= 5:
        delta = c.diff()
        gain  = delta.clip(lower=0).rolling(min(14, len(c))).mean()
        loss  = (-delta.clip(upper=0)).rolling(min(14, len(c))).mean()
        rs    = gain / (loss + 1e-10)
        rsi_h = round(float((100 - (100 / (1 + rs))).iloc[-1]), 2)

    # ══════════════════════════════════════════════════════════
    # SCORING MODEL — original 9 points + 6 new GEX/Skew points
    # Total possible: 15 points (was 9)
    # ══════════════════════════════════════════════════════════
    score = 0

    # ── Original signals (unchanged) ──────────────────────────
    if pcr_oi > 1.2:   score += 2
    elif pcr_oi < 0.8: score -= 2

    if spot_price > vwap: score += 1
    else:                 score -= 1

    if spot_price < max_pain: score += 1
    elif spot_price > max_pain: score -= 1

    if orb_status == 1:   score += 2
    elif orb_status == -1: score -= 2

    if rsi_h < 35:   score += 2
    elif rsi_h > 65: score -= 2

    if cum_ret > 0.3:   score += 1
    elif cum_ret < -0.3: score -= 1

    # ── NEW: GEX signals (+3 points max) ──────────────────────
    # GEX regime: TREND regime = follow direction; PIN = fade extremes
    if gex_regime == "TREND":
        # In trending regime, follow VWAP direction with extra conviction
        if spot_price > vwap: score += 1
        else:                 score -= 1
    elif gex_regime == "PIN":
        # In pinning regime, price pulled toward zero-gamma level
        if spot_price < gex_zero_gamma: score += 1   # expects bounce up
        else:                           score -= 1   # expects fade down

    # Is spot above or below zero-gamma level?
    # Above zero-gamma = dealers add to upside momentum
    # Below zero-gamma = dealers add to downside momentum
    if gex_zero_gamma > 0:
        if spot_price > gex_zero_gamma: score += 1
        else:                           score -= 1

    # High GEX strength = strong dealer influence = more reliable pin/trend
    if gex_strength > 2.0:
        if gex_regime == "PIN":   score += 1   # strong pin = mean revert
        else:                     score += 0   # strong trend but direction unknown

    # ── NEW: IV Skew signals (+3 points max) ──────────────────
    # Positive skew (puts expensive vs calls) = market fears drop = bearish
    # Negative skew (calls expensive vs puts) = market fears squeeze = bullish
    if skew_25d > 0.03:        score -= 2   # strong bearish skew
    elif skew_25d > 0.015:     score -= 1   # mild bearish skew
    elif skew_25d < -0.015:    score += 1   # mild bullish skew
    elif skew_25d < -0.03:     score += 2   # strong bullish skew (rare)

    # PC IV ratio: if puts systematically more expensive than calls
    if pc_iv_ratio > 1.15:     score -= 1   # put buyers paying up = bearish
    elif pc_iv_ratio < 0.90:   score += 1   # call buyers paying up = bullish

    # ── Convert score to probability ──────────────────────────
    total_possible = 15.0   # was 9.0 — updated for new signals
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

    # ATR-based predicted range
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

    print(f"  Score        : {score:+d} / {int(total_possible)} (was /9 before GEX+Skew)")
    print(f"  P(UP)        : {hourly_prob_up*100:.1f}%")
    print(f"  Signal       : {hourly_signal} ({hourly_confidence})")
    print(f"  Pred Range   : {pred_low_h:,.2f} — {pred_high_h:,.2f}")
    print(f"  VWAP         : {vwap:,.2f}")
    print(f"  ORB Status   : {'Above ORB' if orb_status==1 else 'Below ORB' if orb_status==-1 else 'Inside range'}")
    print(f"  GEX Regime   : {gex_regime} (Zero-Gamma={gex_zero_gamma:,.0f})")
    print(f"  Skew Bias    : {skew_bias} (25d={skew_25d:.4f})")
else:
    print("  Not enough data — market may not be open yet")

# ══════════════════════════════════════════════════════════════
# STEP 6 — NEWS CONTEXT  (unchanged)
# ══════════════════════════════════════════════════════════════
print("\n[5/7] Loading news context...")

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
        today_text     = "\n".join([f"  [{h.get('age','today')}] {h['title']}" for h in headlines_today[:8]])
        yesterday_text = "\n".join([f"  [{h.get('age','yesterday')}] {h['title']}" for h in headlines_yesterday[:5]])
        news_context = f"""Today's news ({news_count} headlines, sentiment: {news_sentiment}):
TODAY:\n{today_text if today_text else '  No headlines yet'}
YESTERDAY:\n{yesterday_text if yesterday_text else '  No headlines'}"""
        print(f"  {news_count} headlines — {news_sentiment}")
    except Exception as e:
        print(f"  News load error: {e}")

# ══════════════════════════════════════════════════════════════
# STEP 7 — GEMINI ANALYSIS  (UPGRADED with GEX + Skew context)
# ══════════════════════════════════════════════════════════════
print("\n[6/7] Gemini hourly analysis...")

gemini_analysis    = "Hourly analysis not available."
gemini_summary     = ""
gemini_bull_points = []
gemini_bear_points = []

try:
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key and spot_price > 0:
        client  = genai.Client(api_key=api_key)
        orb_text = ("Above opening range" if orb_status == 1
                    else "Below opening range" if orb_status == -1
                    else "Inside opening range")

        # ── UPGRADED PROMPT with GEX + Skew ───────────────────
        prompt = f"""You are an expert Indian intraday equity trader and options analyst.
Analyse the current NIFTY 50 intraday situation for the next 1-hour session.

CURRENT MARKET DATA (IST: {now_ist.strftime('%H:%M')}):
- NIFTY 50 Spot: {spot_price:,.2f}
- Day Open: {day_open:,.2f} | High: {day_high:,.2f} | Low: {day_low:,.2f}
- Cumulative return from open: {cum_ret:+.2f}%
- VWAP: {vwap:,.2f} | Price vs VWAP: {'Above' if spot_price > vwap else 'Below'}
- Hourly RSI: {rsi_h}
- ORB Status: {orb_text}

OPTION CHAIN — OI DATA:
- PCR (OI): {pcr_oi} — {sentiment_oi}
- Max Pain: {max_pain:,.0f}
- Call Wall (resistance): {call_wall:,.0f}
- Put Wall (support): {put_wall:,.0f}
- ATM IV: {atm_iv}%
- Straddle Price: {straddle_price}
- Total CE OI: {total_ce_oi:,} | Total PE OI: {total_pe_oi:,}

GEX — DEALER GAMMA EXPOSURE (NEW):
- Total GEX: {gex_total:,.0f} ({'positive = dealers long gamma = PIN regime' if gex_total > 0 else 'negative = dealers short gamma = TREND regime'})
- Regime: {gex_regime} ({'price tends to pin/revert' if gex_regime=='PIN' else 'price tends to trend/accelerate'})
- Zero-Gamma Level: {gex_zero_gamma:,.0f} ({'spot above = upside momentum' if spot_price > gex_zero_gamma else 'spot below = downside momentum'})
- GEX Strength: {gex_strength}

IV SKEW — FEAR DIRECTION (NEW):
- 25-Delta Skew: {skew_25d:.4f} ({'put skew = market fears downside' if skew_25d > 0 else 'call skew = market fears upside squeeze'})
- Skew Bias: {skew_bias}
- Put/Call IV Ratio: {pc_iv_ratio} ({'puts expensive = defensive positioning' if pc_iv_ratio > 1.1 else 'balanced'})
- IV Slope: {iv_slope:.4f} ({'fear in OTM options' if iv_slope > 0 else 'normal term structure'})

ML SCORING MODEL — NEXT HOUR:
- Signal: {hourly_signal} ({hourly_confidence})
- P(UP next hour): {hourly_prob_up*100:.1f}%
- Predicted hourly range: {pred_low_h:,.2f} — {pred_high_h:,.2f}
- Score: {score if spot_price > 0 else 0:+d} / 15

{news_context}

Respond in EXACTLY this format:

SUMMARY: [2 sentences — use GEX regime and skew to explain dealer positioning and likely price behaviour]

BULL_CASE: [1 sentence — strongest bullish argument combining GEX/Skew/OI]

BEAR_CASE: [1 sentence — strongest bearish risk combining GEX/Skew/OI]

KEY_LEVELS: [Support: X | Resistance: Y | Zero-Gamma: {gex_zero_gamma:,.0f} | Max Pain: {max_pain:,.0f}]

DISCLAIMER: Intraday analysis for educational purposes only. Not SEBI-registered advice."""

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=prompt
                )
                break
            except Exception as retry_err:
                if '429' in str(retry_err) and attempt < 2:
                    print(f"  Rate limit — waiting 30s (attempt {attempt+1})")
                    time.sleep(30)
                else:
                    raise retry_err

        raw             = response.text.strip()
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
        gemini_analysis = "GEMINI_API_KEY not set." if not api_key else "Market not open."
        print(f"  Skipped — {gemini_analysis}")

except Exception as e:
    gemini_analysis = f"Analysis failed: {str(e)[:100]}"
    print(f"  Gemini error: {e}")

# ══════════════════════════════════════════════════════════════
# SAVE hourly_signal.json  (UPGRADED with GEX + Skew sections)
# ══════════════════════════════════════════════════════════════
print("\n[7/7] Saving hourly_signal.json...")

gen_at   = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
hour_lbl = now_ist.strftime("%H:00 — %H:59 IST")

output = {
    "generated_at": gen_at,
    "ist_time":     now_ist.strftime("%Y-%m-%d %H:%M IST"),
    "hour_label":   hour_lbl,
    "market_open":  spot_price > 0,

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
        "score":      score if spot_price > 0 else 0,
    },

    "options": {
        "pcr_oi":      pcr_oi,
        "max_pain":    max_pain,
        "call_wall":   call_wall,
        "put_wall":    put_wall,
        "atm_iv":      atm_iv,
        "iv_skew":     iv_skew,
        "straddle":    straddle_price,
        "sentiment":   sentiment_oi,
        "total_ce_oi": total_ce_oi,
        "total_pe_oi": total_pe_oi,
    },

    # ── NEW: GEX section ──────────────────────────────────────
    "gex": {
        "total_gex":   round(gex_total, 2),
        "zero_gamma":  round(gex_zero_gamma, 0),
        "regime":      gex_regime,
        "strength":    gex_strength,
        "is_positive": gex_total > 0,
    },

    # ── NEW: Skew section ─────────────────────────────────────
    "skew": {
        "skew_25d":    skew_25d,
        "pc_iv_ratio": pc_iv_ratio,
        "iv_slope":    iv_slope,
        "bias":        skew_bias,
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
    },
}

with open("hourly_signal.json", "w") as f:
    json.dump(output, f, indent=2)

print()
print("=" * 60)
print(f"  TIME        : {hour_lbl}")
print(f"  SPOT        : {spot_price:,.2f}")
print(f"  SIGNAL      : {hourly_signal} ({hourly_confidence})")
print(f"  P(UP)       : {hourly_prob_up*100:.1f}%")
print(f"  RANGE       : {pred_low_h:,.2f} — {pred_high_h:,.2f}")
print(f"  PCR         : {pcr_oi}")
print(f"  MAX PAIN    : {max_pain:,.0f}")
print(f"  GEX REGIME  : {gex_regime} (Zero-γ={gex_zero_gamma:,.0f})")
print(f"  SKEW BIAS   : {skew_bias} (25d={skew_25d:.4f})")
print("=" * 60)
print("Done.")
