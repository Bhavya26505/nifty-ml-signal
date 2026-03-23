"""
upstox_fetch.py
===============
Drop-in replacement for fetch_option_chain_nse() in hourly_model.py
Uses Upstox API v2 with Extended Token (valid 1 year, no daily refresh needed)

SETUP (one time only):
  1. Go to https://account.upstox.com/developer/apps
  2. Open your app → click "Get Extended Token"
  3. Copy the extended token
  4. In your GitHub repo → Settings → Secrets → New secret:
       Name:  UPSTOX_EXTENDED_TOKEN
       Value: (paste token here)
  5. Done. Works for 1 year automatically.
"""

import os
import requests
from datetime import datetime, timedelta
import pytz

IST = pytz.timezone("Asia/Kolkata")

UPSTOX_BASE = "https://api.upstox.com/v2"


def get_access_token() -> str:
    """
    Returns the Extended Token from GitHub Secret.
    This token is valid for 1 year — no daily refresh needed.
    """
    token = os.environ.get("UPSTOX_EXTENDED_TOKEN", "")
    if not token:
        raise ValueError(
            "UPSTOX_EXTENDED_TOKEN not set. "
            "Add it in GitHub → Settings → Secrets → Actions."
        )
    return token


def get_nearest_expiry() -> str:
    """
    Returns nearest weekly Thursday expiry in YYYY-MM-DD format.
    Upstox option chain API requires this format.
    """
    now = datetime.now(IST)
    # Thursday = weekday 3
    days_to_thursday = (3 - now.weekday()) % 7
    if days_to_thursday == 0 and now.hour >= 15:
        days_to_thursday = 7  # today's expiry passed, use next week
    expiry = now + timedelta(days=days_to_thursday)
    return expiry.strftime("%Y-%m-%d")


def fetch_option_chain_upstox(symbol: str = "NSE_INDEX|Nifty 50") -> tuple:
    """
    Fetches live option chain from Upstox API v2.
    Returns: (chain_raw, spot, expiry_str)

    chain_raw format (same as NSE version — compatible with hourly_model.py):
    [{"strike": 24000, "type": "CE", "ltp": 120.5, "oi": 50000, "chg_oi": 1200}, ...]
    """
    token   = get_access_token()
    expiry  = get_nearest_expiry()
    headers = {
        "Accept":        "application/json",
        "Authorization": f"Bearer {token}",
    }

    # ── Step 1: Get option chain ──────────────────────────────
    url    = f"{UPSTOX_BASE}/option/chain"
    params = {"instrument_key": symbol, "expiry_date": expiry}
    resp   = requests.get(url, headers=headers, params=params, timeout=10)

    if resp.status_code == 401:
        raise ValueError(
            "Upstox token expired or invalid. "
            "Generate a new Extended Token from account.upstox.com/developer/apps "
            "and update the UPSTOX_EXTENDED_TOKEN secret."
        )
    resp.raise_for_status()
    data = resp.json()

    # ── Step 2: Get spot price (Nifty 50 index) ───────────────
    ltp_url    = f"{UPSTOX_BASE}/market-quote/ltp"
    ltp_params = {"instrument_key": "NSE_INDEX|Nifty 50"}
    ltp_resp   = requests.get(ltp_url, headers=headers, params=ltp_params, timeout=10)
    ltp_resp.raise_for_status()
    ltp_data   = ltp_resp.json()

    spot = float(
        ltp_data["data"]["NSE_INDEX:Nifty 50"]["last_price"]
    )

    # ── Step 3: Parse option chain into standard format ────────
    chain_raw = []
    for row in data.get("data", []):
        strike = row.get("strike_price")
        if not strike:
            continue

        # Call side
        ce = row.get("call_options", {}).get("market_data", {})
        if ce and ce.get("ltp", 0) > 0:
            chain_raw.append({
                "strike":  int(strike),
                "type":    "CE",
                "ltp":     float(ce.get("ltp",    0)),
                "oi":      int(  ce.get("oi",     0)),
                "chg_oi":  int(  ce.get("change_oi", 0)),
                "bid":     float(ce.get("bid_price",  0)),
                "ask":     float(ce.get("ask_price",  0)),
            })

        # Put side
        pe = row.get("put_options", {}).get("market_data", {})
        if pe and pe.get("ltp", 0) > 0:
            chain_raw.append({
                "strike":  int(strike),
                "type":    "PE",
                "ltp":     float(pe.get("ltp",    0)),
                "oi":      int(  pe.get("oi",     0)),
                "chg_oi":  int(  pe.get("change_oi", 0)),
                "bid":     float(pe.get("bid_price",  0)),
                "ask":     float(pe.get("ask_price",  0)),
            })

    print(f"✓ Upstox: {len(chain_raw)} option rows | Spot={spot:,.2f} | Expiry={expiry}")
    return chain_raw, spot, expiry


# ─────────────────────────────────────────────────────────────
# HOW TO USE IN hourly_model.py
# ─────────────────────────────────────────────────────────────
# Replace the fetch line in main() from:
#
#   chain_raw, spot, expiry = fetch_option_chain_nse("NIFTY")
#
# To:
#
#   from upstox_fetch import fetch_option_chain_upstox
#   chain_raw, spot, expiry = fetch_option_chain_upstox()
#
# Everything else in hourly_model.py stays exactly the same.
# ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Quick test
    chain, spot, expiry = fetch_option_chain_upstox()
    print(f"Spot: {spot:,.2f}")
    print(f"Expiry: {expiry}")
    print(f"Sample row: {chain[0]}")
    print(f"Total rows: {len(chain)}")
