import requests
import feedparser
import json
import os
from datetime import datetime, date, timedelta, timezone
from dateutil import parser as dateparser

print("=" * 55)
print("  NIFTY NEWS FETCHER — 24/7 WITH DATE FILTERING")
print("=" * 55)

# ── Date boundaries ───────────────────────────────────────────
now_utc   = datetime.now(timezone.utc)
today     = now_utc.date()
yesterday = today - timedelta(days=1)
week_ago  = today - timedelta(days=7)
month_ago = today - timedelta(days=30)

print(f"\n  Current UTC : {now_utc.strftime('%Y-%m-%d %H:%M')}")
print(f"  Keeping news from: {week_ago} to {today}")
print(f"  Max age    : 7 days (older headlines rejected)")

# ── RSS feeds — Indian market focused ────────────────────────
FEEDS = [
    # NIFTY specific
    "https://news.google.com/rss/search?q=NIFTY+50+NSE&hl=en-IN&gl=IN&ceid=IN:en",
    # BSE Sensex
    "https://news.google.com/rss/search?q=Sensex+BSE+India+stock+market&hl=en-IN&gl=IN&ceid=IN:en",
    # RBI monetary policy
    "https://news.google.com/rss/search?q=RBI+repo+rate+India+monetary+policy&hl=en-IN&gl=IN&ceid=IN:en",
    # FII DII flows
    "https://news.google.com/rss/search?q=FII+DII+India+equity+flows&hl=en-IN&gl=IN&ceid=IN:en",
    # India economy
    "https://news.google.com/rss/search?q=India+GDP+inflation+economy&hl=en-IN&gl=IN&ceid=IN:en",
    # Global cues
    "https://news.google.com/rss/search?q=US+Fed+interest+rate+global+markets&hl=en-IN&gl=IN&ceid=IN:en",
    # Crude oil dollar
    "https://news.google.com/rss/search?q=crude+oil+price+dollar+index+India&hl=en-IN&gl=IN&ceid=IN:en",
    # Indian corporate earnings
    "https://news.google.com/rss/search?q=India+quarterly+results+earnings+NSE&hl=en-IN&gl=IN&ceid=IN:en",
]

def parse_pub_date(entry):
    """
    Parse the published date from RSS entry.
    Returns a timezone-aware datetime or None.
    """
    for field in ['published', 'updated', 'created']:
        raw = entry.get(field, '')
        if raw:
            try:
                dt = dateparser.parse(raw)
                if dt and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                continue
    return None

def age_label(pub_date):
    """Return human readable age label."""
    if pub_date is None:
        return "unknown"
    diff = now_utc - pub_date
    hours = int(diff.total_seconds() / 3600)
    if hours < 1:
        return "just now"
    elif hours < 24:
        return f"{hours}h ago"
    elif hours < 48:
        return "yesterday"
    else:
        days = diff.days
        return f"{days}d ago"

def priority_label(pub_date):
    """
    Assign priority based on age:
      1 = today (highest)
      2 = yesterday
      3 = last 7 days
      None = older than 7 days (rejected)
    """
    if pub_date is None:
        return None
    diff = (now_utc - pub_date).days
    if diff == 0:
        return 1
    elif diff == 1:
        return 2
    elif diff <= 7:
        return 3
    else:
        return None   # reject

# ── Load existing news.json ───────────────────────────────────
existing_headlines = []
existing_titles    = set()

if os.path.exists("news.json"):
    try:
        with open("news.json", "r") as f:
            existing_data = json.load(f)

        for h in existing_data.get("headlines", []):
            # Keep only headlines within last 7 days
            pub_raw = h.get("published_iso", "")
            if pub_raw:
                try:
                    pub_dt = dateparser.parse(pub_raw)
                    if pub_dt and pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    if pub_dt and (now_utc - pub_dt).days <= 7:
                        existing_headlines.append(h)
                        existing_titles.add(h.get("title", ""))
                except Exception:
                    pass

        print(f"\n  Loaded {len(existing_headlines)} existing headlines (within 7 days)")
    except Exception as e:
        print(f"\n  Could not read existing news.json: {e}")

# ── Fetch new headlines from all feeds ───────────────────────
print(f"\n  Fetching from {len(FEEDS)} RSS feeds...")

new_today     = []
new_yesterday = []
new_week      = []
rejected      = 0
duplicates    = 0

for feed_url in FEEDS:
    try:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:10]:
            title = entry.get("title", "").strip()
            link  = entry.get("link",  "").strip()

            if not title:
                continue

            # Clean title — remove source suffix
            clean_title = title.rsplit(" - ", 1)[0].strip() if " - " in title else title
            source      = title.rsplit(" - ", 1)[-1].strip() if " - " in title else "News"

            # Skip duplicates
            if clean_title in existing_titles or title in existing_titles:
                duplicates += 1
                continue

            # Parse and check date
            pub_dt   = parse_pub_date(entry)
            priority = priority_label(pub_dt)

            if priority is None:
                rejected += 1
                continue   # older than 7 days — skip

            pub_iso = pub_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if pub_dt else ""
            age     = age_label(pub_dt)

            headline = {
                "title":        clean_title,
                "source":       source,
                "link":         link,
                "published":    entry.get("published", ""),
                "published_iso": pub_iso,
                "age":          age,
                "priority":     priority,
                "fetched_at":   now_utc.strftime("%Y-%m-%d %H:%M UTC")
            }

            existing_titles.add(clean_title)
            existing_titles.add(title)

            if priority == 1:
                new_today.append(headline)
            elif priority == 2:
                new_yesterday.append(headline)
            else:
                new_week.append(headline)

    except Exception as e:
        print(f"  Feed error: {e}")

print(f"  New today     : {len(new_today)}")
print(f"  New yesterday : {len(new_yesterday)}")
print(f"  New this week : {len(new_week)}")
print(f"  Rejected (old): {rejected}")
print(f"  Duplicates    : {duplicates}")

# ── Merge: new headlines first, then existing ─────────────────
# Priority order: today first, yesterday second, this week last
all_new = new_today + new_yesterday + new_week

# Combine with existing, sort by priority then by fetched_at
all_headlines = all_new + existing_headlines

# Remove duplicates again (safety check)
seen   = set()
unique = []
for h in all_headlines:
    t = h.get("title", "")
    if t not in seen:
        seen.add(t)
        unique.append(h)

# Sort: priority 1 (today) first, then 2, then 3
unique.sort(key=lambda h: (h.get("priority", 3), h.get("published_iso", "")), reverse=False)

# Cap at 100 headlines max
unique = unique[:100]

# ── Categorise for Gemini context ────────────────────────────
bullish_kw = [
    "rate cut", "gdp growth", "profit", "rally", "surge", "gain",
    "positive", "record high", "buy", "upgrade", "boost", "strong",
    "recovery", "growth", "reform", "investment", "inflow"
]
bearish_kw = [
    "rate hike", "inflation", "loss", "crash", "fall", "decline",
    "negative", "sell", "downgrade", "risk", "deficit", "weak",
    "outflow", "war", "tariff", "recession", "slowdown"
]

bullish_count = sum(
    1 for h in unique
    if any(k in h["title"].lower() for k in bullish_kw)
)
bearish_count = sum(
    1 for h in unique
    if any(k in h["title"].lower() for k in bearish_kw)
)

if bullish_count > bearish_count * 1.4:
    sentiment = "BULLISH"
elif bearish_count > bullish_count * 1.4:
    sentiment = "BEARISH"
else:
    sentiment = "MIXED"

# ── Separate by time window for easy access ───────────────────
today_headlines     = [h for h in unique if h.get("priority") == 1]
yesterday_headlines = [h for h in unique if h.get("priority") == 2]
week_headlines      = [h for h in unique if h.get("priority") == 3]

# ── Build output ──────────────────────────────────────────────
output = {
    "last_updated":       now_utc.strftime("%Y-%m-%d %H:%M UTC"),
    "date_range": {
        "from": week_ago.strftime("%Y-%m-%d"),
        "to":   today.strftime("%Y-%m-%d")
    },
    "total_headlines":    len(unique),
    "today_count":        len(today_headlines),
    "yesterday_count":    len(yesterday_headlines),
    "week_count":         len(week_headlines),
    "sentiment_score":    sentiment,
    "bullish_count":      bullish_count,
    "bearish_count":      bearish_count,
    "today_headlines":    today_headlines,
    "yesterday_headlines": yesterday_headlines,
    "week_headlines":     week_headlines,
    "headlines":          unique   # all combined — for backward compatibility
}

with open("news.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  news.json saved")
print(f"  Total   : {len(unique)} headlines")
print(f"  Today   : {len(today_headlines)}")
print(f"  Yesterday: {len(yesterday_headlines)}")
print(f"  This week: {len(week_headlines)}")
print(f"  Sentiment: {sentiment}")
print("\n  Done.")
