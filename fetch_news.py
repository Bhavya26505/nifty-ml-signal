import requests
import feedparser
import json
import os
from datetime import datetime, date

print("Starting news fetch...")

FEEDS = [
    "https://news.google.com/rss/search?q=NIFTY+50+India+stock&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=NSE+BSE+Indian+market&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=RBI+India+economy+market&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=FII+DII+India+stocks&hl=en-IN&gl=IN&ceid=IN:en",
]

today_str = date.today().strftime("%Y-%m-%d")
fetched_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

# Load existing news.json if it exists and is from today
existing_headlines = []
if os.path.exists("news.json"):
    try:
        with open("news.json", "r") as f:
            existing_data = json.load(f)
        if existing_data.get("date") == today_str:
            existing_headlines = existing_data.get("headlines", [])
            print(f"Loaded {len(existing_headlines)} existing headlines from today")
        else:
            print("New day — starting fresh headlines list")
    except:
        print("Could not read existing news.json — starting fresh")

# Collect existing titles to avoid duplicates
existing_titles = {h["title"] for h in existing_headlines}

# Fetch new headlines from all feeds
new_headlines = []
for feed_url in FEEDS:
    try:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:8]:
            title = entry.get("title", "").strip()
            link  = entry.get("link", "")
            pub   = entry.get("published", fetched_at)

            if not title or title in existing_titles:
                continue

            # Clean title — remove source suffix like " - Economic Times"
            if " - " in title:
                clean_title = title.rsplit(" - ", 1)[0].strip()
            else:
                clean_title = title

            new_headlines.append({
                "title":      clean_title,
                "source":     title.rsplit(" - ", 1)[-1].strip() if " - " in title else "News",
                "link":       link,
                "published":  pub,
                "fetched_at": fetched_at
            })
            existing_titles.add(title)

        print(f"Fetched from feed: {len(feed.entries)} entries")
    except Exception as e:
        print(f"Feed error: {e}")

# Combine old + new, keep max 60 headlines per day
all_headlines = existing_headlines + new_headlines
all_headlines = all_headlines[-60:]

print(f"Total headlines today: {len(all_headlines)} ({len(new_headlines)} new)")

# Categorise headlines for Gemini context
bullish_keywords  = ["rate cut","gdp growth","profit","rally","surge","gain","positive","record","high","buy","upgrade","boost"]
bearish_keywords  = ["rate hike","inflation","loss","crash","fall","decline","negative","sell","downgrade","risk","deficit","weak"]

bullish_count = sum(1 for h in all_headlines if any(k in h["title"].lower() for k in bullish_keywords))
bearish_count = sum(1 for h in all_headlines if any(k in h["title"].lower() for k in bearish_keywords))

sentiment_score = "MIXED"
if bullish_count > bearish_count * 1.5:
    sentiment_score = "BULLISH"
elif bearish_count > bullish_count * 1.5:
    sentiment_score = "BEARISH"

output = {
    "date":            today_str,
    "last_updated":    fetched_at,
    "total_headlines": len(all_headlines),
    "sentiment_score": sentiment_score,
    "bullish_count":   bullish_count,
    "bearish_count":   bearish_count,
    "headlines":       all_headlines
}

with open("news.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"news.json saved — {len(all_headlines)} headlines — sentiment: {sentiment_score}")
print("Done.")
