import os
import re
import csv
import sys
import time
import json
import logging
from datetime import datetime
from typing import List, Dict, Callable

import requests
from bs4 import BeautifulSoup


DATA_DIR = os.path.dirname(__file__)
OUT_CSV = os.path.join(DATA_DIR, "recommended_picks.csv")
LOG_FILE = os.path.join(DATA_DIR, "get_predictions.log")


logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/119.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

TIMEOUT = 20


def fetch_url(url: str) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            logging.warning(f"Fetch non-200 for {url}: {resp.status_code}")
            return ""
        return resp.text
    except Exception as e:
        logging.error(f"Fetch failed for {url}: {e}")
        return ""


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


NBA_TEAM_CODES = {
    "ATL","BOS","BKN","CHA","CHI","CLE","DAL","DEN","DET","GSW","HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK","OKC","ORL","PHI","PHX","POR","SAC","SAS","TOR","UTA","WAS"
}
NBA_TEAM_NAMES = {
    "hawks","celtics","nets","hornets","bulls","cavaliers","mavericks","nuggets","pistons","warriors","rockets","pacers","clippers","lakers","grizzlies","heat","bucks","timberwolves","pelicans","knicks","thunder","magic","76ers","sixers","suns","trail blazers","blazers","kings","spurs","raptors","jazz","wizards"
}


def looks_like_pick(text: str) -> bool:
    t = text.lower()
    if len(text) < 10 or len(text) > 280:
        return False
    # Must contain a matchup cue or team
    has_matchup = (" vs " in t) or (" @ " in t) or any(code in text for code in NBA_TEAM_CODES) or any(name in t for name in NBA_TEAM_NAMES)
    if not has_matchup:
        return False
    # Must include pick-like keywords or market cues
    cues = ["pick", "prediction", "best bet", "over", "under", "ml", "moneyline", "spread", " ats ", " -", "+", " o ", " u "]
    if not any(c in t for c in cues):
        return False
    # Exclude obvious boilerplate
    excludes = ["terms and conditions", "gambling problem", "promo", "bet now", "how to", "what are", "how often"]
    if any(x in t for x in excludes):
        return False
    return True


def extract_candidates_by_keywords(soup: BeautifulSoup, keywords: List[str]) -> List[str]:
    texts = []
    for el in soup.find_all(text=True):
        try:
            t = clean_text(str(el))
        except Exception:
            continue
        low = t.lower()
        if any(k in low for k in keywords):
            if looks_like_pick(t):
                texts.append(t)
    # Dedup while preserving order
    seen = set()
    result = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def parse_cbssports(html: str, url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    items = []
    # Prefer table rows in expert picks tables
    rows = soup.select("table tr")
    for tr in rows:
        txt = clean_text(tr.get_text(" "))
        if looks_like_pick(txt):
            items.append({"source": "cbssports", "title": "Expert Pick", "pick": txt, "url": url})
    # Fallback keyword scan
    if not items:
        picks = extract_candidates_by_keywords(soup, ["pick", "prediction", "against the spread", "ats"]) or []
        for p in picks[:15]:
            items.append({"source": "cbssports", "title": "Expert Pick", "pick": p, "url": url})
    for p in picks[:30]:
        pass
    return items


def parse_oddsshark(html: str, url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    items = []
    # Look for computer picks modules
    blocks = soup.select(".computer-picks, .pick, .prediction, article, .module, .matchup, .score, .panel")
    if not blocks:
        # fallback to keyword scan
        texts = extract_candidates_by_keywords(soup, ["computer pick", "consensus", "prediction"]) or []
        for t in texts[:20]:
            items.append({"source": "oddsshark", "title": "Computer Pick", "pick": t, "url": url})
        return items
    for b in blocks:
        t = clean_text(b.get_text(" "))
        if t and looks_like_pick(t):
            items.append({"source": "oddsshark", "title": "Computer Pick", "pick": t[:500], "url": url})
    return items[:30]


def parse_pickswise(html: str, url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    items = []
    cards = soup.select("article, .PickCard, .prediction-card, .event, .matchup")
    if not cards:
        texts = extract_candidates_by_keywords(soup, ["pick", "prediction", "best bet"]) or []
        for t in texts[:25]:
            items.append({"source": "pickswise", "title": "Pick", "pick": t, "url": url})
        return items
    for c in cards:
        t = clean_text(c.get_text(" "))
        if t and looks_like_pick(t):
            items.append({"source": "pickswise", "title": "Pick", "pick": t[:500], "url": url})
    return items[:30]


def parse_basketballsphere(html: str, url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    items = []
    sections = soup.select("article, .content, .post, .entry-content")
    if not sections:
        texts = extract_candidates_by_keywords(soup, ["prop", "over", "under", "assist", "rebounds", "points"]) or []
        for t in texts[:30]:
            items.append({"source": "basketballsphere", "title": "Player Props", "pick": t, "url": url})
        return items
    for s in sections:
        t = clean_text(s.get_text(" "))
        if t and looks_like_pick(t):
            items.append({"source": "basketballsphere", "title": "Player Props", "pick": t[:800], "url": url})
    return items[:30]


def parse_covers(html: str, url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    items = []
    blocks = soup.select("article, .pick, .expert-picks, .prediction, .content, .event, .matchup")
    if not blocks:
        texts = extract_candidates_by_keywords(soup, ["pick", "prediction", "best bet", "ATS"]) or []
        for t in texts[:25]:
            items.append({"source": "covers", "title": "Pick", "pick": t, "url": url})
        return items
    for b in blocks:
        t = clean_text(b.get_text(" "))
        if t and looks_like_pick(t):
            items.append({"source": "covers", "title": "Pick", "pick": t[:800], "url": url})
    return items[:30]


def parse_actionnetwork(html: str, url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    items = []
    blocks = soup.select("article, .css-*, .pick, .prediction")
    texts = extract_candidates_by_keywords(soup, ["pick", "best bet", "projection", "model", "edge"]) or []
    for t in texts[:30]:
        items.append({"source": "actionnetwork", "title": "Pick", "pick": t, "url": url})
    # Prefer article text if present
    for b in blocks[:5]:
        t = clean_text(b.get_text(" "))
        if looks_like_pick(t):
            items.append({"source": "actionnetwork", "title": "Article", "pick": t[:800], "url": url})
    return items[:30]


def parse_betfirm(html: str, url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    items = []
    blocks = soup.select("article, .post, .entry-content")
    if not blocks:
        texts = extract_candidates_by_keywords(soup, ["free pick", "prediction", "ATS", "vs."]) or []
        for t in texts[:25]:
            items.append({"source": "betfirm", "title": "Free Pick", "pick": t, "url": url})
        return items
    for b in blocks:
        t = clean_text(b.get_text(" "))
        if len(t) > 30:
            items.append({"source": "betfirm", "title": "Free Pick", "pick": t[:800], "url": url})
    return items[:30]


def parse_sportsgambler(html: str, url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    items = []
    blocks = soup.select("article, .predictions, .betting-tips, .content, .match, .event")
    if not blocks:
        texts = extract_candidates_by_keywords(soup, ["predictions", "betting tips", "pick"]) or []
        for t in texts[:20]:
            items.append({"source": "sportsgambler", "title": "Prediction", "pick": t, "url": url})
        return items
    for b in blocks:
        t = clean_text(b.get_text(" "))
        if looks_like_pick(t):
            items.append({"source": "sportsgambler", "title": "Prediction", "pick": t[:800], "url": url})
    return items[:30]


def parse_olbg(html: str, url: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    items = []
    blocks = soup.select("article, .picks, .tip, .content, .prediction, .event, .matchup")
    if not blocks:
        texts = extract_candidates_by_keywords(soup, ["tip", "prediction", "pick", "best bet"]) or []
        for t in texts[:25]:
            items.append({"source": "olbg", "title": "Tip", "pick": t, "url": url})
        return items
    for b in blocks:
        t = clean_text(b.get_text(" "))
        if looks_like_pick(t):
            items.append({"source": "olbg", "title": "Tip", "pick": t[:800], "url": url})
    return items[:30]


SITES: List[Dict[str, str]] = [
    {"url": "https://www.cbssports.com/nba/expert-picks/", "parser": "cbssports"},
    {"url": "https://www.oddsshark.com/nba/computer-picks", "parser": "oddsshark"},
    {"url": "https://www.pickswise.com/nba/picks/", "parser": "pickswise"},
    {"url": "https://basketballsphere.com/en/nba-player-props-nov-11-2025/", "parser": "basketballsphere"},
    {"url": "https://www.covers.com/picks/nba", "parser": "covers"},
    {"url": "https://www.actionnetwork.com/nba/nba-picks-odds-props-predictions-for-tuesday-november-11", "parser": "actionnetwork"},
    {"url": "https://www.betfirm.com/free-nba-picks/", "parser": "betfirm"},
    {"url": "https://www.sportsgambler.com/betting-tips/basketball/nba-predictions/", "parser": "sportsgambler"},
    {"url": "https://www.olbg.com/betting-tips/Basketball/NBA/4", "parser": "olbg"},
]


PARSERS: Dict[str, Callable[[str, str], List[Dict]]] = {
    "cbssports": parse_cbssports,
    "oddsshark": parse_oddsshark,
    "pickswise": parse_pickswise,
    "basketballsphere": parse_basketballsphere,
    "covers": parse_covers,
    "actionnetwork": parse_actionnetwork,
    "betfirm": parse_betfirm,
    "sportsgambler": parse_sportsgambler,
    "olbg": parse_olbg,
}


def run() -> List[Dict]:
    all_rows: List[Dict] = []
    ts = datetime.utcnow().isoformat()
    for site in SITES:
        url = site["url"]
        name = site["parser"]
        parser = PARSERS.get(name)
        if not parser:
            continue
        logging.info(f"Fetching {name}: {url}")
        html = fetch_url(url)
        if not html:
            logging.warning(f"Empty HTML for {url}")
            continue
        try:
            rows = parser(html, url)
        except Exception as e:
            logging.error(f"Parser failed for {name}: {e}")
            rows = []
        for r in rows:
            r = {**r}
            r["scraped_at_utc"] = ts
            all_rows.append(r)
        # polite delay
        time.sleep(1.0)
    return all_rows


def write_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        logging.info("No rows to write")
        return
    # union of keys for header
    keys = set()
    for r in rows:
        keys.update(r.keys())
    header = [
        "source", "title", "pick", "url", "scraped_at_utc",
    ] + sorted([k for k in keys if k not in {"source", "title", "pick", "url", "scraped_at_utc"}])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    try:
        rows = run()
        write_csv(rows, OUT_CSV)
        print(f"Saved {len(rows)} rows to {OUT_CSV}")
    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        print("Scrape failed. See log:", LOG_FILE, file=sys.stderr)
