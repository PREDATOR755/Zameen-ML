"""
zameen_scraper.py – Enhanced for 1500+ listings
================================================
Scrapes property listings for Islamabad from Zameen.com.
Saves to zameen_islamabad.csv.

Improvements:
- Target 1500 listings, up to 60 pages
- Stops after 3 consecutive empty pages
- Longer delays every 5 pages to avoid rate limiting
- Optional location string cleaning (removes area units)
"""

import csv
import time
import random
import re
import logging
import sys
from dataclasses import dataclass, fields, astuple
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("zameen")

# ── Config ────────────────────────────────────────────────────────────────
BASE_URL        = "https://www.zameen.com"
TARGET_COUNT    = 1500          # Increased from 1000
MAX_PAGES       = 75            # Increased from 30
REQUEST_TIMEOUT = 20
MIN_DELAY       = 2
MAX_DELAY       = 4.5
RETRY_ATTEMPTS  = 3
OUTPUT_CSV      = "zameen_islamabad.csv"

# After every 5 pages, add an extra long delay (15-25 seconds)
LONG_DELAY_AFTER_PAGES = 5
LONG_DELAY_MIN = 15
LONG_DELAY_MAX = 25

# User agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
]

# ── Data model ────────────────────────────────────────────────────────────
@dataclass
class Listing:
    price:           Optional[str] = None
    area:            Optional[str] = None
    area_unit:       Optional[str] = None
    city:            str           = "Islamabad"
    location:        Optional[str] = None
    property_type:   Optional[str] = None
    bedrooms:        Optional[str] = None
    bathrooms:       Optional[str] = None
    built_in_year:   Optional[str] = None
    parking_spaces:  Optional[str] = None
    servant_quarters:Optional[str] = None
    store_rooms:     Optional[str] = None
    kitchens:        Optional[str] = None
    drawing_rooms:   Optional[str] = None
    listing_url:     Optional[str] = None

CSV_HEADER = [f.name for f in fields(Listing)]

# ── HTTP helpers ──────────────────────────────────────────────────────────
def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "DNT": "1",
    })
    return s

SESSION = _make_session()

def _get(url: str, params: dict = None) -> Optional[BeautifulSoup]:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            SESSION.headers["User-Agent"] = random.choice(USER_AGENTS)
            resp = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            if resp.status_code == 429:
                wait = 10 * attempt
                log.warning("Rate-limited (429). Waiting %ds …", wait)
                time.sleep(wait)
            elif resp.status_code in (403, 404):
                log.warning("HTTP %d for %s – skipping.", resp.status_code, url)
                return None
            else:
                log.warning("HTTP %d on attempt %d for %s", resp.status_code, attempt, url)
        except requests.RequestException as exc:
            log.warning("Request error on attempt %d: %s", attempt, exc)
        time.sleep(MIN_DELAY * attempt)
    return None

def _polite_sleep(extra_long=False):
    if extra_long:
        wait = random.uniform(LONG_DELAY_MIN, LONG_DELAY_MAX)
        log.info("Taking a longer break (%.1f seconds) to avoid rate limiting...", wait)
    else:
        wait = random.uniform(MIN_DELAY, MAX_DELAY)
    time.sleep(wait)

# ── Location cleaning helper (optional) ───────────────────────────────────
def _clean_location(raw: str) -> str:
    """Remove trailing area numbers and units like '561 Kanal' or '123 Marla'"""
    if not raw:
        return raw
    # Remove pattern: whitespace + digits + space + (Kanal|Marla) at the end
    cleaned = re.sub(r'\s+\d+\s*(Kanal|Marla)\s*$', '', raw, flags=re.I)
    # Also remove "Islamabad" + digits + unit if attached
    cleaned = re.sub(r',?\s*Islamabad\s*\d+\s*(Kanal|Marla)', '', cleaned)
    return cleaned.strip()

# ── Detail-page scraper (unchanged, but kept for completeness) ────────────
_FEATURE_MAP = {
    "bedrooms": "bedrooms",
    "bathrooms": "bathrooms",
    "build year": "built_in_year",
    "year built": "built_in_year",
    "parking spaces": "parking_spaces",
    "servant quarters": "servant_quarters",
    "store rooms": "store_rooms",
    "kitchens": "kitchens",
    "drawing rooms": "drawing_rooms",
    "floors": None,
}

def _scrape_detail(url: str, listing: Listing) -> Listing:
    soup = _get(url)
    if not soup:
        return listing

    full_text = soup.get_text(" ", strip=True)

    detail_patterns = {
        "built_in_year":   r"(?:Build|Built|Year\s+Built)[^\d]*(\d{4})",
        "parking_spaces":  r"Parking\s+Spaces?\s*[:\-]?\s*(\d+)",
        "servant_quarters":r"Servant\s+Quarters?\s*[:\-]?\s*(\d+)",
        "store_rooms":     r"Store\s+Rooms?\s*[:\-]?\s*(\d+)",
        "kitchens":        r"Kitchens?\s*[:\-]?\s*(\d+)",
        "drawing_rooms":   r"Drawing\s+Rooms?\s*[:\-]?\s*(\d+)",
    }
    for field, pattern in detail_patterns.items():
        if getattr(listing, field) is None:
            m = re.search(pattern, full_text, re.I)
            if m:
                setattr(listing, field, m.group(1))

    for dt in soup.select("dt"):
        label = dt.get_text(strip=True).lower().rstrip(":")
        dd = dt.find_next_sibling("dd")
        if not dd:
            continue
        value = dd.get_text(strip=True)
        field = _FEATURE_MAP.get(label)
        if field and getattr(listing, field) is None:
            setattr(listing, field, value)

    for el in soup.find_all(["li", "div", "span"]):
        label = el.get_text(strip=True).lower()
        for key, field in _FEATURE_MAP.items():
            if field and key in label and getattr(listing, field) is None:
                sibling = el.find_next_sibling()
                if sibling:
                    val = sibling.get_text(strip=True)
                    if val and len(val) < 30:
                        setattr(listing, field, val)

    return listing

# ── Page-level scraper (updated to clean location) ────────────────────────
def _scrape_page(page_num: int) -> tuple[list[Listing], bool]:
    url = f"https://www.zameen.com/Houses/Islamabad-3-{page_num}.html"
    log.info("Fetching page %d  →  %s", page_num, url)
    soup = _get(url)
    if not soup:
        log.warning("Page %d returned no content – stopping.", page_num)
        return [], False

    cards = soup.find_all("article")
    log.info("  Found %d article cards on page %d", len(cards), page_num)

    listings = []
    for card in cards:
        try:
            listing = Listing()
            link = card.find("a", href=True)
            if not link:
                continue
            href = link["href"]
            if "/Property/" not in href:
                continue
            listing.listing_url = href if href.startswith("http") else urljoin(BASE_URL, href)

            text = card.get_text(" ", strip=True)

            price_m = re.search(r"PKR\s*([\d.,]+\s*(?:Crore|Lakh|Million|Billion|Thousand)?)", text, re.I)
            if price_m:
                listing.price = price_m.group(1).strip()

            area_m = re.search(r"([\d.,]+)\s*(Marla|Kanal|Sq\.?\s*Ft\.?|Square\s*Feet|Square\s*Yard)", text, re.I)
            if area_m:
                listing.area = area_m.group(1).replace(",", "")
                listing.area_unit = area_m.group(2).strip()

            bed_m = re.search(r"(\d+)\s*Beds?", text, re.I)
            bath_m = re.search(r"(\d+)\s*Baths?", text, re.I)
            if bed_m:
                listing.bedrooms = bed_m.group(1)
            if bath_m:
                listing.bathrooms = bath_m.group(1)

            loc_tag = card.find(
                lambda t: t.name in ("span", "div")
                and "Islamabad" in t.get_text()
                and len(t.get_text(strip=True)) < 120
            )
            if loc_tag:
                raw_location = loc_tag.get_text(strip=True)
                # Clean location string (remove trailing area units)
                listing.location = _clean_location(raw_location)

            listing.property_type = "House"

            if not listing.price and not listing.area:
                continue

            listings.append(listing)

        except Exception as exc:
            log.warning("Card parse error: %s", exc)

    has_next = len(cards) > 0
    return listings, has_next

# ── Detail enrichment (unchanged) ─────────────────────────────────────────
def _enrich(listings: list[Listing]) -> list[Listing]:
    enriched = []
    for lst in tqdm(listings, desc="Enriching detail pages", unit="listing"):
        needs_detail = any([
            lst.built_in_year is None,
            lst.parking_spaces is None,
            lst.servant_quarters is None,
            lst.store_rooms is None,
            lst.kitchens is None,
            lst.drawing_rooms is None,
        ])
        if needs_detail and lst.listing_url:
            lst = _scrape_detail(lst.listing_url, lst)
            _polite_sleep(extra_long=False)  # normal delay between detail pages
        enriched.append(lst)
    return enriched

# ── CSV writer ────────────────────────────────────────────────────────────
def _save_csv(listings: list[Listing], path: str):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for lst in listings:
            writer.writerow(astuple(lst))
    log.info("Saved %d listings → %s", len(listings), path)

# ── Main with empty page detection and long delays ────────────────────────
def main():
    log.info("━━━  Zameen.com Islamabad Scraper (Enhanced) ━━━")
    log.info("Target: %d listings  |  Max pages: %d", TARGET_COUNT, MAX_PAGES)

    all_listings: list[Listing] = []
    empty_page_count = 0

    for page in range(1, MAX_PAGES + 1):
        page_listings, has_next = _scrape_page(page)
        if len(page_listings) == 0:
            empty_page_count += 1
            log.info("  Page %d returned 0 listings. Empty count: %d", page, empty_page_count)
            if empty_page_count >= 3:
                log.info("3 consecutive empty pages → stopping pagination.")
                break
        else:
            empty_page_count = 0
            all_listings.extend(page_listings)
            log.info("  Running total: %d / %d", len(all_listings), TARGET_COUNT)

        if len(all_listings) >= TARGET_COUNT:
            log.info("Target reached. Stopping pagination.")
            all_listings = all_listings[:TARGET_COUNT]
            break

        if not has_next:
            log.info("No further pages detected (no article tags).")
            break

        # After every LONG_DELAY_AFTER_PAGES pages, take a longer rest
        if page % LONG_DELAY_AFTER_PAGES == 0 and page < MAX_PAGES:
            _polite_sleep(extra_long=True)
        else:
            _polite_sleep(extra_long=False)

    if not all_listings:
        log.error("No listings scraped. Zameen HTML may have changed.")
        sys.exit(1)

    log.info("Starting detail-page enrichment for %d listings …", len(all_listings))
    all_listings = _enrich(all_listings)

    _save_csv(all_listings, OUTPUT_CSV)
    log.info("Done! ✔")

if __name__ == "__main__":
    main()
