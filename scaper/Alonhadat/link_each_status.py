import time
import random
import requests
import subprocess
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

from selenium.common import TimeoutException

INPUT_PAGE = 1
BASE_URL = "https://alonhadat.com.vn"
LIST_PAGE = BASE_URL + "/can-ban-nha/ho-chi-minh/trang-{}" # changing the url for others region if need
OUTPUT_FILE = r"data\raw\alonhadat_listings.csv"

# Simulator for not 403 forbidden 
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# check if the website markers blocked
BLOCKED_MARKERS = (
    "xác thực pageid",
    "vui lòng xác minh",
    "thông báo hiện tại website",
    "bị kẻ xấu",
    "máy của bạn đã truy cập quá nhiều",
    "vui lòng thử lại sau",
    "recaptchcontainer",
)
def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()

def _looks_blocked(html_text: str) -> bool:
    lowered = html_text.lower()
    return any(marker in lowered for marker in BLOCKED_MARKERS)


def _is_verification_page(soup: BeautifulSoup, html_text: str) -> bool:
    title = (soup.title.get_text(" ", strip=True).lower() if soup.title else "")
    if "vui lòng xác minh không phải robot" in title:
        return True

    form = soup.select_one("form[action*='xac-thuc-nguoi-dung.html']")
    captcha_input = soup.select_one("input[name='captcha']")
    captcha_image = soup.select_one("img.captchagenerator, img[src*='ImageCaptcha.ashx']")
    verify_button = soup.select_one("#verify")

    if form and captcha_input and captcha_image and verify_button:
        return True

    lowered = html_text.lower()
    return "xác minh \"tôi không phải người máy\"" in lowered or "xác thực" in lowered and "imagecaptcha.ashx" in lowered

def fetch_soup_with_manual_captcha(
    url: str,
    target_selector: str,
    captcha_selector: str = "[name='g-recaptcha-response']",
    max_retries: int = 3,
    retry_delay: int = 5,
) -> BeautifulSoup | None:
    last_error = None

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            if _is_verification_page(soup, resp.text):
                raise TimeoutException("Verification page detected")

            if soup.select_one(target_selector):
                return soup

            raise TimeoutException("Target selector not found")

        except Exception as e:
            last_error = e

            try:
                print(run(["warp-cli", "disconnect"]))
                print(run(["warp-cli", "connect"]))
            except EOFError:
                pass
            except Exception as warp_error:
                print(f"WARP reconnect failed: {warp_error}")

            if attempt < max_retries - 1:
                time.sleep(retry_delay)

    print(f"Không thể tải trang: {url}")
    print(last_error)
    return None

def extract_list_page(page_url: str) -> list[dict]:
    soup = fetch_soup_with_manual_captcha(page_url, "article.property-item")
    if soup is None:
        return []

    rows = []

    articles = soup.select("article.property-item")
    for article in articles:
        link_tag = article.select_one("a.link")
        new_addr_tag = article.select_one("p.new-address")
        old_addr_tag = article.select_one("p.old-address")

        title_tag = article.select_one("h3.property-title")
        post_day_tag = article.select_one("time.created-date")
        price_tag = article.select_one("span.price [itemprop='price'], span.price")
        area_tag = article.select_one("span.area [itemprop='value'], span.area")

        if not link_tag:
            continue

        new_street_tag = new_addr_tag.select_one("span[itemprop='streetAddress']") if new_addr_tag else None
        new_locality_tag = new_addr_tag.select_one("span[itemprop='addressLocality']") if new_addr_tag else None
        new_region_tag = new_addr_tag.select_one("span[itemprop='addressRegion']") if new_addr_tag else None

        old_address = old_addr_tag.get_text(" ", strip=True) if old_addr_tag else None

        price_value = None
        if price_tag:
            price_value = price_tag.get("content")
            if price_value is None:
                price_value = price_tag.get_text(" ", strip=True)

        area_value = None
        if area_tag:
            area_value = area_tag.get_text(" ", strip=True).replace("Diện tích:", "").strip()

        rows.append({
            "link": urljoin(BASE_URL, link_tag.get("href", "").strip()),
            "title": title_tag.get_text(" ", strip=True) if title_tag else None,
            "post_day": post_day_tag.get("datetime") if post_day_tag else None,
            "price": price_value,
            "street": new_street_tag.get_text(strip=True) if new_street_tag else None,
            "locality": new_locality_tag.get_text(strip=True) if new_locality_tag else None,
            "region": new_region_tag.get_text(strip=True) if new_region_tag else None,
            "old_address": old_address,
            "area": area_value,
        })

    return rows

def save_listings(rows: list[dict]) -> None:
    new_df = pd.DataFrame(rows)

    if os.path.exists(OUTPUT_FILE):
        old_df = pd.read_csv(OUTPUT_FILE)
        df = pd.concat([old_df, new_df], ignore_index=True)
        df = df.drop_duplicates(subset=["link"], keep="first").reset_index(drop=True)
    else:
        df = new_df.drop_duplicates(subset=["link"], keep="first").reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


def main():
    for page in range(INPUT_PAGE, INPUT_PAGE + 50):
        page_url = LIST_PAGE.format(page)
        print(f"Scraping: {page_url}")

        try:
            rows = extract_list_page(page_url)
            print(f"  found {len(rows)} listings")
            if rows:
                save_listings(rows)
        except Exception as e:
            print(f"  failed on page {page}: {e}")

        time.sleep(random.randint(1, 3))

    print("Done. Detail scraping is handled by Link2details.py.")

if __name__ == "__main__":
    main()