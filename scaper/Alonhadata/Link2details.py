import time
import random
import requests
import subprocess
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import WebDriverException
import os

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
}

BLOCKED_MARKERS = [
    "vui lòng xác minh",
    "máy của bạn đã truy cập quá nhiều",
    "imagecaptcha.ashx",
]

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--window-size=1280,1600")

def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()

def looks_blocked(html: str) -> bool:
    lowered = html.lower()
    return any(m in lowered for m in BLOCKED_MARKERS)


def parse_detail_page(soup: BeautifulSoup) -> dict:
    data = {}

    # find table likely containing property meta
    tables = soup.find_all("table")
    target_table = None
    for tbl in tables:
        txt = tbl.get_text(" ", strip=True).lower()
        if any(k in txt for k in ("mã tin", "ngày đăng", "diện tích")):
            target_table = tbl
            break

    if not target_table and tables:
        target_table = tables[0]

    if target_table:
        rows = target_table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            # pairwise keys/values across the row
            i = 0
            while i < len(cells):
                key = cells[i].get_text(" ", strip=True)
                # normalize key
                key = key.replace(':', '').strip()
                val = ""
                if i + 1 < len(cells):
                    val = cells[i + 1].get_text(" ", strip=True)

                if key:
                    data[key] = val

                i += 2

    # address
    addr = None
    adr_tag = soup.select_one("div.address .value") or soup.select_one("div.address")
    if adr_tag:
        addr = adr_tag.get_text(" ", strip=True)
        data["Địa chỉ"] = addr

    # price fallback
    if "Giá" not in data:
        price_tag = soup.select_one(".price, .ct_price")
        if price_tag:
            data["Giá"] = price_tag.get_text(" ", strip=True)
    else:
        # normalize price value if it contains prefix like 'Giá:'
        if isinstance(data.get("Giá"), str) and data.get("Giá").lower().startswith("giá"):
            # remove leading 'Giá:' or similar
            data["Giá"] = data["Giá"].split(":", 1)[-1].strip()

    # some pages include an extra section with more info (project, duplicate blocks)
    section = soup.select_one("section.moreinfor1, div.moreinfor1")
    if section:
        table = section.find("table")
        if table:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                i = 0
                while i < len(cells):
                    key = cells[i].get_text(" ", strip=True)
                    key = key.replace(':', '').strip()
                    val = ""
                    if i + 1 < len(cells):
                        val = cells[i + 1].get_text(" ", strip=True)

                    if key:
                        if not data.get(key):
                            data[key] = val
                    i += 2

        # project extraction removed per user request

    # also check anywhere else for project link
    # project extraction intentionally omitted

    return data


def fetch(url: str) -> (BeautifulSoup | None):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        if looks_blocked(resp.text):
            # attempt manual resolution via Selenium
            return fetch_via_selenium_manual(url)
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        # on network/error, try manual Selenium path
        print(f"Failed fetch {url}: {e}")
        return fetch_via_selenium_manual(url)


def fetch_via_selenium_manual(url: str) -> (BeautifulSoup | None):
    print(f"Opening browser for manual verification: {url}")
    try:
        driver = webdriver.Chrome(options=chrome_options)
    except WebDriverException as e:
        print("Could not start Chrome WebDriver:", e)
        return None

    try:
        driver.get(url)

        # initial check
        page_html = driver.page_source
        if not looks_blocked(page_html):
            return BeautifulSoup(page_html, "html.parser")

        try:
            print(run(["warp-cli", "disconnect"]))
            print(run(["warp-cli", "connect"]))   
        except EOFError:
            ans = ""

        if ans == "s":
            return None

        # wait for user to solve — poll page source for removal of block markers
        timeout_seconds = 300
        poll = 5
        waited = 0
        while waited < timeout_seconds:
            time.sleep(poll)
            waited += poll
            page_html = driver.page_source
            if not looks_blocked(page_html):
                return BeautifulSoup(page_html, "html.parser")

        print("Timed out waiting for manual verification.")
        return None
    except WebDriverException as e:
        print("Selenium error while loading page:", e)
        return None
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def main():
    try:
        df = pd.read_csv(r"HousePricePrediction\data\raw\alonhadat_listings.csv")
    except Exception as e:
        print("Could not read alonhadat_listings.csv:", e)
        return
    # Process links in a resilient pipeline:
    # - For each link: try fetch and parse; on success append to details CSV and remove the link from listings CSV
    # - On blocked/fetch failure: move the link to the end of the listings CSV for retry later (do not delete)
    if df.empty:
        print("No links to process in alonhadat_listings.csv")
        return

    while not df.empty:
        listing_row = df.iloc[0].to_dict()
        url = listing_row.get("link")
        if not url or not isinstance(url, str):
            # drop malformed row
            df = df.iloc[1:].reset_index(drop=True)
            df.to_csv(r"HousePricePrediction\data\raw\alonhadat_listings.csv", index=False, encoding="utf-8-sig")
            continue

        print(f"Processing (remaining {len(df)}): {url}")
        soup = fetch(url)
        if soup is None:
            print("  skipped (blocked or fetch error)")
            # move this link to the end to allow others to be processed
            if len(df) == 1:
                print("Only one link left and it is blocked; stopping to avoid tight loop.")
                break
            first = df.iloc[0:1]
            df = pd.concat([df.iloc[1:].reset_index(drop=True), first]).reset_index(drop=True)
            df.to_csv(r"HousePricePrediction\data\raw\alonhadat_listings.csv", index=False, encoding="utf-8-sig")
            time.sleep(random.uniform(1, 2))
            continue

        data = dict(listing_row)
        data["link"] = url
        detail = parse_detail_page(soup)
        data.update(detail)

        out_df = pd.DataFrame([data])

        # enforce canonical column order and align with existing file if present
        canonical = [
            "link", "street", "locality", "region", "Diện tích",
            "Mã tin", "Hướng", "Phòng ăn", "Loại tin", "Đường trước nhà", "Nhà bếp",
            "Loại BDS", "Pháp lý", "Sân thượng", "Chiều ngang", "Số lầu", "Chổ để xe hơi",
            "Chiều dài", "Số phòng ngủ", "Chính chủ", "Giá",
        ]

        # existing file -> reuse its header order
        details_path = r"HousePricePrediction\data\raw\alonhadat_details.csv"
        if os.path.exists(details_path):
            try:
                existing_df = pd.read_csv(details_path)
                combined = pd.concat([existing_df, out_df], ignore_index=True, sort=False)
                ordered_cols = [c for c in canonical if c in combined.columns] + [c for c in combined.columns if c not in canonical]
                combined = combined.reindex(columns=ordered_cols)
                combined.to_csv(details_path, index=False, encoding="utf-8-sig")
            except Exception:
                # fallback: append with columns present in this row
                out_df.to_csv(details_path, mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            # write with canonical first, then any extra columns
            desired = [c for c in canonical if c in out_df.columns]
            extras = [c for c in out_df.columns if c not in desired]
            cols = desired + extras
            out_df = out_df.reindex(columns=cols)
            out_df.to_csv(details_path, mode="w", header=True, index=False, encoding="utf-8-sig")

        # remove processed link from listings and persist progress
        df = df.iloc[1:].reset_index(drop=True)
        df.to_csv(r"HousePricePrediction\data\raw\alonhadat_listings.csv", index=False, encoding="utf-8-sig")
        print("  saved detail and removed from listings")
        time.sleep(random.uniform(1, 2))

    print("Processing finished.")


if __name__ == "__main__":
    main()
