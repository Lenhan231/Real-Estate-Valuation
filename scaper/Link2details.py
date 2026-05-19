import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
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

    return data


def fetch(url: str) -> (BeautifulSoup | None):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        if looks_blocked(resp.text):
            print(f"Blocked or verification page detected, skipping: {url}")
            return None
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"Failed fetch {url}: {e}")
        return None


def main():
    try:
        df = pd.read_csv("inhadat_listings.csv")
    except Exception as e:
        print("Could not read inhadat_listings.csv:", e)
        return
    # Process links in a resilient pipeline:
    # - For each link: try fetch and parse; on success append to details CSV and remove the link from listings CSV
    # - On blocked/fetch failure: move the link to the end of the listings CSV for retry later (do not delete)
    if df.empty:
        print("No links to process in inhadat_listings.csv")
        return

    while not df.empty:
        url = df.iloc[0].get("link")
        if not url or not isinstance(url, str):
            # drop malformed row
            df = df.iloc[1:].reset_index(drop=True)
            df.to_csv("inhadat_listings.csv", index=False, encoding="utf-8-sig")
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
            df.to_csv("inhadat_listings.csv", index=False, encoding="utf-8-sig")
            time.sleep(random.uniform(1, 2))
            continue

        data = {"link": url}
        detail = parse_detail_page(soup)
        data.update(detail)

        out_df = pd.DataFrame([data])
        # append to details CSV (create if missing)
        if os.path.exists("inhadat_details.csv"):
            out_df.to_csv("inhadat_details.csv", mode="a", header=False, index=False, encoding="utf-8-sig")
        else:
            out_df.to_csv("inhadat_details.csv", mode="w", header=True, index=False, encoding="utf-8-sig")

        # remove processed link from listings and persist progress
        df = df.iloc[1:].reset_index(drop=True)
        df.to_csv("inhadat_listings.csv", index=False, encoding="utf-8-sig")
        print("  saved detail and removed from listings")
        time.sleep(random.uniform(1, 2))

    print("Processing finished.")


if __name__ == "__main__":
    main()
