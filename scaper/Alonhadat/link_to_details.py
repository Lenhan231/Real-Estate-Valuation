import time
import random
import requests
import subprocess
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

from selenium.common import TimeoutException

INPUT_FILE = r".\data\raw\alonhadat_listings.csv"
OUTPUT_FILE = r".\data\raw\alonhadat_details.csv"
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

    # ---- contact info ----
    name_tag = soup.select_one(".contact-info .name")
    if name_tag:
        data["Tên liên hệ"] = _clean_value(name_tag.get_text(" ", strip=True))

    phone_tag = soup.select_one(".contact-info .fone a[href^='tel:']")
    if phone_tag:
        data["Số Điện Thoại"] = _clean_value(phone_tag.get_text(" ", strip=True))
    else:
        phone_fallback = soup.select_one(".contact-info .fone")
        if phone_fallback:
            data["Số Điện Thoại"] = _clean_value(phone_fallback.get_text(" ", strip=True))

    # ---- review score ----
    review_label = soup.select_one(".review-box .review-score")
    review_rate = soup.select_one(".review-box .review-star .rate")
    if review_label or review_rate:
        score_text = ""
        if review_label:
            score_text = _clean_value(review_label.get_text(" ", strip=True)).replace("Được đánh giá:", "").strip()

        if review_rate:
            style = review_rate.get("style", "")
            # width:60%; -> 3/5 stars if the site uses 5-star scaling
            if "width:" in style:
                width_part = style.split("width:", 1)[1].split(";", 1)[0].strip().replace("%", "")
                try:
                    pct = float(width_part)
                    data["được đánh giá"] = f"{pct/20:.1f}/5"
                except ValueError:
                    if score_text:
                        data["được đánh giá"] = score_text
            elif score_text:
                data["được đánh giá"] = score_text
        elif score_text:
            data["được đánh giá"] = score_text        
        # project extraction removed per user request

    desc_tag = soup.select_one(
    "section.detail.text-content p[itemprop='description']"
    )

    if desc_tag:
        data["Thông tin chi tiết"] = " ".join(
            desc_tag.get_text(" ", strip=True).split()
        )

    BASE_URL = "https://alonhadat.com.vn"

    img_tags = soup.select("ul.image-list li img[src]")

    # fallback for single-image pages
    if not img_tags:
        img_tags = soup.select("section.images div.imageview img[src]")

    images = [
        urljoin(BASE_URL, img["src"])
        for img in img_tags
    ]

    data["Hình ảnh"] = images

        # also check anywhere else for project link
    # project extraction intentionally omitted

    return data

def _clean_key(text: str) -> str:
    return text.replace(":", "").strip()

def _clean_value(text: str) -> str:
    return " ".join(text.split()).strip()

def fetch(url: str) -> (BeautifulSoup | None):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 404:
            return None, "404"
        resp.raise_for_status()
        if looks_blocked(resp.text):
            # attempt manual resolution via Selenium
            return fetch_via_selenium_manual(url), "OK"
        return BeautifulSoup(resp.text, "html.parser"), "OK"
    except Exception as e:
        # on network/error, try manual Selenium path
        print(f"Failed fetch {url}: {e}")
        return fetch_via_selenium_manual(url), "OK"


def fetch_via_selenium_manual(url: str) -> (BeautifulSoup | None):
    print(f"Retrying with WARP reconnect: {url}")

    try:
        print(run(["warp-cli", "disconnect"]))
        time.sleep(2)

        print(run(["warp-cli", "connect"]))
        time.sleep(5)

    except Exception as e:
        print("WARP reconnect failed:", e)
        return None

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()

        if looks_blocked(resp.text):
            print("Still blocked after reconnect.")
            return None

        return BeautifulSoup(resp.text, "html.parser")

    except Exception as e:
        print("Retry fetch failed:", e)
        return None

def link_to_detail():
    try:
        df = pd.read_csv(INPUT_FILE)
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
            df.to_csv(INPUT_FILE, index=False, encoding="utf-8-sig")
            continue

        print(f"Processing (remaining {len(df)}): {url}")
        soup, status = fetch(url)
        if soup is None:
            # Retry fetch failed: 404 Client Error: Not Found for url delete
            if status == "404":
                print("  link not found (404), removing from listings")
                df = df.iloc[1:].reset_index(drop=True)
                df.to_csv(INPUT_FILE, index=False, encoding="utf-8-sig")
                continue
            else:
                print("  skipped (blocked or fetch error)")
                # move this link to the end to allow others to be processed
                if len(df) == 1:
                    print("Only one link left and it is blocked; stopping to avoid tight loop.")
                    break
                first = df.iloc[0:1]
                df = pd.concat([df.iloc[1:].reset_index(drop=True), first]).reset_index(drop=True)
                df.to_csv(INPUT_FILE, index=False, encoding="utf-8-sig")
                time.sleep(random.uniform(1, 2))
                continue

        data = dict(listing_row)
        data["link"] = url
        detail = parse_detail_page(soup)
        data.update(detail)

        out_df = pd.DataFrame([data])

        # enforce canonical column order and align with existing file if present
        canonical = [
            "link", "title", "post_day", "price", "street", "locality", "region", "area", "old_address",
            "Mã tin", "Hướng", "Phòng ăn", "Loại tin", "Đường trước nhà", "Nhà bếp",
            "Loại BDS", "Pháp lý", "Sân thượng", "Chiều ngang", "Số lầu", "Chổ để xe hơi",
            "Chiều dài", "Số phòng ngủ", "Chính chủ", "Giá", "Thông tin chi tiết",
            "Hình ảnh", "Số Điện Thoại", "được đánh giá"
        ]

        # existing file -> reuse its header order
        details_path = OUTPUT_FILE
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
        df.to_csv(INPUT_FILE, index=False, encoding="utf-8-sig")
        print("  saved detail and removed from listings")
        time.sleep(random.uniform(1, 2))

    print("Processing finished.")


if __name__ == "__main__":
    main()
