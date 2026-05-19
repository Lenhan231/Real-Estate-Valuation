import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from selenium import webdriver
from selenium.common import TimeoutException
from selenium.common import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

BASE_URL = "https://i-nhadat.com"
LIST_PAGE = BASE_URL + "/can-ban-nha-dat/ben-tre-t13/p{}.htm"

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

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--window-size=1280,1600")


BLOCKED_MARKERS = (
    "xác thực pageid",
    "vui lòng xác minh",
    "thông báo hiện tại website",
    "bị kẻ xấu",
    "máy của bạn đã truy cập quá nhiều",
    "vui lòng thử lại sau",
    "recaptchcontainer",
)

def fetch_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


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


def _safe_quit(driver):
    try:
        driver.quit()
    except WebDriverException:
        pass


def fetch_soup_with_manual_captcha(url: str, target_selector: str, captcha_selector: str = "[name='g-recaptcha-response']") -> BeautifulSoup:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        if _is_verification_page(soup, resp.text):
            raise TimeoutException()
        if soup.select_one(target_selector):
            return soup
        raise TimeoutException()
    except Exception:
        driver = webdriver.Chrome(options=chrome_options)
        try:
            driver.get(url)

            page_soup = BeautifulSoup(driver.page_source, "html.parser")
            if _is_verification_page(page_soup, driver.page_source):
                print(f"Captcha detected on: {url}")
                print("Hãy giải captcha / xác thực trong cửa sổ Chrome rồi quay lại terminal và nhấn Enter để tiếp tục.")
                try:
                    input()
                except EOFError:
                    pass

            wait = WebDriverWait(driver, 60)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, target_selector)))
            return BeautifulSoup(driver.page_source, "html.parser")
        except Exception as e:
            print(f"Không thể tải trang: {url}")
            print(e)
            return None
        finally:
            _safe_quit(driver)

def extract_list_page(page_url: str) -> list[dict]:
    soup = fetch_soup_with_manual_captcha(page_url, ".content-items .content-item")
    if soup is None:
        return []

    rows = []
    items = soup.select(".content-items .content-item")
    for item in items:
        # prefer the main link inside thumbnail or title; fall back to first anchor
        link_tag = item.select_one(".thumbnail a") or item.select_one(".ct_title a") or item.select_one("a")
        if not link_tag:
            continue

        href = link_tag.get("href", "").strip()
        if not href:
            continue

        rows.append({"link": urljoin(BASE_URL, href)})

    return rows


def extract_more_info(url: str):
    try:
        soup = fetch_soup_with_manual_captcha(url, ".moreinfor1, .property-details")
        if soup is None:
            return None

        data = {"link": url}

        title_tag = soup.select_one("h1, .title, .property-title")
        if title_tag:
            data["title"] = title_tag.get_text(" ", strip=True)

        price_tag = soup.select_one(".price [itemprop='price'], .price .value, .price")
        if price_tag:
            data["Giá"] = price_tag.get_text(" ", strip=True)

        area_tag = soup.select_one(".area [itemprop='value'], .square [itemprop='value'], .square .value, .area")
        if area_tag:
            data["Diện tích"] = area_tag.get_text(" ", strip=True)

        address_tag = soup.select_one(".address .value, .property-address [itemprop='addressLocality'], .property-address")
        if address_tag:
            data["Địa chỉ"] = address_tag.get_text(" ", strip=True)

        section = soup.select_one("section.moreinfor1, div.moreinfor1")
        if not section:
            return data

        table = section.find("table") or section
        rows = table.find_all("tr")

        for row in rows:
            cells = row.find_all("td")
            i = 0
            while i < len(cells) - 1:
                key = cells[i].get_text(strip=True)
                value = cells[i + 1].get_text(" ", strip=True)

                if key:
                    data[key] = value

                i += 2

        return data

    except Exception as e:
        print(f"ERROR: {url}")
        print(e)
        return None

def main():
    all_rows = []

    for page in range(1, 4):
        page_url = LIST_PAGE.format(page)
        print(f"Scraping: {page_url}")

        try:
            rows = extract_list_page(page_url)
            print(f"  found {len(rows)} listings")
            all_rows.extend(rows)
        except Exception as e:
            print(f"  failed on page {page}: {e}")

        time.sleep(random.randint(1, 3))

    df = pd.DataFrame(all_rows)

    if not df.empty:
        df = df.drop_duplicates(subset=["link"]).reset_index(drop=True)

    print(df)

    # write listings only (details will be fetched in a later step)
    df.to_csv("inhadat_listings.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} listings to inhadat_listings.csv. Detail extraction skipped.")

if __name__ == "__main__":
    main()