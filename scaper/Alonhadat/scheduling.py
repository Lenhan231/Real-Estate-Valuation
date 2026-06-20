import random
import time
from itertools import product
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scaper.Alonhadat.link_each_status import extract_list_page, save_listings
else:
    from .link_each_status import extract_list_page, save_listings

BASE_URL = "https://alonhadat.com.vn"

REGIONS = [
    "ho-chi-minh",
]

TYPES = [
    "can-ban-nha-mat-tien",
    "can-ban-nha-trong-hem",
]

START_PAGE = 1
END_PAGE = START_PAGE + 50


def build_list_page(region_slug: str, type_slug: str) -> str:
    # Adjust this path if the site uses a different URL order
    return f"{BASE_URL}/{type_slug}/{region_slug}/trang-{{}}"


def crawl_list_pages():
    for region_slug, type_slug in product(REGIONS, TYPES):
        list_page = build_list_page(region_slug, type_slug)

        for page in range(START_PAGE, END_PAGE + 1):
            page_url = list_page.format(page)
            print(f"Scraping: {page_url}")

            try:
                rows = extract_list_page(page_url)
                print(f"  found {len(rows)} listings")
                if rows:
                    save_listings(rows)
            except Exception as e:
                print(f"  failed on {region_slug} / {type_slug} / page {page}: {e}")

            time.sleep(random.randint(1, 3))


def main():
    crawl_list_pages()


if __name__ == "__main__":
    main()