import argparse
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

DEFAULT_START_PAGE = 1
DEFAULT_END_PAGE = 50


def build_list_page(region_slug: str, type_slug: str) -> str:
    # Adjust this path if the site uses a different URL order
    return f"{BASE_URL}/{type_slug}/{region_slug}/trang-{{}}"


def crawl_list_pages(
    start_page: int = DEFAULT_START_PAGE,
    end_page: int = DEFAULT_END_PAGE,
) -> None:
    for region_slug, type_slug in product(REGIONS, TYPES):
        list_page = build_list_page(region_slug, type_slug)

        for page in range(start_page, end_page + 1):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Alonhadat listing pages.")
    parser.add_argument("--start-page", type=int, default=DEFAULT_START_PAGE)
    parser.add_argument("--end-page", type=int, default=DEFAULT_END_PAGE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crawl_list_pages(start_page=args.start_page, end_page=args.end_page)
    print("Done. Detail scraping is handled by Link2details.py.")


if __name__ == "__main__":
    main()
