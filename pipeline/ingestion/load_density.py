import pandas as pd
import requests
from bs4 import BeautifulSoup


HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
}

SCRAPE_XA_URL = 'https://vi.wikipedia.org/wiki/Danh_s%C3%A1ch_x%C3%A3_t%E1%BA%A1i_Vi%E1%BB%87t_Nam'
SCRAPE_PHUONG_URL = 'https://vi.wikipedia.org/wiki/Danh_s%C3%A1ch_ph%C6%B0%E1%BB%9Dng_t%E1%BA%A1i_Vi%E1%BB%87t_Nam'

REGIONS = ["Hà Nội", "Thành phố Hồ Chí Minh"]
REGION_MAPPING = {
    "thành phố hồ chí minh": "hồ chí minh",
    "hà nội": "hà nội"
}

COLUMN_RENAME = {
    "Tên": "locality",
    "Thuộc tỉnh/thành phố": "region",
    "Diện tích (km²)": "locality_square",
    "Mật độ dân số (người/km²)": "locality_population_density"
}


def _scrape_wiki_table(url: str) -> pd.DataFrame:
    response = requests.get(url, headers=HEADERS, timeout=30)
    print(f'HTTP status: {response.status_code}')

    if response.status_code != 200:
        raise Exception(f'Failed to fetch the page: {response.status_code}')

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.select_one('table.wikitable.sortable') or soup.find('table')

    if table is None:
        raise Exception('Không tìm thấy bảng nào trên trang.')

    rows = []
    for tr in table.find_all('tr'):
        cells = [cell.get_text(' ', strip=True) for cell in tr.find_all(['th', 'td'])]
        if cells:
            rows.append(cells)

    if len(rows) < 2:
        raise Exception('Bảng không có đủ dữ liệu để chuyển thành DataFrame.')

    df = pd.DataFrame(rows[1:], columns=rows[0])
    df = df.replace(r'\s*\[\d+\]', '', regex=True)

    return df


def _clean_density_data(df: pd.DataFrame, locality_type: str, population_col: str, drop_cols: list) -> pd.DataFrame:
    df = df[df["Thuộc tỉnh/thành phố"].isin(REGIONS)].copy()

    df["Thuộc tỉnh/thành phố"] = (
        df["Thuộc tỉnh/thành phố"]
        .str.strip()
        .str.lower()
        .replace(REGION_MAPPING)
    )

    df["Tên"] = (
        locality_type + " " +
        df["Tên"]
        .str.strip()
        .str.lower()
    )

    df = df.drop(columns=drop_cols)

    rename_cols = COLUMN_RENAME.copy()
    if population_col in rename_cols:
        rename_cols[population_col] = "locality_population"

    df = df.rename(columns=rename_cols)

    return df


def _normalize_case(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        df[col] = df[col].str.lower()
    return df


def load_density() -> pd.DataFrame:
    print("Scraping density xã from Wikipedia...")
    density_xa = _scrape_wiki_table(SCRAPE_XA_URL)
    density_xa = _clean_density_data(
        density_xa,
        locality_type="xã",
        population_col="Dân số năm 2025 (người)",
        drop_cols=["STT", "Năm thành lập", "Dân số năm 2025 (người)"]
    )

    print("Scraping density phường from Wikipedia...")
    density_phuong = _scrape_wiki_table(SCRAPE_PHUONG_URL)
    density_phuong = _clean_density_data(
        density_phuong,
        locality_type="phường",
        population_col="Dân số năm 2024 (người)",
        drop_cols=["STT", "Năm thành lập", "Loại đô thị [ 2 ] (năm công nhận)", "Dân số năm 2024 (người)"]
    )

    density = pd.concat([density_xa, density_phuong], ignore_index=True)
    print(f"✓ Loaded {len(density)} density records")

    return density


def merge_density_with_alonhadat(
    df: pd.DataFrame,
    density_df: pd.DataFrame
) -> pd.DataFrame:
    
    df = _normalize_case(df, ["locality", "region"])
    density_df = _normalize_case(density_df, ["locality", "region"])

    density_df.to_csv("data/external/density_data.csv", index=False)

    merged_df = pd.merge(
        df,
        density_df[["locality", "region", "locality_square", "locality_population_density"]],
        on=["locality", "region"],
        how="left"
    )
    
    return merged_df


if __name__ == "__main__":
    density_df = load_density()
    print(density_df.head())