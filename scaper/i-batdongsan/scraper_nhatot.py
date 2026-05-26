from selenium import webdriver

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--window-size=1280,1600")

driver = webdriver.Chrome(options=chrome_options)

driver.get("https://example.com")