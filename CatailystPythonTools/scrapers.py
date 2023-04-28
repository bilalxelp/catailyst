import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import requests
from newspaper import Article
import pickle
import time
from CatailystPythonTools.helpers import base_name, abbr_name

USER_AGENT = "Catailyst Inc. catailyst333@gmail.com"  # SEC requires company name/email in user agent

#####################
# Reusable Scrapers
#####################


def scrape_generic(df, url_col, out_name, title_col="Article Title", fulltext_col="Article Full Text"):
    """
    Generic article text scraper using Newspaper3K package. Takes a DataFrame with a column of URLS and
    scrapes title and fulltext for each, saving in new columns. Saves progress as pickle file in case of
    interruption.
    :param df: the df containing URLs to scrape, pandas DataFrame
    :param url_col: the name of the column containing URLS, str
    :param out_name: a name for saving the resulting dataframe .csv file, str
    :param title_col: a name for the new title column, str, defaults to "Article Title"
    :param fulltext_col: a name for the new full text column, str, defaults to "Article Full Text"
    :return: a copy of the original dataframe with title and full text columns added, pandas DataFrame
    """
    save_point = out_name.replace(".csv", "") + ".pkl"
    user_agent = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0'

    if os.path.isfile(save_point):
        with open(save_point, 'rb') as f:
            df = pickle.load(f)
    else:
        df = df.copy()
        df[title_col] = [None] * df.shape[0]
        df[fulltext_col] = [None] * df.shape[0]

    for i, row in df.iterrows():
        if pd.notna(row[fulltext_col]):
            continue  # skip already downloaded articles

        # Otherwise download the article text
        try:
            print(f"Downloading text for row {i}, {row[url_col]}...")
            raw_html = requests.get(row[url_col], headers={'User-agent': user_agent})
            article = Article('')
            article.download(raw_html.content)
            article.parse()
            df.loc[i, title_col] = article.title
            df.loc[i, fulltext_col] = article.text

            with open(save_point, 'wb') as f:
                pickle.dump(df, f)  # save progress

        except Exception as e:
            print(f"Error downloading text for row {i} ({row['URL']}):", e)
            continue

    df.to_csv(out_name)
    print("DONE!")
    return df


def scrape_custom(df, url_col, text_col, outfile, tag, attr_dict, alt_tag=None, alt_attr_dict=None):
    """
    A customizable and interruptable selenium scraper
    :param df: dataframe containing the urls to scrape, pandas DataFrame
    :param url_col: column name in df containing the urls to scrape, str
    :param text_col: column name in which to store scraped text, str
    :param outfile: name for the final csv and intermediate (save point) pickle files
    :param tag: the html tag containing the desired content, str (e.g. "div")
    :param attr_dict: a dict of attr and values to further specify the desired html tag, dict
    (e.g. {"class": "abstract"})
    :param alt_tag: an optional second choice for the html tag to search for if first tag is not successful
    (defaults to None), str
    :param alt_attr_dict: an optional second choice for the attr dict in case first tag and attr dict are not
    successful (defaults to None), dict
    :return: a copy of df with the new scraped text stored in text_col
    """
    save_point = outfile.replace(".csv", "") + '.pkl'

    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome("C:\\Users\\arind\\desktop\\Final Codes good to run\\Chromedriver\\chromedriver.exe", options=chrome_options)

    # Start from save point if interrupted
    if os.path.isfile(save_point):
        with open(save_point, "rb") as f:
            df = pickle.load(f)
    else:
        df = df.copy()
        df[text_col] = [None] * len(df)  # initial new fulltext column

    for i in range(len(df)):
        if pd.notna(df.loc[i, text_col]):
            continue  # skip if fulltext already downloaded

        try:
            url = df.loc[i, url_col]
            driver.get(url)
            time.sleep(2)
            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            selected_soup = soup.find(tag, attr_dict)
            if not selected_soup and alt_tag:  # try second choice if provided
                selected_soup = soup.find(alt_tag, alt_attr_dict)
            df.loc[i, text_col] = selected_soup.get_text(separator=" ")
        except AttributeError:
            df.loc[i, text_col] = "Text Not Found"
        except TimeoutException:
            driver.quit()
            time.sleep(60)
            driver = webdriver.Chrome()
            continue
        except Exception as e:
            print(f"An unexpected error occurred while extracting text from URL '{url}': {e}")
            continue

        with open(save_point, "wb") as f:
            pickle.dump(df, f)  # save progress after every download

        if i % 100 == 0:
            df.to_csv(outfile)  # save to file every 100 downloads just in case

    driver.quit()
    df.to_csv(outfile)
    print("DONE!")
    return df


def scrape_company_pages(df, outfile):
    """
    Given a dataframe with 'Company Name' and 'SEC Company Page' columns, scrape all available info from
    that company's SEC page, and save the resulting dataframe to outfile (as well as returning it)
    """
    save_point = "sec_save_point.pkl"

    if os.path.isfile(save_point):
        # Start from save point if present
        with open(save_point, "rb") as f:
            df = pickle.load(f)
    else:
        # Initialize dataframe
        df = df.copy()
        cols = ["SIC", "Ticker", "Fiscal Year End", "Exchange", "Business Address", "Mailing Address",
                "Phone Number", "Alt CIK", "Alt CIK Company Name", "Alt Company/Owner Names", "Alt Tickers",
                "Company Website", "Base Name", "Abbr Name"]
        for col in cols:
            if col not in df.columns:
                df[col] = [None] * len(df)

    chrome_options = Options()
    chrome_options.add_argument("--headless")  # using selenium b/c page requires javascript
    chrome_options.add_argument(f"--user-agent={USER_AGENT}")
    driver = webdriver.Chrome(options=chrome_options)

    for i, row in df.iterrows():
        if pd.notna(row["Business Address"]) or pd.isna(row["SEC Company Page"]):
            continue  # skip already downloaded companies or companies with no SEC url

        print(f"Scraping company {i}, {row['Company Name']}...")
        driver.get(row["SEC Company Page"])
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # Extract info from tables (not always same tables on each page)
        trs = soup.find_all("tr")
        for tr in trs:
            left_col = tr.find("td")
            if not left_col:
                continue  # no cells in this row
            label = left_col.get_text()
            value = tr.find_all("td")[1].get_text(separator=" ")
            if label == "SIC" and pd.isna(df.loc[i, "SIC"]):  # keep first SIC if there are multiple
                df.loc[i, "SIC"] = value
            elif label == "Business Address":
                df.loc[i, "Business Address"] = value
            elif label == "Business Phone":
                df.loc[i, "Phone Number"] = value
            elif label == "Mailing Address":
                df.loc[i, "Mailing Address"] = value
            elif label == "Fiscal Year End":
                df.loc[i, "Fiscal Year End"] = value
            elif label == "SEC Alt" and value:
                df.loc[i, "Alt CIK"], df.loc[i, "Alt CIK Company Name"] = value[0:10], value[10:].strip()
            elif label == "Ticker":
                value = value.split(":")
                if len(value) == 2:
                    df.loc[i, "Exchange"], df.loc[i, "Ticker"] = value
                else:
                    df.loc[i, "Ticker"] = value
            elif label == "Website":
                df.loc[i, "Company Website"] = value.split()[0]

        # Find the alternate company name table and join those names together
        alt_names_header = soup.find("h4", text="Company Names & Stock Symbols")
        if alt_names_header:
            alt_names_table = alt_names_header.nextSibling
            alt_names = []
            alt_tickers = []
            if alt_names_table:
                for tr in alt_names_table.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) > 0:
                        alt_names.append(tds[0].get_text())
                    if len(tds) > 1:
                        alt_tickers.append(tds[1].get_text())
            df.loc[i, "Alt Company/Owner Names"] = "; ".join(alt_names)
            df.loc[i, "Alt Tickers"] = "; ".join(alt_tickers)

        # Add standardized and abbreviated names
        df.loc[i, "Base Name"] = base_name(row["Company Name"])
        df.loc[i, "Abbr Name"] = abbr_name(row["Company Name"])

        with open(save_point, "wb") as f:
            pickle.dump(df, f)

    df.to_excel(outfile, engine="openpyxl")
    return df
