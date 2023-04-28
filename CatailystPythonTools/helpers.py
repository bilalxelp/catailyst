"""
File: helpers.py
Author: Liz Codd
Last Updated: 07/14/2021
Helper functions for biotech event analysis
"""
import os
import sys
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer
import pandas_datareader.data as web
from yahoofinancials import YahooFinancials
from datetime import datetime
import pandas as pd
import numpy as np
import re
import math
from importlib import resources
import dateutil.parser as dparser
import tempfile

###########
# Finance
###########


def update_price_data(infile, col, outfile):
    """
    Downloads current (1980 -> today) stock prices for all tickers in a given event dataframe and saves as .csv file
    :param infile: the event dataset filepath (must be .csv file), str
    :param col: name of column with ticker symbols, e.g. "Symbol", str
    :param outfile: filepath (func will NOT create new directories) for the price data output file (will be .csv), str
    """
    df = pd.read_csv(infile)
    tickers = list(df[col].dropna().unique())  # unique ticker symbols in events dataset
    tickers.append('SPY')  # include S&P 500 for market control
    yahoo_financials = YahooFinancials(tickers)
    print(f"Retrieving stock data for {len(tickers)} companies...")
    data = yahoo_financials.get_historical_price_data(start_date='1980-01-01',
                                                      end_date=datetime.today().strftime('%Y-%m-%d'),
                                                      time_interval='daily')
    data = {k: v for k, v in data.items() if 'prices' in v.keys()}  # drop failed ticker requests (those with no prices)
    tickers = data.keys()
    prices_df = pd.DataFrame({ticker: {x['formatted_date']: x['adjclose'] for x in data[ticker]['prices']}
                              for ticker in tickers})
    prices_df = prices_df.sort_index()
    prices_df.to_csv(outfile)


def download_adj_close(ticker, start, end):
    """
    Download adj close data for a single ticker from start date until end date
    :param ticker: the ticker to retrieve, str
    :param start: the start date, date obj
    :param end: the end date, date obj
    :return: a Series of adj close prices for the ticker with a datetime index, pandas Series
    """
    date_fmt = '%a, %m/%d/%y'
    prices = None
    try:
        df = web.DataReader(ticker, 'yahoo', start, end)
        prices = df['Adj Close']
        prices.name = ticker
        print(f"Downloaded {ticker} data from {start.strftime(date_fmt)} to {end.strftime(date_fmt)}")
    except Exception as e:
        print(f"Failed to download stock data for {ticker}: {e}")
    return prices


def closes_2_grosses(ticker, closes):
    """
    Convert a series of adj close prices to daily gross returns
    :param ticker: a ticker symbol (must match the name of the Series closes), str
    :param closes: the adj close prices of a stock, pd.Series with datetime index
    """
    prices = closes.reset_index()[ticker]
    grosses = [np.nan] * len(prices)
    for i in range(1, len(prices)):
        try:
            grosses[i] = prices[i] / prices[i-1]
        except ZeroDivisionError:
            pass
    return pd.Series(grosses, index=closes.index)


def compound_grosses(grosses, start, end):
    """
    Convert a series of daily gross returns to compounded value of $1.00
    :param grosses: the daily gross returns of a stock, pd.Series with datetime index
    :param start: the date to start compounding $1, str or datetime
    :param end: the date to stop compounding, str or datetime
    """
    compound_vals = list(grosses[start:end])
    assert compound_vals[1] > 0, 'Missing price data, check company IPO date'
    compound_vals[0] = 1  # start with $1.00
    for i in range(1, len(compound_vals)):
        compound_vals[i] *= compound_vals[i-1]

    compound_vals = map(lambda x: round(x, 2), compound_vals)
    return pd.Series(compound_vals, name='Compound Value', index=grosses[start:end].index)


def extract_prices(text):
    """
    Extract price "tags": price amount text and up to two words after if present
    :param text: text to search, str
    :return: text "tags" separated by semicolons, e.g. "$10 million grant; EUR 65,000,000 in funding", str
    """
    # REGEX EXPLANATION: USD (and occasionally EUR) prices can have multiple "," but only one "." However for most
    # EUR prices there are multiple "." and only one ",". Also, we want to capture abbreviated units like M,
    # plus two words following the price if available
    reg = re.compile(r"((€|\beur)( )?\d+[\d\.]*,?(\d)*[mbk]?\b( [a-z]*)?( [a-z]*)?|" +               # euro only
                     r"(\$|€|\bus(d)?|\beur?)( )?\d+[\d,]*(\.)?(\d)*[mbk]?\b( [a-z]*)?( [a-z]*)?)")  # usd/euro
    if pd.notna(text):
        matches = reg.findall(text.lower())
        if matches:
            amounts = []
            for match in matches:
                amount = match[0]
                if re.search(r"€|\beur", amount):  # price in euros, may need reformatting
                    amount = reformat_eur(amount)
                amounts.append(amount)
            return "; ".join(amounts)
    return None


def reformat_eur(price):
    """ Reformat "."s to "," and vice versa if currency is euros """
    price = re.sub(r"\.(?=(\d){3})", ",", price)
    price = re.sub(r",(?=(\d){1,2}\b)", ".", price)
    return price


def get_currency(text):
    """ Determine the currency of the given amount text """
    if pd.isna(text):
        return None

    text = text.lower()
    if re.search(r"€|\beur", text):
        return "EUR"  # Euro
    if re.search(r"\bjpy", text):
        return "JPY"  # Japanese yen
    if re.search(r"\bcny", text):
        return "CNY"  # Chinese yuan
    if re.search(r"\binr", text):
        return "INR"  # Indian rupee
    if re.search(r"\bmxn", text):
        return "MXN"  # Mexican peso
    if re.search(r"\$|\bus|dollar", text):
        return "USD"  # US dollar
    return None


def price_in_millions(text):
    """
    Convert price text with various units to an amount in millions
    :param text: some text containing a price, e.g. "$10 million dollars"
    :return: the price in millions, e.g. "10.0", str
    """
    match = re.search(r"([\d\.,]+)(.*)?", text.lower())

    if not match:
        return None

    number = match.group(1).replace(",", "").strip().strip(".")  # remove commas from number
    unit = match.group(2) if len(match.groups()) == 2 else ''

    if not unit:  # no unit provided
        return str(round(float(number) / 1_000_000, 2))
    elif re.search(r"per|\bord|share", unit):  # throw away prices per share
        return None
    elif re.search(r"thousand", unit):  # unit is thousands
        return str(round(float(number) / 1000, 2))
    elif re.search(r"m\b|million|mil\b", unit):  # unit is millions
        return str(float(number))
    elif re.search(r"b\b|billion|bil\b", unit):  # unit is billions
        return str(float(number) * 1000)
    else:  # not a recognizable unit
        return str(round(float(number) / 1_000_000, 2))


def extract_amt_curr(text):
    """
    Get currency and amount in millions, filtering out invalid/duplicate prices (less than threshold or followed
    by words like "per" or "shares")
    :param text: some extracted text containing prices, may be multiple price text separated by semicolon, str
    :return: a tuple of currencies and price-in-millions (each separated by semicolons if multiple prices were valid),
    e.g. ("USD; EUR", "10.0; 11.5"), tuple of str
    """
    if pd.isna(text):
        return None, None

    currs = []
    amts = []

    amounts = text.split("; ")

    for amount in amounts:
        amt_millions = price_in_millions(amount)
        curr = get_currency(amount)
        if not amt_millions or not curr:
            continue  # no viable prices found
        # Only include if amount is greater than 10,000 (0.01 million) and not a duplicate
        if float(amt_millions) > 0.01 and amt_millions not in amts:
            amts.append(amt_millions)
            currs.append(curr)

    if not amts:
        return None, None
    return "; ".join(currs), "; ".join(amts)


#########
# Utils
#########

def merge_duplicate_cols(df):
    """ After merging two dataframes we are often left with a df containing columns with _x and _y extensions
    since they had other duplicate columns which were not used for mergine. Merge these by keeping info in _x
    column if present, otherwise keep it from the _y column, then drop the _x, _y extension cols """
    x_cols = [col for col in df.columns if re.search(r"_x$", col)]
    y_cols = [col for col in df.columns if re.search(r"_y$", col)]

    assert sorted([x_col.replace("_x", "") for x_col in x_cols]) == \
           sorted([y_col.replace("_y", "") for y_col in y_cols]), "Must have equivalent _x and _y columns"

    base_cols = [x_col.replace("_x", "") for x_col in x_cols]
    new_df = df.copy()
    for col in base_cols:
        new_df[col] = new_df[col + "_x"].fillna(new_df[col + "_y"])
    new_df = new_df.drop(columns=x_cols + y_cols)
    return new_df


def split_csv_approx(filename, max_mb=50):
    """
    Split a given csv file into approx max_mb MB as this is the file size limit for Xelp team
    :param filename: the name (full path) of the large file you wish to split, str
    :param max_mb: the maximum size of the resulting files in MB, defaults to 50 as that is Xelp's limit, float
    """
    num_chunks = math.ceil(os.stat(filename).st_size / (max_mb * 1_000_000))  # approx 50 MB chunks
    df = pd.read_csv(filename, index_col=0)
    sub_dfs = np.array_split(df, num_chunks)
    for i, sub_df in enumerate(sub_dfs, start=1):
        sub_filename = re.sub(r"\.csv", "", filename) + "_part" + str(i) + ".csv"
        sub_df.to_csv(sub_filename)  # e.g. save as filename_part1.csv ... filename_partN.csv

# TODO: create non-recursive version of function or optimize
def split_csv(filename, max_mb=50):
    """
    Recursively split dataframe in half until all chunks are guaranteed to be < max_mb MB, save these N smaller
    csv files as filename_part1.csv ... filename_partN.csv
    :param filename: the name (full path) of the large file you wish to split, str
    :param max_mb: the maximum size of the resulting files in MB, defaults to 50 as that is Xelp's limit, float
    """
    df = pd.read_csv(filename, index_col=0)

    def split_df(df):
        """ Recursive inner function """
        with tempfile.NamedTemporaryFile() as f:
            df.to_csv(f)
            file_size = os.stat(f.name).st_size
        if file_size < (max_mb * 1_000_000):
            return [df]
        else:
            df1, df2 = np.array_split(df, 2)
            return split_df(df1) + split_df(df2)

    sub_dfs = split_df(df)

    for i, sub_df in enumerate(sub_dfs, start=1):
        sub_filename = re.sub(r"\.csv", "", filename) + "_part" + str(i) + ".csv"
        sub_df.to_csv(sub_filename)  # e.g. save as filename_part1.csv ... filename_partN.csv

    print(f"Split {filename} into {i} sub files, each < {max_mb} MB")


def extract_date(string, fuzzy=True):
    """
    Return a date object if the string can be interpreted as a date.
    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    :return: a date object if the string contains a date else None, date object or None
    """
    try:
        date_obj = dparser.parse(string, fuzzy=fuzzy)
        return date_obj
    except ValueError:
        return None


############
# Cleaning
############


def clean_extra_newlines(text):
    """ Remove redundant newlines/spaces from a given string of text """
    if pd.isna(text):
        return text
    text = text.strip()
    return re.sub(r"\s*\n+[\s\n]*", os.linesep, text)


def clean_multispace(text):
    """ Remove sequences of > 1 space from a string of text (convert to one space only) """
    if pd.isna(text):
        return text
    text = text.strip()
    return re.sub(r" {2,}", " ", text)


def clean_name(name):
    """ Clean company names """
    if not isinstance(name, str):
        return name

    name = name.replace("\xa0", " ")  # \xa0 is non-breaking space character
    name = re.sub(r"\.|®|™|\(|\)", "", name).strip().upper()
    name = re.sub(r",|\||;", " ", name)  # often list item separators
    name = name.replace("&", " AND ")
    name = name.replace("-", " ")  # no hyphens, too likely to cause mismatching
    name = name.replace("’", "'")
    name = re.sub(r"'S\b", "", name)  # get rid of possessives
    name = name.replace("INCORPORATED", "INC").replace("CORPORATION", "CORP")
    name = re.sub(" A/S| A S", " AS", name)
    name = name.replace("&", " AND ")
    name = re.sub(r"INTERNATIONAL|\bINT\b", "INTL", name)
    name = re.sub(r"PHARMACEUTICAL[S]?|PHARM[A]?[S]?\b", "PHARMA", name)
    name = re.sub(r"TECHNOLOGY|TECHNOLOGIES|TECHS", "TECH", name)
    name = re.sub(r"LABORATORY|LABORATORIES", "LABS", name)
    name = re.sub(r"COMPANY|COMPANIES", "CO", name)
    name = re.sub(f"SOLUTION[S]?|SLTNS", "SLTN", name)
    name = re.sub("HOLDING[S]?|HLDGS", "HLDG", name)
    name = name.replace("PUBLIC LIMITED CORPORATION", "PLC").replace("LIMITED", "LTD").replace("GROUP", "GRP")
    name = re.sub(r"/( )?[A-Z]{2,3}(/)?$", "", name)  # get rid of "/DE/" or "/ MA" endings and the like
    name = re.sub(r"\\( )?[A-Z]{2,3}(\\)?$", "", name)  # get rid of "\DE\" or "\ MA" endings and the like
    name = re.sub(r"\s{2,}", " ", name)  # convert 2 or more whitespace chars to a single space
    return name.strip()


def base_name(name):
    """ Return just the distinguishing part of a company name - Inc causes many mismatches """
    if not isinstance(name, str):
        return name

    name = clean_name(name)
    name = re.sub(r"\b(INC|LTD|CORP|LLC|PLC|AND CO|CO|AG|AS|NV|SA|AIH|AB|BV)\b", "", name)
    name = re.sub(r"\b(GRP|GMBH|US[A]?$|SPA$)", "", name)
    name = re.sub(r"\s{2,}", " ", name)  # convert 2 or more whitespace chars to a single space
    return name.strip()


def safe_base_name(name):
    """
    Try to standardize company names while avoiding false positive matches by only removing suffixes from a
    company's name if it would not be shorter than 5 characters and would not result in an English word
    """
    if not isinstance(name, str):
        return name

    comp_basename = base_name(name)
    wnl = WordNetLemmatizer()
    lemma_name = wnl.lemmatize(comp_basename.lower())
    if len(comp_basename) < 5 or lemma_name in words.words():
        return clean_name(name)  # not safe to remove common suffixes, just use clean_name()
    return comp_basename  # probably safe to remove endings


def vc_clean_name(name):
    """ Return a standardized version of a VC investor name """
    if not isinstance(name, str):
        return name

    name = name.replace("\xa0", " ")  # \xa0 is non-breaking space character
    name = re.sub(r"\.|®|™|\(|\)|,", "", name).strip().upper()
    name = name.replace("&", " AND ")
    name = name.replace("-", " ").strip()  # no hyphens, too likely to cause mismatching
    name = name.replace("’", "'")
    name = re.sub(r"'S\b", "", name)  # get rid of possessives
    name = name.replace("INCORPORATED", "INC").replace("CORPORATION", "CORP")
    name = re.sub(" A/S| A S", " AS", name)
    name = re.sub(r"INTERNATIONAL|\bINT\b", "INTL", name)
    name = re.sub(r"PHARMACEUTICAL[S]?|PHARM[A]?[S]?\b", "PHARMA", name)
    name = re.sub(r"TECHNOLOGY|TECHNOLOGIES|TECHS", "TECH", name)
    name = re.sub(r"LABORATORY|LABORATORIES", "LABS", name)
    name = re.sub(r"COMPANY|COMPANIES", "CO", name)
    name = name.replace("PARTNERS", "PARTNER").replace("VENTURES", "VENTURE").replace("INVESTMENTS", "INVESTMENT")
    name = name.replace("SCIENCES", "SCIENCE")
    name = re.sub(r"SOLUTION[S]?|SLTNS", "SLTN", name)
    name = re.sub(r"HOLDING[S]?|HLDGS", "HLDG", name)
    name = name.replace("PUBLIC LIMITED CORPORATION", "PLC").replace("LIMITED", "LTD").replace("GROUP", "GRP")
    name = re.sub(r"/( )?[A-Z]{2,3}(/)?$", "", name)  # get rid of "/DE/" or "/ MA" type endings
    name = re.sub(r"\\( )?[A-Z]{2,3}(\\)?$", "", name)  # get rid of "\DE\" or "\ MA" type endings
    name = re.sub(r"\b(INC|LTD|CORP|LLC|PLC|LP|PTY|PTE|AG|AS|NV|SA|AIH|AB|BV|S(A)?RL)\b", "", name)
    name = re.sub(r"\b(GMBH|US[A]?|SPA)$", "", name)
    name = re.sub(r"\s{2,}", " ", name)  # convert 2 or more whitespace chars to a single space
    return name.strip()


def abbr_name(name):
    """ Remove common biotech company name endings """
    if pd.isna(name):
        return name
    name = base_name(name)
    name = name.replace("PHARMA", "")
    name = name.replace("THERAPEUTICS", "")
    name = name.replace("INTL", "")
    return name.strip()


def remove_company_duplicates(text):
    """
    Remove company name duplicates (considered duplicate if their base name transformations match) from a list
    separated by semicolons
    """
    if not isinstance(text, str):
        return text

    items = text.split(";")
    basename_list = []
    unique_list = []
    for item in items:
        item_basename = base_name(item)
        if item_basename not in basename_list:
            basename_list.append(item_basename)
            unique_list.append(clean_multispace(item))  # preserve original item name, just remove extra spaces

    return "; ".join(unique_list)


def clean_string_list(text):
    """
    Clean up (capitalize first word and remove extra spaces) and remove duplicates (case-insensitive) from a string
    of terms separated by semicolons, e.g. a list of drug indications
    """
    if not isinstance(text, str):
        return text

    unique_items = []
    for item in text.split(";"):
        item = clean_multispace(item).capitalize()
        if item not in unique_items:
            unique_items.append(item)

    return "; ".join(unique_items)


def names_2_ciks(names):
    """
    Get CIK for each company in a list from the given lookup table:
    :param names: a list of company names to look up, list of str
    :return: a list of the CIKs or 'NO CIK' if a CIK was not found for a company (order maintained), list of str
    """
    with resources.path("CatailystPythonTools.data", "Comp_Info_CIK_TickerEC210820.xlsx") as path:
        lookup = pd.read_excel(path)

    ciks = []
    for name in names:
        name = safe_base_name(name)  # Pfizer, Inc. -> PFIZER

        search = lookup[lookup["Base Name"] == name]
        if search.empty:  # try alternative column...
            search = lookup[lookup["Abbr Name"] == name]
        if not search.empty:  # found a match
            ciks.append(str(list(search["CIK"])[0]))
        else:  # neither contained a match
            ciks.append("NO CIK")

    return ciks


###########
# Mapping
###########


def map_fin(fin):
    """ Map a series of "Financing Type" to their standard Catailyst "Financing Category" """
    with resources.path('CatailystPythonTools.data', 'FinanceMapping071421.xlsx') as path:
        df_fin = pd.read_excel(path, engine="openpyxl")

    fin = fin.apply(lambda x: x if pd.isna(x) else re.sub(r" ?/ ?", "/", x))  # remove spaces around slash

    mapped_fin = pd.Series(index=fin.index)
    for i, row in df_fin.iterrows():
        term = row['Financing Type']
        # Select all rows that contain this term and map them to TA Grouping
        mapped_fin[fin.apply(lambda x: False if pd.isna(x) else term.lower() in x.lower())] = row['Financing Category']
    return mapped_fin



def map_TA(TAs):
    """
    Map a series of TAs (Therapeutic Area) to standard Catailyst terms
    :param TAs: a column of Therapeutic Area terms to be mapped, pandas Series of str
    :return: the mapped terms, pandas Series of str
    """
    with resources.path('CatailystPythonTools.data', 'CatailystMapping070621.xlsx') as path:
        df_TA = pd.read_excel(path, engine="openpyxl")

    mapped_TAs = pd.Series(index=TAs.index)
    for i, row in df_TA.iterrows():
        term = row['Therapeutic Area']
        # Select all rows that contain this term and map them to TA Grouping
        mapped_TAs[TAs.apply(lambda x: False if pd.isna(x) else term.lower() in x.lower())] = row['TA Grouping']
    return mapped_TAs


def map_TM(TMs):
    """
    Map a series of TMs (Therapeutic Modalities) to standard Catailyst terms
    :param TMs: a column of Therapeutic Modality terms to be mapped, pandas Series of str
    :return: the mapped terms, pandas Series of str
    """
    with resources.path('CatailystPythonTools.data', 'CatailystMapping070621.xlsx') as path:
        df_TM = pd.read_excel(path, sheet_name='TM', engine="openpyxl")

    mapped_TMs = pd.Series(index=TMs.index)
    for i, row in df_TM.iterrows():
        term = row['Therapeutic Modality']
        # Select all rows that contain this term and map them to TM Grouping
        mapped_TMs[TMs.apply(lambda x: False if pd.isna(x) else term.lower() in x.lower())] = row['TM Grouping']
    return mapped_TMs


def map_DS(DSs):
    """
    Map a series of DSs (Development Stages) to standard Catailyst terms
    :param DSs: a column of Development Stage terms to be mapped, pandas Series of str
    :return: the mapped terms, pandas Series of str
    """
    # Map DSs
    return DSs.apply(map_one_DS)


def map_one_DS(ds):
    """
    Standardize development stage tag
    :param ds: the development stage tag, str
    :return: the clean development stage tag, str
    """
    if pd.isna(ds):
        return None  # NaN encountered
    ds = ds.lower().replace(".", "").strip()    # lowercase, remove periods and extra whitespace
    ds = re.sub(r"( )*/( )*", "/", ds)          # remove spaces around slashes
    ds = re.sub(r" {2,}", " ", ds)              # remove double spaces
    if re.search(r"pre(-| )?clinical", ds):
        return "Preclinical"
    if re.search(r"phase 4|phase iv", ds):
        return "Phase 4"
    if re.search(r"phase 2/3|phase ii/iii", ds):
        return "Phase 2/3"
    if re.search(r"phase 1/2|phase i/ii", ds):
        return "Phase 1/2"
    if re.search(r"phase (3(a|b)?|iii(a|b)?)", ds):
        return "Phase 3"
    if re.search(r"phase (2(a|b)?|ii(a|b)?)", ds):
        return "Phase 2"
    if re.search(r"phase (1(a|b)?|i(a|b)?)", ds):  # e.g. phase ia -> Phase 1
        return "Phase 1"
    if "market" in ds:
        return "Marketed"
    if "approv" in ds:
        return "Approved"
    if "pilot" in ds:
        return "Pilot"
    else:
        return np.nan  # no mapping