import streamlit as st
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import datetime as dt
import os
import sys
import traceback
import re
import time
import requests
from CatailystPythonTools.helpers import names_2_ciks, extract_date, clean_name, base_name, abbr_name
import nltk

USER_AGENT = "Catailyst Inc Catailyst333@gmail.com"

def on_demand_SEC_search(comp_name, re_query_1, re_query_2, re_query_3, start_date, end_date, partner=None, 
                         comp_cik=None, partner_cik=None, form_types=['10-K', '10-Q', '8-K', '6-K', '20-F'], 
                         search_exhibits=False):
    """
    Search specified form types filed between start date and end date for a given company for all keywords 
    in the given query string.
    """
    # Only one partner supported at this time...
    if partner:
        partner = partner.split("|")[0].strip()
    
    if partner_cik:
        partner_cik = partner_cik.split("|")[0].strip()
        
    # Convert names to CIKs if not provided, zero pad to 10 digits as required by SEC
    comp_cik = zero_pad_cik(comp_cik if comp_cik else get_cik(comp_name))
    partner_cik = zero_pad_cik(partner_cik if partner_cik else get_cik(partner))
    
    if comp_cik == "NO CIK" and partner_cik and partner_cik != "NO CIK":
        # Switch partner and comp
        comp_name, partner = partner, comp_name
        comp_cik, partner_cik = partner_cik, comp_cik
    
    if comp_cik == "NO CIK" and (not partner or partner_cik == "NO CIK"):
        print(f"No CIK found for {comp_name}, must provide CIK or update lookup table in CatailystPythonTools")
        return pd.DataFrame()
        
    print(f"Searching SEC forms for {comp_name} {'and ' + partner if partner else ''}" + 
          f" from {start_date} to {end_date}...")
        
    # Get relevant forms urls...    
    df = get_relevant_urls(comp_name, comp_cik, start_date, end_date, form_types, search_exhibits=search_exhibits)
    if partner and partner_cik and partner_cik != "NO CIK":
        partner_forms = get_relevant_urls(partner, partner_cik, start_date, end_date, form_types, 
                                          search_exhibits=search_exhibits)
        df = pd.concat([df, partner_forms], axis=0).reset_index(drop=True)
    
    # For debug purposes:
    # df.to_csv(f"{comp_name}{'_' + partner if partner else ''}_{start_date}_{end_date}.csv")
    
    # Prep regex from raw strings
    if not re_query_1:
        print("Error: RE_Query_1 column missing value")
        exit()
        
    regex_1 = re.compile(re_query_1, re.IGNORECASE)
    regex_2 = re.compile(re_query_2, re.IGNORECASE) if isinstance(re_query_2, str) else None
    regex_3 = re.compile(re_query_3, re.IGNORECASE) if isinstance(re_query_3, str) else None
    
    # Search each form for keywords and return dataframe containing extracted paragraphs and sentences
    results = []
    for i, row in df.iterrows():
        other_comp_regex = None
        if partner:  # other company (if provided) should be a required search term
            other_comp_regex = re.compile(partner, re.IGNORECASE) if row["Company Name"] == comp_name else re.compile(comp_name, re.IGNORECASE)

        soup = get_form_text(row["Form URL"])
        
        # Run search and create matching rows for:
        # 1) Other company name and just regex_1
        results.extend(get_matching_rows(soup, [other_comp_regex, regex_1], row))
        
        # 2) Other company name, regex_1, regex_2, and regex_3
        results.extend(get_matching_rows(soup, [other_comp_regex, regex_1, regex_2, regex_3], row))
        
    return pd.DataFrame(results)

def get_matching_rows(soup, req_re_list, row):
    """
    Given a bs4 soup object, list of required regex strings to match, and the current row (contains information 
    about the query) return a list of new rows to add to the results dataframe. New rows with have the same 
    columns as given row PLUS "Matching Paragraph" and "Matching Sentence" columns
    """
    req_re_list = [regex for regex in req_re_list if isinstance(regex, re.Pattern)]  # filter out nulls
    assert req_re_list, "Must supply at least one valid regex query term"
    
    matching_rows=[]
    paras = extract_matching_paras(soup, req_re_list)
    
    for para, para_keywords in paras.items():
        sents = extract_matching_sents(para, req_re_list)
        new_row = row.copy()
        new_row["Matching Paragraph"] = para
        new_row["Matching Sentence"] = None
        new_row["Keywords Found"] = para_keywords

        for sent, sent_keywords in sents.items():  # append a row for each unique matching sentence
            sent_row = new_row.copy()
            sent_row["Matching Sentence"] = sent
            sent_row["Keywords Found"] = sent_keywords  # replace with sent-specific keywords
            matching_rows.append(sent_row)
        if not sents:  # append one row for the paragraph even if there were no matching sentences
            matching_rows.append(new_row)
    
    return matching_rows

def get_cik(comp_name):
    """ Get the CIK for the given company name """
    if comp_name == None:
        return None
    cik = names_2_ciks([comp_name])[0]
    return cik

def zero_pad_cik(cik):
    """ SEC search requires 10-digit CIK codes (left padded with zeros) """
    if pd.isna(cik) or cik == "NO CIK":
        return "NO CIK"
    else:
        return f"{cik:0>10}"
    
def get_relevant_urls(comp_name, cik, start_date, end_date, form_types, search_exhibits=False):
    """ Get relevant form urls """    
    url_start = "https://www.sec.gov/Archives/edgar/data/"  # base URL for the SEC Edgar browser
    endpoint = "https://www.sec.gov/cgi-bin/browse-edgar"
    
    entries = []
    for form_type in form_types:
        # Define our parameters dictionary
        param_dict = {'action': 'getcompany',
                      'CIK': cik,
                      'type': form_type,
                      'dateb': '',
                      'owner': 'exclude',
                      'start': '',
                      'output': 'atom',
                      'count': '100'}

        # Request the url and parse the results
        try:
            r = requests.get(endpoint, params=param_dict, headers={"User-agent": USER_AGENT})
            soup = BeautifulSoup(r.content, 'lxml')
            entries += soup.find_all('entry')
        except:
            traceback.print_exc()
            continue  # just keep going...

    rows = []
    for entry in entries:
        try:
            # Only look within specified date range...
            form_date = extract_date(entry.find('filing-date').get_text())
            if extract_date(start_date) < form_date < extract_date(end_date):
                index_link = entry.find('filing-href').get_text()
                r = requests.get(index_link, headers={"User-agent": USER_AGENT})
                soup = BeautifulSoup(r.text, "html.parser")
                table = soup.find('table', {"summary": "Document Format Files"})
                trs = table.find_all('tr')
                for tr in trs:
                    tds = tr.find_all('td')
                    if tds:
                        # Only look within specified form types...
                        form_type = tds[3].get_text()
                        if form_type in form_types or (search_exhibits and 'EX' in form_type):
                            form_link = tds[2].find('a')['href']
                            form_url = "https://www.sec.gov" + re.sub(r'.*/ix\?doc=/', '/', form_link)
                            rows.append({"Company Name": comp_name, "Company CIK": cik, "Form Type": form_type, 
                                         "Form Date": form_date, "Form URL": form_url})
        except Exception as e:
            print(f"Error encountered processing form entry for {comp_name}: {e}")
            traceback.print_exc()
            continue
            
    return pd.DataFrame(rows)

def get_form_text(url):
    """ Return the text from a given url """
    r = requests.get(url, headers={"User-agent": USER_AGENT})
    return BeautifulSoup(r.text, "lxml")

def clean_extra_newlines(text):
    """ Remove redundant newlines/spaces from a given string of text """
    if pd.isna(text):
        return text
    return re.sub(r"\s*\n+[\s\n]*", os.linesep, text)

def extract_matching_paras(soup, req_re_list):
    """
    Split fulltext into paragraphs and return a dict of paragraphs which match all regex strings in req_re_list.
    The paragraphs will be the keys (guarantees uniqueness) and values are the keywords found in the paragraph
    :param soup: a BeautifulSoup object, str
    :param req_re_list: a dict of compiled regex strings, list of re.Pattern objects
    :return: a dict containing all matching paragraphs and the keywords found for each, {str: str}
    """
    matching_paras = {}
    for div in soup.find_all(["div", "p"]):
        para = div.get_text(separator=" ")
        matches = [regex.search(para) for regex in req_re_list]
        if all(matches):
            matching_paras[para] = "; ".join([match.group(0) for match in matches])
        
    # Clean up keys
    matching_paras = {clean_extra_newlines(key).replace("\xa0", " ").strip(): val for key, val in matching_paras.items()}
    return matching_paras
    
def extract_matching_sents(para, req_re_list):
    """
    Split paragraph into sentences and return a dict of sentences which contain all regex strings in req_re_list
    :param para: a paragraph, str
    :param req_re_list: a list of re.Pattern objects
    :return: dict of matching sentences and the keywords found, {str: str} """
    sents = split_sents(para)
    matching_sents = {}
    for sent in sents:
        matches = [regex.search(sent) for regex in req_re_list]
        if all(matches):
            matching_sents[sent] = "; ".join([match.group(0) for match in matches])
    matching_sents = {clean_extra_newlines(key).replace("\xa0", " ").strip(): val for key, val in matching_sents.items()}
    return matching_sents

def split_sents(text):
    """ Split text into a list of sentences """
    text = re.sub(r"\.(\s|$)|\n", ".</SENT>", text)
    return [sent.strip() for sent in text.split("</SENT>") if sent]

query_df = pd.read_excel("data/cik-list.xlsx")

query_df["RE_Query_1"] = "Net product sale|Global revenue|Net Sales|Net Product Revenues|Net Revenue|net product revenue |Product Sales|Product revenues|sales of"
query_df["RE_Query_2"] = "Increase |Decrease"
query_df["RE_Query_3"] = "projected|projection"
query_df["Partner_CIK"] = ""


def get_rev_search(start_date, end_date, partners, company):
    global query_df
    query_df["Start Date"] = start_date
    query_df["End Date"] = end_date
    query_df["Partners"] = partners
    #company = "ABEONA THERAPEUTICS INC."
    
    df = pd.DataFrame()
    for x in range(len(query_df)):
        # if re.findall(company, query_df['Company Name'][x], re.IGNORECASE):
        if company.lower() in query_df['Company Name'][x].lower():
            df = df.append(query_df.iloc[x])
    print(df)

    results = pd.DataFrame()
    for i, row in df.iterrows():
        df = on_demand_SEC_search(row["Company Name"], row["RE_Query_1"], row["RE_Query_2"], row["RE_Query_3"], 
                                row["Start Date"], row["End Date"], partner=row["Partners"], 
                                comp_cik=row["CIK"], partner_cik=row["Partner_CIK"], search_exhibits=True)
        results = pd.concat([results, df], axis=0).reset_index(drop=True)

    print(results)
    return results

def clean_df(results):
    #CHECKING FOR '€', '$', 'DKK', 'EUR', 'Usd', 'kEUR', '%'
    patterns = ['€', '$', 'DKK', 'EUR', 'Usd', 'kEUR', '%']
    results = results.dropna(subset='Matching Sentence', how='any')
    results['Matching Sentence'] = results['Matching Sentence'].astype(str)

    df1 = pd.DataFrame()
    for x in range(len(results)):
        for pattern in patterns:
            if re.search(pattern, str(results['Matching Sentence'].iloc[x])):
                df1 = df1.append(results.iloc[x], ignore_index = True)
                print(str(results['Matching Sentence'].iloc[x]), pattern)
            else:
                print("NO Characters")
    df1 = df1.drop_duplicates()
    df1.reset_index(drop=True, inplace=True)
    

    #REMOVES IF WORDS MORE THAN 200
    for x in range(len(df1)):
        if len(df1['Matching Sentence'].loc[x].split()) >= 200:
            print("MORE THAN 200 words")
            df1 = df1.drop(labels=x, axis=0)
    df1.reset_index(drop=True, inplace=True)

    #REMOVES IF WORDS MORE THAN 120 AND RATIO LESS THAN 6
    for x in range(len(df1)):
        if len(df1['Matching Sentence'].loc[x].split()) >= 120:
            if len(df1['Matching Sentence'].loc[x].split())/len(df1['Matching Sentence'].loc[x].split()) < 6:
                print("BAD RATIO")
                df1 = df1.drop(labels=x, axis=0)
    df1.reset_index(drop=True, inplace=True)

    #REMOVE IF IT CONTAINS '$' MORE THAN 6 TIMES
    for x in range(len(df1)):
        if df1['Matching Sentence'].loc[x].count('$') > 6:
            print("more than 6 dollar signs")
            df1 = df1.drop(labels=x, axis=0)
    df1.reset_index(drop=True, inplace=True)

    for x in range(len(df1)): 
        if re.search("table", df1['Matching Sentence'].loc[x], re.IGNORECASE):
            print("removed %s for table" %x)
            df1 = df1.drop(labels=x, axis=0)
    df1.reset_index(drop=True, inplace=True)

    if len(df1) > 0:
        for x in range(len(df1)): 
            df1['Keywords Found'].loc[x] = df1['Keywords Found'].loc[x].split(";")[0]
            
        df1 = df1.drop("Company CIK", axis = 1)
        df1.rename(columns = {'Matching Sentence':'Revenue Data'}, inplace = True)
        df1.rename(columns = {'Keywords Found':'Asset Name'}, inplace = True)
        df1.rename(columns = {'Matching Paragraph':'Reference Data'}, inplace = True)
        df1.insert(1, 'Revenue Data', df1.pop('Revenue Data'))
        df1.insert(1, 'Asset Name', df1.pop('Asset Name'))

    print(df1)
    return df1






#STREAMLIT BLOCK-------------------------------------------------------------

def process_data(start_date, end_date, company_name, partner_name):
    
    df_1 = get_rev_search(str(start_date), str(end_date), partner_name, company_name)
    df_clean = clean_df(df_1)
    return df_clean

# Set up the Streamlit app
st.title('Ondemand Revenue Search')
start_date = st.date_input('Start Date')
end_date = st.date_input('End Date')
company_name = st.text_input('Company Name')
partner_name = st.text_input('Partner Name')

# Process the data when the user clicks the button
if st.button('Search'):
    df = process_data(start_date, end_date, company_name, partner_name)
    # Display the dataframe
    st.write(df)
    # Add a button to download the dataframe
    st.download_button(
        label="Download Data",
        data=df.to_csv().encode('utf-8'),
        file_name='processed_data.csv',
        mime='text/csv'
    )
