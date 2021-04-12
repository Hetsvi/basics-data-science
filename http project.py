import os
import pandas as pd
import numpy as np
import requests
import bs4
import time
import glob
import re



def answers():
    """
    Returns two lists with your answers

    """
    lst = [1, 2, 2, 1]
    lst1 = ['https://www.pinterest.com/robots.txt', 'https://twitter.com/robots.txt',
            'https://en.wikipedia.org/robots.txt', 'https://www.tumblr.com/robots.txt',
            'https://www.yahoo.com/robots.txt', 'https://www.traderjoes.com/robots.txt']
    return lst, lst1



def find_countries(url):
    """
    Scrapes the site to extract the name of the countries.

    """

    lst = []
    for i in range(26):
        url = "http://example.webscraping.com/places/default/index/" + str(i)
        resp = requests.get(url)
        urlText = resp.text
        soup = bs4.BeautifulSoup(urlText, 'html.parser')
        time.sleep(1)
        a = soup.find('table')
        for i in a.findAll('div'):
            white = i.text.strip()
            lst.append(white)

    df = pd.DataFrame()
    df['Countries'] = lst

    return df


def first_letters_count(df):
    """
    Counts number of countries that begin with the same letter

    """
    df['First Letter'] = df['Countries'].apply(lambda x: x[:1])
    a = df.groupby(
        ['First Letter']).count()

    new_df = pd.DataFrame(a)

    new_df = new_df.rename(columns={'Countries': 'Count'})
    new_df = new_df.sort_values('Count', ascending=False)

    return new_df

def extract_book_links(text):
    """
    :Example:
    >>> fp = os.path.join('data', 'products.html')
    >>> out = extract_book_links(open(fp).read())
    >>> url = 'scarlet-the-lunar-chronicles-2_218/index.html'
    >>> out[0] == url
    True
    """
    lst = []
    soup = bs4.BeautifulSoup(text, 'html.parser')
    section = soup.find('section')
    div = section.findAll('div')[1]
    new_list = div.findAll('article', attrs={"class": "product_pod"})

    for i in new_list:
        price = i.find(attrs={"class": "price_color"})
        price = price.get_text()
        decimal = re.compile(r'[^\d.]+')
        price = decimal.sub('', price)
        price = float(price)
        star = i.find('p').attrs['class'][1]
        if (price < 20 and star in ["Four", "Five"]):
            lst.append(i.find('a')['href'])

    return lst


def get_product_info(text):
    """
    :Example:
    >>> fp = os.path.join('data', 'Frankenstein.html')
    >>> out = get_product_info(open(fp).read())
    >>> isinstance(out, dict)
    True
    >>> 'UPC' in out.keys()
    True
    >>> out['Rating']
    'Two'
    """

    soup = bs4.BeautifulSoup(text, 'html.parser')
    div = soup.findAll('div')
    table = soup.findAll('table')[0]
    available = table.find('th', text="Availability").find_next_sibling("td").text
    reviews = table.find('th', text="Number of reviews").find_next_sibling("td").text
    price_exc = table.find('th', text="Price (excl. tax)").find_next_sibling("td").text
    price_inc = table.find('th', text="Price (incl. tax)").find_next_sibling("td").text
    product_type = table.find('th', text="Product Type").find_next_sibling("td").text
    UPC = table.find('th', text="UPC").find_next_sibling("td").text
    tax = table.find('th', text="Tax").find_next_sibling("td").text
    description = soup.findAll('p')[3].text
    rating = soup.findAll('p')[2].attrs['class'][1]
    title = soup.find('h1').text

    new_dict = {"Availablity": available,
                "Number of Reviews": reviews,
                "Price (excl. tax)": price_exc,
                "Price (incl. tax)": price_inc,
                "Product Type": product_type,
                "Tax": tax,
                "UPC": UPC,
                "Description": description,
                "Rating": rating,
                "Title": title
                }

    return new_dict


def scrape_books(k):
    """

    :Example:
    >>> out = scrape_books(1)
    >>> out.shape
    (1, 10)
    >>> out['Rating'][0] == 'Five'
    True
    >>> out['UPC'][0] == 'ce6396b0f23f6ecc'
    True
    """
    if (k > 50):
        print("Invalid page number")
        return

    df = pd.DataFrame
    count = 0
    url_list = []
    for i in range(1, k + 1):
        url = 'http://books.toscrape.com/catalogue/page-' + str(i) + '.html'
        resp = requests.get(url)
        urltext = resp.text
        a = extract_book_links(urltext)
        for url in a:
            url_list.append("http://books.toscrape.com/catalogue/" + url)

    l1 = []

    for i in url_list:
        resp_1 = requests.get(i)
        urltext_1 = resp_1.text
        dict1 = get_product_info(urltext_1)
        l1.append(dict1)


    return pd.DataFrame(l1)


def depth(comments):
    """
    
    :Example:
    >>> fp = os.path.join('data', 'comments.csv')
    >>> comments = pd.read_csv(fp, sep='|')
    >>> (depth(comments) == [1, 2, 2, 3, 4, 1, 2, 1]).all()
    True
    """
    df = pd.DataFrame()
    df['col'] = comments['reply_to']
    df['col'] = df['col'].fillna(1)
    children = comments[comments['reply_to'].isnull()]['post_id']
    comments = comments.dropna(subset=['reply_to'])
    counter = 2

    while (len(comments.index != 0)):
        index = []
        for i in children:
            df['col'] = df['col'].replace(i, counter)
            indx = comments.index[comments['reply_to'] == i].tolist()
            index = index + indx

        children = comments[comments['reply_to'].isin(children)]['post_id']
        comments = comments.drop(index)
        counter = counter + 1


    return df['col']
