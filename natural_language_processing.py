import re
import datetime
import requests
import json
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote

headers = {
    'Key': '',
    'Content-Type': 'application/json',
    'Accept': 'application/json',
}

target_url_before_keyword = 'https://www.bloomberg.com/search?query='
before_endtime = '&endTime='
endtime = str(datetime.datetime.utcnow().isoformat())     # UTC timezone of ISO8601
before_page = 'Z&page='     # Z means UTC
URL_pattern = re.compile('\{"storyType":"Article","url":"([^"]+)"')

def bloomberg_article_URL(keyword, start_page, end_page):
    URL_list = []
    page_num = start_page - 1
    end = False
    while not end:
        page_num += 1
        URL = target_url_before_keyword + keyword + before_endtime + endtime + before_page + str(page_num)
        # example: https://www.bloomberg.com/search?query=bitcoin&endTime=2017-11-03T08:19:08.909Z&page=2

        source_code = urllib.request.urlopen(URL)
        soup = str(BeautifulSoup(source_code, 'lxml', from_encoding='utf-8'))
        URL_list_single_page = URL_pattern.findall(soup)

        if URL_list_single_page == []:
            end = True
        elif page_num == end_page + 1:
            end = True
        else:
            URL_list += URL_list_single_page

    return URL_list


def article_crawling(keyword):
    source_code_from_URL = urllib.request.urlopen(bloomberg_article_URL(keyword, 1, 1)[1])
    soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')
    content_of_article = soup.select('div.body-copy')
    article = []
    for item in content_of_article:
        string_item = str(item.find_all(text=True))
        string_item = string_item.replace("\\n", "")
        string_item = string_item.replace("\\xa0", " ")
        string_item = string_item.replace("\'\'", "")
        string_item = string_item.replace("\\'", "")
        string_item = string_item.replace(", ,", "")

        # string_item = string_item.replace(re.compile('{"([^"]+)"}', ''))

        article.append(string_item)

    return article


def twitter_crawling(twit_ID):
    return None


def nlp_sentiment(data):
    r = requests.post('URL_for_nlp', data=json.dumps(data), headers=headers)
    emo_score = r.json()["documents"][0]["score"]
    return emo_score

