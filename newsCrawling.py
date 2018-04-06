import sys
import re
import datetime
import pytz
from bs4 import BeautifulSoup
import urllib.request
from urllib.parse import quote

# key for using azure api 
headers = {
    'Ocp-Apim-Subscription-Key': 'c5ea992048b040ddb65333e156f9fed9',
    'Content-Type': 'application/json',
    'Accept': 'application/json',
}

target_url_before_keyword = 'https://www.bloomberg.com/search?query='
#keyword='' #입력받음
before_endtime = '&endTime='
endtime=str(datetime.datetime.utcnow().isoformat()) #iso8601의 UTC timezone
before_page = 'Z&page=' #Z는 UTC 의미
#page_num = #입력받음

URL_pattern = re.compile('\{"storyType":"Article","url":"([^"]+)"') #의 dict형태의 part에서 url이 다 튀어나오더라 ㅇㅇ
time_pattern = re.compile('data-type="updated" datetime="([^"]+)"') #의 형태에서 최초 올린 시간 등장

def article_URL(keyword, start_page, end_page):
    URL_list = []
    page_num = start_page - 1
    end = False
    while not end:
        page_num += 1
        URL = target_url_before_keyword + keyword + before_endtime + endtime + before_page +  str(page_num)
        # 그래서 https://www.bloomberg.com/search?query=bitcoin&endTime=2017-11-03T08:19:08.909Z&page=2 형태로 나타남
        
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
       
bitcoin_URL = article_URL('bitcoin', 1, 100)
print(bitcoin_URL)

def article_analysis(article_URL):
    for URL in article_URL:
        source_code_from_URL = urllib.request.urlopen(URL)
        soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')
        
        time_pattern = re.compile('data-type="updated" datetime="([^"]+)"')
        time = time_pattern.findall(str(soup))
        
        content_of_article = soup.select('div.body-copy') # bloomberg 기사 본문은 여기에 ㅇㅇ
        article = []
        for item in content_of_article:
            string_item = str(item.find_all(text=True))
            string_item = re.sub('({["contentId"][^"]+["containerId"][^"]+})', '', string_item) # 고쳐야함. {} 형태의 dict구문 지울 수 있게끔
            string_item = string_item.replace("\'\\n\',", "")
            string_item = string_item.replace("\'", "")
            string_item = string_item.replace("\\xa0", "")
            string_item = string_item.replace(", ,", "")
            string_item = string_item.replace(".,", ".")
            string_item = string_item.replace('[', '')
            string_item = string_item.replace(']', '') # 너저분...한데 간단히 할 방법 없나
    
            article.append(string_item)
        
        sentences = article.split('.')
        emo_point = []
        
        for sentence in sentences:
            data = {
                "documents": [{
                    "id": "1", 
                    "text":sentence,
                }]
            }            
            
            r = requests.post("https://westus.api.cognitive.microsoft.com/text/analytics/v2.0/sentiment",
                              data=json.dumps(data), headers=headers)
            emo_score = r.json()["documents"][0]["score"]
            
            if emo_score != 0.5:
                emo_point.append(emo_score)
                
        emotional_score = sum(emo_point)/len(emo_point)
        return [time, emotional_score]
        
        

'''        
def main(argv):
    if len(argv) != 4:
        print('python [module name] [keyword] [crawling page num] [output file name]')
        return
    
    keyword = argv[1]
    page_num = int(argv[2])
    output_file_name = argv[3]
    target_URL = target_url_before_keyword + keyword + before_endtime + endtime + before_page + str(page_num) #영문 받을때
    output_file = open(output_file_name, 'w')
    get_link_from_news_title(page_num, target_URL, output_file)
    output_file.close()
    
if __name__ == '__main__':
    main(sys.argv)
'''
