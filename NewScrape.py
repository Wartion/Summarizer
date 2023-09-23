import requests
from bs4 import BeautifulSoup

def get_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    article_content = soup.find('div', class_='article-content')
    article_text = article_content.get_text(separator=' ')
    return article_text
