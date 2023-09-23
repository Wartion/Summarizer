from newspaper import Article
import gradio as gr

def get_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return article.text

