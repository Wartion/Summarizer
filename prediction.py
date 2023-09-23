import Dbias
from dbias.predict import get_sentiment

def predict_bias(summary):
    sentiment = get_sentiment(summary)
    return sentiment


summary = "My name is Pratiyush"
predict_bias(summary)
print(summary)