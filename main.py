import gradio as gr
from NewsScrape import get_article
from summerize import summarize_article
from predictions import predict_bias
from prediction_type import predict_label

def pipeline(url):
    article_text = get_article(url)
    summary = summarize_article(article_text)
    bias_label = predict_bias(summary)

    return article_text, summary, bias_label




iface = gr.Interface(
    fn=pipeline,
    inputs=gr.inputs.Textbox(lines=1, label="Enter Article Text"),
    outputs=[
        gr.outputs.Textbox(label="Article"),
        gr.outputs.Textbox(label="Summary"),
        gr.outputs.Textbox(label="Bias Label"),
    ],
    iterate=True  # Enable iterating over outputs
)

iface.queue()  # Enable the queue
iface.launch()

