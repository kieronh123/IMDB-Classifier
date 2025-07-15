from transformers import pipeline
import gradio as gr

# Use your uploaded model
classifier = pipeline("sentiment-analysis", model="khushon123/imdb-sentiment-model")

def predict_sentiment(review_text):
    result = classifier(review_text)[0]
    label_map = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "POSITIVE",
        "NEGATIVE": "NEGATIVE",
        "POSITIVE": "POSITIVE"
    }
    label = label_map.get(result["label"], result["label"])
    score = round(result["score"], 2)
    return f"{label} ({score})"

gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter a movie review..."),
    outputs="text",
    title="IMDB Sentiment Classifier",
    description="Enter a movie review and get a prediction from a fine-tuned DistilBERT model."
).launch()
