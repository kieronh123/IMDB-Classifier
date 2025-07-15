# Import Gradio for web app UI
import gradio as gr

# Import Hugging Face pipeline for inference
from transformers import pipeline

# Load your fine-tuned sentiment analysis model from local folder
# This folder should contain a Hugging Face-compatible model (via save_pretrained)
classifier = pipeline("sentiment-analysis", model="./exported_model")

# Define the function that runs on user input
def predict_sentiment(review_text):
    result = classifier(review_text)[0]
    raw_label = result["label"]
    score = round(result["score"], 2)

    # Map internal labels to human-readable ones
    label_map = {
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "POSITIVE",
        "NEGATIVE": "NEGATIVE",  # for pre-mapped models
        "POSITIVE": "POSITIVE"
    }

    label = label_map.get(raw_label, raw_label)  # fallback in case label is already mapped
    return f"{label} ({score})"


# Build the Gradio interface
interface = gr.Interface(
    fn=predict_sentiment,                       # Function to call with input
    inputs=gr.Textbox(lines=4, placeholder="Enter a movie review..."),  # User input box
    outputs="text",                             # Display prediction as text
    title="🎬 IMDB Sentiment Classifier",       # App title
    description="Enter a movie review and get a predicted sentiment from your fine-tuned DistilBERT model."  # Subtitle
)

# Launch the app (opens in browser)
interface.launch()
