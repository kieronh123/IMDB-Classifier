from transformers import pipeline

# Load from the results directory
classifier = pipeline("sentiment-analysis", model="./exported_model")

# Test it
print(classifier("This movie was absolutely amazing!"))
print(classifier("The acting was bad and the story was boring."))
