# === Import Required Libraries ===
from datasets import load_dataset         # For loading the IMDB dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate                           # For computing accuracy metric
import numpy as np                        # For numerical operations

# === Step 1: Load the IMDB Dataset ===
# This dataset has 50,000 movie reviews labeled as "positive" or "negative"
dataset = load_dataset("imdb")

# For quicker training/testing, we use small subsets:
# - 2,000 training samples
# - 500 test samples
small_train = dataset["train"].shuffle(seed=42).select(range(2000))
small_test = dataset["test"].shuffle(seed=42).select(range(500))

# === Step 2: Load a Pretrained Tokenizer ===
# We're using the tokenizer from DistilBERT, a smaller version of BERT.
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Define a function to tokenize the input text
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding=True)

# Apply the tokenizer to the train and test sets
tokenized_train = small_train.map(tokenize_fn, batched=True)
tokenized_test = small_test.map(tokenize_fn, batched=True)

# === Step 3: Load a Pretrained Model ===
# We load DistilBERT configured for binary classification (2 output labels)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# === Step 4: Define Evaluation Metrics ===
# We’ll use accuracy to evaluate our model on the validation set
accuracy = evaluate.load("accuracy")

# Function to compute accuracy after each evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Convert logits to predicted class indices
    return accuracy.compute(predictions=predictions, references=labels)

# === Step 5: Set Up Training Configuration ===
# TrainingArguments defines how the model will train and evaluate
training_args = TrainingArguments(
    output_dir="./results",                     # Where to save model checkpoints
    evaluation_strategy="epoch",                # Evaluate at the end of every epoch
    per_device_train_batch_size=8,              # Batch size for training
    per_device_eval_batch_size=8,               # Batch size for evaluation
    num_train_epochs=2,                         # Total training epochs
    weight_decay=0.01,                          # Weight decay for regularization
)

# === Step 6: Create Trainer Instance ===
# Trainer is Hugging Face’s high-level API to train models easily
trainer = Trainer(
    model=model,                                # The model to train
    args=training_args,                         # Training configurations
    train_dataset=tokenized_train,              # Training data
    eval_dataset=tokenized_test,                # Evaluation data
    compute_metrics=compute_metrics,            # Metrics to compute
)

# === Step 7: Train the Model ===
# This starts the training loop and saves the model in the output directory
trainer.train()
