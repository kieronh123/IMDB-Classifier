# 🎬 IMDB Sentiment Classifier (Hugging Face + RTX 4080)

This project fine-tunes a [DistilBERT](https://huggingface.co/distilbert-base-uncased) model on the IMDB movie review dataset using Hugging Face Transformers and runs with GPU acceleration.

---

## 📦 Setup

```powershell
# Clone the repo
git clone https://github.com/YOUR_USERNAME/hf-imdb-sentiment-classifier.git
cd hf-imdb-sentiment-classifier

# Create and activate a virtual environment (Windows / PowerShell)
python -m venv .venv
.venv\Scripts\Activate

# Install dependencies
pip install -r requirements.txt
