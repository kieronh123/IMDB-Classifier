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
## ⚙️ PyTorch Installation (Required)

This project uses PyTorch, but it is **not included in `requirements.txt`** because the install differs by system.

Please install the correct version for your system using the official PyTorch site:

👉 https://pytorch.org/get-started/locally/

For most users (Windows + NVIDIA GPU):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
