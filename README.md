# 🧠 Mental Health Emotion Analyzer

A deep learning-based emotion detection chatbot built using HuggingFace Transformers and DistilBERT. It helps users express their feelings and provides empathetic, context-aware responses.

---

## 🚀 Features

* Trained DistilBERT model for multi-class emotion classification
* Label mapping for emotions like `sadness`, `joy`, `anger`, etc.
* Chatbot-style CLI interface for supportive replies
* Custom responses for user replies like "yes", "no", "help", etc.
* Clean modular structure (train, predict, chatbot)

---

## 🗂️ Folder Structure

```
Mental-Health-Emotion-Analyzer/
├── saved_model/           # Trained model and tokenizer
├── train_model.py         # Model training script
├── predict.py             # CLI chatbot script
├── requirements.txt       # Dependencies
├── .gitignore             # Files to ignore in Git
└── README.md              # Project documentation
```

---

## 📦 Installation

```bash
# Clone the repo
https://github.com/yourusername/Mental-Health-Emotion-Analyzer.git
cd Mental-Health-Emotion-Analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Training the Model

```bash
python train_model.py
```

This will:

* Load and preprocess the data
* Fine-tune `distilbert-base-uncased`
* Save model and tokenizer to `saved_model/`

---

## 💬 Running the Chatbot

```bash
python predict.py
```

Type any sentence and the bot will:

* Predict the emotion
* Give a friendly, empathetic response

Example:

```
🤖 Hi! I'm your Mental Health Companion. Type how you're feeling. (Type 'exit' to end)
You: I’m feeling tired and anxious
🤖 Bot: It’s okay to feel this way. Want to talk more about it? 💙
```

---

## 🌐 Deployment (Upcoming)

You can deploy this as a web app using:

* Streamlit or Flask for frontend
* Render.com or HuggingFace Spaces for hosting

(Coming soon in `web_app/` branch)

---

## 🤝 Contribution

Pull requests are welcome. For major changes, please open an issue first.

---

## 📄 License

MIT License. Feel free to use for educational or personal use.

---

## 🙌 Acknowledgments

* HuggingFace 🤗 Transformers
* Emotion classification dataset (Kaggle or Emotion Dataset by Saravia et al.)
* Python, PyTorch, Scikit-learn
