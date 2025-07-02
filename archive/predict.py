# predict.py
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline

# Load fine-tuned model from saved directory (replace with your actual path if needed)
model_path = "saved_model"  # or "saved_model" if you renamed
model = DistilBertForSequenceClassification.from_pretrained("saved_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("saved_model")

# Create sentiment analysis pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Predict function
def predict_emotion(text):
    result = classifier(text)
    label = result[0]['label']
    score = round(result[0]['score'], 4)
    print(f"Text: {text}")
    print(f"Predicted Emotion: {label} (Confidence: {score})")

# Example use
if __name__ == "__main__":
    test_text = input("Enter a sentence to analyze emotion: ")
    predict_emotion(test_text)
