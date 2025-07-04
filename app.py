from flask import Flask, render_template, request
from transformers import pipeline
import random
import os

app = Flask(__name__)

# Use HuggingFace hosted model (small, free, works on Render)
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Map model labels to emotions (as per SST-2)
label_mapping = {
    "NEGATIVE": "sadness",
    "POSITIVE": "joy"
}

# Emotion-based responses
emotion_responses = {
    "sadness": [
        "I'm really sorry you're feeling this way 💔. You matter, and I'm here to listen.",
        "It's okay to not be okay sometimes. Want to talk more about it? 💙",
        "Tough moments don’t define you. Let’s get through it together 🌧️➡️🌤️"
    ],
    "joy": [
        "That’s lovely to hear! 😊 What made your day brighter?",
        "I'm so happy for you! Keep shining 🌟",
        "Spreading joy like yours makes the world better 💫"
    ]
}

# Helper to generate a friendly response
def get_bot_response(user_input, emotion):
    text = user_input.lower().strip()

    if text in ["exit", "quit", "bye"]:
        return "Take care of yourself 💙. I'm always here if you need to talk."

    if any(word in text for word in ["yes", "yeah", "sure", "okay", "ok"]):
        return "Thanks for sharing that. I’m here for you 💙"

    if any(word in text for word in ["no", "nah", "not really"]):
        return "No worries at all. I'm right here if you change your mind. 🤗"

    if any(kw in text for kw in ["help", "what should", "do now", "suggest", "advice"]):
        return "Would you like me to suggest an activity to lift your mood or calm your thoughts? 🌈"

    return random.choice(emotion_responses.get(emotion, ["I'm here for you. You can tell me anything. 🫂"]))


@app.route("/", methods=["GET", "POST"])
def index():
    user_message = ""
    bot_response = ""
    if request.method == "POST":
        user_message = request.form["message"]
        result = classifier(user_message)
        label = result[0]['label']
        emotion = label_mapping.get(label, "joy")  # Default to joy if unknown
        bot_response = get_bot_response(user_message, emotion)
    return render_template("index.html", user_message=user_message, bot_response=bot_response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render requires port from env
    app.run(debug=True, host="0.0.0.0", port=port)
