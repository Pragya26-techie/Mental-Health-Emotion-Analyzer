from flask import Flask, render_template, request
from transformers import pipeline
import random

app = Flask(__name__)

# Load model
classifier = pipeline("text-classification", model="saved_model", tokenizer="saved_model")
# Label mapping
label_mapping = {
    "LABEL_0": "sadness",
    "LABEL_1": "joy",
    "LABEL_2": "love",
    "LABEL_3": "anger",
    "LABEL_4": "fear",
    "LABEL_5": "surprise"
}

# Empathetic emotion-based responses
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
    ],
    "love": [
        "Love is powerful. Cherish it always ❤️",
        "That’s heartwarming! Love brings strength and peace 💕",
        "So sweet! Who’s bringing this love into your life? 😊"
    ],
    "anger": [
        "I hear you. Your feelings are valid. Want to talk it out calmly? 😌",
        "It's okay to feel angry. Let’s unpack that together 🧘‍♂️",
        "Breathe in... breathe out. I’m here to listen, no judgment 🙏"
    ],
    "fear": [
        "That sounds tough. But you're not alone 🫂",
        "Let’s face this fear together. One step at a time 🤝",
        "Want to share more about what's worrying you?"
    ],
    "surprise": [
        "Wow, that sounds unexpected! 😲 Tell me more.",
        "Surprises can be exciting or scary. What kind was it?",
        "Interesting! I'm curious — what happened?"
    ]
}

# Intent-aware helper
def get_bot_response(user_input, emotion):
    text = user_input.lower().strip()

    if text in ["exit", "quit", "bye"]:
        return "Take care of yourself 💙. I'm always here if you need to talk."

    # Intent detection
    if any(word in text for word in ["yes", "yeah", "sure", "okay", "ok"]):
        if emotion == "sadness":
            return "Thanks for sharing that with me. You’re safe here 💙"
        elif emotion == "joy":
            return "That’s amazing! 😊 Tell me more!"
        elif emotion == "anger":
            return "Let’s work through that together calmly 🙏"
        else:
            return "Go ahead, I'm listening..."

    if any(word in text for word in ["no", "nah", "not really"]):
        return "No worries at all. I'm right here if you change your mind. 🤗"

    if any(kw in text for kw in ["help", "what should", "do now", "suggest", "advice"]):
        return "Would you like me to suggest an activity that might lift your mood or calm your thoughts? 🌈"

    # Fallback based on emotion
    return random.choice(emotion_responses.get(emotion, ["I'm here for you. You can tell me anything. 🫂"]))


# Main Chatbot Loop
def chatbot():
    print("\n🤖 Hi! I'm your Mental Health Companion. Share how you feel (type 'exit' to leave).\n")

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in ["exit", "quit", "bye"]:
            print("\n🤖 Bot: Take care of yourself 💙. I'm always here when you need to talk.")
            break

        result = classifier(user_input)
        label = result[0]['label']
        emotion = label_mapping.get(label, "unknown")
        response = get_bot_response(user_input, emotion)

        print(f"🤖 Bot: {response}\n")



@app.route("/", methods=["GET", "POST"])
def index():
    user_message = ""
    bot_response = ""
    if request.method == "POST":
        user_message = request.form["message"]
        result = classifier(user_message)
        label = result[0]['label']
        emotion = label_mapping.get(label, "unknown")
        bot_response = get_bot_response(user_message, emotion)
    return render_template("index.html", user_message=user_message, bot_response=bot_response)

if __name__ == "__main__":
    app.run(debug=True)
