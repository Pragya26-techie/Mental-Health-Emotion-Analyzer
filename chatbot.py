import random

from transformers import pipeline

# Load model and tokenizer
classifier = pipeline("text-classification", model="saved_model", tokenizer="saved_model")

# Label mapping from model output
label_mapping = {
    "LABEL_0": "sadness",
    "LABEL_1": "joy",
    "LABEL_2": "love",
    "LABEL_3": "anger",
    "LABEL_4": "fear",
    "LABEL_5": "surprise"
}

# Pre-defined empathetic responses for each emotion
emotion_responses = {
    "sadness": [
        "I'm really sorry you're feeling down 💔. You're not alone, I'm here for you.",
        "Tough times never last, but tough people do. You've got this! 💪",
        "Do you want to talk about what's making you feel this way? I'm all ears."
    ],
    "joy": [
        "That’s amazing! 😊 What’s making you smile today?",
        "I love hearing good news like that! Keep spreading the positivity 💫",
        "Sounds like a happy moment! Tell me more about it 🌞"
    ],
    "love": [
        "Love is a beautiful feeling ❤️. Hold on to it tightly.",
        "That’s so sweet. You truly deserve love and happiness! 🥰",
        "I'm glad you're feeling the warmth of love. Who's the lucky person? 😉"
    ],
    "anger": [
        "I hear your frustration 😤. Want to talk about what made you angry?",
        "Take a deep breath. It's okay to feel this way. Let's work through it together 🧘",
        "Sometimes venting helps. I'm here for you — no judgment."
    ],
    "fear": [
        "It's okay to be scared sometimes. You're not alone 🫂",
        "I understand. Let's take a deep breath together 🤝",
        "Want to share what's worrying you? Talking can help."
    ],
    "surprise": [
        "Whoa! That sounds unexpected 😲. What happened exactly?",
        "Sounds like something caught you off guard. I'm listening!",
        "That must have been quite a surprise! Want to share more?"
    ]
}
def get_bot_response(text, emotion):
    text_lower = text.lower()

    # Intent detection
    if any(word in text_lower for word in ["yes", "yeah", "yup", "okay", "ok", "sure"]):
        if emotion == "sadness":
            return "Thanks for opening up. I'm listening. 💙"
        elif emotion == "anger":
            return "I understand. Want to talk about what made you feel this way?"
        elif emotion == "joy":
            return "That's wonderful! 😊 Tell me more about it!"
        else:
            return "Go on, I'm listening..."

    elif any(word in text_lower for word in ["no", "nah", "not really"]):
        return "That's okay too. I'm here whenever you feel ready to talk. 🤗"
    elif any(word in text_lower for word in ["help", "what should", "do now", "advice"]):
        return "Would you like me to suggest some activities to uplift your mood or help you focus?"
    
   # Default emotion-based response
    responses = emotion_responses.get(emotion, ["I'm here for you. Feel free to share more. 🫂"])
    return random.choice(responses)

# Main chatbot function for interaction
def chatbot():
    print("\n🤖 Hi! I'm your Mental Health Companion. Type how you're feeling. (Type 'exit' to end)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("\n🤖 Bot: Take care of yourself 💙. I'm always here when you need to talk.")
            break

        result = classifier(user_input)
        label = result[0]['label']
        emotion = label_mapping.get(label, "unknown")
        response = get_bot_response(user_input,emotion)

        print(f"🤖 Bot: {response}\n")

if __name__ == "__main__":
    chatbot()
