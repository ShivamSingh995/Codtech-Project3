import nltk
import spacy
import random

# Download NLTK data (only first time)
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Simple intents and responses
intents = {
    "greeting": ["hello", "hi", "hey", "good morning", "good evening"],
    "goodbye": ["bye", "goodbye", "see you", "take care"],
    "thanks": ["thanks", "thank you"],
    "name": ["what is your name", "who are you"],
    "age": ["how old are you", "what is your age"],
}

responses = {
    "greeting": ["Hello! How can I help you?", "Hi there!", "Hey! What can I do for you?"],
    "goodbye": ["Goodbye!", "See you later!", "Take care!"],
    "thanks": ["You're welcome!", "No problem!", "Happy to help!"],
    "name": ["I'm a chatbot built using NLP!", "You can call me ChatNLP."],
    "age": ["I'm ageless!", "I exist in the digital realm, so I do not age."],
    "default": ["I'm sorry, I don't understand. Could you please rephrase?"]
}

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Lowercase, tokenize, and lemmatize
    tokens = word_tokenize(text.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized)

def match_intent(user_input):
    processed = preprocess(user_input)
    for intent, patterns in intents.items():
        for pattern in patterns:
            # Use spaCy similarity
            doc1 = nlp(processed)
            doc2 = nlp(pattern)
            if doc1.similarity(doc2) > 0.85:
                return intent
            # Fallback to substring matching
            if pattern in processed:
                return intent
    return "default"

def get_response(user_input):
    intent = match_intent(user_input)
    return random.choice(responses[intent])

def chat():
    print("Chatbot: Hello! I am your NLP-powered chatbot. Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["bye", "exit", "quit"]:
            print("Chatbot:", random.choice(responses["goodbye"]))
            break
        response = get_response(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    chat()