import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training data (example intents)
training_sentences = [
    "Hi", "Hello", "Hey",
    "How are you?", "How is it going?",
    "Bye", "Goodbye", "See you later",
    "What is your name?", "Who are you?"
]

labels = [
    "greeting", "greeting", "greeting",
    "status", "status",
    "goodbye", "goodbye", "goodbye",
    "identity", "identity"
]

# Responses for each intent
responses = {
    "greeting": ["Hello!", "Hi there!", "Hey!"],
    "status": ["I'm doing great!", "All good here!"],
    "goodbye": ["Goodbye!", "See you soon!"],
    "identity": ["I'm a simple ML chatbot!", "I am your assistant."]
}

# Convert text to numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(training_sentences)

# Train classifier
model = LogisticRegression()
model.fit(X, labels)

# Chat function
def chatbot():
    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)[0]
        
        print("Bot:", random.choice(responses[prediction]))

chatbot()
