import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load Resources
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Helper function to clean up the sentence


def clean_up_sentence(sentence):
    """Tokenize and lemmatize the input sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    """Convert sentence into a bag-of-words (array of 0s and 1s)."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    """Predict the class of the input sentence."""
    # 1. Preprocess user input
    bow = bag_of_words(sentence)
    # 2 . Predict intent
    res = model.predict(np.array([bow]))[0]

    # 3. Filter out results below a threshold (uncertainty)
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # 4. Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    """Get a random response based on the predicted intent"""
    if not intents_list:
        return "I'm sorry, I don't understand that."

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


# Main Chat Loop
print("Chatbot is running! Type 'quit' to exit.")

while True:
    message = input("You: ")
    if message.lower() == 'quit':
        print("Chatbot: Goodbye!")
        break

    try:
        ints = predict_class(message)
        res = get_response(ints, intents)
        print(res)
    except Exception as e:
        print("Error:", e)
