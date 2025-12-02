import os
import json
import random
from flask import Flask, render_template, request, jsonify, session
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import requests

# Initialize NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "very-secret-key")

lemmatizer = WordNetLemmatizer()

# Load model artifacts (ensure these exist)
MODEL_PATH = "model/chatbot_model.h5"
WORDS_PKL = "model/words.pkl"
CLASSES_PKL = "model/classes.pkl"
INTENTS_JSON = "intents.json"
MEMORY_FILE = "data/memory.json"

model = load_model(MODEL_PATH)
words = pickle.load(open(WORDS_PKL, "rb"))
classes = pickle.load(open(CLASSES_PKL, "rb"))
with open(INTENTS_JSON, "r", encoding="utf-8") as f:
    intents = json.load(f)

# Ensure memory file exists
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists(MEMORY_FILE):
    json.dump({"users": {}}, open(MEMORY_FILE, "w"))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(w.lower()) for w in sentence_words]


def bag_of_words(sentence):
    s_words = clean_up_sentence(sentence)
    bag = [1 if w in s_words else 0 for w in words]
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append(
            {"intent": classes[r[0]], "probability": float(r[1])})
    return return_list


def get_response(ints, intents_json):
    if not ints:
        return "👻 Ghost doesn't understand that yet — I can learn though!"
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "👻 I couldn't find the right response."

# --- Routes ---


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get")
def get_reply():
    msg = request.args.get('msg', '')
    # Basic RAG trigger: if user says "search:" do rag query (example)
    if msg.lower().startswith("search:"):
        query = msg.split(":", 1)[1].strip()
        # call rag endpoint internally
        from rag import rag_query
        ans = rag_query(query)
        return jsonify(ans)
    ints = predict_class(msg)
    res = get_response(ints, intents)

    # If user asked to remember something, store in memory
    if ints and ints[0]['intent'] == 'memory_set':
        # naive parse: "remember that X is Y"
        try:
            _, kv = msg.lower().split("remember", 1)
            # store as raw string under session user
            user = session.get('user', 'guest')
            store_memory(user, msg)
            res = "Okay — I stored that for you."
        except:
            res = "Tell me what to remember like: 'remember that I like python.'"
    return jsonify(res)

# Simple login demo (not for production)


@app.route("/login", methods=["POST"])
def login():
    data = request.json or {}
    u = data.get("username", "")
    p = data.get("password", "")
    # Demo creds
    if u == "user" and p == "pass":
        session['user'] = u
        return jsonify({"success": True, "user": u})
    return jsonify({"success": False})

# Memory endpoints


def load_memory():
    return json.load(open(MEMORY_FILE, "r"))


def save_memory(mem):
    json.dump(mem, open(MEMORY_FILE, "w"), indent=2)


def store_memory(user, text):
    mem = load_memory()
    if user not in mem['users']:
        mem['users'][user] = []
    mem['users'][user].append({"text": text})
    save_memory(mem)


@app.route("/memory", methods=["GET", "POST"])
def memory():
    user = session.get('user', 'guest')
    mem = load_memory()
    if request.method == 'POST':
        # add memory
        payload = request.json or {}
        text = payload.get('text', '')
        if not text:
            return jsonify({"error": "no text"}), 400
        store_memory(user, text)
        return jsonify({"stored": True})
    else:
        # retrieve
        return jsonify(mem['users'].get(user, []))

# External API example: weather


@app.route("/weather")
def weather():
    city = request.args.get("city", "")
    if not city:
        return jsonify({"error": "provide city param like ?city=Accra"}), 400
    api_key = os.environ.get("OPENWEATHER_API_KEY", "")  # set in your env
    if not api_key:
        return jsonify({"error": "OPENWEATHER_API_KEY not configured in environment"}), 500
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={api_key}"
    resp = requests.get(url)
    if resp.status_code != 200:
        return jsonify({"error": "failed to fetch weather", "detail": resp.text}), 500
    data = resp.json()
    summary = f"{data['name']}: {data['weather'][0]['description']}, {data['main']['temp']}°C"
    return jsonify({"weather": summary})

# RAG endpoint uses rag.py utilities


@app.route("/rag/query", methods=["POST"])
def rag_route():
    payload = request.json or {}
    query = payload.get("query", "")
    if not query:
        return jsonify({"error": "no query"}), 400
    from rag import rag_query
    ans = rag_query(query)
    return jsonify({"answer": ans})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
