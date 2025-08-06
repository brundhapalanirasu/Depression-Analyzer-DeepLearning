from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import sqlite3
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.model_selection import train_test_split

nltk.download('wordnet')
nltk.download('omw-1.4')  
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import pandas as pd



with open('model/lgbm_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)



app = Flask(__name__)
app.secret_key = '123'

model = load_model('model/chatbot_model.h5')
lemmatizer = WordNetLemmatizer()

with open("model/words.pkl", "rb") as f:
    words = pickle.load(f)

with open("model/classes.pkl", "rb") as f:
    classes = pickle.load(f)

with open('intents.json', encoding='utf-8') as file:
    data_file = file.read()
intents = json.loads(data_file)

dataset = pd.read_csv('emotions.csv')
texts = dataset['text']        
labels = dataset['label']
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
import pickle
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)


database = "new.db"
conn = sqlite3.connect(database)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS register (Id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, usermail TEXT UNIQUE, password INT, age INT, depression_level TEXT, number INT)")
cur.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            rating INTEGER NOT NULL,
            message TEXT NOT NULL
        )
    ''')
conn.commit()

@app.route('/')
def index():
    return render_template("register.html")

label_map = {
    0: 'Anxiety',
    1: 'Bipolar',
    2: 'Depression',
    3: 'Normal',
    4: 'Personality Disorder',
    5: 'Stress',
    6: 'Suicidal'
}

def emotion(new_text):
    new_text_vec = vectorizer.transform([new_text])  # <-- wrapped in list

    prediction = loaded_model.predict(new_text_vec)
    prediction_proba = loaded_model.predict_proba(new_text_vec)[0]

    emotions = label_map.get(prediction[0], "Unknown")
    depression_level = f"{prediction_proba[prediction[0]] * 100:.2f}%" 
    level = f"{emotions}-{depression_level}"
    
    return level

@app.route('/mick')
def mick_page():
    return render_template('mick_chat.html')

@app.route('/register', methods=['POST','GET'])
def register():
    if request.method == 'POST':
        username=request.form["username"]
        usermail=request.form["usermail"]
        password=request.form["password"]
        age = request.form['age']
        number = request.form['number']

        conn = sqlite3.connect(database)
        cur = conn.cursor()
        cur.execute("INSERT INTO register (username, usermail, password, age, depression_level, number) VALUES (?, ?, ?, ?, ?, ?)", (username, usermail, password, age, 0, number))
        conn.commit()
        return render_template("index.html")
    return render_template("register.html")



@app.route('/login', methods=['POST','GET'])
def login():
    global usermail
    if request.method == 'POST':        
        usermail = request.form["usermail"]
        password = request.form["password"]
        conn = sqlite3.connect(database)
        cur = conn.cursor()
        cur.execute("SELECT * FROM register WHERE usermail=? AND password=?", (usermail, password))
        data = cur.fetchone()  
        if data:
            return render_template("chatbot.html")
        else:
            return render_template('index.html', message= 'Password Mismatch')
    return render_template('index.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    global usermail

    email = usermail
    rating = request.form['rating']
    message = request.form['message']
    print(rating)
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO feedback (email, rating, message)
        VALUES (?, ?, ?)
    ''', (email, rating, message))
    conn.commit()
    conn.close()

    return render_template('index.html', name=email)

@app.route('/view_feedback')
def view_feedback():
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("SELECT email, rating, message FROM feedback ORDER BY id DESC")
    all_feedback = cursor.fetchall()
    conn.close()
    return render_template('view_feedback.html', feedback=all_feedback)



def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

def predict_class(sentence):
    bow = [0] * len(words)
    sentence_words = clean_up_sentence(sentence)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bow[i] = 1
    return model.predict(np.array([bow]))[0]

def get_intent(predictions):
    ERROR_THRESHOLD = 0.25
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    if confidence > ERROR_THRESHOLD:
        return classes[predicted_class]
    return None

def get_response(intent):
    for i in intents['intents']:
        if i['tag'] == intent:
            return random.choice(i['responses'])
    return "I'm sorry, I didn't understand that."



@app.route('/profile-info')
def profile_info():
    global usermail
    conn = sqlite3.connect(database)
    cur = conn.cursor()
    cur.execute("SELECT * FROM register WHERE usermail=?", (usermail,))
    data = cur.fetchone()
    
    data = [
        ("Name", data[1]),
        ("Email Id", data[2]),
        ("Age", data[4]),
        ("Depression Level", data[5]),
        ("Mobile Number", data[6])
    ]
    return jsonify(data)



def update_sentiment_score(user_id, level):
    print(level)
    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute("UPDATE register SET depression_level = ? WHERE usermail = ?", (level, user_id,))    
    conn.commit()
    conn.close()




@app.route('/chatbot')
def chatbot():
    
    return render_template('chatbot.html')

@app.route('/feedback')
def feedback():
    
    return render_template('feedback.html')


@app.route('/mick_chat', methods=['POST'])
def mick_chat():
    global usermail
    user_message = request.json.get('message')
    percen = emotion(user_message)

    update_sentiment_score(usermail, percen)

    predictions = predict_class(user_message)
    intent = get_intent(predictions)
    response = get_response(intent)
    return jsonify({'reply': response})

@app.route('/chat', methods=['POST'])
def chat():
    global usermail
    user_message = request.json['message']
    percen = emotion(user_message)
    
    update_sentiment_score(usermail, percen)
    

    predictions = predict_class(user_message)
    intent = get_intent(predictions)
    response = get_response(intent)
    return jsonify({'reply': response})

if __name__ == '__main__':
    app.run(debug=False, port=560)
