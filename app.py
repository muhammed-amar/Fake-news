from flask import Flask, render_template, request, jsonify
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)

model = joblib.load('model.pkl')
stop_words = set(stopwords.words('english'))
port_stem = PorterStemmer()
vectorizer = joblib.load("vectorizer.pkl")

# Preprocess text
def preprocess_text(text):
    # Add your preprocessing steps here
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [port_stem.stem(word) for word in text if not word in stop_words]
    return ' '.join(text)

@app.route('/')
def index():
    return render_template('temp.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    content = data['author'] + ' ' + data['title'] + ' ' + data['text']
    preprocessed_content = preprocess_text(content)
    transformed_content = vectorizer.transform([preprocessed_content])
    prediction = model.predict(transformed_content)
    if prediction==1:
        return "Fake news"
    else:
        return "Real news"

if __name__ == '__main__':
    app.run(debug=True)
