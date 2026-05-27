from flask import Flask,render_template,request
from utils import text_preprocessing
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'

app = Flask(__name__)

# load the model
model = joblib.load(MODEL_DIR / 'model.pkl')

# load the vectorizer
# vectorizer = joblib.load(MODEL_DIR / 'vectorizer.pkl')

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def spam_predict():
    message = request.form['message']
    
    # preprocessing input message
    message = text_preprocessing(message)
    
    # msg_vec = vectorizer.transform([message]).toarray()
    
    pred = model.predict([message])[0]
    
    # result = 'spam' if pred == 1 else 'ham'
    result = ''
    if pred == 1:
        result = 'spam'
    else:
        result = 'ham'
    
    return render_template('index.html',prediction_text = result)

if __name__ == '__main__':
    app.run(debug=True)
    
    
