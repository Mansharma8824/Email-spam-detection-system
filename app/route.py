from flask import Flask,render_template,request
from .utils import text_preprocessing
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'

app = Flask(__name__)

# load the model
model = joblib.load(MODEL_DIR / 'model.pkl')


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
    
    # check how much spam prabelity
    probability = model.predict_proba([message])
    ham_prob = probability[0][0]
    spam_prob = probability[0][1]
    spam_percentage = round(spam_prob *100,2)
    
    
    result = ''
    if pred == 1:
        result = f'🚨 SPAM DETECTED'
    else:
        result = f'✅ NOT SPAM'
    
    return render_template('index.html',prediction_text = result,confidence = spam_percentage)

if __name__ == '__main__':
    app.run()
    
    
