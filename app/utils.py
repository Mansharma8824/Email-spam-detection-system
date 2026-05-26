import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
import re

nltk.download('punkt_tab')
nltk.download('stopwords')

def text_preprocessing(text):
    # converting to lower
    text = text.lower()
    
    # removing any special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # creating tokens
    tokens = word_tokenize(text)
    
    # removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # steamming 
    steamer = PorterStemmer()
    stemmed_tokens = [steamer.stem(word) for word in tokens ]
    
    return " ".join(stemmed_tokens)