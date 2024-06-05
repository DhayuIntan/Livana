from flask import Flask, render_template, request
import joblib
import logging
import re
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
app.debug = True

# Tambahkan StreamHandler dan atur level logging
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Tambahkan formatter ke handler
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Tambahkan handler ke logger Flask
if not app.logger.handlers:
    app.logger.addHandler(handler)

# Load the saved Logistic Regression model and TFIDF vectorizer
lr_clf = joblib.load('logistic_regression_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Assuming you have a saved vectorizer
nltk.download('punkt')

norm = {
    " dgn ": " dengan", " gue ": " saya ", " bgmn ": " bagaimana ", " gmn ": " bagaimana",
    " tdk ": " tidak ", " g ": " tidak ", " ngga ": " tidak ", " nggak ": " tidak ",
    " blm ": " belum ", " blum ": " belum ", " apk ": " aplikasi ", " gk ": " tidak ",
    " knp ": " kenapa ", " mbois ": " bagus ", " mantul ": " bagus ",
    " nice ": " bagus ", " top ": " bagus ", " sblm ": " sebelum ", " kmrn ": " kemarin ",
    " sip ": " bagus ", " skrg ": " sekarang ", " like ": " suka ", " good ": " bagus ",
    " jos ": " bagus ", " perfect ": " bagus ", " dgn ": " dengan ", " tp ": " tetapi ",
    " tapi ": " tetapi "
}

def casefolding(Review):
    return Review.lower()

def normalisasi(str_text):
    for i in norm:
        str_text = str_text.replace(i, norm[i])
    return str_text

def tokenize_text(kalimat):
    tokens = word_tokenize(kalimat)
    return tokens

factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()

def stopword_text(tokens):
    cleaned_tokens = [token for token in tokens if token not in stopwords]
    return cleaned_tokens

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def stemming_text(tokens):
    hasil = [stemmer.stem(token) for token in tokens]
    return hasil

def join_tokens(tokens):
    return ' '.join(tokens)

def map_sentiment(prediction):
    if prediction == 1:
        return "positive"
    elif prediction == 0:
        return "negative"

# Predict sentiment function
def predict_sentiment(review):
    review_processed = casefolding(review)
    review_processed = normalisasi(review_processed)
    review_processed = tokenize_text(review_processed)
    review_processed = stopword_text(review_processed)
    review_processed = stemming_text(review_processed)
    review_processed = join_tokens(review_processed)
    app.logger.info("test txt "+review_processed)
    review_tfidf = tfidf_vectorizer.transform([review_processed])
    prediction = lr_clf.predict(review_tfidf)
    return map_sentiment(prediction[0])

@app.route('/', methods=['GET', 'POST'])
def index():
    app.logger.info("haloo")
    if request.method == 'POST':
        inp = request.json["inp"]
        app.logger.info(inp)

        if inp:
            result = predict_sentiment(inp)
            app.logger.info(result)
            return result
    return render_template('index.html', result="Result", inp=None)

if __name__ == "__main__":
    app.run(debug=True)
