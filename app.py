from flask import Flask, request, render_template
import joblib
import re
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure required NLTK data is available (for servers like Render)
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained models from the models folder
sentiment_model = joblib.load("models/naive_bayes_model.joblib")
category_model = joblib.load("models/logistic_regression_model.joblib")
vectorizer = joblib.load("models/vectorizer.joblib")

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Preprocessing function
def preprocess_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Lowercase and remove non-word characters
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Lemmatization and stop-word removal
    text = " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    )
    return text

# Prediction function
def predict_sentiment_and_category(title: str):
    # Preprocess the input title
    clean_title = preprocess_text(title)

    # Transform title using the pre-trained TF-IDF vectorizer
    title_vectorized = vectorizer.transform([clean_title])

    # Predict sentiment and category from ML models
    sentiment = sentiment_model.predict(title_vectorized)[0]
    category = category_model.predict(title_vectorized)[0]

    # VADER Sentiment Analysis
    vader_score = analyzer.polarity_scores(title)["compound"]
    sentiment_label = (
        "positive" if vader_score > 0
        else "negative" if vader_score < 0
        else "neutral"
    )

    return sentiment, category, sentiment_label, vader_score

# Route for homepage
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle prediction
@app.route("/predict", methods=["POST"])
def predict():
    title = request.form.get("title", "")

    sentiment, category, sentiment_label, vader_score = predict_sentiment_and_category(
        title
    )

    return render_template(
        "result.html",
        title=title,
        sentiment=sentiment_label,
        ml_sentiment=sentiment,
        category=category,
        vader_score=vader_score,
    )

if __name__ == "__main__":
    # For local development; Render will use gunicorn app:app
    app.run(host="0.0.0.0", port=5000, debug=True)
