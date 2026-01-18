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
    
    # Keep sentiment-carrying stopwords
    sentiment_words = {
        'not', 'no', 'nor', 'never', 'neither', 'none', 'nobody', 'nothing',
        'very', 'so', 'too', 'more', 'most', 'much', 'really', 'absolutely',
        'only', 'just', 'even', 'still', 'but', 'however', 'yet',
        'against', 'down', 'off', 'over', 'up', 'above', 'below'
    }
    stop_words = stop_words - sentiment_words

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

# Validation function
def validate_text(text: str) -> tuple[bool, str]:
    """
    Validates input text for meaningful content.
    Returns (is_valid, error_message)
    """
    # Check if text is empty or only whitespace
    if not text or not text.strip():
        return False, "Please enter some text to analyze."
    
    # Check minimum length (at least 3 characters)
    text_stripped = text.strip()
    if len(text_stripped) < 3:
        return False, "Text is too short. Please enter at least 3 characters."
    
    # Check for meaningful words (at least 2 alphanumeric characters)
    words = re.findall(r'\b\w{2,}\b', text_stripped)
    if len(words) == 0:
        return False, "Please enter meaningful text with actual words."
    
    # Check minimum word count (at least 2 words for better analysis)
    if len(words) < 2:
        return False, "Text is too short. Please enter at least 2 words for accurate analysis."
    
    # Check if text contains only numbers
    if text_stripped.replace(" ", "").isdigit():
        return False, "Please enter text with words, not just numbers."
    
    # Check if text is just repeated characters (e.g., "aaaaaaa", "!!!!!!")
    unique_chars = set(text_stripped.replace(" ", ""))
    if len(unique_chars) <= 2 and len(text_stripped) > 5:
        return False, "Please enter meaningful text, not just repeated characters."
    
    return True, ""

# Route to handle prediction
@app.route("/predict", methods=["POST"])
def predict():
    title = request.form.get("title", "")
    
    # Validate input text
    is_valid, error_message = validate_text(title)
    if not is_valid:
        return render_template("index.html", error=error_message, input_text=title)

    try:
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
    except Exception as e:
        # Handle any unexpected errors during prediction
        return render_template(
            "index.html", 
            error=f"An error occurred during analysis. Please try different text.",
            input_text=title
        )

if __name__ == "__main__":
    # For local development; Render will use gunicorn app:app
    app.run(host="0.0.0.0", port=5000, debug=True)
