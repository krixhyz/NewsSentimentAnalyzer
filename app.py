from flask import Flask, request, render_template
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained models from the models folder
sentiment_model = joblib.load('models/naive_bayes_model.joblib')
category_model = joblib.load('models/logistic_regression_model.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Convert to lower case and remove punctuation
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Lemmatization and stop words removal
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Prediction function
def predict_sentiment_and_category(title):
    # Preprocess the input title
    clean_title = preprocess_text(title)
    
    # Transform the title using the pre-trained TF-IDF vectorizer
    title_vectorized = vectorizer.transform([clean_title])
    
    # Predict sentiment and category
    sentiment = sentiment_model.predict(title_vectorized)[0]
    category = category_model.predict(title_vectorized)[0]
    
    # VADER Sentiment Analysis
    vader_score = analyzer.polarity_scores(title)['compound']
    sentiment_label = 'positive' if vader_score > 0 else 'negative' if vader_score < 0 else 'neutral'
    
    return sentiment, category, sentiment_label, vader_score

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        title = request.form['title']
        
        # Get predictions
        sentiment, category, sentiment_label, vader_score = predict_sentiment_and_category(title)
        
        return render_template('result.html', title=title, sentiment=sentiment_label, category=category, vader_score=vader_score)

if __name__ == '__main__':
    app.run(debug=True)
