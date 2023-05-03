import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Convert to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Remove non-alphabetic characters and words less than 3 characters
    tokens = [token for token in tokens if token.isalpha() and len(token) > 2]
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens into string and return
    return ' '.join(tokens)

df = pd.read_csv('combined.csv')
df1 = df.copy()
df1 = df1[["Title", "Body"]]  
df1.Body = df1.Body.fillna(" ")
df1['text'] = df1.Body + " " + df1.Title
df1 = df1.drop(["Body", "Title"], axis=1)

df1['text'] = df1['text'].apply(preprocess_text)

def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)['compound']
    return sentiment

df1['sentiment_score'] = df1['text'].apply(analyze_sentiment)

# Add a new column to hold the sentiment labels
df1['sentiment'] = df1['sentiment_score'].apply(lambda x: 'positive' if x >= 0 else 'negative')

df1.to_csv('combined_sentiment.csv', index=False)

