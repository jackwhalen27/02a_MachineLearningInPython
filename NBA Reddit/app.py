import streamlit as st
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from PIL import Image
from google.cloud import firestore
import requests
import json
import os
import matplotlib.pyplot as plt
from matplotlib import dates as mpl_dates
from datetime import datetime
import seaborn as sns
import datetime as dt
from datetime import datetime, timedelta
import schedule
import time

# Define the contents of the "Home" tab
def home():
    st.title("r/NBA Sentiment Analysis Project")
    st.subheader("Created by: Gabriel Kusiatin, Jack Whalen, Abe Zaidman, Kayvan Khoobehi, Daniel Cohen")
    "---"
    st.subheader("Project Design and Objective")
    st.write("The goal of this project is to create a model that accuratley predicts the sentiment of posts pulled from the r/NBA reddit page. We're hoping that those who visit our web application can accurately understand the sentiment surrounding any topic being discussed in the r/NBA page.")
    st.write("This project was created as a part of the Applied Machine Learning course in the Master's of Business Analytics program at the A.B. Freeman School of Business at Tulane University. This project is the culmination of everything we learned over the course of our Spring semester in Applied Machine Learning and we hope that those who visit our site enjoy all the work we've done!")
    "---"
    st.subheader("Connect With Us!")
    images = {
    "Gabriel Kusiatin": {
        "path": "1647971708087.jpg",
        "link": "https://www.linkedin.com/in/gabriel-kusiatin/"
    },
    "Jack Whalen": {
        "path": "Johnelite.jpg",
        "link": "https://www.linkedin.com/in/jackjwhalen/"
    },
    "Abe Zaidman": {
        "path": "Tore.jpg",
        "link": "https://www.linkedin.com/in/abezaidman/"
    },
    "Kayvan Khoobehi": {
        "path": "Kayvan.jpg",
        "link": "https://www.linkedin.com/in/kayvankhoobehi/"
    },
    "Daniel Cohen": {
        "path": "Landlord.jpg",
        "link": "https://www.linkedin.com/in/daniel-cohen17/"
    }
}

# Display the pictures side by side with names and hyperlinks
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        image = Image.open(images["Gabriel Kusiatin"]["path"])
        st.image(image)
        st.markdown(f"[{list(images.keys())[0]}]({images['Gabriel Kusiatin']['link']})")
    
    with col2:
        image = Image.open(images["Jack Whalen"]["path"])
        st.image(image)
        st.markdown(f"[{list(images.keys())[1]}]({images['Jack Whalen']['link']})")

    with col3:
        image = Image.open(images["Abe Zaidman"]["path"])
        st.image(image)
        st.markdown(f"[{list(images.keys())[2]}]({images['Abe Zaidman']['link']})")
    
    with col4:
        image = Image.open(images["Kayvan Khoobehi"]["path"])
        st.image(image)
        st.markdown(f"[{list(images.keys())[3]}]({images['Kayvan Khoobehi']['link']})")
    
    with col5:
        image = Image.open(images["Daniel Cohen"]["path"])
        st.image(image)
        st.markdown(f"[{list(images.keys())[4]}]({images['Daniel Cohen']['link']})")

# Define the contents of the "Model Training" tab
def model_training():
    st.title("Model Training")
    st.write("This is where you can train your machine learning model.")

# Define the contents of the "Model Prediction and Results" tab
def model_prediction():
    st.title("Model Prediction and Results")
    st.write("This is where you can make predictions and view the results.")

# Define the contents of the "Google Cloud Data Automation" tab
def google_cloud():
    st.title("Google Cloud Data Automation")
    "---"
    st.subheader("About:")
    st.write("In order to keep our model and data up-to-date, we utilized Google Cloud Platform to create a machine learning/data collection workflow that updates everyday at 8 am. This means that every morning at 8, our Google Cloud Functions collect posts from r/NBA, processes them into a structure our model can interpret, and then predict the sentiment based on the text of each individual post.")
    "---"
    # Set Google Cloud credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "aml-final-project-384918-b801224e3e8c.json"
    # Create a Firestore client
    db = firestore.Client(project = 'aml-final-project-384918')
    # Get a reference to a collection
    collection_ref = db.collection(u'reddit_NBA')
    posts = list(collection_ref.stream())
    docs_dict = []
    for post in posts:
        doc_dict = post.to_dict()
        doc_dict['created_utc'] = post.create_time.timestamp()
        docs_dict.append(doc_dict)
    df = pd.DataFrame(docs_dict)
    df1 = df.copy()
    df1 = df1[["title", "selftext", 'created_utc']]
    df1['created_utc'] = pd.to_datetime(df1['created_utc'], unit='s').dt.date
    df1.selftext = df1.selftext.fillna(" ")
    df1['text'] = df1.selftext + " " + df1.title
    df1 = df1.drop(["title", "selftext"], axis=1)
    def preprocess_text(text):
        if not isinstance(text, (str, bytes)):
            return ''
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
    df1['text'] = df1['text'].apply(preprocess_text)
    def analyze_sentiment(text):
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(text)['compound']
        return sentiment
    df1['sentiment_score'] = df1['text'].apply(analyze_sentiment)
# Add a new column to hold the sentiment labels
    df1['sentiment'] = df1['sentiment_score'].apply(lambda x: 'positive' if x >= 0 else 'negative')
    df1 = df1.drop('sentiment_score', axis= 1)
    df1 = df1.sort_values('created_utc', ascending= False)
    st.subheader("Reddit posts from r/NBA, pulled everyday at 8 am")
    st.dataframe(df1)
    "---"
    st.subheader("Number of posts per day")
    # Group the data by date and count the number of posts for each date
    date_counts = df1.groupby('created_utc').size().reset_index(name='count')
    fig, ax = plt.subplots()
    ax.bar(date_counts['created_utc'], date_counts['count'])
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Posts')
    ax.set_title('Posts Pulled by Date')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    st.write("The bar chart above shows the number of posts pulled from r/NBA each day since our cloud functions were established (04/26/2023). The number of posts stored in our Google Cloud Firestore is dependent on the number of posts posted and can not exceed more than 100 as limited by Reddit's API.")
    "---"
    st.subheader("Model Performance on Live Data")
    with open('reddit_sentiment_model.pkl', 'rb') as f:
        trained_model = pickle.load(f)
    df2 = df1.copy()
    df2 = df1.drop('created_utc', axis = 1)
    y_pred = trained_model.predict(df2['text'])
    test_accuracy = accuracy_score(df2['sentiment'], y_pred)
    test_f1_score = f1_score(df2['sentiment'], y_pred, pos_label='positive')
    st.write(f"Accuracy: {test_accuracy}")
    st.write(f"F1_Score: {test_f1_score}")

    # Define a function to run the code
    def update_dataframe():
        # Load the existing dataframe or create a new one if it doesn't exist
        try:
            performance_by_date = pd.read_csv('performance_by_date.csv')
        except FileNotFoundError:
            performance_by_date = pd.DataFrame(columns=['created_utc', 'accuracy', 'f1_score'])

        # get the current date
        today = pd.Timestamp.now().normalize()

        # append the current date and performance to the dataframe
        performance_by_date = performance_by_date.append({
            'created_utc': today,
            'accuracy': test_accuracy,
            'f1_score': test_f1_score
        }, ignore_index=True)

        # save the updated dataframe to a csv file
        performance_by_date.to_csv('performance_by_date.csv', index=False)

    # Define a function to schedule the update_dataframe function to run at 8:10am every day
    def schedule_update():
        schedule.every().day.at("08:10").do(update_dataframe)

        while True:
            schedule.run_pending()
            time.sleep(1)

    # Call the function to schedule the update
    schedule_update()
    df_model = pd.DataFrame('performance_by_date.csv')
    st.dataframe(df_model)

    # Create the tabs
tabs = ["Home", "Model Training", "Model Prediction and Results", "Google Cloud Data Automation"]
page = st.sidebar.selectbox("Select a page", tabs)

# Display the selected page with its corresponding content
if page == "Home":
    home()
elif page == "Model Training":
    model_training()
elif page == "Model Prediction and Results":
    model_prediction()
elif page == "Google Cloud Data Automation":
    google_cloud()

