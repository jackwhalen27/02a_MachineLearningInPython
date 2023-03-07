import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

df = pd.read_csv('IMDB_movie_reviews_labeled.csv')

st.title('Streamlit App!')
st.header('IMDB Pipeline Test Example')
st.write('Below is a summary of the dataset:')
st.write(df.head())

X = df.loc[:, ['review']]
y = df.sentiment

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

X_train_docs = [doc for doc in X_train.review]

pipeline = Pipeline([
('vect', TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=1000)), 
('cls', LinearSVC())
])

pipeline.fit(X_train_docs, y_train)

cross_val_score(pipeline, X_train_docs, y_train, cv=5).mean()

predicted = pipeline.predict([doc for doc in X_test.review])

accuracy_score(y_test, predicted)

df_sample = df = pd.read_csv("IMDB_movie_reviews_test_sample.csv")

predicted_sample = pipeline.predict([doc for doc in df_sample.review])

accuracy_score(df_sample.sentiment, predicted_sample)