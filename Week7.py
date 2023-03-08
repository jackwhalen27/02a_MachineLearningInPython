import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

# Data
df = pd.read_csv('IMDB_movie_reviews_labeled.csv')

# Train-Test Split
X = df.loc[:, ['review']]
y = df.sentiment

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Training the Model
X_train_docs = [doc for doc in X_train.review]

pipeline = Pipeline([
('vect', TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=1000)), 
('cls', LinearSVC())
])

pipeline.fit(X_train_docs, y_train)

# Model Evaluation
cv_score = cross_val_score(pipeline, X_train_docs, y_train, cv=5).mean()
st.write('Cross-validation score:', cv_score)

predicted = pipeline.predict([doc for doc in X_test.review])
test_accuracy = accuracy_score(y_test, predicted)
st.write('Test set accuracy:', test_accuracy)

cm = confusion_matrix(y_test, predicted)
st.write('Confusion Matrix:', cm)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],
       title='Confusion Matrix',
       ylabel='True label',
       xlabel='Predicted label')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")

# Create form for user input
st.write('### Make a Prediction')
review_text = st.text_input('Enter review text:')
prediction = pipeline.predict([review_text])
if review_text:
    st.write('Prediction:', prediction[0])
else:
    st.write('Please enter review text to make a prediction.')
