import streamlit as st
import numpy as np
import pandas as pd
import pickle
import sklearn.metrics

#Load testing data and preprocessing:
test = pd.read_csv("MedsTest.csv")

test['Review'] = test['benefits_review'] + ' ' + test['side_effects_review'] + ' ' + test['comments_review']
test['BinSentiment'] = test['rating'].apply(lambda x: 1 if x > 5 else 0)

#Create example table globally
Example = test.iloc[:, : 4].sample(n=1, random_state=1234)
    
ExampleDF = pd.DataFrame(Example)

#Load model and results using Pickle
with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('predictions.pickle', 'rb') as f:
    predictions = pickle.load(f)

with open('scores.pickle', 'rb') as f:
    accuracy, f1 = pickle.load(f)

with open('vectorizer', 'rb') as f:
    vectorizer = pickle.load(f)

#first, define functions needed for app:
def input_processing(text):
    processed_input = text.lower().strip()
    return processed_input

def sentiment(text):
    processed_input = input_processing(text)
    user_test = vectorizer.transform([processed_input])
    user_pred = model.predict(user_test)[0]
    return user_pred

#define the app itself:
def ind_assgn_app():

    #Introduction and problem description
    st.title("Medication Sentiment Classification Model (By Jack Whalen)")
    
    st.header("Problem Description:")
    st.write("For this assignment, we were tasked with creating non-deep learning models to accurately predict sentiment towards medications based on text reviews and numeric ratings. Data to analyze included comments on medication benefits, comments on side effects, general comments, and a rating for the drug overall between 1 and 10.")
    st.write("Below, you will see the process used to test and improve various models, as well as the basis on which the final model was selected for end-user testing.")

    #examples of medication reviews
    st.header("Review Example:")
    st.subheader("A review found in the dataset will look like the following:")
    st.table(ExampleDF)

    #Data Breakdown:
    st.header("Dataset Breakdown and Summary:")
    st.subheader("Below are summary statistics of the dataset used for training and validating the model:")
    st.write("Total Number of Reviews: " + str(len(test)))
    st.write("Count of Positive Reviews: " + str(sum(test['BinSentiment']==1)))
    st.write("Count of Negative Reviews: " + str(sum(test['BinSentiment']==0)))
    st.write("Ratio of Positive to Negative Reviews: " + str((sum(test['BinSentiment']==1))/len(test))[:6])

    #Preprocessing and Vectorization steps:
    st.header("Text Preprocessing and Data Tokenization using TF-IDF:")
    st.subheader("Before training the model, the data was preprocessed using the following methods:")
    st.write("Text Preprocessing: Basic data cleaning techniques were used to combine the three review columns into one overall review column, and all capital letters and punctuation were removed from all text data.")
    st.write("Binary Sentiment Indicator: A new column was created to classify the review's sentiment as either 1 (meaning positive) or 0 (meaning negative) based on its numerical rating between 1 and 10, where a review rated 5 or lower is considered negative.")
    st.write("Text Feature Extraction: The model utilizes a Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer to extract features from the review data. The vectorizer uses english stop-words and has a maximum feature count of 8,000.")

    #Model Training/Tuning/Selection:
    st.header("Model Training, Testing, and Tuning:")
    st.subheader("Three Models Tested: Logistic Regression, Naive Bayes, and Support Vector Machine (SVM):")
    st.write("The three models tested yielded similar accuracy and f1 scores, as shown below:")
    st.write("Logistic Regression: Accuracy = 73.65%. F-1 Score = 66.19%")
    st.write("Naive Bayes: Accuracy = 70.37%. F-1 Score = 58.23%")
    st.write("SVM: Accuracy = 72.59%. F-1 Score = 63.78%")

    st.subheader("Logistic Regression and SVM scores after changing parameters using Gridsearch hyperparameter tuning:")
    st.write("Logistic Regression: Accuracy = 77.99%. F-1 Score = 85.64%")
    st.write("SVM: Accuracy = 76.83%. F-1 Score = 85.20%")

    st.subheader("Other model types, such as XGBoost and a RandomForest model, were also trained and tuned, but resulted in much lower accuracy scores. Therefore, Logistic Regression is chosen as the final model.")

    #Final Model:
    st.header("Final Model Chosen: Logistic Regression")
    st.write("The logistic regression model was first created using the sklearn package, with the maximum number of iterations set to 8,000.")
    st.write("As mentioned above, our model's parameters were tuned using Gridsearch, with 5 cross-validations and a total of 600 possible parameter combinations evaluated.")
    st.write("While other model improvement methods were performed, no other significant improvements to model accuracy were discovered.")
    st.subheader("Final Model Scores: Accuracy = 77.99%. F-1 Score = 85.64%")

    #User-Testing of Model
    st.header("Test the model yourself!")
    st.subheader("Enter a medication review below, and the model will make a best-guess prediction as to the sentiment of the review.")
    st.write("DISCLAIMER: Model is not perfect. Certain inputs may be classified incorrectly.")

    text = st.text_input("Medication Review:", "")
    if text:
        input_processing(text)
        y_pred = sentiment(text)
        if y_pred == 1:
            st.write("This review is: Positive")
        else:
            st.write("This review is: Negative")

if __name__ == "__main__":
    ind_assgn_app()
