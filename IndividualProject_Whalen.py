#Load Packages
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.linear_model import LinearRegression

#Load Training and Testing Data
TrainSet = pd.read_csv("MedsTrain.csv")
TestSet = pd.read_csv("MedsTest.csv")

#Text Preprocessing to remove capitalization and punctuation
TrainSet['benefits_review'] = TrainSet["benefits_review"].str.lower().replace('[^\w\s]', '', regex=True)
TrainSet['side_effects_review'] = TrainSet['side_effects_review'].str.lower().replace('[^\w\s]', '', regex=True)
TrainSet['comments_review'] = TrainSet['comments_review'].str.lower().replace('[^\w\s]', '', regex=True)

TestSet['benefits_review'] = TestSet["benefits_review"].str.lower().replace('[^\w\s]', '', regex=True)
TestSet['side_effects_review'] = TestSet['side_effects_review'].str.lower().replace('[^\w\s]', '', regex=True)
TestSet['comments_review'] = TestSet['comments_review'].str.lower().replace('[^\w\s]', '', regex=True)

#Concatenating Comment columsn to create single features variable
TrainSet['Review'] = TrainSet['benefits_review'] + ' ' + TrainSet['side_effects_review'] + ' ' + TrainSet['comments_review']
TestSet['Review'] = TestSet['benefits_review'] + ' ' + TestSet['side_effects_review'] + ' ' + TestSet['comments_review']

#training and testing split
x_train = TrainSet['Review']
x_test = TestSet['Review']
y_train = TrainSet['rating']
y_test = TestSet['rating']

#bag-of-words feature extraction
vect = CountVectorizer()
x_train = vect.fit_transform(x_train)
x_test = vect.transform(x_test)

