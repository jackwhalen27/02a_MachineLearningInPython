import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn import metrics

traindf = pd.read_csv("train.csv")
testdf = pd.read_csv("test.csv")

traindf.head()
testdf.head()

traindf.rating.value_counts()
testdf.rating.value_counts()

