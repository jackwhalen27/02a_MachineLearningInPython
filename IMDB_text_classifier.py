import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn import metrics

df = pd.read_csv("IMDB_movie_reviews_labeled.csv")

df.shape
df.head()
df.isna().sum()

df.sentiment.value_counts()

X = df.loc[:,['review']]
Y = df.sentiment
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3, stratify=Y)
Y_train.value_counts()

X_train_docs = [doc for doc in X_train.review]
vect = CountVectorizer(ngram_range=(1,3), stop_words='english', max_features=1000).fit(X_train_docs)
X_train_features = vect.transform(X_train_docs)
print('X_train_features:\n{}'.format(repr(X_train_features)))

feature_names = vect.get_feature_names_out()
print("Number of features: {}".format(len(feature_names)))
print("First 100 features:\n{}".format(feature_names[:100]))
print("Every 100th feature:\n{}".format(feature_names[::100]))

lin_svc = LinearSVC(max_iter = 120000)
scores = cross_val_score(lin_svc, X_train_features, Y_train, cv = 5)
print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))

lin_svc.fit(X_train_features, Y_train)

X_test_docs = [doc for doc in X_test.review]
X_test_features = vect.transform(X_test_docs)
Y_test_pred = lin_svc.predict(X_test_features)
metrics.accuracy_score(Y_test, Y_test_pred)
