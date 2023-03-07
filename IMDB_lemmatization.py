import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn import metrics

df = pd.read_csv('IMDB_movie_reviews_labeled.csv')

df.sentiment.value_counts()

X = df.loc[:,['review']]
y = df.sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
y_train.value_counts()
X_train_docs = [doc for doc in X_train.review]

#pip install -U spacy
import spacy
import re
from spacy.lang.en.stop_words import STOP_WORDS

en_nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
pattern = re.compile('(?u)\\b\\w\\w+\\b')

def custom_tokenizer(document):
    doc_spacy = en_nlp(document)
    lemmas = [token.lemma_ for token in doc_spacy]
    return [token for token in lemmas if token not in STOP_WORDS and pattern.match(token)]

vect = TfidfVectorizer(tokenizer=custom_tokenizer, ngram_range=(1, 2), max_features=1000).fit(X_train_docs)

X_train_features = vect.transform(X_train_docs)

feature_names = vect.get_feature_names()
print("Number of features: {}".format(len(feature_names)))
print("First 100 features:\n{}".format(feature_names[:100]))
print("Every 100th feature:\n{}".format(feature_names[::100]))

lin_svc = LinearSVC(max_iter=120000)
scores = cross_val_score(lin_svc, X_train_features, y_train, cv=5)
print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores)))

lin_svc.fit(X_train_features, y_train)

X_test_docs = [doc for doc in X_test.review]
X_test_features = vect.transform(X_test_docs)

y_test_pred = lin_svc.predict(X_test_features)
metrics.accuracy_score(y_test, y_test_pred)
