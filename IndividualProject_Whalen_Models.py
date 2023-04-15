#Load Packages
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

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

#Changing ratings to binary values (1 is positive, 0 is negative)
TrainSet['BinSentiment'] = TrainSet['rating'].apply(lambda x: 1 if x > 5 else 0)
TestSet['BinSentiment'] = TestSet['rating'].apply(lambda x: 1 if x > 5 else 0)

#training and testing split
train_comments, val_comments, train_sentiment, val_sentiment = train_test_split(
    TrainSet['Review'], TrainSet['BinSentiment'], test_size=0.3, stratify=TrainSet['BinSentiment'], random_state=1234
)
y_test = TestSet['BinSentiment']

#TF-IDF Vectorizer for Feature Extraction
vectorizer = TfidfVectorizer(stop_words = "english", max_features = 8000)
x_train_vectorized = vectorizer.fit_transform(train_comments)
x_val_vectorized = vectorizer.transform(val_comments)
x_test_vectorized = vectorizer.transform(TestSet['Review'])

#MODEL TRAINING:
#Model 1: Logistic Regression:
log_reg = LogisticRegression(max_iter=8000)
log_reg.fit(x_train_vectorized, train_sentiment)

#Model 2: Naive Bayes:
nb = MultinomialNB()
nb.fit(x_train_vectorized, train_sentiment)

#Model 3: SVC:
svc = SVC()
svc.fit(x_train_vectorized, train_sentiment)

#MODEL EVALUATION:
y_pred_logreg = log_reg.predict(x_test_vectorized)
y_pred_nb = nb.predict(x_test_vectorized)
y_pred_svc = svc.predict(x_test_vectorized)

#Accuracy Scores:
print("Logistic Regression: ", metrics.accuracy_score(y_test, y_pred_logreg))
print("Naive Bayes: ", metrics.accuracy_score(y_test, y_pred_nb))
print("SVC: ", metrics.accuracy_score(y_test, y_pred_svc))

#Precision Scores:
print("Logistic Regression: \n", metrics.precision_score(y_test, y_pred_logreg, average='weighted', zero_division=0))
print("Naive Bayes: \n", metrics.precision_score(y_test, y_pred_nb, average='weighted'))
print("SVC: \n", metrics.precision_score(y_test, y_pred_svc, average='weighted'))

#F1-Scores:
print("Logistic Regression: ", metrics.f1_score(y_test, y_pred_logreg, average='weighted'))
print("Naive Bayes: ", metrics.f1_score(y_test, y_pred_nb, average='weighted'))
print("SVC: ", metrics.f1_score(y_test, y_pred_svc, average='weighted'))

#Gridsearch Parameter Tuning: SVC Model
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'sigmoid', 'poly']}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

grid.fit(x_train_vectorized, train_sentiment)
print(grid.best_estimator_)

y_pred_grid = grid.predict(x_test_vectorized)
print(metrics.accuracy_score(y_test, y_pred_grid))
print(metrics.f1_score(y_test, y_pred_grid))

#Gridsearch Parameter Tuning: Logistic Regression
param_grid2 = parameters = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty':['none', 'elasticnet', 'l1', 'l2'],
    'C':[0.001, 0.01, 0.1, 1, 10, 100]}

grid2 = GridSearchCV(LogisticRegression(), param_grid2, refit=True, cv=5, verbose=3)

grid2.fit(x_train_vectorized, train_sentiment)
print(grid2.best_estimator_)

y_pred_grid2 = grid2.predict(x_test_vectorized)
print(metrics.accuracy_score(y_test, y_pred_grid2))
print(metrics.f1_score(y_test, y_pred_grid2))

#Tuned SVC and LogReg models improve in both accuracy and f-1 scores. Train one more model and pick best one
import xgboost as xgb

xgb_model = xgb.XGBClassifier(random_state = 1234)
xgb_param_grid = {'learning_rate': [.05,.01,.1,.2,.3],
                  'max_depth': range(2,10,1),
                  'n_estimators': range(50,300,50)}

xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv = 5)
xgb_grid.fit(x_train_vectorized, train_sentiment)

xgb_params = xgb_grid.best_params_

best_xgb = xgb.XGBClassifier(random_state = 1234, **xgb_params)
best_xgb.fit(x_train_vectorized, train_sentiment)

y_pred_xgb = best_xgb.predict(x_test_vectorized)

print(metrics.accuracy_score(y_test, y_pred_xgb))
print(metrics.f1_score(y_test, y_pred_xgb))

#All accuracy scores still under 78%. Will try Random Forrest and select best model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=1234)
rf.fit(x_train_vectorized, train_sentiment)

y_pred_rf = rf.predict(x_test_vectorized)
print(metrics.accuracy_score(y_test, y_pred_rf))
print(metrics.f1_score(y_test, y_pred_rf))

rf_param_grid = {'n_estimators': range(0,500,10),
                 'max_depth': range(1,10,1)
}

rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv = 5)
rf_grid.fit(x_train_vectorized, train_sentiment)

y_pred_rf_grid = rf_grid.predict(x_test_vectorized)
print(metrics.accuracy_score(y_test, y_pred_rf_grid))
print(metrics.f1_score(y_test, y_pred_rf_grid))

#FINAL CHOSEN MODEL: LogisticRegression
model_vF = grid2
predictions_vF = y_pred_grid2

