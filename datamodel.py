import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

# Load preprocessed data
df = pd.read_csv('combined_sentiment.csv')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2, random_state=42)

# Build the model pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english')),
    ('classifier', LogisticRegression())
])

# Fit the model
pipeline.fit(X_train, y_train)

# Predict on test set and evaluate the model performance
y_pred = pipeline.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, pos_label='positive')

# Save the model
with open('reddit_sentiment_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# Print the model performance
print(f'Test dataset accuracy: {test_acc:.4f}')
print(f'Test dataset F1 score: {test_f1:.4f}')
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
classes = [0, 1]

# create visualization
fig, ax = plt.subplots()
im = ax.imshow(cm, cmap="Blues")

# add colorbar
cbar = ax.figure.colorbar(im, ax=ax)

# set tick labels and positions
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

# rotate x tick labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

for i in range(len(classes)):
    for j in range(len(classes)):
        text = ax.text(j, i, cm[i, j],
                       ha="center", va="center", color="white")
        
        # highlight incorrect predictions
        if i != j:
            text.set_color("red")

# add axis labels
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

# display visualization
plt.show()