import pickle
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load the trained model
with open('reddit_sentiment_model.pkl', 'rb') as f:
    trained_model = pickle.load(f)


new_df = pd.read_csv("reddit_basketball_posts_sentiment.csv")

# Make predictions on the new dataset
new_dataset_predictions = trained_model.predict(new_df['text'])

# If you have the true labels for the new dataset, you can evaluate the model's performance
new_y_true = new_df['sentiment']  # Replace 'sentiment' with the actual column name containing the true labels
new_acc = accuracy_score(new_y_true, new_dataset_predictions)
new_f1 = f1_score(new_y_true, new_dataset_predictions, pos_label='positive')
new_cm = confusion_matrix(new_y_true, new_dataset_predictions)

print(f'New dataset accuracy: {new_acc:.4f}')
print(f'New dataset F1 score: {new_f1:.4f}')
print("New dataset confusion matrix:")
print(new_cm)