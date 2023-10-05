"""
Will be importing code from ThreadsSentimentAnalysis.ipynb

BERT can be used in replacement, but it does take long to compile, therefore, XGBoost is being used
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from gensim.models import Word2Vec
df = pd.read_csv('data/threads_review.csv')
df['sentiment'] = df['rating'].apply(lambda x: 1 if x in [4, 5] else 0)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=123)
train_reviews = [review.split() for review in train_df['review_description']]
test_reviews = [review.split() for review in test_df['review_description']]
vec_model = Word2Vec(sentences=train_reviews, vector_size=100, window=25, min_count=10, sg=0)
def review_to_vector(review):
    vectors = [vec_model.wv[word] for word in review if word in vec_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(100)
X_train = np.array([review_to_vector(review) for review in train_reviews])
X_test = np.array([review_to_vector(review) for review in test_reviews])
model = xgb.XGBClassifier()
model.fit(X_train, train_df['sentiment'])

"""
Creating GUI window for the Sentiment Analysis
"""

import tkinter as tk

# GUI window
window = tk.Tk()
window.title("Sentiment Analysis")

# Creating text box
input_label = tk.Label(window, text="Enter Text Below:")
input_label.pack()
input_text = tk.Text(window, height=8, width=45)
input_text.pack()

# Analyze using XGBoost
def analyze_sentiment_xgboost():
    text = input_text.get("1.0", "end-1c")  # Get the input text

    # Preprocess text
    review_vector = review_to_vector(text.split())

    # Passing through XGBoost model
    sentiment = model.predict([review_vector])[0]

    # Sentiment review labels
    sentiment_labels = ["Negative", "Positive"]

    # Displaying results from sentiment_labels
    result_label.config(text=f"Sentiment: {sentiment_labels[sentiment]}")

# Pass text through XGBoost
analyze_button_xgboost = tk.Button(window, text="Analyze Sentiment", command=analyze_sentiment_xgboost)
analyze_button_xgboost.pack()

# Display results
result_label = tk.Label(window, text="")
result_label.pack()

# Run the GUI
window.mainloop()