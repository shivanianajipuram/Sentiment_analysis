# train.py

import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("dataset.csv")

# Take smaller random sample (faster training)
df = df.sample(5000, random_state=42)

# Check dataset
print(df.head())
print(df.columns)

# =========================
# TEXT CLEANING FUNCTION
# =========================

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters/numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Split into words
    text = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))

    text = [word for word in text if word not in stop_words]

    # Join words again
    return " ".join(text)

# =========================
# CLEAN TEXT
# =========================

df['cleaned_text'] = df['review'].apply(clean_text)

# =========================
# FEATURES AND LABELS
# =========================

X = df['cleaned_text']

y = df['sentiment']

# =========================
# TF-IDF VECTORIZATION
# =========================

vectorizer = TfidfVectorizer(max_features=5000)

X_vectorized = vectorizer.fit_transform(X)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL
# =========================

model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2f}")

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "model.pkl")

joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")