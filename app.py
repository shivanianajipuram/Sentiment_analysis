# app.py

import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.split()

    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]

    return " ".join(text)

# Emoji mapping
emoji_dict = {
    "positive": "😀",
    "negative": "😡",
    "neutral": "😐"
}

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis App")

st.title("🧠 Sentiment Analysis Web App")

st.write("Enter a sentence or review to analyze sentiment.")

# User input
user_input = st.text_area("Enter Text")

if st.button("Analyze Sentiment"):

    if user_input.strip() != "":

        # Clean text
        cleaned = clean_text(user_input)

        # Vectorize
        vectorized_text = vectorizer.transform([cleaned])

        # Prediction
        prediction = model.predict(vectorized_text)[0]

        # Probability
        probabilities = model.predict_proba(vectorized_text)[0]

        # Show result
        st.subheader("Prediction")

        st.success(f"{prediction.upper()} {emoji_dict[prediction]}")

        # Confidence score
        st.subheader("Confidence Scores")

        labels = model.classes_

        for label, prob in zip(labels, probabilities):
            st.write(f"{label}: {prob:.2f}")

        # Bar chart
        st.bar_chart(probabilities)

        # Word cloud
        st.subheader("Word Cloud")

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white'
        ).generate(cleaned)

        fig, ax = plt.subplots()

        ax.imshow(wordcloud, interpolation='bilinear')

        ax.axis("off")

        st.pyplot(fig)

    else:
        st.warning("Please enter some text.")