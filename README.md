# SentimentAnalysisWebApp

A machine learning based sentiment analysis web application that predicts the emotional tone of user-entered text or reviews. The application classifies text into sentiments such as Positive, Negative, and Neutral using Natural Language Processing (NLP) techniques and an interactive web interface.

The project helps users analyze opinions, reviews, and feedback instantly while demonstrating the implementation of NLP preprocessing, text vectorization, machine learning model training, and web application deployment.

---

# Dataset Used

The project uses the IMDb Movie Reviews Dataset for training the sentiment classification model.
```bash
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
```

Dataset contains:
- Movie reviews
- Sentiment labels (Positive / Negative)

---

# NLP Techniques Used

- Text Cleaning
- Stopword Removal
- TF-IDF Vectorization
- Sentiment Classification
- Machine Learning Prediction

---

# Tech Stack

## Frontend
- Streamlit

## Backend / Machine Learning
- Python
- Scikit-learn
- NLP Preprocessing

## Libraries Used
- Pandas
- NLTK
- Joblib
- WordCloud
- Matplotlib

## Machine Learning Model
- Logistic Regression

---

# Tools & Platforms

- VS Code
- Git
- GitHub
- Streamlit Cloud

---

# Project Structure

```txt
SentimentAnalysis/
│
├── app.py
├── train.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── dataset.csv
└── README.md
```

# How to Run Locally

## Clone the repository

```bash
git clone https://github.com/shivanianajipuram/Sentiment_analysis.git
```

---

## Open the project folder

```bash
cd Sentiment_analysis
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the application

```bash
streamlit run app.py
```

---
## live demo 
```bash
https://sentiment-analysis-i1cq.onrender.com/
```
# Technologies Used

- Python
- Streamlit
- Scikit-learn
- NLTK
- Pandas
- TF-IDF Vectorizer
- Logistic Regression
- WordCloud
- Matplotlib
