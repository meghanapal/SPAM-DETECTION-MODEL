---

# AI Spam Message Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![API](https://img.shields.io/badge/API-FastAPI-green)
![NLP](https://img.shields.io/badge/NLP-TFIDF-purple)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

# Project Overview

This project implements an **AI-powered Spam Detection System** that classifies SMS or text messages as **Spam or Ham (Not Spam)** using **Machine Learning and Natural Language Processing (NLP)**.

Spam messages are commonly used for advertising, phishing, and scams. This system automatically identifies such messages using a trained machine learning model.

The model is trained using **Multinomial Naive Bayes** with **TF-IDF text vectorization** and is deployed using a **FastAPI REST API** so it can be integrated into real-world web applications.

---

# Project Features

* AI-based **Spam Message Classification**
* Natural Language Processing for text analysis
* TF-IDF vectorization for feature extraction
* Multinomial Naive Bayes machine learning model
* Model serialization using Pickle
* REST API deployment using FastAPI
* Ready for web or mobile application integration

---

# System Architecture

```
                +------------------+
                |   User Message   |
                +------------------+
                         |
                         v
                +------------------+
                |  Web Application |
                +------------------+
                         |
                         v
                +------------------+
                |   FastAPI Server |
                +------------------+
                         |
                         v
                +------------------+
                |  TF-IDF Vector   |
                |   Transformation |
                +------------------+
                         |
                         v
                +------------------+
                | Naive Bayes Model|
                +------------------+
                         |
                         v
                +------------------+
                | Spam / Not Spam  |
                +------------------+
```

---

# Machine Learning Pipeline

```
Dataset
   ↓
Data Cleaning
   ↓
Text Preprocessing
   ↓
TF-IDF Feature Extraction
   ↓
Train/Test Split
   ↓
Naive Bayes Model Training
   ↓
Model Evaluation
   ↓
Model Saving (.pkl)
   ↓
API Integration
```

---

# Technologies Used

| Technology        | Purpose                     |
| ----------------- | --------------------------- |
| Python            | Programming language        |
| Pandas            | Data manipulation           |
| Scikit-learn      | Machine learning algorithms |
| NLP               | Text preprocessing          |
| TF-IDF Vectorizer | Feature extraction          |
| Naive Bayes       | Spam classification         |
| Pickle            | Model storage               |
| FastAPI           | API development             |
| Uvicorn           | API server                  |

---

# Project Structure

```
SpamDetectionModel
│
├── dataset
│   └── spam.csv
│
├── model_training.ipynb
│
├── predict_spam.py
│
├── api.py
│
├── spam_model.pkl
│
├── vectorizer.pkl
│
└── README.md
```

---

# Dataset

The dataset contains SMS messages labeled as **Spam or Ham**.

Example:

| Label | Message                                  |
| ----- | ---------------------------------------- |
| ham   | Hey, are we meeting later?               |
| spam  | Congratulations! You've won a free prize |

Dataset fields:

* **label** → spam or ham
* **message** → SMS text content

---

# Model Training

The model uses **Multinomial Naive Bayes**, a probabilistic classifier suitable for text classification tasks.

### Training Steps

1. Load dataset using Pandas
2. Clean and preprocess text
3. Convert text into vectors using TF-IDF
4. Split dataset into training and testing sets
5. Train Naive Bayes classifier
6. Evaluate model performance
7. Save trained model

---

# Model Training Code

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

df = pd.read_csv("spam.csv")

X = df["message"]
y = df["label"]

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

pickle.dump(model, open("spam_model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))
```

---

# Prediction Script

File: `predict_spam.py`

```python
import pickle

model = pickle.load(open("spam_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

def predict(message):
    vector = vectorizer.transform([message])
    prediction = model.predict(vector)
    return prediction[0]

print(predict("Congratulations! You won a free prize"))
```

---

# API Integration (FastAPI)

File: `api.py`

```python
from fastapi import FastAPI
import pickle

app = FastAPI()

model = pickle.load(open("spam_model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.get("/")
def home():
    return {"message": "Spam Detection API Running"}

@app.get("/predict")
def predict_spam(message: str):
    vector = vectorizer.transform([message])
    prediction = model.predict(vector)
    return {"prediction": prediction[0]}
```

---

# Running the Project

## Install Dependencies

```
pip install fastapi uvicorn scikit-learn pandas
```

---

## Start the API

```
uvicorn api:app --reload
```

---

## Open in Browser

```
http://127.0.0.1:8000
```

---

## Test Prediction

```
http://127.0.0.1:8000/predict?message=You won a free lottery
```

---

# Example API Response

```
{
  "prediction": "spam"
}
```

---

# Model Advantages

* Simple and efficient
* Fast training
* Works well for text classification
* Low computational cost
* Easy deployment

---

# Future Improvements

* Use **Deep Learning models (LSTM / BERT)**
* Create a **Web UI**
* Deploy to **AWS / Google Cloud**
* Add **Email spam detection**
* Implement **real-time filtering**

---

# Author

AI / Machine Learning Academic Project

---

# License

This project is open-source and available under the **MIT License**.

---
