# Filmception_AIpowered_model


# 🎬 Movie Summary AI System

This project develops a comprehensive AI-powered system for **processing movie summaries**, **predicting movie genres**, and **converting summaries into audio** across multiple languages. It integrates data preprocessing, machine learning, translation, and text-to-speech technologies into an interactive menu-driven application.

---

## 🧩 Project Overview

The system accepts user-input movie summaries and provides two main functionalities:

1. **Summary to Audio Conversion**  
   Converts movie summaries into speech in multiple languages (Arabic, Urdu, Korean) using translation APIs and Text-to-Speech (TTS) engines.

2. **Movie Genre Prediction**  
   Predicts one or more genres for a given movie summary using a multi-label classification machine learning model trained on the CMU Movie Summary Dataset.

---

## 🚀 Components

### 1. Data Preprocessing & Cleaning
- Remove special characters, stopwords, numbers, and redundant spaces
- Tokenize, lowercase, and lemmatize text
- Extract multi-label genres from metadata
- Train-test split ensuring no data leakage

### 2. Text Translation & Audio Generation
- Translate summaries into Arabic, Urdu, and Korean using APIs (e.g., Google Translate, MarianMT)
- Generate audio from translated text using TTS libraries (e.g., gTTS)
- Save audio files for at least 50 summaries

### 3. Genre Prediction Model
- Build a multi-label classifier (options: Logistic Regression, Random Forest, LSTM, Transformer models)
- Feature extraction via TF-IDF, word embeddings, or transformers
- Evaluate with accuracy, precision, recall, F1-score, and confusion matrix

### 4. Interactive Menu Interface
- User inputs movie summary
- Choose to convert to audio or predict genre
- Select preferred language for audio playback

---

## 📂 Files Included

- `data/`: Preprocessed movie summaries and metadata
- `model/`: Trained machine learning model files
- `scripts/`: Data preprocessing, translation, TTS, and prediction scripts
- `app.py` or `main.py`: Interactive menu-driven system
- `requirements.txt`: Python dependencies

---

## 📚 Technologies Used

- Python (Pandas, Scikit-learn, TensorFlow/PyTorch)
- NLP libraries (NLTK, SpaCy, Hugging Face Transformers)
- Translation APIs (Google Translate, MarianMT)
- TTS libraries (gTTS, pyttsx3)
- Multi-label classification techniques

---

## 🔧 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
