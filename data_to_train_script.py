import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# Load the dataset
df = pd.read_csv("train_data.csv")

# Check necessary columns
if 'Cleaned_Summary' not in df.columns or 'Genre_List' not in df.columns:
    raise ValueError("Missing required columns: 'Cleaned_Summary' and/or 'Genre_List'")

# Convert genre string lists to actual Python lists (if stored as strings)
df['Genre_List'] = df['Genre_List'].apply(eval)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(df['Cleaned_Summary'])

# Multi-label Binarization for genres
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['Genre_List'])

# Save preprocessed features and labels
with open("tfidf_features.pkl", "wb") as f:
    pickle.dump(X, f)

with open("genre_labels.pkl", "wb") as f:
    pickle.dump(y, f)

# Save vectorizer and label encoder for reuse
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("label_binarizer.pkl", "wb") as f:
    pickle.dump(mlb, f)

print("✅ Preprocessing complete. Saved tfidf_features.pkl, genre_labels.pkl, tfidf_vectorizer.pkl, and label_binarizer.pkl")
