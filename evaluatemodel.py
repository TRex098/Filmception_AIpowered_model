import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
df = pd.read_csv("train_data.csv")
df["Genre_List"] = df["Genre_List"].apply(lambda x: x.strip("[]").replace("'", "").split(", "))

# Load saved components
model = joblib.load("genre_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
mlb = joblib.load("label_binarizer.pkl")

# Prepare features and labels
X = vectorizer.transform(df["Cleaned_Summary"])
y_true = mlb.transform(df["Genre_List"])

# Predict on full data (or use test split if available)
y_pred = model.predict(X)

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

print("Model Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
