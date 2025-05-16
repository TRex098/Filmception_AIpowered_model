import joblib

# Load saved files
model = joblib.load("genre_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
mlb = joblib.load("label_binarizer.pkl")

# Input summary
summary = input("Enter movie summary: ")

# Vectorize input
X = vectorizer.transform([summary])

# Predict genres
y_pred = model.predict(X)

# Inverse transform
predicted_genres = mlb.inverse_transform(y_pred)

# Display results
if predicted_genres and predicted_genres[0]:
    print("Predicted Genres:", list(predicted_genres[0]))
else:
    print("No genres predicted with confidence.")
