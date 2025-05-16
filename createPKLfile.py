import pandas as pd
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer

# 1. Load Data
df = pd.read_csv('train_data.csv')

# 2. Clean and Preprocess Movie Summaries
def clean_text(text):
    if pd.isnull(text):
        return ""
    return text.lower().strip()

df['clean_summary'] = df['Cleaned_Summary'].apply(clean_text)

# 3. Parse Genre List
def parse_genre_list(genre_str):
    try:
        return ast.literal_eval(genre_str)
    except (ValueError, SyntaxError):
        return []

df['genre_list'] = df['Genre_List'].apply(parse_genre_list)

# 4. Generate SBERT Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
df['embedding'] = df['clean_summary'].apply(lambda x: model.encode(x))

# 5. Encode Genres with MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['genre_list'])

# Save genre classes
pd.Series(mlb.classes_).to_csv('genre_classes.csv', index=False)

# 6. Combine genre data
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
df = df.reset_index(drop=True)
genre_df = genre_df.reset_index(drop=True)
df = pd.concat([df, genre_df], axis=1)

# 7. Create Feature and Label Vectors
df['feature_vector'] = df['embedding'].apply(lambda x: np.array(x))
df['label_vector'] = genre_encoded.tolist()

# 8. Save to Pickle
df.to_pickle('movie_genre_features.pkl')
print("✅ Preprocessing complete. Saved to 'movie_genre_features.pkl'.")
