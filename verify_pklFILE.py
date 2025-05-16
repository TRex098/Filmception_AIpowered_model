import pickle
import os
from scipy.sparse import csr_matrix

# File paths
features_file = "tfidf_features.pkl"
labels_file = "genre_labels.pkl"
vectorizer_file = "tfidf_vectorizer.pkl"
binarizer_file = "label_binarizer.pkl"

# Helper function to verify and load pickle file
def load_and_check(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    with open(file_path, "rb") as f:
        data = pickle.load(f)
        print(f"✅ Loaded: {file_path}")
        print(f"   Type: {type(data)}")
        if isinstance(data, csr_matrix):
            print(f"   Shape: {data.shape}, Sparse matrix")
        elif hasattr(data, 'shape'):
            print(f"   Shape: {data.shape}")
        elif hasattr(data, '__len__'):
            print(f"   Length: {len(data)}")
        return data

# Verify each file
X = load_and_check(features_file)
y = load_and_check(labels_file)
vectorizer = load_and_check(vectorizer_file)
binarizer = load_and_check(binarizer_file)

# Optional: preview some genre labels
if y is not None and hasattr(y, 'shape'):
    print("\nSample label vector (first row):", y[0])
