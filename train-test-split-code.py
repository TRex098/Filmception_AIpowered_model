import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split

# Load the dataset
df = pd.read_csv('cleaned_data.csv')

# Use Cleaned_Summary as input feature, Genre_List as label
X = df[['MovieID', 'Cleaned_Summary']]
Y = df['Genre_List'].apply(eval)  # Convert stringified list to actual Python list

# Binarize the genre labels
mlb = MultiLabelBinarizer()
Y_bin = mlb.fit_transform(Y)

# Convert X to numpy array for the split function
X_np = X.to_numpy()

# Perform multi-label stratified train-test split (80% train, 20% test)
X_train, Y_train, X_test, Y_test = iterative_train_test_split(X_np, Y_bin, test_size=0.2)

# Convert back to DataFrames
X_train_df = pd.DataFrame(X_train, columns=['MovieID', 'Cleaned_Summary'])
Y_train_genres = mlb.inverse_transform(Y_train)
Y_train_df = pd.DataFrame({'Genre_List': [list(genres) for genres in Y_train_genres]})
train_df = pd.concat([X_train_df, Y_train_df], axis=1)

X_test_df = pd.DataFrame(X_test, columns=['MovieID', 'Cleaned_Summary'])
Y_test_genres = mlb.inverse_transform(Y_test)
Y_test_df = pd.DataFrame({'Genre_List': [list(genres) for genres in Y_test_genres]})
test_df = pd.concat([X_test_df, Y_test_df], axis=1)

# Save the split datasets to new CSV files
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

print("✅ Train-test split completed. Files saved as 'train_data.csv' and 'test_data.csv'.")
