import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import torch

# Define device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to translate a list of texts
def translate_texts(texts, model_name, batch_size=32):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    model.eval()

    translations = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Translating with {model_name}"):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            output = model.generate(**encoded)

        decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in output]
        translations.extend(decoded)

    return translations

# Load CSV file
df = pd.read_csv("train_data.csv")

# For testing with a small dataset (100 random samples)
df = df.sample(60).reset_index(drop=True)

# Check for 'Cleaned_Summary' column
if 'Cleaned_Summary' not in df.columns:
    raise KeyError("'Cleaned_Summary' column not found in CSV")

# Arabic
df['summary_arabic'] = translate_texts(df['Cleaned_Summary'].astype(str).tolist(), "Helsinki-NLP/opus-mt-en-ar")

# Urdu
df['summary_urdu'] = translate_texts(df['Cleaned_Summary'].astype(str).tolist(), "Helsinki-NLP/opus-mt-en-ur")

# Spanish
df['summary_spanish'] = translate_texts(df['Cleaned_Summary'].astype(str).tolist(), "Helsinki-NLP/opus-mt-en-es")

# Save result
df.to_csv("translated_data.csv", index=False, encoding="utf-8-sig")

print("✅ Translation completed and saved to translated_data.csv")
