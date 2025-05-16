import torch
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS

# Define device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to translate text using a MarianMT model
def translate_text(text, model_name):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    model.eval()

    encoded = tokenizer([text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output = model.generate(**encoded)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded

# Take user input summary
summary = input("Enter a movie summary: ")

# Perform translations
translations = {
    "arabic": translate_text(summary, "Helsinki-NLP/opus-mt-en-ar"),
    "urdu": translate_text(summary, "Helsinki-NLP/opus-mt-en-ur"),
    "spanish": translate_text(summary, "Helsinki-NLP/opus-mt-en-es")
}

# Show all translations
print("\n✅ Translations:")
for lang, translated_text in translations.items():
    print(f"{lang.capitalize()}: {translated_text}")

# Ask for preferred language
preferred = input("\nWhich language would you like to hear? (arabic/urdu/spanish): ").strip().lower()

# Map to gTTS supported language codes
lang_codes = {
    "arabic": "ar",
    "urdu": "ur",
    "spanish": "es"
}

if preferred not in translations or preferred not in lang_codes:
    print("❌ Invalid language selection.")
else:
    text_to_speak = translations[preferred]
    tts = gTTS(text=text_to_speak, lang=lang_codes[preferred])
    output_file = f"summary_{preferred}.mp3"
    tts.save(output_file)
    print(f"🔊 Audio saved as {output_file}")
