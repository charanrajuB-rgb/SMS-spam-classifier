import joblib

# Load saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Test with new messages
samples = [
    "Claim your free lottery ticket now!!!",
    "click this link you yor class is live."
]

for msg in samples:
    pred = model.predict(vectorizer.transform([msg]))[0]
    print(f"Message: {msg} → Prediction: {'Spam' if pred==1 else 'Ham'}")
