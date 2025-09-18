import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ------------------------------
# 1. Load Dataset (tab separated)
# ------------------------------
df = pd.read_csv("SMSSpam.csv", sep="\t", names=["label", "message"], on_bad_lines="skip")

# Drop missing values
df.dropna(inplace=True)

print("🔹 First 5 rows:")
print(df.head())

# ------------------------------
# 2. Encode Labels
# ------------------------------
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ------------------------------
# 3. Split Dataset
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# ------------------------------
# 4. Vectorize Text
# ------------------------------
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------------------
# 5. Train Model
# ------------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ------------------------------
# 6. Evaluate Model
# ------------------------------
y_pred = model.predict(X_test_vec)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))

# ------------------------------
# 7. Test Custom Messages
# ------------------------------
samples = [
    "Congratulations! You won a free iPhone, click here.",
    "Hey bro, are we meeting tomorrow?",
    "Exclusive offer just for you, win cash now!"
]

for msg in samples:
    pred = model.predict(vectorizer.transform([msg]))[0]
    print(f"Message: {msg} → Prediction: {'Spam' if pred==1 else 'Ham'}")

from sklearn.metrics import classification_report, confusion_matrix

# Extra evaluation
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

print("\n🌀 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import joblib

# Save model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n💾 Model & Vectorizer saved successfully!")
