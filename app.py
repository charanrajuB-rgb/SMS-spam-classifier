import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="📩 SMS Spam Classifier", page_icon="📨")

st.title("📩 SMS Spam Classifier")
st.write("Type any message below and see if it is Spam or Ham.")

# Session state to keep history
if "history" not in st.session_state:
    st.session_state.history = []

# Input box for user
user_input = st.text_area("Enter your message here:")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        vectorized = vectorizer.transform([user_input])
        prediction = model.predict(vectorized)[0]
        # Add icon and color
        if prediction == 1:
            result = "🚨 Spam"
            st.error(f"Prediction: {result}")
        else:
            result = "✅ Ham"
            st.success(f"Prediction: {result}")
        # Save to history
        st.session_state.history.append((user_input, result))

# Show message history
if st.session_state.history:
    st.write("---")
    st.subheader("📝 Message History")
    for msg, pred in reversed(st.session_state.history):
        if pred == "🚨 Spam":
            st.markdown(f"**Message:** {msg} → **Prediction:** <span style='color:red'>{pred}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"**Message:** {msg} → **Prediction:** <span style='color:green'>{pred}</span>", unsafe_allow_html=True)

# -----------------------------
# Optional: Show model metrics
# -----------------------------
st.write("---")
st.subheader("📊 Model Evaluation Metrics")

# Load dataset to evaluate
df = pd.read_csv("SMSSpam.csv", sep="\t", names=["label", "message"], on_bad_lines='skip')
df.dropna(inplace=True)
df['label'] = df['label'].map({'ham':0,'spam':1})

X_train_vec = vectorizer.transform(df['message'])
y_true = df['label']
y_pred_full = model.predict(X_train_vec)

# Accuracy
acc = accuracy_score(y_true, y_pred_full)
st.write(f"**Accuracy:** {acc:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_full)
st.write("**Confusion Matrix:**")
cm_df = pd.DataFrame(cm, index=["Ham","Spam"], columns=["Pred Ham","Pred Spam"])
st.dataframe(cm_df)

# Optional: plot confusion matrix
st.write("**Confusion Matrix Heatmap:**")
fig, ax = plt.subplots()
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# Classification Report
st.write("**Classification Report:**")
report = classification_report(y_true, y_pred_full, target_names=["Ham","Spam"], output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)
