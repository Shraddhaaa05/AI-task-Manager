import streamlit as st
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import os

# Download only what's needed (safe for Streamlit Cloud)
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

download_nltk_data()

# Load saved model and vectorizer
@st.cache_resource
def load_models():
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    output_labels = joblib.load('multi_output_labels.pkl')
    return model, vectorizer, output_labels

try:
    model, vectorizer, output_labels = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

# Preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    filtered = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(filtered)

# --- UI ---
st.set_page_config(page_title="AI-Powered Task Management System", page_icon="🔮", layout="centered")

st.title("🔮 AI-Powered Task Management System")
st.markdown("Enter a task or bug summary and get AI predictions for multiple labels.")

if not models_loaded:
    st.error(f"❌ Failed to load model files: {load_error}")
    st.info("Make sure model.pkl, tfidf_vectorizer.pkl, and multi_output_labels.pkl are in the same directory.")
    st.stop()

summary_input = st.text_area("✍️ Enter Task Summary:", height=120, placeholder="e.g. Fix login bug on mobile app causing crash for Android users")

if st.button("🎯 Predict", type="primary"):
    if not summary_input.strip():
        st.warning("⚠️ Please enter a valid task summary.")
    else:
        with st.spinner("Analysing task..."):
            try:
                processed = preprocess(summary_input)
                vect = vectorizer.transform([processed])
                preds = model.predict(vect)[0]

                # --- Predictions table ---
                st.subheader("📌 Predicted Outputs")
                cols = st.columns(2)
                for i, (label, pred) in enumerate(zip(output_labels, preds)):
                    with cols[i % 2]:
                        st.metric(label=label.replace('_', ' ').title(), value=str(pred))

                # --- Pie chart ---
                st.subheader("📊 Prediction Summary Chart")
                combined_labels = [f"{label.replace('_',' ').title()}: {pred}" for label, pred in zip(output_labels, preds)]
                percentages = [100 / len(combined_labels)] * len(combined_labels)
                colors_list = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b', '#fa709a', '#fee140', '#a18cd1', '#fda085', '#f6d365']

                fig, ax = plt.subplots(figsize=(8, 6))
                wedges, texts, autotexts = ax.pie(
                    percentages,
                    labels=combined_labels,
                    autopct='%1.1f%%',
                    colors=colors_list[:len(combined_labels)],
                    pctdistance=0.85,
                )
                for text in texts:
                    text.set_fontsize(8)
                ax.axis('equal')
                plt.tight_layout()
                st.pyplot(fig)
                st.success("✅ Prediction complete!")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")