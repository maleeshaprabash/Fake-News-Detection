import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import pickle
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')

# Preprocessing logic from notebook
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Lemmatization and stopword removal
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# App Configuration
st.set_page_config(page_title="Fake News Detector", layout="centered", page_icon="📰")

# Styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .stTextArea>div>div>textarea {
        border-radius: 10px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        margin-top: 20px;
    }
    .real { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .fake { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("📰 Fake News Detection System")
st.write("Determine the authenticity of news articles using machine learning.")

# Model and Vectorizer Loading
MODEL_PATH = 'lr_ngram_model.pkl'
VECTORIZER_PATH = 'vectorizer_ngram.pkl'

@st.cache_resource
def load_models():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    return None, None

model, vectorizer = load_models()

# Sidebar instructions / Upload (In case files aren't there)
if not model or not vectorizer:
    st.sidebar.warning("⚠️ Model files not found locally.")
    st.info("Please ensure `lr_ngram_model.pkl` and `vectorizer_ngram.pkl` are in the app directory.")
    
    st.subheader("Train Model (Optional)")
    st.write("Upload `Fake.csv` and `True.csv` to train the model if it's missing.")
    fake_file = st.file_uploader("Upload Fake.csv", type=['csv'])
    true_file = st.file_uploader("Upload True.csv", type=['csv'])
    
    if fake_file and true_file:
        if st.button("Train and Save Model"):
            with st.spinner("Training model... This might take a minute."):
                fake_news = pd.read_csv(fake_file)
                real_news = pd.read_csv(true_file)
                fake_news['label'] = 0
                real_news['label'] = 1
                df = pd.concat([fake_news, real_news], ignore_index=True)
                df = df.dropna(subset=['text'])
                df['text'] = df['text'].apply(preprocess_text)
                
                vectorizer_ngram = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
                X = vectorizer_ngram.fit_transform(df['text'])
                y = df['label']
                
                lr_ngram = LogisticRegression(max_iter=1000)
                lr_ngram.fit(X, y)
                
                with open(MODEL_PATH, 'wb') as f:
                    pickle.dump(lr_ngram, f)
                with open(VECTORIZER_PATH, 'wb') as f:
                    pickle.dump(vectorizer_ngram, f)
                
                st.success("Model trained and saved! Please refresh the app.")
                st.rerun()

# Prediction Interface
if model and vectorizer:
    user_input = st.text_area("Enter the news text below:", placeholder="Paste your article here...", height=250)
    
    if st.button("Analyze News"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing text..."):
                # Preprocess
                cleaned_text = preprocess_text(user_input)
                # Vectorize
                vectorized_text = vectorizer.transform([cleaned_text])
                # Predict
                prediction = model.predict(vectorized_text)[0]
                probability = model.predict_proba(vectorized_text)[0]
                
                # Display Results
                if prediction == 1:
                    st.markdown('<div class="result-box real">✅ Authentic News Content</div>', unsafe_allow_html=True)
                    st.write(f"Confidence: {probability[1]*100:.2f}%")
                else:
                    st.markdown('<div class="result-box fake">❌ Potentially Fake News</div>', unsafe_allow_html=True)
                    st.write(f"Confidence: {probability[0]*100:.2f}%")
                
                with st.expander("Show Analyzed Text"):
                    st.write(f"**Cleaned Text:** {cleaned_text}")
else:
    st.error("Application cannot proceed without a trained model. Use the sidebar/upload options to provide data for training.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | AI Model: Logistic Regression")
