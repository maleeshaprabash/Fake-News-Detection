import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
st.set_page_config(page_title="Fake News Detector", layout="wide", page_icon="📰")

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
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Model and Vectorizer constants
MODEL_PATH = 'lr_ngram_model.pkl'
VECTORIZER_PATH = 'vectorizer_ngram.pkl'
METRICS_PATH = 'model_metrics.pkl'

@st.cache_resource
def load_models():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        
        metrics = None
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, 'rb') as f:
                metrics = pickle.load(f)
        
        return model, vectorizer, metrics
    return None, None, None

model, vectorizer, model_metrics = load_models()

# App Header
st.title("📰 Fake News Detection System")
st.write("Determine the authenticity of news articles using machine learning.")

# Sidebar / Training Section
with st.sidebar:
    st.header("⚙️ Configuration")
    if not model or not vectorizer:
        st.warning("⚠️ Model files not found.")
    
    st.subheader("Train Model")
    st.write("Upload datasets to (re)train.")
    fake_file = st.file_uploader("Upload Fake.csv", type=['csv'])
    true_file = st.file_uploader("Upload True.csv", type=['csv'])
    
    if fake_file and true_file:
        if st.button("Train and Save Model"):
            with st.spinner("Training model..."):
                fake_news = pd.read_csv(fake_file)
                real_news = pd.read_csv(true_file)
                fake_news['label'] = 0
                real_news['label'] = 1
                df = pd.concat([fake_news, real_news], ignore_index=True)
                
                # Store EDA data before full preprocessing
                subject_dist = df['subject'].value_counts()
                df['text_len'] = df['text'].fillna('').apply(len)
                
                # Preprocess
                df = df.dropna(subset=['text'])
                df['text_clean'] = df['text'].apply(preprocess_text)
                
                vectorizer_ngram = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
                X = vectorizer_ngram.fit_transform(df['text_clean'])
                y = df['label']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                lr_ngram = LogisticRegression(max_iter=1000)
                lr_ngram.fit(X_train, y_train)
                
                # Calculate Metrics
                y_pred = lr_ngram.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred, output_dict=True)
                
                # Save
                with open(MODEL_PATH, 'wb') as f:
                    pickle.dump(lr_ngram, f)
                with open(VECTORIZER_PATH, 'wb') as f:
                    pickle.dump(vectorizer_ngram, f)
                
                metrics = {
                    'accuracy': acc,
                    'confusion_matrix': cm,
                    'classification_report': report,
                    'subject_dist': subject_dist,
                    'text_len_stats': df.groupby('label')['text_len'].describe()
                }
                with open(METRICS_PATH, 'wb') as f:
                    pickle.dump(metrics, f)
                
                st.success("Model trained! Refreshing...")
                st.rerun()

# Tabs
tab1, tab2, tab3 = st.tabs(["🔍 News Analysis", "📊 Data Insights (EDA)", "📈 Model Evaluation"])

with tab1:
    if model and vectorizer:
        st.subheader("Detect Fake News")
        user_input = st.text_area("Enter news text:", placeholder="Paste your article here...", height=250)
        
        if st.button("Analyze News"):
            if user_input.strip() == "":
                st.warning("Please enter some text.")
            else:
                with st.spinner("Analyzing..."):
                    cleaned_text = preprocess_text(user_input)
                    vectorized_text = vectorizer.transform([cleaned_text])
                    prediction = model.predict(vectorized_text)[0]
                    probability = model.predict_proba(vectorized_text)[0]
                    
                    if prediction == 1:
                        st.markdown('<div class="result-box real">✅ Authentic News Content</div>', unsafe_allow_html=True)
                        st.metric("Authenticity Confidence", f"{probability[1]*100:.2f}%")
                    else:
                        st.markdown('<div class="result-box fake">❌ Potentially Fake News</div>', unsafe_allow_html=True)
                        st.metric("Fake News Confidence", f"{probability[0]*100:.2f}%")
    else:
        st.error("Please train the model first by uploading data in the sidebar.")

with tab2:
    st.subheader("Exploratory Data Analysis")
    if model_metrics and 'subject_dist' in model_metrics:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Subject Distribution**")
            fig, ax = plt.subplots()
            sns.barplot(x=model_metrics['subject_dist'].index, y=model_metrics['subject_dist'].values, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
        with col2:
            st.write("**Text Length Comparison**")
            stats = model_metrics['text_len_stats']
            st.dataframe(stats)
            st.info("Comparison of average text length between Fake (0) and Real (1) news.")
    else:
        st.info("Train the model on the full dataset to see EDA insights here.")

with tab3:
    st.subheader("Model Performance Metrics")
    if model_metrics:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Overall Accuracy</h3>
                <h1 style='color: #007bff;'>{model_metrics['accuracy']*100:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Classification Report Summary**")
            report_df = pd.DataFrame(model_metrics['classification_report']).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0))
            
        with col2:
            st.write("**Confusion Matrix**")
            fig, ax = plt.subplots()
            sns.heatmap(model_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Predicted Fake', 'Predicted Real'],
                        yticklabels=['Actual Fake', 'Actual Real'])
            st.pyplot(fig)
    else:
        st.info("Train the model to see evaluation metrics here.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | AI Model: Logistic Regression")
