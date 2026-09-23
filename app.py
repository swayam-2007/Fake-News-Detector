import streamlit as st
import sys
import os

# Ensure src is in the python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import predict, load_artifacts

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom CSS for a premium look
st.markdown("""
<style>
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #ddd;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
    }
    .result-card-real {
        background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .result-card-fake {
        background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .result-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .confidence-text {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📰 AI Fake News Detector")
st.write("Enter a news article or headline below, and our machine learning model will predict whether it is **REAL** or **FAKE**.")

# Load models from predict.py
@st.cache_resource
def load_models_artifacts():
    return load_artifacts()

try:
    model, vectorizer = load_models_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please train the model first by running `python src/train.py`.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

user_input = st.text_area("News Text", height=200, placeholder="Paste a news article or headline here...")

if st.button("Analyze Text"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text patterns..."):
            try:
                label, confidence = predict(user_input, model, vectorizer)
                
                # Render results
                if label.upper() == "REAL":
                    st.markdown(f'''
                        <div class="result-card-real">
                            <h1 class="result-title">✅ REAL NEWS</h1>
                            <div class="confidence-text">Confidence: {confidence:.2%}</div>
                        </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                        <div class="result-card-fake">
                            <h1 class="result-title">❌ FAKE NEWS</h1>
                            <div class="confidence-text">Confidence: {confidence:.2%}</div>
                        </div>
                    ''', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
