import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ==============================
# Load model + tokenizer
# ==============================
MODEL_PATH = "./bert-fakenews-model"  # change if needed

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ==============================
# Prediction function
# ==============================
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=1).item()
    return pred, probs[0].tolist()

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom CSS
st.markdown(
    """
    <style>
        body {
            background-color: #0d1117;
        }
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); 
        }
        h1 {
            color: #ffffff;
            text-align: center;
            font-size: 34px;
            font-weight: 800;
            margin-bottom: 8px;
        }
        p {
            color: #cccccc;
            text-align: center;
            margin-top: -5px;
            margin-bottom: 25px;
        }
        label {
            color: white !important;
            font-weight: 600 !important;
        }
        textarea, input {
            border-radius: 8px !important;
            font-size: 14px !important;
        }
        .prediction-card {
            padding: 20px;
            border-radius: 10px;
            font-size: 22px;
            font-weight: bold;
            color: white;
            text-align: center;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title inside a centered rounded rectangle + subtitle
st.markdown(
    """
    <div style='
        display: flex;
        justify-content: center;
        margin-bottom: 15px;
    '>
        <div style='
            background: rgba(255, 255, 255, 0.08);
            width: 500px;
            height: 80px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.4);
        '>
            <h1 style='color: white; font-size: 40px; text-align: left; font-weight: 800; margin: 0;'>
                Fake News Detector 
            </h1>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: white; font-weight: 600; font-size: 16px;'>Enter a news article below to check if it is Fake or True.</p>", 
    unsafe_allow_html=True
)

# Inputs (centered column)
title = st.text_input("Enter News Title (optional)")
text = st.text_area("Enter News Content (required)", height=180)

predict_btn = st.button("Predict", use_container_width=True, disabled=not bool(text.strip()))

# Prediction (below inputs)
if predict_btn:
    full_input = title + " " + text if title else text
    pred, probs = predict(full_input)

    label = "True News" if pred == 1 else "Fake News"
    color = "#2ecc71" if pred == 1 else "#e74c3c"

    st.markdown(
        f"""
        <div class="prediction-card" style="background-color:{color};">
            {label}
        </div>
        """,
        unsafe_allow_html=True
    )
