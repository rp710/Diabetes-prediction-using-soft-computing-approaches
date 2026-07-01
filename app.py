import os
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
import torch

from src.models import ANFISThenANN, FeedForwardANN

st.set_page_config(page_title="Diabetes Prediction | RP", layout="centered")
st.markdown("""
    <style>
        .reportview-container { background: #ffffff; }
        .main { max-width: 850px; padding: 2rem; font-family: 'Helvetica Neue', sans-serif; }
        h1 { font-weight: 300; font-size: 2.5rem; letter-spacing: -1px; margin-bottom: 0.5rem; }
        h2 { font-weight: 400; font-size: 1.5rem; border-bottom: 1px solid #f0f0f0; padding-bottom: 0.5rem; margin-top: 2rem;}
        .stButton>button { background-color: #000000; color: white; border-radius: 0px; padding: 0.6rem 2rem; border: none; font-weight: 400; width: 100%; transition: all 0.2s ease;}
        .stButton>button:hover { background-color: #333333; color: white; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Load Pretrained Models & Pipeline (.pkl / .pth)
# -----------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FNN_PATH = MODELS_DIR / "fnn_model.pth"
ANFIS_PATH = MODELS_DIR / "anfis_model.pth"


@st.cache_resource
def load_pretrained_assets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not SCALER_PATH.exists() or not FNN_PATH.exists() or not ANFIS_PATH.exists():
        return None, None, None, device

    scaler = joblib.load(SCALER_PATH)

    fnn = FeedForwardANN(input_size=8).to(device)
    fnn.load_state_dict(torch.load(FNN_PATH, map_location=device))
    fnn.eval()

    anfis = ANFISThenANN(in_features=8, n_mfs=6, hidden_size=32).to(device)
    anfis.load_state_dict(torch.load(ANFIS_PATH, map_location=device))
    anfis.eval()

    return scaler, fnn, anfis, device

scaler, fnn_model, anfis_model, device = load_pretrained_assets()

# -----------------------------------------
# UI Layout
# -----------------------------------------
st.title("Diabetes Prediction Engine")
st.markdown("**Project by RP** | Diagnostic Analytics")

if scaler is None:
    st.error("Pretrained pipeline not found. Please run `python train_and_save.py` in your terminal to generate the .pkl and .pth files.")
    st.stop()

st.markdown("Enter custom clinical parameters below to generate a real-time risk assessment using pre-trained inference models.")

selected_model_name = st.selectbox("Select Inference Engine", ["Feedforward ANN", "Hybrid ANN-ANFIS"])
active_model = fnn_model if selected_model_name == "Feedforward ANN" else anfis_model

st.markdown("## Clinical Parameters")
col1, col2 = st.columns(2, gap="large")

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
    bp = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
    skin = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

with col2:
    insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=1000.0, value=79.0, step=1.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.000, max_value=3.000, value=0.500, step=0.010)
    age = st.number_input("Age (Years)", min_value=21, max_value=120, value=33, step=1)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------
# Inference Logic
# -----------------------------------------
if st.button("Run Diagnostic Scan"):
    custom_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    custom_data_scaled = scaler.transform(custom_data)
    input_tensor = torch.tensor(custom_data_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        prediction_prob = active_model(input_tensor).cpu().numpy()[0][0]
    
    st.markdown("## Diagnostic Result")
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric(label="Risk Probability", value=f"{prediction_prob * 100:.1f}%")
    with res_col2:
        if prediction_prob >= 0.5:
            st.error("High Risk: The model indicates a strong likelihood of diabetes based on the provided clinical parameters. Further medical evaluation is recommended.")
        else:
            st.success("Low Risk: The model indicates a low likelihood of diabetes based on the provided clinical parameters.")
