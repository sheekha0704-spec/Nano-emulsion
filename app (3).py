
import streamlit as st
import joblib
import pandas as pd
import os

def get_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)

@st.cache_resource
def load_assets():
    try:
        model = joblib.load(get_path('drug_nano_predictor_v1.pkl'))
        features = joblib.load(get_path('model_features.pkl'))
        drugs = joblib.load(get_path('drug_list.pkl'))
        oils = joblib.load(get_path('oil_list.pkl'))
        surfs = joblib.load(get_path('surfactant_list.pkl'))
        return model, features, drugs, oils, surfs
    except:
        return None, [], [], [], []

st.title("🧪 Nanoemulsion Optimizer")
model, features, drugs, oils, surfs = load_assets()

if model:
    user_inputs = {}
    cols = st.columns(2)
    for i, feat in enumerate(features):
        with cols[i % 2]:
            user_inputs[feat] = st.number_input(f"{feat}", value=0.0)
    
    if st.button("Analyze Stability"):
        input_df = pd.DataFrame([user_inputs])[features]
        prob = model.predict_proba(input_df)[0][1]
        res = "STABLE" if prob > 0.5 else "UNSTABLE"
        st.success(f"Result: {res} ({prob*100:.1f}%)")
