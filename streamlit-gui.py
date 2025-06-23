# streamlit-gui.py

import streamlit as st
import numpy as np
import pickle
import os
import scipy.io.wavfile as wav
from manualSVM import extract_features
from settings import DEMO_FILE

st.set_page_config(page_title="Spoken Digit Detection", layout="centered")

# --------------------------
# Load model
# --------------------------
@st.cache_resource
def load_model():
    with open("model/svm_speech_model.pkl", "rb") as f:
        return pickle.load(f)

model_data = load_model()
model = model_data['model']
label_names = model_data['label_names']
accuracy = model_data['accuracy']
precision = model_data['precision']
recall = model_data['recall']
f1 = model_data['f1']
mean = model_data.get('mean', None)
std = model_data.get('std', None)

def preprocess_features(features):
    if mean is not None and std is not None:
        return (features - mean) / std
    return features

# --------------------------
# UI
# --------------------------
st.title("Real-Time Spoken Digit Detection")
st.caption("Free Spoken Digit Detection using SVM and MFCC + Frequency/Time Domain Features")
st.divider()

# --------------------------
# Demo
# --------------------------
st.subheader("🔊 Demo")
if os.path.exists(DEMO_FILE):
    st.audio(DEMO_FILE, format="audio/wav")
    if st.button("📊 Predict from Recording"):
        if os.path.exists(DEMO_FILE):
            sr, signal = wav.read(DEMO_FILE)
            features = extract_features(signal.flatten(), sr)
            features = preprocess_features(features)
            prediction = model.predict([features])[0]
            st.success(f"**Predicted Digit (from recording):** {label_names[prediction]}")

# --------------------------
# Upload section
# --------------------------
st.divider()
st.subheader("📁 Upload File and Predict")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    if st.button("📊 Predict from File"):
        sr, signal = wav.read(uploaded_file)
        features = extract_features(signal.flatten(), sr)
        features = preprocess_features(features)
        prediction = model.predict([features])[0]
        st.success(f"**Predicted Digit (from file):** {label_names[prediction]}")

# --------------------------
# Model stats
# --------------------------
st.divider()
st.subheader("📈 Model Performance Summary")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Precision", f"{precision:.2%}")
col3.metric("Recall", f"{recall:.2%}")
col4.metric("F1 Score", f"{f1:.2%}")


st.divider()
st.subheader("Authors")

data = [
    ["Danishwara Pracheta", "2308561050", "@dash4k"],
    ["Maliqy Numurti", "2308561068", "@Maliqytritata"],
    ["Krisna Udayana", "2308561122", "@KrisnaUdayana"],
    ["Dewa Sutha", "2308561137", "@DewaMahattama"]
]

for row in data:
    cols = st.columns(3)
    for i, col in enumerate(cols):
        with col:
            st.write(row[i])