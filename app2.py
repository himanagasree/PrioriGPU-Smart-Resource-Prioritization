
import streamlit as st
import pandas as pd
import joblib as jb

# Load model and encoder
model = jb.load("greenland_priority_model.pkl")
label_encoder = jb.load("label_encoder.pkl")

# Mapping for user-friendly dropdown
project_types = list(label_encoder.classes_)
project_map = {t: label_encoder.transform([t])[0] for t in project_types}

st.title("Amazon Greenland - Project Priority Predictor")

# Input fields
roi = st.slider("ROI Score (0 to 1)", 0.0, 1.0, 0.5)
urgency = st.slider("Urgency Score (1 to 10)", 1, 10, 5)
gpu_hours = st.number_input("GPU Hours Requested", min_value=10, max_value=2000, step=50)
team_size = st.number_input("Team Size", min_value=1, max_value=50, step=1)
project_type = st.selectbox("Project Type", project_types)

# Predict
if st.button("Predict Priority"):
    input_df = pd.DataFrame([{
        "roi_score": roi,
        "urgency_score": urgency,
        "gpu_hours_requested": gpu_hours,
        "team_size": team_size,
        "project_type_encoded": project_map[project_type]
    }])
    
    prediction = model.predict(input_df)
    predicted_label = label_encoder.inverse_transform(prediction)
    st.success(f"Predicted Project Priority: **{predicted_label[0]}**")
