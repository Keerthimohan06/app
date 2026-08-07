import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Diabetes Progression Risk Predictor")
st.write(
    "Adjust the clinical indicators in the sidebar to assess diabetes risk progression probability."
)

# -----------------------------
# 2. Data Preparation & Model Training
# -----------------------------
@st.cache_resource
def load_and_train_model():
    # Load scikit-learn diabetes dataset
    raw_data = load_diabetes(as_frame=True)
    df = raw_data.frame

    # Convert continuous target into a binary risk class (Above median = High Risk)
    median_target = df['target'].median()
    df['risk'] = (df['target'] > median_target).astype(int)

    feature_cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    X = df[feature_cols]
    y = df['risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    
    return model, feature_cols, accuracy, df

model, feature_names, accuracy, dataset = load_and_train_model()

st.sidebar.header("📊 Patient Indicators")
st.sidebar.caption(f"Model Test Accuracy: **{accuracy:.1%}**")

# -----------------------------
# 3. Sidebar User Inputs
# -----------------------------
age = st.sidebar.slider("Age (Normalized)", -0.10, 0.10, 0.0, step=0.01)
sex = st.sidebar.selectbox("Sex (Normalized)", options=[-0.044, 0.050], format_func=lambda x: "Female" if x < 0 else "Male")
bmi = st.sidebar.slider("Body Mass Index (BMI)", -0.10, 0.12, 0.0, step=0.01)
bp = st.sidebar.slider("Average Blood Pressure", -0.10, 0.12, 0.0, step=0.01)

st.sidebar.subheader("Blood Serum Measurements")
s1 = st.sidebar.slider("Total Cholesterol (s1)", -0.12, 0.12, 0.0, step=0.01)
s2 = st.sidebar.slider("Low-Density Lipoproteins (s2)", -0.11, 0.12, 0.0, step=0.01)
s3 = st.sidebar.slider("High-Density Lipoproteins (s3)", -0.10, 0.10, 0.0, step=0.01)
s4 = st.sidebar.slider("Total Cholesterol / HDL (s4)", -0.08, 0.10, 0.0, step=0.01)
s5 = st.sidebar.slider("Log of Serum Triglycerides (s5)", -0.12, 0.13, 0.0, step=0.01)
s6 = st.sidebar.slider("Blood Sugar Level (s6)", -0.10, 0.10, 0.0, step=0.01)

input_data = pd.DataFrame([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]], columns=feature_names)

# -----------------------------
# 4. Main Dashboard Output
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Patient Summary")
    st.dataframe(input_data, hide_index=True)

    if st.button("Predict Risk Level", type="primary"):
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        high_risk_prob = probabilities[1]

        st.divider()

        if prediction == 1:
            st.error(f"⚠️ **High Diabetes Progression Risk Detected**")
        else:
            st.success(f"✅ **Low Diabetes Progression Risk Detected**")

        st.metric(label="High-Risk Probability", value=f"{high_risk_prob * 100:.1f}%")
        st.progress(high_risk_prob)

with col2:
    st.subheader("📈 Model Feature Importance")
    importances = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=True)

    fig = px.bar(
        importances,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Drivers of Prediction",
        color="Importance",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption("Developed with Streamlit & Scikit-Learn | Data source: Scikit-learn Diabetes Dataset")