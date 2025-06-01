import streamlit as st
import pandas as pd
import joblib
import os
from src.data_preprocessing import DataPreprocessing
from src.feature import FeatureEngineering
import src.data_loader as data

# def show_model():
df = data.load_data(os.path.join(os.path.dirname(__file__), '..', 'data', 'Train.csv'))
model = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'model', 'stacking_model.pkl'))

st.title("Customer Segmentation Classification")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.toggle("Married")
    age = st.slider("Age", 0, 100)
    graduated = st.toggle("Graduated")
with col2:
    profession = st.selectbox("Profession", [
        'Artist', 'Doctor', 'Engineer', 'Entertainment', 'Executive',
        'Healthcare', 'Homemaker', 'Lawyer', 'Marketing'])
    work = st.slider("Work Experience", 0, 10)
    spend = st.selectbox("Spending Score", ["Low", "Average", "High"])
    fa_size = st.slider("Family Size", 0, 10)

var1 = st.selectbox("Var_1", ["Cat_1", "Cat_2", "Cat_3", "Cat_4", "Cat_5", "Cat_6", "Cat_7"])

x_test = pd.DataFrame([{
    "Gender": gender,
    "Ever_Married": "Yes" if married else "No",
    "Age": age,
    "Graduated": "Yes" if graduated else "No",
    "Profession": profession,
    "Work_Experience": work,
    "Spending_Score": spend,
    "Family_Size": fa_size,
    "Var_1": var1
}])

df, x_test = DataPreprocessing(df).process(x_test)
df, x_test = FeatureEngineering(df).feature(x_test)

if st.button("Predict"):
    label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    prediction = model.predict(x_test)
    prediction_proba = model.predict_proba(x_test)

    st.success(f"Customer segmentation: **{label_map[prediction[0]]}**")

    st.markdown("### Prediction Probability:")
    prob_df = pd.DataFrame(prediction_proba, columns=[label_map[i] for i in range(prediction_proba.shape[1])])
    for label in prob_df.columns:
        percent = int(prob_df[label][0] * 100)
        st.write(f"**Class {label}: {percent}%**")
        st.progress(percent)
