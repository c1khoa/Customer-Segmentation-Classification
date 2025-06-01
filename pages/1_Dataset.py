import streamlit as st
import os
import src.data_loader as data
# def show_dataset():
df = data.load_data(os.path.join(os.path.dirname(__file__), '..', 'data', 'Train.csv'))
df_pre = data.load_data(os.path.join(os.path.dirname(__file__), '..', 'data', 'dataPreprocessing.csv'))
    
st.title("📂 Dataset Viewer")
select_data = st.selectbox("Choose forrmat of data:", ["Raw data", "Preprocessed data"])
df if select_data == "Raw data" else df_pre