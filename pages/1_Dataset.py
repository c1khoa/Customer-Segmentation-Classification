import streamlit as st
import os
import src.data_loader as data
MODEL_DIR = os.getenv("MODEL_DIR", r"E:\CS - CSNgành\CS116 - Lập trình Python cho ML\Project\Deploy Model")
# def show_dataset():
df = data.load_data(os.path.join(MODEL_DIR, 'data/Train.csv'))
df_pre = data.load_data(os.path.join(MODEL_DIR, 'data/dataPreprocessing.csv'))
    
st.title("📂 Dataset Viewer")
select_data = st.selectbox("Choose forrmat of data:", ["Raw data", "Preprocessed data"])
df if select_data == "Raw data" else df_pre