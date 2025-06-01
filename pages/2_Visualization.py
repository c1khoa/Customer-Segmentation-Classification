import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import os
import src.data_loader as data
from src.data_preprocessing import DataPreprocessing

# def show_visualization():
df = data.load_data(os.path.join(os.path.dirname(__file__), '..', 'data', 'Train.csv'))
df_null = df.copy()
df_encoding, _ = DataPreprocessing(df_null).process(df_null)

numerics = ["Age", "Work_Experience", "Family_Size"]
catelog = ["Gender", "Ever_Married", "Graduated", "Profession", "Spending_Score", "Var_1", "Segmentation"]

st.title("📊 Dataset Visualization")
sub_option = st.radio("**Choose type of visualization:**",
                        ["One variable", "Two variables", "Many variables"])

if sub_option == "One variable":
    select_col = st.selectbox("Choose variable:", numerics + catelog)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    if select_col in numerics:
        sns.histplot(data=df, x=select_col, kde=True, ax=axs[0])
        axs[0].set_title(f"{select_col} Distribution")
        sns.boxplot(data=df, x=select_col, ax=axs[1])
        axs[1].set_title(f"{select_col} Boxplot")
    else:
        sns.countplot(data=df, x=select_col, color='cyan', ax=axs[0])
        axs[0].set_title(f"{select_col} Count")
        count_data = df[select_col].value_counts()
        axs[1].pie(count_data, labels=count_data.index, autopct='%1.1f%%')
        axs[1].set_title(f"{select_col} Pie Chart")

    st.pyplot(fig)

elif sub_option == "Two variables":
    col1, col2 = st.columns(2)
    with col1:
        select_col_1 = st.selectbox("Choose first variable:", numerics + catelog)
    with col2:
        options_2 = [c for c in numerics + catelog if c != select_col_1]
        select_col_2 = st.selectbox("Choose second variable:", options_2)

    fig, ax = plt.subplots(figsize=(10, 6))
    if select_col_1 in numerics and select_col_2 in catelog:
        sns.histplot(x=df[select_col_1], hue=df[select_col_2], ax=ax)
    elif select_col_1 in catelog and select_col_2 in numerics:
        sns.histplot(x=df[select_col_2], hue=df[select_col_1], ax=ax)
    elif select_col_1 in catelog and select_col_2 in catelog:
        sns.countplot(x=df[select_col_1], hue=df[select_col_2], ax=ax)
    else:
        sns.lineplot(x=df[select_col_1], y=df[select_col_2], ax=ax)
    ax.set_title(f"{select_col_1} vs {select_col_2}")
    st.pyplot(fig)

elif sub_option == "Many variables":
    st.write("The correlation matrix")
    fig, ax = plt.subplots(figsize=(18, 10))
    sns.heatmap(df_encoding.corr(), annot=True, fmt='.2f', ax=ax)
    st.pyplot(fig)
