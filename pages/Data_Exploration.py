import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load dataset ---
@st.cache_data
def load_data(path):
    return pd.read_excel(path)

file_path = 'data/883eax2-sup-0002.xlsx'
df = load_data(file_path)

st.title("Exploratory Data Analysis")

# --- Basic info ---
st.header("Basic Info")
st.write(f"Dataset shape: {df.shape}")
st.write("Column types:")
st.dataframe(df.dtypes)
st.subheader("First 5 rows")
st.dataframe(df.head())

# --- Missing values ---
st.header("Missing Values")
missing_values = df.isnull().sum()
st.dataframe(missing_values[missing_values > 0])

# --- Categorical columns ---
cat_cols = df.select_dtypes(include='object').columns
st.header("Categorical Features")
for col in cat_cols:
    st.subheader(f"Column: {col}")
    counts = df[col].value_counts().head(10)
    st.write(counts)
    
    # Bar plot
    fig, ax = plt.subplots(figsize=(8,4))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index[:10], ax=ax)
    st.pyplot(fig)

# --- Numeric columns ---
num_cols = df.select_dtypes(include='number').columns
if len(num_cols) > 0:
    st.header("Numeric Features")
    st.subheader("Summary Statistics")
    st.dataframe(df[num_cols].describe())

    # Histogram
    st.subheader("Distributions")
    fig, ax = plt.subplots(figsize=(12,5))
    df[num_cols].hist(ax=ax, bins=20)
    st.pyplot(fig)

    # Correlation heatmap
    if len(num_cols) > 1:
        st.subheader("Correlation Heatmap")
        corr_matrix = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

# --- Target distribution ---
st.header("Target Variable: Winter_mortality")
fig, ax = plt.subplots(figsize=(8,4))
sns.histplot(df['Winter_mortality'], bins=20, kde=True, ax=ax)
st.pyplot(fig)
