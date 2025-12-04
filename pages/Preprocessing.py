import streamlit as st
import numpy as np
import pandas as pd

st.title("Preprocessed Data Overview")

# --- Load preprocessed arrays ---
@st.cache_data
def load_processed_data():
    X_train = np.load("X_train_processed.npy", allow_pickle=True)
    X_val = np.load("X_val_processed.npy", allow_pickle=True)
    X_test = np.load("X_test_processed.npy", allow_pickle=True)
    y_train = np.load("y_train.npy", allow_pickle=True)
    y_val = np.load("y_val.npy", allow_pickle=True)
    y_test = np.load("y_test.npy", allow_pickle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()

# --- Display shapes ---
st.header("Dataset Shapes")
st.write(f"X_train: {X_train.shape}")
st.write(f"X_val:   {X_val.shape}")
st.write(f"X_test:  {X_test.shape}")
st.write(f"y_train: {y_train.shape}")
st.write(f"y_val:   {y_val.shape}")
st.write(f"y_test:  {y_test.shape}")

# --- Quick data preview ---
st.header("Preview of Feature Arrays")
st.subheader("X_train (first 10 rows)")
st.dataframe(pd.DataFrame(X_train).head(10))

st.subheader("y_train (first 10 values)")
st.dataframe(pd.DataFrame(y_train).head(10))

# --- Basic statistics ---
st.header("Basic Statistics")
st.subheader("Numeric summary of X_train")
st.dataframe(pd.DataFrame(X_train).describe())

st.subheader("Target variable summary")
st.dataframe(pd.DataFrame(y_train).describe())

st.success("Preprocessed data loaded successfully! Ready for modeling.")
