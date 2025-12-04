import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title("Model Evaluation (Preprocessed Data)")

# --- Load preprocessed validation/test sets ---
@st.cache_data
def load_data():
    X_val = np.load("X_val_processed.npy", allow_pickle=True)
    X_test = np.load("X_test_processed.npy", allow_pickle=True)
    y_val = np.load("y_val.npy", allow_pickle=True)
    y_test = np.load("y_test.npy", allow_pickle=True)
    return X_val, X_test, y_val, y_test

X_val, X_test, y_val, y_test = load_data()
st.write(f"Validation set shape: {X_val.shape}, Test set shape: {X_test.shape}")

# --- Load trained models ---
model_files = {
    "Linear Regression": "lr_model.pkl",
    "Random Forest": "rf_best_model.pkl",
    "Gradient Boosting": "gb_best_model.pkl"
}

@st.cache_data
def load_models(files_dict):
    models = {}
    for name, path in files_dict.items():
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models

models = load_models(model_files)
st.write("Trained models loaded successfully.")

# --- Evaluate models ---
metrics_list = []
predictions = {}

st.header("Validation Set Metrics")
for name, model in models.items():
    y_pred = model.predict(X_val)
    predictions[name] = y_pred
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    metrics_list.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    })

metrics_df = pd.DataFrame(metrics_list)
st.dataframe(metrics_df)

# --- Predicted vs Actual Plots ---
st.header("Predicted vs Actual Plots")
for name, y_pred in predictions.items():
    st.subheader(name)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_val, y_pred, alpha=0.5)
    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{name}: Predicted vs Actual")
    st.pyplot(fig)

# --- Residual Plots ---
st.header("Residual Plots")
for name, y_pred in predictions.items():
    st.subheader(name)
    residuals = y_val - y_pred
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_xlabel("Residual")
    ax.set_title(f"{name}: Residual Distribution")
    st.pyplot(fig)
