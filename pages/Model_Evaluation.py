import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title("Model Evaluation")

# --- Load validation/test set ---
@st.cache_data
def load_data(path):
    return pd.read_excel(path)

df = load_data("../data/883eax2-sup-0002.xlsx")
target_col = "Winter_mortality"

X = df.drop(columns=[target_col])
y_true = df[target_col]

# --- Load trained models ---
models = {}
model_files = {
    "Linear Regression": "../models/lr_model.pkl",
    "Random Forest": "../models/rf_best_model.pkl",
    "Gradient Boosting": "../models/gb_best_model.pkl"
}

preprocessor_file = "../models/preprocessor.pkl"

# Load preprocessor
with open(preprocessor_file, "rb") as f:
    preprocessor = pickle.load(f)

# Transform features
X_processed = preprocessor.transform(X)

# Load models
for name, path in model_files.items():
    with open(path, "rb") as f:
        models[name] = pickle.load(f)

# --- Evaluate models ---
metrics_list = []
predictions = {}

st.header("Validation/Test Metrics")
for name, model in models.items():
    y_pred = model.predict(X_processed)
    predictions[name] = y_pred
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
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
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{name}: Predicted vs Actual")
    st.pyplot(fig)

# --- Optional: Residual plots ---
st.header("Residual Plots")
for name, y_pred in predictions.items():
    st.subheader(name)
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_xlabel("Residual")
    ax.set_title(f"{name}: Residual Distribution")
    st.pyplot(fig)
