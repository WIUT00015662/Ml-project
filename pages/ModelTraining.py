import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title("Model Evaluation and Inference")

# --- Load preprocessed data ---
@st.cache_data
def load_data():
    X_val = np.load("X_val_processed.npy", allow_pickle=True)
    X_test = np.load("X_test_processed.npy", allow_pickle=True)
    y_val = np.load("y_val.npy", allow_pickle=True)
    y_test = np.load("y_test.npy", allow_pickle=True)
    return X_val, X_test, y_val, y_test

X_val, X_test, y_val, y_test = load_data()
st.write("Validation and Test data loaded.")

# --- Load saved models ---
model_files = {
    "Linear Regression": "lr_model.pkl",
    "Random Forest": "rf_best_model.pkl",
    "Gradient Boosting": "best_model.pkl"
}

@st.cache_data
def load_models(files_dict):
    models = {}
    for name, path in files_dict.items():
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    return models

models = load_models(model_files)
st.write("Models loaded successfully.")

# --- Helper function for evaluation ---
def evaluate_model(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R^2": r2_score(y_true, y_pred)
    }

# --- Evaluate each model ---
st.header("Model Evaluation on Validation Set")
metrics_list = []

for name, model in models.items():
    y_val_pred = model.predict(X_val)
    metrics = evaluate_model(y_val, y_val_pred)
    metrics_list.append({"Model": name, **metrics})

metrics_df = pd.DataFrame(metrics_list)
st.dataframe(metrics_df)

# --- Optional: allow user to select a model for predictions ---
st.header("Predict with a Selected Model")
selected_model_name = st.selectbox("Select a model:", list(models.keys()))
input_index = st.number_input("Enter row index from validation set for prediction:", min_value=0, max_value=len(X_val)-1, value=0)

if st.button("Predict"):
    model = models[selected_model_name]
    pred = model.predict([X_val[input_index]])[0]
    st.write(f"Predicted Winter Mortality for row {input_index}: {pred}")
    st.write(f"Actual value: {y_val[input_index]}")
