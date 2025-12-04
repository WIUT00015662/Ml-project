import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- Load preprocessed dataset ---
@st.cache_data
def load_data(path):
    return pd.read_excel(path)

file_path = '../data/883eax2-sup-0002.xlsx'
df = load_data(file_path)

# --- Sidebar: Model selection ---
st.sidebar.title("Model Training Options")
model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest", "Gradient Boosting"])

# Hyperparameters
if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, step=50)
    max_depth = st.sidebar.slider("max_depth", 1, 20, 10)
    min_samples_split = st.sidebar.slider("min_samples_split", 2, 10, 2)
    min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 5, 1)
elif model_choice == "Gradient Boosting":
    n_estimators = st.sidebar.slider("n_estimators", 50, 500, 200, step=50)
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.1, step=0.01)
    max_depth = st.sidebar.slider("max_depth", 1, 10, 3)
    subsample = st.sidebar.slider("subsample", 0.5, 1.0, 0.9, step=0.1)
    min_samples_split = st.sidebar.slider("min_samples_split", 2, 10, 2)
    min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 5, 2)

st.sidebar.markdown("---")

# --- Prepare data ---
target_col = "Winter_mortality"
categorical_cols = ['Activity','Country','Production','Breed','Management',
                    'MidSeason_Target','Environment','Program']
ordinal_cols = ['Age','Beekeep_for','Bee_population_size','Apiary_Size',
                'Swarm_bought','Swarm_produced','Queen_bought','Queen_produced']

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, ordinal_cols),
    ('cat', categorical_pipeline, categorical_cols)
], remainder='passthrough')

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)

# --- Train model ---
if st.button("Train Model"):
    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    
    # Fit
    with st.spinner("Training..."):
        model.fit(X_train_processed, y_train)
    
    # Predict
    y_val_pred = model.predict(X_val_processed)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    
    st.subheader(f"{model_choice} Metrics on Validation Set")
    st.write(f"RMSE: {rmse:.3f}")
    st.write(f"MAE: {mae:.3f}")
    st.write(f"R²: {r2:.3f}")
    
    # Predicted vs Actual Plot
    st.subheader("Predicted vs Actual")
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_val, y_val_pred, alpha=0.5)
    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    st.pyplot(fig)
    
    # Feature importance for RF / GB
    if model_choice in ["Random Forest", "Gradient Boosting"]:
        st.subheader("Feature Importance")
        try:
            importances = model.feature_importances_
            feature_names = preprocessor.get_feature_names_out()
            fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            fi_df = fi_df.sort_values('importance', ascending=False).head(20)
            st.bar_chart(fi_df.set_index('feature'))
        except Exception as e:
            st.write("Could not compute feature importance:", e)
