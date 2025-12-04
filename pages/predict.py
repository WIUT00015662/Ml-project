import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

st.title("Predict Winter Mortality")

# --- Load trained model ---
with open("../best_model.pkl", "rb") as f:  # path to your saved model
    model = pickle.load(f)

with open("../preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# --- User inputs ---
st.header("Provide Feature Values")

ordinal_cols = ['Age','Beekeep_for','Bee_population_size','Apiary_Size',
                'Swarm_bought','Swarm_produced','Queen_bought','Queen_produced']
binary_cols = ['Qualif','Training','Coop_treat','Apiarist_book','Org_member','Continue',
               'Chronic_Depop','ClinSign_Brood','ClinSign_Honeybees','H_Rate_ColMortality',
               'H_Rate_HoneyMortality','OtherEvent','VarroaMites','QueenProblems',
               'VarroosisV1','ChronicParalysisV1','AmericanFoulbroodV1','NosemosisV1',
               'EuropeanFoulbroodV1','Migration','Merger']
categorical_cols = ['Activity','Country','Production','Breed','Management',
                    'MidSeason_Target','Environment','Program']

user_input = {}

# Ordinal / numeric inputs
for col in ordinal_cols:
    val = st.number_input(f"{col}", value=0)
    user_input[col] = val

# Binary inputs
for col in binary_cols:
    val = st.selectbox(f"{col}", [0, 1], index=1)
    user_input[col] = val

# Categorical inputs
for col in categorical_cols:
    val = st.selectbox(f"{col}", ["Category1","Category2","Category3"])  # replace with actual categories
    user_input[col] = val

# Convert to dataframe
input_df = pd.DataFrame([user_input])

# Preprocess
input_processed = preprocessor.transform(input_df)

# Predict
prediction = model.predict(input_processed)[0]

st.subheader("Predicted Winter Mortality")
st.write(f"{prediction:.2f}")
