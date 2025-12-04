import streamlit as st
import pandas as pd

# --- Load dataset ---
@st.cache_data
def load_data(path):
    return pd.read_excel(path)

file_path = '../data/883eax2-sup-0002.xlsx'
df = load_data(file_path)

st.title("Data Preprocessing")

# --- Drop unnecessary columns ---
df = df.drop(columns=['ID_api'])
st.header("Dropped Columns")
st.write("Dropped column: ID_api")

# --- Handle missing values ---
st.header("Missing Values Handling")
st.write("Fill missing values in 'Environment' with mode")
df['Environment'] = df['Environment'].fillna(df['Environment'].mode()[0])

missing = df.isnull().sum()
st.subheader("Remaining missing values per column")
st.dataframe(missing[missing > 0] if any(missing>0) else "No missing values left!")

# --- Numeric-like columns processing ---
st.header("Numeric-like Columns")
num_like_cols = ['Age', 'Beekeep_for', 'Bee_population_size', 'Apiary_Size',
                 'Swarm_bought', 'Swarm_produced', 'Queen_bought', 'Queen_produced']

st.write("Extract numeric part from columns with mixed strings (e.g., '1___Less than 30')")
for col in num_like_cols:
    df[col] = df[col].apply(lambda x: int(str(x).split('___')[0].split('__')[0]))
st.dataframe(df[num_like_cols].head(10))
st.write("Summary statistics for numeric-like columns:")
st.dataframe(df[num_like_cols].describe())

# --- Binary columns processing ---
st.header("Binary Columns")
binary_cols = ['Qualif','Training','Coop_treat','Apiarist_book','Org_member','Continue',
               'Chronic_Depop','ClinSign_Brood','ClinSign_Honeybees','H_Rate_ColMortality',
               'H_Rate_HoneyMortality','OtherEvent','VarroaMites','QueenProblems',
               'VarroosisV1','ChronicParalysisV1','AmericanFoulbroodV1','NosemosisV1',
               'EuropeanFoulbroodV1','Migration','Merger']

st.write("Convert string responses to 0/1")
for col in binary_cols:
    df[col] = df[col].map({'Yes':1,'No':0,'Suffering':1,'Not_Suffering':0})

st.dataframe(df[binary_cols].head(10))
st.write("Check unique values for binary columns:")
for col in binary_cols:
    st.write(f"{col}: {df[col].unique()}")

# --- Categorical columns ---
st.header("Categorical Columns")
categorical_cols = ['Activity','Country','Production','Breed','Management',
                    'MidSeason_Target','Environment','Program']

st.write("Display top categories for each categorical column")
for col in categorical_cols:
    st.subheader(col)
    counts = df[col].value_counts().head(10)
    st.bar_chart(counts)

st.success("Preprocessing completed! Data ready for modeling.")

