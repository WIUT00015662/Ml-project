import streamlit as st

st.set_page_config(page_title="Bee Mortality ML App", layout="wide")

st.title("🐝 Honey Bee Winter Mortality Prediction App")
st.write("Welcome! Use the pages on the left to explore data, preprocessing, model training, and evaluation.")

st.info("""
This Streamlit application allows you to:
- Explore the dataset  
- Review preprocessing steps  
- Train machine learning models  
- Evaluate model performance  
- Make predictions using the final deployed model  
""")
