import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


st.set_page_config(
    page_title="Data Cleaning Results",
    page_icon="🧠",
    layout="wide"
)

@st.cache_data
def load_data():
    original_data = pd.read_csv('healthcare-dataset-stroke-data.csv')
    cleaned_data = pd.read_csv('processed_stroke_data.csv')
    return original_data, cleaned_data

original_data, cleaned_data = load_data()

st.title("Data Cleaning Results")
st.markdown("---")

st.header("Data Cleaning Summary")

original_missing = original_data.replace('N/A', np.nan).isna().sum()
cleaned_missing = cleaned_data.isna().sum()

original_data_processed = original_data.copy()
original_data_processed['bmi'] = pd.to_numeric(original_data_processed['bmi'], errors='coerce')

male_bmi_missing = original_data_processed[(original_data_processed['gender'] == 'Male') & 
                                          (original_data_processed['bmi'].isna())].shape[0]
female_bmi_missing = original_data_processed[(original_data_processed['gender'] == 'Female') & 
                                            (original_data_processed['bmi'].isna())].shape[0]
other_bmi_missing = original_data_processed[(~original_data_processed['gender'].isin(['Male', 'Female'])) & 
                                           (original_data_processed['bmi'].isna())].shape[0]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Duplicate Rows Removed", original_data.shape[0] - cleaned_data.shape[0])
with col2:
    st.metric("Missing BMI Values Filled", original_missing['bmi'])
with col3:
    st.metric("Missing Age Values Filled", original_missing['age'])

st.subheader("Gender Breakdown of Missing BMI Values")
fig, ax = plt.subplots(figsize=(8, 5))
gender_missing = [male_bmi_missing, female_bmi_missing, other_bmi_missing]
gender_labels = ['Male', 'Female', 'Other']
ax.bar(gender_labels, gender_missing)
ax.set_ylabel('Number of Missing Values')
ax.set_title('Missing BMI Values by Gender')
st.pyplot(fig)


st.subheader("Raw Data Comparison")
tab1, tab2 = st.tabs(["Original Data", "Cleaned Data"])

with tab1:
    st.dataframe(original_data.head(10))

with tab2:
    st.dataframe(cleaned_data.head(10)) 