import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Stroke Prediction Project",
    page_icon="🧠",
    layout="wide"
)

# Main landing page
st.title("Stroke Prediction Project")
st.markdown("---")

st.write("""
Welcome to the Stroke Prediction Project dashboard. Use the sidebar to navigate between different sections:

- **Project Proposal**: View the project proposal details
- **Data Cleaning Results**: Explore the data cleaning process and results
""")

# Footer
st.markdown("---")
st.markdown("Ben Proell, Nima Mollaei, Yoomin Choi, Taylor West, Tyler Chevalier")
