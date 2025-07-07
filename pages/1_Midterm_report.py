import streamlit as st
import import_ipynb
import Project
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# Set page configuration
st.set_page_config(
    page_title="Project Proposal",
    page_icon="🧠",
    layout="wide"
)

# Main title
st.title("Stroke Prediction Project Proposal")
st.markdown("---")

# Introduction and Background
st.header("Introduction and Background")
st.write("""
This project focuses on predicting the likelihood of stroke occurrence using machine learning techniques. 
Stroke is a critical medical condition that requires early detection and intervention to prevent severe consequences.
""")
st.write("""
         We will be using machine learning to predict strokes before they happen, by aiming to identify high risk
         individuals using medical and demographic data. Stroke prediction is an active area of research due to its
         potential to reduce mortality and improve preventive care. We will be using The Stroke Prediction Dataset 
         from Kaggle which has features such as age, gender, hypertension, heart disease, average glucose level,
         BMI, and work type. The target label is a binary indicator of whether the individual has had a stroke. 
         """)
#data set link
st.subheader("Dataset Link")
st.info("""Dataset: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data""")

# Problem Definition
st.header("Problem Definition")
st.write("""
         Stroke is one of the leading causes of death and long-term disability worldwide. Early detection of stroke
         risk is critical for timely intervention, but current clinical approaches often rely on reactive diagnosis
         rather than proactive risk assessment. A data-driven model that can estimate an individual’s probability 
         of experiencing a stroke would support preventive care, improve resource allocation, and potentially save
         lives. Given the growing availability of electronic health records, there is a clear opportunity to apply 
         machine learning for stroke risk prediction at scale. 
         """)

# Methods
st.header("Methods")
st.subheader("Data Preprocessing Methods")
st.write("""
For data preprocessing, we filled any missing values with the median of the respective column. We also converted
columns that had categorical values into numerical values, and we plan to use one-hot encoding in the future.

(see [Data Cleaning Results] for more details)
""")

st.subheader("Naive Bayes Model")
st.write("""
We decided to use Naive Bayes as our initial model due to its simplicity and effectiveness for classification tasks.
Naive Bayes is particularly suitable for high-dimensional data and can handle both categorical and continuous features.
That being said, we will be using supervised learning methods to train our model(s)
""")


# Results and Discussion
st.header("Results and Discussion")
st.subheader("Quantitative Metrics")
st.write("""
While implementing Naive Bayes as our baseline model, We tested three different feature configurations:
- **Original**: All features included.
- **Variant 1**: Dropped gender, heart disease, residence type, and smoking status.
- **Variant 2**: Used only age, hypertension, BMI, and smoking status.

**Results Summary:**
- **Accuracy**: Minimal difference between configurations, with variant two slightly outperforming the others.
- **F1 Score**: Variant one achieved the best F1 score, suggesting balanced performance.
- **Likelihood Ratios**: All variants showed high LR+, indicating features are better at identifying true positives than true negatives.

**Key Insight:**  
Our features currently favor positive prediction. To improve specificity, we will focus on dimensionality reduction and feature engineering in the next steps.
""")
st.image("Images_for_visualizations/Accuracies.png", caption="Variant Accuracies")
st.image("Images_for_visualizations/F1Scores.png", caption="Variant F1 Scores")
st.image("Images_for_visualizations/Other.png", caption="Likelihood Ratios")
st.subheader("Visualizations of dataset")

st.image("Images_for_visualizations/vis1.png",caption="Percent of stroke based on gender")

st.image(["Images_for_visualizations/freq_based_on_Age.png", "Images_for_visualizations/freq_glucose.png"],
         caption=["Frequency based on age","Frequency base on glucose"])
st.image(["Images_for_visualizations/freq_heart_disease.png","Images_for_visualizations/freq_hypertension.png"],
         caption=["Frequency given heart disease","Frequency given hypertension"])
st.image(["Images_for_visualizations/freq_married.png", "Images_for_visualizations/freq_residence.png"],
         caption=["Frequency given married status", "Frequency given residence type"])
st.image(["Images_for_visualizations/freq_smoking.png", "Images_for_visualizations/freq_work_type.png"],
         caption=["Frequency given smoking status", "Frequency given work type"])
st.image(["Images_for_visualizations/percent_given_age.png", "Images_for_visualizations/percent_BMI.png"],
         caption=["Percent given age", "Percent BMI"])


st.subheader("Next Steps")
st.write("""
Based on our current analysis and model performance, our next steps are as follows:

✅ **1. Feature Engineering**
- Create interaction features (e.g. age × hypertension).
- Binarize or bucket continuous features to test Naive Bayes assumptions.

✅ **2. Dimensionality Reduction**
- Implement PCA or SelectKBest to reduce redundant features and improve generalization.

✅ **3. Model Exploration**
- Test alternative algorithms including:
  - Logistic Regression
  - Random Forest
  - Support Vector Machines

✅ **4. Hyperparameter Tuning**
- Perform grid search for each algorithm to optimize performance.

✅ **5. Cross-validation**
- Use k-fold cross-validation to ensure robust evaluation and reduce overfitting risks.

✅ **6. Deployment Preparation**
- Finalize the Streamlit app for user testing.
- Document all code, methods, and results clearly for reproducibility.
""")

st.subheader("Long-term Considerations")
st.write("""
🔬 **Ethical Analysis:** Assess bias in prediction outcomes by gender, age, or socioeconomic factors.

💡 **Model Interpretability:** Implement SHAP or feature importance plots to explain predictions to healthcare stakeholders.

📈 **Future Data Collection:** Explore acquiring larger and more balanced datasets to improve model performance and generalizability.
""")

# References
st.header("References")
st.info("""Resources 

[1] N. Ghaffar Nia, E. Kaplanoglu, and A. Nasab, “Evaluation of artificial intelligence techniques in disease diagnosis and prediction,” Discov Artif Intell 3, 5 (2023). [Online] Available: https://doi.org/10.1007/s44163-023-00049-5 [Accessed June 12, 2025] 

[2] Y. Kumar, A. Koul, R. Singla et al. “Artificial intelligence in disease diagnosis: a systematic literature review, synthesizing framework and future research agenda,”  J Ambient Intell Human Comput 14, 8459–8486 (2023).  [Online] Available: https://doi.org/10.1007/s12652-021-03612-z [Accessed June 12, 2025] 

[3] V. Jackins, S. Vimal, M. Kaliappan et al. “AI-based smart prediction of clinical disease using random forest classifier and Naive Bayes.,” J Supercomput 77, 5198–5219 (2021).  [Online] Available: https://doi.org/10.1007/s11227-020-03481-x [Accessed June 12, 2025] 

[4] “Naive Bayes,” Scikit Learn. [Online] Available: https://scikit-learn.org/stable/modules/naive_bayes.html [Accessed June 13, 2025] 

[5] “Support Vector Machines,” Scikit Learn. [Online] Available: https://scikit-learn.org/stable/modules/svm.html [Accessed June 13, 2025] 

[6] “Random Forest and Other Randomized Tree Ensembles,” Scikit Learn. [Online] Available: https://scikit-learn.org/stable/modules/ensemble.html#forest [Accessed June 13, 2025] 
""")

##team responsibilities
st.header("Team Responsibilities")
## responsibility table
st.subheader("Responsibility Table")
st.table(
    {
        "Team Member": ["Ben Proell", "Nima Mollaei", "Yoomin Choi", "Taylor West", "Tyler Chevalier"],
        "Responsibilities": [
            "* Data cleaning\n* Updating website/visualizations for data cleaning",
            "* Vizualisations",
            "* Updating gantt chart\n* Next steps ",
            "* Data cleaning\n* Implementing bayes\n* Quantitative metrics ",
            "* Set up python notebook\n* Data cleaning\n* Organization of Midterm Report/putting on streamlit"
        ]
    }
)
## gantt chart
st.subheader("Gantt Chart")
st.image("pages/gantt.png", caption="Gantt Chart", use_column_width=True)
# Footer
st.markdown("---")
st.markdown("Ben Proell, Nima Mollaei, Yoomin Choi, Taylor West, Tyler Chevalier") 