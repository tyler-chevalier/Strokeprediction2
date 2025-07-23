import streamlit as st
# import import_ipynb
# import Project
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
st.title("Stroke Prediction Midterm Report")
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
For our data, we used the Stroke Prediction Dataset on Kaggle, downloaded in the form of a csv (healthcare-dataset-stroke-data.csv). 
To transform the data into a form that was usable for our model, we used pandas to read the csv file and load it into a dataframe.
First, duplicate rows were scanned for and removed to reduce overfitting. Next, N/A and NaN values, which would cause an error when running the model, were removed.
Because the data that was invalid were continuous features (age, bmi, and glucose level), they were replaced with the median value of their feature.
In the case of bmi, this median was calculated according to gender. Finally, yes/no and categorical features were changed into integers using maps so that the model would be able to read the features, and then normalized using a Standard Scaler. 
The process data was then saved to an output csv file (processed_stroke_data.csv).
In order to get a better idea of which features were significant to the dataset, we created visualizations of each feature, first by frequency, to see the raw amount of the label given each feature, and then as a percentage based on the label.
Because our dataset had a large amount of negative labels and a small amount of positive labels, the percentage graph gave a better idea of which features had a significant difference between labels.

""")

st.subheader("Naive Bayes Model")
st.write("""
Our first method used was Naive Bayes. We used GaussianNB to train our model, due to the fact that we had features with continuous values. 
In order to look at how different feature combinations would affect the model, we evaluated three different models using three metrics: 
the accuracy (a percentage of how many datapoints were evaluated correctly), the F1 score (a measurement taking into account the precision and 
recall), and the class likelihood ratio (a measurement evaluating the model predictions taking into account the frequency of a true positive and 
true negative). While we had originally planned on using the precision-recall curve as our third metric, we found that this wasn’t a useful metric
for Naive Bayes due to the lack of hyperparameters, and replaced it with accuracy instead.
Our dataset was split into 70% training data and 30% testing data.


The first model used all features of the dataset. The metrics achieved for this model were:
* Accuracy: 87.2%
* F1: 23.7%
* LR+: 4.03
* LR-: 0.64

The second model removed features that showed a less than 10% variation between categories on the percentage by label graphs. The features that remained were age, hypertension, ever_married, work_type, avg_glucose_level, and bmi. The metrics achieved for this model were:
Accuracy: 88.3%
* F1: 24.6%
* LR+: 4.39
* LR-: 0.65

The third model reduced the features again to the combination of features that created the highest accuracy. These features were age, hypertension, bmi, and smoking_status. The metrics achieved for this model were:
Accuracy: 89.8%
* F1: 19.1%
* LR+: 3.67
* LR-: 0.80


""")


# Results and Discussion
st.header("Results and Discussion")
st.subheader("Quantitative Metrics")
st.write("""
Overall, the Naive Bayes method achieved a high accuracy which was above our goal (80%) for this model. However, the F1 scores achieved were low.
Our metrics showed that we were very successful at predicting true negatives, but unsuccessful at predicting true positives. In real world terms,
predicting true positives would be more important in order to enable medical intervention reaching the people who need it. Due to this, our model
should be considered to have performed poorly despite the high accuracy. One reason that our model performed poorly was likely due to the fact that
there were a large amount of negative labels compared to the amount of positive labels, which lead to overfitting. Naive Bayes is also a simple 
model that we expected not to achieve as good results as our other models. For our next steps, we plan to implement Support Vector Machine 
and Random Forest as our next two models. These models have been shown to achieve greater success at identifying true positives in medical 
contexts, and we will hopefully be able to utilize them to achieve better F1 scores. If our scores do not improve, we will have to reevaluate 
our data cleaning and training methods, with one possible solution being to find another dataset with a larger set of positive labels to reduce
the overfitting.
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
For next steps, Support Vector Machine and Random Forest will be implemented as the next two models. These models have been shown to achieve 
greater success at identifying true positives in medical contexts, and will hopefully be able to achieve better F1 scores. If scores do not 
improve, data cleaning and training methods will need to be reevaluated, with one possible solution being to find another dataset with a larger 
set of positive labels to reduce the overfitting.
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