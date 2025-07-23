import streamlit as st
# import import_ipynb
# import Project


st.set_page_config(
    page_title="Final Report",
    page_icon="🧠",
    layout="wide"
)

st.title("Stroke Prediction Final Report")
st.markdown("---")


# Introduction and Background
st.header("Introduction and Background")
st.write("""
This project aims to use machine learning to predict strokes before they happen, by identifying high risk individuals using medical and demographic data. It uses the Stroke Prediction Dataset from Kaggle which contains features such as age, gender, BMI, and work type.  
Currently, Support Vector Machine (SVM) is the most widely used ML method for using AI to predict diagnoses, with many studies finding it achieves the best performance [1]. One study found it to have a 97.56% accuracy rate in detecting Alzheimer's [2]. Random Forest is another method that’s widely used and has achieved good results, in one study having a 98.4% accuracy in detecting TD antibodies [1]. Hybrid systems, such as between SVM and artificial neural network, have also found good results [3], and Naive Bayes was able to achieve 82.35% test accuracy in evaluating heart disease data [3]. 

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
st.subheader("Visualizations of dataset")

st.image("Images_for_visualizations/vis1.png",caption="Percent of stroke based on gender")

st.image(["Images_for_visualizations/freq_based_on_age.png", "Images_for_visualizations/freq_glucose.png"],
         caption=["Frequency based on age","Frequency base on glucose"])
st.image(["Images_for_visualizations/freq_heart_disease.png","Images_for_visualizations/freq_hypertension.png"],
         caption=["Frequency given heart disease","Frequency given hypertension"])
st.image(["Images_for_visualizations/freq_married.png", "Images_for_visualizations/freq_residence.png"],
         caption=["Frequency given married status", "Frequency given residence type"])
st.image(["Images_for_visualizations/freq_smoking.png", "Images_for_visualizations/freq_work_type.png"],
         caption=["Frequency given smoking status", "Frequency given work type"])
st.image(["Images_for_visualizations/percent_given_age.png", "Images_for_visualizations/percent_BMI.png"],
         caption=["Percent given age", "Percent BMI"])

st.subheader("Naive Bayes")
st.write("""
The first method used was Naive Bayes. GaussianNB was the best choice to train our model, due to the fact that there were features with continuous values. In order to look at how different feature combinations would affect the model, they were evaluated three different models using four metrics: the accuracy (a percentage of how many datapoints were evaluated correctly), the F1 score (a measurement taking into account the precision and recall), the recall (which evaluated true positives and gave a better idea at how well the model was able to identify at risk patients) and the class likelihood ratio (a measurement evaluating the model predictions taking into account the frequency of a true positive and true negative). While it was originally planned to use the precision-recall curve as an additional metric, it was found that this wasn’t a useful metric for Naive Bayes due to the lack of hyperparameters, and was replaced with accuracy instead.
Our dataset was split into 70% training data and 30% testing data.

The first model used all features of the dataset. The metrics achieved for this model were:
* Accuracy: 87.2%
* F1: 24.6%
* Recall: 42.1%
* LR+: 4.03
* LR-: 0.65

The second model removed features that showed a less than 10% variation between categories on the percentage by label graphs. The features that remained were age, hypertension, ever_married, work_type, avg_glucose_level, and bmi. The metrics achieved for this model were:
* Accuracy: 89.0%
* F1: 28.1%
* Recall: 43.4%
* LR+: 5.02
* LR-: 0.62

The third model used all features, but used smote on the dataset. The metrics achieved for this model were:
* Accuracy: 74.4%
* F1: 21.0%
* Recall: 72.4%
* LR+: 2.83
* LR-: 0.37



""")

st.subheader("Support Vector Machine")
st.write("""
The second model was Support Vector Machine. This was the model that had achieved the highest metrics in the literature reviewed, so it was expected that it was possible to use SVM to achieve greater than 90% for the accuracy and F1 score metrics. However, there were limitations in the chosen dataset that made it not possible for this to be achieved.
The first limitation was that the data was not linearly separable given the features. As shown in the visualization of the SVM model, positive datapoints are mixed in with the negative datapoints, so that they can’t be easily separated by a line. This characteristic is retained across different combinations of features. There was also a low amount of positively labelled datapoints in the dataset, which made it difficult to train the model to recognize them.
To combat this, class weights of {0:1, 1:17} were used, which gave more weight to the positive label. A gamma of 0.4 and a cost of 0.1 were also used. Combined, these hyperparameters were aimed towards preventing overfitting of the negative data. For the kernel, the radial ‘rbf’ kernel was used, which is the best fit for data that is not linearly separable and does not have a clear pattern.
Because in a real-world situation it would be more important to identify true positives, or a patient having a stroke, as opposed to true negatives, the models were aimed towards maximizing recall even if it came at the expense of some accuracy.

For the first SVM model, which used all features of the dataset, the metrics achieved were:
* Accuracy: 68.7%
* F1: 19.9%
* Recall: 83.7%
* LR+: 2.62
* LR-: 0.24

For the second SVM model, which removed features that showed a less than 10% variation between categories on the percentage by label graphs, the metrics achieved were:
* Accuracy: 68.0%
* F1: 20.2%
* Recall: 87.3%
* LR+: 2.66
* LR-: 0.19

For the third model, which used all features and SMOTE, the metrics achieved were:
* Accuracy: 36.0%
* F1: 12.4%
* Recall: 97.6%
* LR+: 1.46
* LR-: 0.07


""")

st.subheader("Random Forest")
st.write("""
Our third model was Random Forest. On first attempts, using all the features from the dataset as is made our accuracy for negatives 
reach 95%, while our accuracy for positives sat at 0% . Our dataset has about 5% of the data points being stroke victims, meaning our model is 
only guessing “no stroke”. Our F1 score for positives was also 0 when using all features. We tweaked our features and used age, hypertension, 
work type, residence type, average glucose level, and smoking status for our first variant. This variant had an F1 score of 0.07. Next, we 
implemented PCA to try and raise our F1, and we only managed to get it to 0.03. Lastly we used SMOTE to try and generate some synthetic data
points, which helped our metrics improve. The SMOTE implementation of Random Forest gave us our best F1 Score, and our best recall so far, 
peaking at 0.25 and 0.38 respectively.
""")

# Results and Discussion
st.header("Results and Discussion")
st.subheader("Naive Bayes Results")
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

st.subheader("Model Comparison and Analysis")
st.write("""
The Naive Bayes achieved high accuracy but very low recall and F1 scores. This shows using Naive Bayes for predicting the majority class, in this 
case non-stroke patients, is efficient. However, in the case of identifying true positives, it was not efficient, critical in clinical settings. 
The class imbalance possibly could lead the model to overfit to the negative class. Feature reduction helped slightly to improve accuracy but made 
recall worse. Even though Naive Bayes method was fast and simple, it was not suitable when the recall is important.The Support Vector Machine (SVM)
approach underperformed on accuracy and F1, but it achieved the highest recall rate. Since it is a clinical setting, having high recall made it the
best fit at identifying true stroke cases. This is because of class weight tuning also with adjusting hyperparameters and using the RBF kernel. 
Although it has lower overall accuracy, SVM has a recall-oriented performance which makes it more appropriate for healthcare applications, since
missing true positives is more risky. The Random Forest model showed strong overall accuracy but failed to detect stroke cases by having 0% 
recall and F1 initially. Even after applying to PCA, the performance of the minority class was poor. However, after implementing SMOTE, it 
improved. This showcases that the model tends to overfit to the majority class with imbalanced datasets. In this case, Random Forest is not 
suitable. In the comparison of the three models, Naive Bayes resulted in the highest accuracy but lowest recall. So it is the least efficient 
at detecting at-risk patients even though it is simple and computationally efficient. SVM maximized recall and outperformed Naive Bayes for true
positive detection, keeping with a balance better suited to clinical needs. Random Forest performed worse because initially it could not deal with 
the imbalance in the dataset to make any meaningful positive class predictions. The performance plot of all models emphasized that an imbalanced 
dataset impacts the result a lot overall, since all models were not good with poor F1 scores and poor performance on the positive class. SVM’s 
high recall rate is best suited for this medical application. To deal with imbalance, Random Forest is a possibility but requires significant 
preprocessing such as SMOTE. Although Naive Bayes has a high accuracy rate, it was too simplistic for this job. All the models’ future development 
will come in the form of balancing the set, through increased data intake or synthesizing data creation. It can more adequately catch true 
positives.

""")


st.subheader("Next Steps")
st.write("""
         To improve stroke prediction models, the main focus moving to the next step would be addressing the dataset’s class imbalance. 
         Applying Synthetic Minority Over-sampling Technique (SMOTE) would be helpful to generate synthetic stroke cases by improving recall 
         and F1 scores across all the three models. Additionally, utilizing KNN imputation or acquiring more datasets with more positive cases 
         would enhance preprocessing would increase generalizability.Model-specific improvements such as testing ComplementNB for Naive 
         Bayes would help handle imbalance better. Hyperparameter tuning for SVM using grid search and applying class weighting with SMOTE 
         to Random Forest would be a good approach as well. Also to validate even better, getting more external data and building a streamlined
         evaluation pipeline continuously will be a good approach to refine the models.  
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
            "Introduction/Background, Problem Definition, SVM Hyperparameter Tuning/Visualizations, Website, Website Hosting, Precision Recall Curves, Data Cleaning",
            "PCA Method, Random Forest Method, SMOTE, Exploratory Data Analysis, Visualizations, Introduction/Background",
            "Model Comparison and Analysis, Next Steps, Website",
            "Research/Resources, Introduction/Background, Problem Definition, Naive Bayes Model, Naive Bayes Metrics/Visualizations, Naive Bayes Analysis, SVM Model, SVM Metrics/Visualizations, SVM Analysis",
            "Introduction/Background, Problem Definition,Data cleaning, PCA method,Random Forest Method and its visualizations, SMOTE, Random Forest analysis"
        ]
    }
)
st.write("""The team worked together to create the presentation and write the report.""")
## gantt chart
st.subheader("Gantt Chart")
st.image("pages/gantt.png", caption="Gantt Chart", use_column_width=True)
# Footer
st.markdown("---")  
st.markdown("Ben Proell, Nima Mollaei, Yoomin Choi, Taylor West, Tyler Chevalier") 