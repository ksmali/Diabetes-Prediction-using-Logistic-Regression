# Diabetes-Prediction-using-Logistic-Regression

This project involves predicting whether a person has diabetes or not based on various factors such as pregnancies, glucose levels, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age. The dataset used in the project is loaded using Pandas and exploratory data analysis is performed using Matplotlib and Seaborn. The missing values in certain columns are replaced with mean values of respective columns.

After the data is cleaned and prepared, logistic regression is applied to the dataset to make predictions. Before that, feature selection is performed using Recursive Feature Elimination (RFE) technique to select the most relevant features. The dataset is then split into training and testing sets using train_test_split from Scikit-learn.

The training dataset is scaled using StandardScaler from Scikit-learn to normalize the data. Logistic regression is then applied to the training set and the model is evaluated using various metrics such as accuracy, precision, recall, and F1 score. The confusion matrix and ROC curve are plotted to visualize the performance of the model.

Finally, the AUC score is calculated to determine the overall performance of the model. This project can be useful for healthcare professionals to predict the likelihood of diabetes in patients based on their medical history and other factors.
