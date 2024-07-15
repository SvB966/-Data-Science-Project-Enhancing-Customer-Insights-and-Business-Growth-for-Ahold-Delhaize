# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:54:30 2023

@author: SGvan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('preprocessed_customer_marketing_data.csv')  # Update the path to your dataset

# Feature Engineering: Creating new combined features
data['Age'] = 2023 - data['Year_Birth']  # Assuming the current year is 2023
data['Income_Per_Household_Member'] = data['Income'] / (data['Kidhome'] + data['Teenhome'] + 1)
data['Total_Alcohol_Spending'] = data['MntWines'] + data['MntMeatProducts']  # Example combination
data['Luxury_Spending'] = data['MntGoldProds'] + data['MntSweetProducts']  # Another example
data['Engagement_Recency'] = data['EngagementScore'] * data['Recency']
data['Tenure_Spending_Ratio'] = data['CustomerTenure'] / data['TotalSpending']

# Standardizing the entire dataset (including newly created features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.drop(columns=['Categorized_Response']))

# Applying PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_pca = pca.fit_transform(X_scaled)

# Calculating IQR for each PCA component
Q1 = np.percentile(X_pca, 25, axis=0)
Q3 = np.percentile(X_pca, 75, axis=0)
IQR = Q3 - Q1

# Defining the outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Function to replace outliers with the 25th or 75th percentile values
def cap_outliers(data, lower_bound, upper_bound, percentiles):
    data_capped = np.where(data < lower_bound, percentiles[0], data)
    data_capped = np.where(data_capped > upper_bound, percentiles[1], data_capped)
    return data_capped

# Capping outliers in the PCA-transformed dataset
X_pca_capped = np.copy(X_pca)
for i in range(X_pca.shape[1]):
    X_pca_capped[:, i] = cap_outliers(X_pca[:, i], lower_bound[i], upper_bound[i], [Q1[i], Q3[i]])

# Splitting the dataset into training and testing sets
y = data['Categorized_Response']
X_train, X_test, y_train, y_test = train_test_split(X_pca_capped, y, test_size=0.2, random_state=42)

# Training the models
random_forest = RandomForestClassifier(random_state=42)
gbm = GradientBoostingClassifier(random_state=42)
adaboost = AdaBoostClassifier(random_state=42)

random_forest.fit(X_train, y_train)
gbm.fit(X_train, y_train)
adaboost.fit(X_train, y_train)

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[('rf', random_forest), ('gbm', gbm), ('adaboost', adaboost)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# Function to evaluate a model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Evaluating each model
accuracy_rf, report_rf = evaluate_model(random_forest, X_test, y_test)
accuracy_gbm, report_gbm = evaluate_model(gbm, X_test, y_test)
accuracy_adaboost, report_adaboost = evaluate_model(adaboost, X_test, y_test)
accuracy_voting, report_voting = evaluate_model(voting_clf, X_test, y_test)

# Hyperparameter Tuning Setup for Random Forest (Example)
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 4, 6]
}

grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                              param_grid=param_grid_rf, 
                              cv=3, 
                              n_jobs=-1, 
                              verbose=2)

# Fitting the grid search to the data (this step may require adjustments based on computational resources)
# grid_search_rf.fit(X_train, y_train)
