import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc)
from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Load the standardized dataset
file_path = 'standardized_customer_marketing_data.csv'
data = pd.read_csv(file_path)

# Convert 'Dt_Customer' to datetime and then to ordinal format
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer']).apply(lambda x: x.toordinal())

# Splitting the data into features and target
X = data.drop('Response', axis=1)
y = data['Response']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Preprocessing
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', 'passthrough', categorical_features)
    ])

# Model pipelines
log_reg_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression(random_state=42))])

rf_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', RandomForestClassifier(random_state=42))])

# Hyperparameter tuning using GridSearchCV
log_reg_param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__penalty': ['l2']
}

rf_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

log_reg_grid_search = GridSearchCV(log_reg_pipe, log_reg_param_grid, cv=5, scoring='accuracy')
rf_grid_search = GridSearchCV(rf_pipe, rf_param_grid, cv=5, scoring='accuracy')

# Fit models
log_reg_grid_search.fit(X_train, y_train)
rf_grid_search.fit(X_train, y_train)

# Best params
print("Best parameters for Logistic Regression: ", log_reg_grid_search.best_params_)
print("Best parameters for Random Forest: ", rf_grid_search.best_params_)

# Function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision_recall = precision_recall_curve(y_test, y_proba)
    
    return accuracy, precision, recall, f1, roc_auc, conf_matrix, fpr, tpr, precision_recall

log_reg_metrics = evaluate_model(log_reg_grid_search, X_test, y_test)
rf_metrics = evaluate_model(rf_grid_search, X_test, y_test)

# Visualization of Confusion Matrix, ROC Curve, Precision-Recall Curve, Feature Importance
# Add code for visualization here

# Feature Importance for Random Forest
feature_importance_rf = rf_grid_search.best_estimator_.named_steps['classifier'].feature_importances_
# Sorting the feature importances
sorted_idx = np.argsort(feature_importance_rf)[::-1]

# 3D Scatter Plot of Top 3 Features
# Modify the code for 3D scatter plot here

plt.show()
