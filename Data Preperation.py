import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Load all the provided datasets
purchasing_behavior_data = pd.read_csv('purchasing_behavior_data.csv')
household_data = pd.read_csv('household_data.csv')
marketing_responses_data = pd.read_csv('marketing_responses_data.csv')
promotional_response_data = pd.read_csv('promotional_response_data.csv')
economic_data = pd.read_csv('economic_data.csv')
geographic_data = pd.read_csv('geographic_data.csv')
demographic_data = pd.read_csv('demographic_data.csv')
customer_engagement_data = pd.read_csv('customer_engagement_data.csv')

# Displaying the first few rows of each dataset to understand their structure and contents
datasets = {
    "Purchasing Behavior Data": purchasing_behavior_data,
    "Household Data": household_data,
    "Marketing Responses Data": marketing_responses_data,
    "Promotional Response Data": promotional_response_data,
    "Economic Data": economic_data,
    "Geographic Data": geographic_data,
    "Demographic Data": demographic_data,
    "Customer Engagement Data": customer_engagement_data
}

dataset_overviews = {name: data.head() for name, data in datasets.items()}
dataset_overviews

# Data Cleaning: Analyzing missing values and inconsistencies in each dataset

missing_values_summary = {name: data.isnull().sum() for name, data in datasets.items()}
missing_values_summary

# Step 1: Removing rows with missing 'ID' values from all datasets

cleaned_datasets = {name: data.dropna(subset=['ID']) for name, data in datasets.items()}

# Re-evaluating the missing values after removing rows with missing 'ID'
cleaned_missing_values_summary = {name: cleaned_data.isnull().sum() for name, cleaned_data in cleaned_datasets.items()}
cleaned_missing_values_summary

# Step 2: Addressing missing values in other columns with imputation strategies

# Defining a function for imputation
def impute_missing_values(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == 'float64' or dataset[column].dtype == 'int64':
            # Impute numerical columns with mean
            dataset[column].fillna(dataset[column].mean(), inplace=True)
        else:
            # Impute categorical columns with mode
            dataset[column].fillna(dataset[column].mode()[0], inplace=True)
    return dataset

# Applying the imputation to all datasets
imputed_datasets = {name: impute_missing_values(data.copy()) for name, data in cleaned_datasets.items()}

# Checking the missing values post-imputation
imputed_missing_values_summary = {name: imputed_data.isnull().sum() for name, imputed_data in imputed_datasets.items()}
imputed_missing_values_summary



# Step 3: Standardization and Encoding of Categorical Data

# Applying label encoding to categorical columns in the Demographic and Geographic Data
def encode_categorical_data(dataset):
    label_encoder = LabelEncoder()
    for column in dataset.columns:
        if dataset[column].dtype == 'object':
            dataset[column] = label_encoder.fit_transform(dataset[column])
    return dataset

# Encoding categorical data in Demographic and Geographic Data
demographic_data_encoded = encode_categorical_data(imputed_datasets['Demographic Data'].copy())
geographic_data_encoded = encode_categorical_data(imputed_datasets['Geographic Data'].copy())

# Updating the datasets dictionary with the encoded datasets
imputed_datasets['Demographic Data'] = demographic_data_encoded
imputed_datasets['Geographic Data'] = geographic_data_encoded

# Displaying the first few rows of the encoded datasets to verify the changes
encoded_dataset_overviews = {
    "Encoded Demographic Data": demographic_data_encoded.head(),
    "Encoded Geographic Data": geographic_data_encoded.head()
}
encoded_dataset_overviews

# Step 4: Data Integration - Merging all datasets on the 'ID' column

# Merging datasets one by one on the 'ID' column
integrated_data = imputed_datasets['Purchasing Behavior Data']
for name, dataset in imputed_datasets.items():
    if name != 'Purchasing Behavior Data':
        integrated_data = integrated_data.merge(dataset, on='ID', how='left')

# Checking the first few rows of the integrated dataset to ensure successful merging
integrated_data_overview = integrated_data.head()
integrated_data_overview, integrated_data.shape

# Final Quality Check on the Integrated Dataset

# Checking for data types and any remaining inconsistencies
data_types = integrated_data.dtypes
any_inconsistencies = integrated_data.isnull().sum().any()

# Checking the summary statistics to understand the distribution of data
summary_statistics = integrated_data.describe()

quality_check_results = {
    "Data Types": data_types,
    "Any Inconsistencies": any_inconsistencies,
    "Summary Statistics": summary_statistics
}
quality_check_results



# Excluding non-numerical and ID columns from scaling
scaling_columns = integrated_data.select_dtypes(include=['float64', 'int64']).columns.drop('ID')

# Applying StandardScaler
scaler = StandardScaler()
integrated_data_scaled = integrated_data.copy()
integrated_data_scaled[scaling_columns] = scaler.fit_transform(integrated_data[scaling_columns])

# Displaying the first few rows of the scaled dataset to verify the changes
scaled_data_overview = integrated_data_scaled.head()
scaled_data_overview

