import pandas as pd

# integrated dataset 
integrated_data = pd.read_csv('prepared_and_scaled_dataset.csv')

# Feature Engineering: Creating additional features as requested

# 1. Total Spending: Sum of all 'Mnt' (amount spent) columns
mnt_columns = [col for col in integrated_data.columns if col.startswith('Mnt')]
integrated_data['TotalSpending'] = integrated_data[mnt_columns].sum(axis=1)

# 2. Avg Spending Per Purchase: Total Spending divided by the sum of all purchase columns
purchase_columns = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
total_purchases = integrated_data[purchase_columns].sum(axis=1)
integrated_data['AvgSpendingPerPurchase'] = integrated_data['TotalSpending'] / total_purchases

# 3. Income Spending Ratio: Total Spending divided by Income
integrated_data['IncomeSpendingRatio'] = integrated_data['TotalSpending'] / integrated_data['Income']

# 4. Engagement Score: Sum of all 'AcceptedCmp' columns and 'Response'
cmp_columns = [col for col in integrated_data.columns if col.startswith('AcceptedCmp') or col == 'Response']
integrated_data['EngagementScore'] = integrated_data[cmp_columns].sum(axis=1)

# 5. Education Categories: One-hot encoding for Education
education_categories = pd.get_dummies(integrated_data['Education'], prefix='Education')
integrated_data = pd.concat([integrated_data, education_categories], axis=1)

# 6. Marital Status Categories: One-hot encoding for Marital Status
marital_status_categories = pd.get_dummies(integrated_data['Marital_Status'], prefix='Marital_Status')
integrated_data = pd.concat([integrated_data, marital_status_categories], axis=1)

# 7. Customer Tenure: Days between 'Dt_Customer' and a reference date (e.g., today)
reference_date = pd.to_datetime('today')
integrated_data['Dt_Customer'] = pd.to_datetime(integrated_data['Dt_Customer'])
integrated_data['CustomerTenure'] = (reference_date - integrated_data['Dt_Customer']).dt.days

# 8. Categorized Response: Categorizing 'Response' into binary (0 or 1)
integrated_data['Categorized_Response'] = integrated_data['Response'].apply(lambda x: 1 if x > 0 else 0)

# Displaying the first few rows of the dataset with new features
new_features_overview = integrated_data.head()
new_features_overview

