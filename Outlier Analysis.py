
import pandas as pd

# Load the dataset
file_path = 'standardized_customer_marketing_data.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
data_info = data.info()
data_head = data.head()

#2. Initial Outlier Identification (Box Plots)": 
import matplotlib.pyplot as plt
import seaborn as sns

# Selecting key columns for outlier analysis
columns_to_analyze = ['Income', 'TotalSpending', 'NumWebVisitsMonth']

# Creating box plots for each selected column
plt.figure(figsize=(18, 6))
for i, col in enumerate(columns_to_analyze, 1):
    plt.subplot(1, len(columns_to_analyze), i)
    sns.boxplot(y=data[col])
    plt.title(f'Box Plot of {col}')

plt.tight_layout()
plt.show()
#3. Quantitative Outlier Analysis (IQR Method)": 
# Function to calculate IQR and identify outliers
def identify_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Applying the function to each selected column
outlier_info = {}
for col in columns_to_analyze:
    outliers, lower_bound, upper_bound = identify_outliers(data, col)
    outlier_info[col] = {
        "Number of Outliers": len(outliers),
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound
    }

 #4. Comprehensive Outlier Visualization (Optimized Box Plots)": 
# Optimizing the visual appeal of the box plots with aesthetic formatting and colors

# Setting a style for the plots
sns.set_style("whitegrid")

# Choosing a color palette
palette = sns.color_palette("coolwarm", n_colors=len(all_columns_for_outlier_analysis))

# Adjusting the layout to accommodate all variables for visualization
num_rows = 5  # Number of rows in the subplot grid
num_cols = 3  # Number of columns in the subplot grid

plt.figure(figsize=(20, 25))
for i, col in enumerate(all_columns_for_outlier_analysis, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(y=data[col], color=palette[i-1])
    plt.title(f'Box Plot of {col}', fontsize=15)
    plt.ylabel(col, fontsize=12)
    plt.xlabel('')

plt.tight_layout()
plt.suptitle('Box Plots for Various Variables in the Dataset', fontsize=20, y=1.02)
plt.show()

 #5. Outlier Handling and Post-Handling Visualization
import numpy as np

# Setting percentile thresholds for capping
percentile_threshold = 95

# Function to cap values at a specified percentile
def cap_values(df, column, percentile):
    threshold = df[column].quantile(percentile / 100)
    df[column] = np.where(df[column] > threshold, threshold, df[column])
    return df

# Applying capping to spending and purchase behavior variables
spending_behavior_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
                             'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases',
                             'NumCatalogPurchases', 'NumStorePurchases']

for col in spending_behavior_columns:
    data = cap_values(data, col, percentile_threshold)

# Applying log transformation to 'NumWebVisitsMonth' to normalize distribution
data['NumWebVisitsMonth'] = np.log1p(data['NumWebVisitsMonth'])

# Visualizing the updated distributions
plt.figure(figsize=(20, 15))
for i, col in enumerate(spending_behavior_columns + ['NumWebVisitsMonth'], 1):
    plt.subplot(4, 3, i)
    sns.boxplot(y=data[col], color=palette[i-1])
    plt.title(f'Box Plot of {col} (Post Handling)', fontsize=15)

plt.tight_layout()
plt.suptitle('Box Plots Post Outlier Handling', fontsize=20, y=1.02)
plt.show()

python_scripts

