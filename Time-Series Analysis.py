import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller
from mpl_toolkits.mplot3d import Axes3D

# Set the visual style of the plots
sns.set_style("whitegrid")

def load_and_process_data(file_path):
    data = pd.read_csv(file_path)
    data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'])
    spending_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    data['TotalSpending'] = data[spending_columns].sum(axis=1)
    return data

def plot_total_spending_over_time(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Dt_Customer'], data['TotalSpending'], color='blue', linestyle='--')
    plt.title('Total Spending Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Spending')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_spending_distribution(data):
    plt.figure(figsize=(12, 6))
    sns.histplot(data['TotalSpending'], kde=True, color='green')
    plt.title('Distribution of Total Spending')
    plt.xlabel('Total Spending')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_monthly_sales(data):
    data['MonthYear'] = data['Dt_Customer'].dt.to_period('M')
    monthly_sales = data.groupby('MonthYear')['TotalSpending'].sum()
    plt.figure(figsize=(12, 6))
    monthly_sales.plot(color='purple')
    plt.title('Monthly Sales Over Time')
    plt.xlabel('Month-Year')
    plt.ylabel('Total Sales')
    plt.tight_layout()
    plt.show()
    return monthly_sales

def perform_seasonal_decompose(monthly_sales):
    monthly_sales.index = monthly_sales.index.to_timestamp()
    decomposition = seasonal_decompose(monthly_sales, model='additive')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
    decomposition.trend.plot(ax=ax1, color='red')
    ax1.set_title('Trend')
    ax1.set_xlabel('Month-Year')
    decomposition.seasonal.plot(ax=ax2, color='orange')
    ax2.set_title('Seasonality')
    ax2.set_xlabel('Month-Year')
    decomposition.resid.plot(ax=ax3, color='teal')
    ax3.set_title('Residuals')
    ax3.set_xlabel('Month-Year')
    plt.tight_layout()
    plt.show()

def augmented_dickey_fuller_test(monthly_sales):
    adf_test_result = adfuller(monthly_sales)
    adf_p_value = adf_test_result[1]
    stationarity_status = "non-stationary" if adf_p_value > 0.05 else "stationary"
    print(f"ADF P-Value: {adf_p_value}, Stationarity Status: {stationarity_status}")

# Main execution
file_path = 'standardized_customer_marketing_data.csv'
data_sorted = load_and_process_data(file_path).sort_values('Dt_Customer')

plot_total_spending_over_time(data_sorted)
plot_spending_distribution(data_sorted)
monthly_sales = plot_monthly_sales(data_sorted)
perform_seasonal_decompose(monthly_sales)
augmented_dickey_fuller_test(monthly_sales)

# Example data: Error values decreasing over iterations
iterations = np.arange(1, 101)  # 100 iterations
error = np.exp(-0.05 * iterations)  # Exponential decay of error

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create a color palette
palette = sns.color_palette("coolwarm", n_colors=1)

# Plot
plt.figure(figsize=(10, 6))
sns.lineplot(x=iterations, y=error, palette=palette, linewidth=2.5)
plt.title('Error Reduction Over Iterations', fontsize=16)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.show()


# Example data: Cost function landscape
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

# Set the aesthetic style of the plots
sns.set_style("white")

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_title('Cost Function Landscape', fontsize=16)
ax.set_xlabel('X-axis', fontsize=14)
ax.set_ylabel('Y-axis', fontsize=14)
ax.set_zlabel('Cost', fontsize=14)
plt.show()
