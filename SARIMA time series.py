import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
# Load and Prepare Data
file_path = 'standardized_customer_marketing_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Convert 'Dt_Customer' to datetime and set as index
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'])
data.set_index('Dt_Customer', inplace=True)

# Aggregate 'TotalSpending' by month
monthly_sales = data['TotalSpending'].resample('M').sum()

# Splitting Dataset (80% training, 20% test)
split_ratio = 0.8
split_index = int(len(monthly_sales) * split_ratio)
train_data, test_data = monthly_sales[:
                                      split_index], monthly_sales[split_index:]

# Grid Search for SARIMA Parameters
p = d = q = range(0, 3)  # AR, differencing, and MA components
seasonal_pdq = [(x[0], x[1], x[2], 12)
                for x in itertools.product(p, d, q)]  # Seasonal components

best_aic = np.inf
best_model = None
best_params = None
for param in itertools.product(p, d, q):
    for param_seasonal in seasonal_pdq:
        try:
            model = SARIMAX(train_data,
                            order=param,
                            seasonal_order=param_seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_model = results
                best_params = (param, param_seasonal)
        except:
            continue

print('Best SARIMA Model:', best_params, 'AIC:', best_aic)

# Model Training
model = SARIMAX(monthly_sales,
                order=best_params[0],
                seasonal_order=best_params[1],
                enforce_stationarity=False,
                enforce_invertibility=False)
results = model.fit()
# MAPE Function Definition
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Model Evaluation
predictions = results.get_prediction(start=test_data.index[0], end=test_data.index[-1])
predictions_mean = predictions.predicted_mean
mae = mean_absolute_error(test_data, predictions_mean)
rmse = np.sqrt(mean_squared_error(test_data, predictions_mean))
mape = mean_absolute_percentage_error(test_data, predictions_mean)  # Calculate MAPE

# Print evaluation metrics
print('RMSE:', rmse)
print('MAE:', mae)
print('MAPE:', mape)  # Print MAPE


# Set the aesthetic style of the plots
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# Plotting Actual vs Predicted Sales with Improved Aesthetics
plt.figure(figsize=(14, 7))
plt.plot(monthly_sales.index, monthly_sales,
         label='Actual Sales', color='dodgerblue', linewidth=2)
plt.plot(predictions_mean.index, predictions_mean,
         label='Predicted Sales', color='crimson', linewidth=2)
plt.fill_between(predictions_mean.index,
                 predictions.conf_int().iloc[:, 0],
                 predictions.conf_int().iloc[:, 1], color='palevioletred', alpha=0.3)
plt.title('Actual vs Predicted Sales', fontsize=20)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Standardized Sales', fontsize=14)
plt.legend(frameon=True, fontsize=12)
plt.tight_layout()
sns.despine(trim=True)  # Remove the top and right spines from plot
plt.show()


# Forecasting Future Sales
future_forecast = results.get_forecast(steps=12)  # Next 12 months
forecast_values = future_forecast.predicted_mean
forecast_conf_int = future_forecast.conf_int()
print(forecast_values)
print(forecast_conf_int)

# Plotting the Sales Forecast with Improved Aesthetics
plt.figure(figsize=(14, 7))
plt.plot(monthly_sales.index, monthly_sales,
         label='Past Sales', color='dodgerblue', linewidth=2)
plt.plot(forecast_values.index, forecast_values,
         label='Forecasted Sales', color='forestgreen', linewidth=2)
plt.fill_between(forecast_values.index,
                 forecast_conf_int.iloc[:, 0],
                 forecast_conf_int.iloc[:, 1], color='lightgreen', alpha=0.5)
plt.title('Sales Forecast', fontsize=20)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Standardized Sales', fontsize=14)
plt.legend(frameon=True, fontsize=12)
plt.tight_layout()
sns.despine(trim=True)  # Remove the top and right spines from plot
plt.show()
