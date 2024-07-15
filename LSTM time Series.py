import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Load and preprocess the dataset
file_path = 'standardized_customer_marketing_data.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Convert 'Dt_Customer' to datetime and create a 'MonthYear' column for aggregation
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')
df['MonthYear'] = df['Dt_Customer'].dt.to_period('M').dt.to_timestamp()

# Check and remove duplicates
duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    df.drop_duplicates(inplace=True)
    print(f"Removed {duplicate_count} duplicate rows")

# Aggregate monthly sales and normalize
monthly_sales = df.groupby('MonthYear')['TotalSpending'].sum().reset_index()
scaler = MinMaxScaler()
monthly_sales['TotalSpending'] = scaler.fit_transform(monthly_sales[['TotalSpending']])

# Create lagged and rolling features
num_lags = 3  # Number of lagged features
for lag in range(1, num_lags + 1):
    monthly_sales[f'TotalSpending_Lag{lag}'] = monthly_sales['TotalSpending'].shift(lag)

window_size = 3  # Window size for rolling statistics
monthly_sales['Rolling_Mean'] = monthly_sales['TotalSpending'].rolling(window=window_size).mean()
monthly_sales['Rolling_Std'] = monthly_sales['TotalSpending'].rolling(window=window_size).std()
monthly_sales.dropna(inplace=True)

# Split data for training, validation, and testing
train_size = 0.7
validation_size = 0.15
test_size = 0.15  # Ensure these add up to 1
train_data, validation_data, test_data = np.split(monthly_sales.sample(frac=1), [int(train_size*len(monthly_sales)), int((train_size+validation_size)*len(monthly_sales))])

# Define a function to reshape data for LSTM input
def reshape_data(df, features, target):
    X = df[features].values.reshape((df.shape[0], 1, len(features)))
    y = df[target].values
    return X, y

# Select features and target for reshaping
features = [col for col in monthly_sales.columns if col.startswith('TotalSpending_Lag') or col.startswith('Rolling')]
target = 'TotalSpending'

# Reshape the datasets
train_X, train_y = reshape_data(train_data, features, target)
validation_X, validation_y = reshape_data(validation_data, features, target)
test_X, test_y = reshape_data(test_data, features, target)

model = Sequential([
    LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]), activation='relu', return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.1),
    LSTM(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    Dropout(0.1),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)


# Train the model
history = model.fit(
    train_X, train_y,
    batch_size=32,  # Consider tuning batch size if needed
    epochs=75,  # Set to a number that allows learning without overfitting
    validation_data=(validation_X, validation_y),
    callbacks=[early_stopping],
    verbose=1
)

# ... [previous code] ...

# R-squared Function Definition (not typically used for time-series, but included for demonstration)
def r_squared(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)

# Evaluate the model
predictions = model.predict(test_X).flatten()
test_mse = mean_squared_error(test_y, predictions)
test_mae = mean_absolute_error(test_y, predictions)
test_mape = mean_absolute_percentage_error(test_y, predictions)  # Calculate MAPE
test_r_squared = r_squared(test_y, predictions)  # Calculate R-squared

# Print evaluation metrics
print(f"Test MSE: {test_mse}")
print(f"Test MAE: {test_mae}")
print(f"Test MAPE: {test_mape}")  # Print MAPE
print(f"Test R-squared: {test_r_squared}")  # Print R-squared

# ... [rest of the LSTM script] ...


# Plotting training and validation loss and MAE
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.tight_layout()
plt.show()

# Actual vs Predicted Monthly Sales Plot
plt.figure(figsize=(10, 6))
plt.plot(range(len(test_y)), test_y, label='Actual Sales', color='blue')
plt.plot(range(len(predictions)), predictions, label='Predicted Sales', color='red', linestyle='--')
plt.title('Actual vs Predicted Monthly Sales')
plt.xlabel('Time Index')
plt.ylabel('Normalized Sales')
plt.legend()
plt.show()

# Calculate the prediction errors
errors = test_y - predictions

# Plotting prediction error over time
plt.figure(figsize=(10, 6))
plt.plot(range(len(errors)), errors, marker='o', linestyle='', color='red')
plt.title('Prediction Error Over Time')
plt.xlabel('Time Index')
plt.ylabel('Prediction Error')
plt.axhline(0, color='black', lw=1)  # Add a line at zero error level
plt.show()
# Plotting training and validation loss and MAE on the same graph
plt.figure(figsize=(12, 5))

# First subplot for the loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Second subplot for the MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()

plt.tight_layout()
plt.show()
# Plot residuals
residuals = test_y - predictions
plt.figure(figsize=(10, 6))
plt.plot(residuals, label='Residuals', color='blue')
plt.hlines(y=0, xmin=0, xmax=len(residuals), colors='red', linestyles='--')
plt.title('Residuals of Predictions')
plt.xlabel('Time Index')
plt.ylabel('Residuals')
plt.legend()
plt.show()

# Distribution of Predictions and Actual Values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(test_y, bins=20, alpha=0.7, label='Actual Values')
plt.title('Distribution of Actual Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(predictions, bins=20, alpha=0.7, label='Predicted Values', color='red')
plt.title('Distribution of Predicted Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# MAE Over Time
mae_over_time = np.abs(residuals)
plt.figure(figsize=(10, 6))
plt.plot(mae_over_time, label='MAE Over Time', color='orange')
plt.title('Mean Absolute Error Over Time')
plt.xlabel('Time Index')
plt.ylabel('MAE')
plt.legend()
plt.show()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# Define the color map using a nice palette
cmap = plt.get_cmap('Spectral')

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
sc = ax.scatter(np.arange(len(test_y)), test_y, predictions, c=predictions, cmap=cmap)
plt.colorbar(sc, label='Predicted Sales')
ax.set_title('3D Scatter Plot: Actual vs Predicted Sales Over Time')
ax.set_xlabel('Time Index')
ax.set_ylabel('Actual Sales')
ax.set_zlabel('Predicted Sales')
plt.show()
from matplotlib import cm

# Assuming 'history.history' contains loss values for both training and validation
X = np.arange(0, len(history.history['loss']))
Y = np.arange(0, len(history.history['val_loss']))
X, Y = np.meshgrid(X, Y)
Z = np.array(history.history['loss'])

fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_title('Surface Plot: Model Loss Over Epochs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Epoch')
ax.set_zlabel('Loss')
plt.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
# Assuming 'feature_importance' contains importance of features
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

import seaborn as sns

# Assuming 'df' is your DataFrame
correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='YlGnBu')
plt.title('Heatmap of Correlation Matrix')
plt.show()
