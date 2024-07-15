import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import style
import numpy as np

# Load the dataset
file_path = 'standardized_customer_marketing_data.csv'
df = pd.read_csv(file_path)

# Convert 'Dt_Customer' to datetime and extract year
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
df['Year'] = df['Dt_Customer'].dt.year

# Display the first few rows of the dataset
print(df.head())

# Histograms for continuous data
continuous_columns = ['Income', 'TotalSpending', 'CustomerTenure']
for col in continuous_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Correlation heatmap
purchase_columns = ['Income', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts']
correlation_matrix = df[purchase_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation between Income and Purchasing Behaviors')
plt.show()

## Ensure only unique columns are included for groupby mean calculation
family_columns = ['Kidhome', 'Teenhome']
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove any family columns from numeric_cols to avoid duplication
numeric_cols = [col for col in numeric_cols if col not in family_columns]

# Group data by family composition and calculate average spending
family_grouped = df[family_columns + numeric_cols].groupby(family_columns).mean().reset_index()

# 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(family_grouped['Kidhome'], family_grouped['Teenhome'], family_grouped['TotalSpending'])
ax.set_xlabel('Kidhome')
ax.set_ylabel('Teenhome')
ax.set_zlabel('TotalSpending')
plt.title('Customer Segments Based on Family Composition')
plt.show()


# Bar charts for categorical data (education and marital status)
education_columns = [col for col in df.columns if 'Education_' in col]
marital_status_columns = [col for col in df.columns if 'Marital_Status_' in col]
fig, ax = plt.subplots(1, 2, figsize=(20, 5))
sns.barplot(x=education_columns, y=df[education_columns].sum(), ax=ax[0])
sns.barplot(x=marital_status_columns, y=df[marital_status_columns].sum(), ax=ax[1])
ax[0].set_title('Distribution of Education Levels')
ax[1].set_title('Distribution of Marital Status')
ax[0].tick_params(axis='x', rotation=45)
ax[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# Histograms for engagement and purchasing columns
engagement_columns = ['Recency', 'Complain']
purchasing_columns = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
plt.figure(figsize=(20, 10))
for i, col in enumerate(engagement_columns + purchasing_columns, 1):
    plt.subplot(2, len(engagement_columns + purchasing_columns), i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Interactive 3D scatter plot
fig = px.scatter_3d(df, x='Income', y='TotalSpending', z='CustomerTenure', color='Income')
fig.update_layout(title='Interactive 3D Scatter Plot of Income, Total Spending, and Customer Tenure')
fig.show()

# Assuming columns
style.use('fivethirtyeight')

# Ensure that 'Year' is sorted if it's not in chronological order
df = df.sort_values('Year')

fig, ax = plt.subplots()

# Make sure for plotting purposes
x = df['Year'].astype(str)
y = df['TotalSpending']

line, = ax.plot_date(x, y, '-', alpha=0.0)  # Start with an invisible line

def animate(i):
    line.set_alpha(1.0)  # Make line visible
    line.set_data(x[:i], y[:i])  # Set data for animation
    return line,

# Use frames instead of the length of x to control the animation range
ani = animation.FuncAnimation(fig, animate, frames=len(x), interval=100, blit=True)

plt.title('Animated Line Plot of Total Spending Over Years')
plt.show()
# Event Handling
fig, ax = plt.subplots()

def onclick(event):
    print('Clicked at:', event.xdata, event.ydata)

fig.canvas.mpl_connect('button_press_event', onclick)
ax.plot(df['Income'], df['TotalSpending'], 'o')
plt.title('Click on the Plot to Get Coordinates')
plt.xlabel('Income')
plt.ylabel('Total Spending')
plt.show()
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Sample data for 3D bar plot
categories = np.arange(len(df['MntWines']))
zs = [10, 20, 30]  # Replace with real category indices or names
for z in zs:
    xs = categories
    ys = df['MntWines']  # Replace with actual data for each z (category)
    ax.bar(xs, ys, zs=z, zdir='y', alpha=0.8)

ax.set_xlabel('Product Category')
ax.set_ylabel('Customer Segment')
ax.set_zlabel('Spending')

plt.show()

fig, ax = plt.subplots()

def onclick(event):
    ix, iy = event.xdata, event.ydata
    print(f'x = {ix}, y = {iy}')
    ax.plot(ix, iy, 'xr')
    plt.draw()

fig.canvas.mpl_connect('button_press_event', onclick)

ax.plot(df['Income'], df['TotalSpending'], 'o')
plt.title('Click on the Plot to Display Values')
plt.xlabel('Income')
plt.ylabel('Total Spending')
plt.show()

fig, ax = plt.subplots()

x = df['Year']
y = df['CustomerBase']  
points, = ax.plot([], [], 'o')

def init():
    points.set_data([], [])
    return points,

def animate(i):
    points.set_data(x[:i], y[:i])
    return points,

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x), interval=100, blit=True)

plt.title('Evolution of Customer Base Over Time')
plt.show()
# Calculate age from Year_Birth
df['Age'] = df['Year'].max() - df['Year_Birth']

# 3D Scatter Plot: Age, Income, and Total Spending
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['Age'], df['Income'], df['TotalSpending'], c=df['Income'], cmap='viridis')
ax.set_xlabel('Age')
ax.set_ylabel('Income')
ax.set_zlabel('TotalSpending')
plt.title('3D Scatter Plot: Age, Income, and Total Spending')
plt.show()
# Assuming 'NumWebVisitsMonth' column exists
# Create a new DataFrame for animation
animation_data = df[['Year', 'NumWebVisitsMonth', 'TotalSpending', 'CustomerTenure']].groupby('Year').mean().reset_index()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def animate(i):
    ax.clear()
    ax.bar(animation_data['NumWebVisitsMonth'][:i], animation_data['TotalSpending'][:i], zs=animation_data['CustomerTenure'][:i], zdir='y', alpha=0.8)
    ax.set_xlabel('Monthly Web Visits')
    ax.set_ylabel('Customer Tenure')
    ax.set_zlabel('Total Spending')

ani = animation.FuncAnimation(fig, animate, frames=len(animation_data), interval=100, repeat=False)
plt.show()
# Interactive 3D Scatter Plot
fig = px.scatter_3d(df, x='Income', y='NumStorePurchases', z='NumWebPurchases', color='Income')
fig.update_layout(title='Interactive 3D Scatter Plot of Income, Store Purchases, and Web Purchases')
fig.show()
