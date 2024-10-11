import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('data.csv')

# Data Exploration
# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Handle Missing Values
for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        data[column].fillna(data[column].mean(), inplace=True)

# Feature Engineering - Create Lag Features (last 3 days sales, etc.)
data['lag_1'] = data.groupby('product_id')['sales'].shift(1)
data['lag_2'] = data.groupby('product_id')['sales'].shift(2)
data['lag_3'] = data.groupby('product_id')['sales'].shift(3)

# Handling Date Features
data['day'] = pd.to_datetime(data['date']).dt.day
data['month'] = pd.to_datetime(data['date']).dt.month
data['year'] = pd.to_datetime(data['date']).dt.year
data['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek

# Train-Test Split
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save processed datasets
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print("Data preprocessing complete!")
