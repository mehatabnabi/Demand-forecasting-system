import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
data = pd.read_csv('train_data.csv')

# Time Series Plot of Sales
plt.figure(figsize=(12, 6))
plt.plot(data['date'], data['sales'], label='Sales')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Correlation Heatmap
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True)
plt.title('Feature Correlation Heatmap')
plt.show()

# Sales by Product
sns.boxplot(x='product_id', y='sales', data=data)
plt.title('Sales Distribution by Product')
plt.show()
