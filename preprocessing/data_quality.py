import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from common.config import INPUT_CSV

np.random.seed(42)  # Seed for reproducibility

df = pd.read_csv(INPUT_CSV)

# Missing values
missing = df.isnull().sum()
print("Missing values per column:\n", missing[missing > 0])

# Duplicates check (based on 'id' as primary key)
duplicates = df.duplicated(subset=['id']).sum()
print("Duplicate IDs:", duplicates)

# Data types check
print("\nData types:\n", df.dtypes)

# Unique values for categorical columns
print("\nUnique room_types:", df['room_type'].unique())
print("Unique neighbourhood_groups:", df['neighbourhood_group'].unique())

# Basic stats for numerical columns
print("\nNumerical stats:\n", df.describe())

# Special check for reviews: Rows with number_of_reviews == 0 should have NaN in last_review and reviews_per_month
zero_reviews = df[df['number_of_reviews'] == 0]
print("\nRows with 0 reviews:", zero_reviews.shape[0])

# Plot missing values heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap (Before Cleaning)')
plt.show()

# Plot distribution of price (target variable)
plt.figure(figsize=(8, 4))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Price Distribution')
plt.show()

# Additional plot for availability (for forecasting task)
plt.figure(figsize=(8, 4))
sns.histplot(df['availability_365'], bins=50, kde=True)
plt.title('Availability Distribution')
plt.show()

# Additional plot for reviews_per_month (for demand prediction)
plt.figure(figsize=(8, 4))
sns.histplot(df['reviews_per_month'].dropna(), bins=50, kde=True)
plt.title('Reviews Per Month Distribution')
plt.show()
