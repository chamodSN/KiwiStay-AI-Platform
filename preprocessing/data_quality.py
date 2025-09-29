# Code for Data Quality Check
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from common.config import INPUT_CSV

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
print("Missing last_review in 0 reviews rows:", zero_reviews['last_review'].isnull().sum())
print("Missing reviews_per_month in 0 reviews rows:", zero_reviews['reviews_per_month'].isnull().sum())

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

# Seed for reproducibility
np.random.seed(42)

