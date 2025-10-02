import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from common.config import INPUT_CSV, CLEANING_OUTPUT_CSV

np.random.seed(42)  # Seed for reproducibility

df = pd.read_csv(INPUT_CSV)

# Handle missing values, remove duplicates, correct data types.
# For reviews_per_month: Set to 0 when number_of_reviews == 0 (meaningful, as no reviews mean 0 per month).
# For last_review: Convert to days_since_last_review (using Oct 01, 2025 - updated date to match query) and impute missing with max+1.
# License is often missing – drop if >80% missing.
# Remove duplicates based on 'id'.
# Handle outliers: Cap price at 99th percentile, minimum_nights at 95th percentile (data-driven).
# Don't drop 0 availability; add 'is_inactive' flag.
# Impute remaining numerical NaNs with median.
# Output: Show before/after shape, missing values, plots.

# UPDATE: Added capping for number_of_reviews_ltm (outliers skew clustering and classification).
# UPDATE: Added 'recent_reviews_ratio' = number_of_reviews_ltm / number_of_reviews (for availability forecasting and demand prediction; captures "recent activity" without dropping zeros).
# WHY: Helps in forecasting high/low demand; ratio=0 means no recent reviews (low demand). For clustering, aids in segments like "recently popular". Doesn't affect regression.

# Remove duplicates based on 'id'
df = df.drop_duplicates(subset=['id'])
print("Shape after removing duplicates:", df.shape)

# Handle missing reviews_per_month: Set to 0 where number_of_reviews == 0 (otherwise NaN)
df.loc[df['number_of_reviews'] == 0, 'reviews_per_month'] = 0

# Handle other missing: Drop rows with missing price (target)
df.dropna(subset=['price'], inplace=True)

# For license: Drop if >80% missing
if df['license'].isnull().sum() / len(df) > 0.8:
    df.drop('license', axis=1, inplace=True)
    print("Dropped 'license' column due to high missing values.")

# Convert data types: id/host_id to string, last_review to datetime
df['id'] = df['id'].astype(str)
df['host_id'] = df['host_id'].astype(str)
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

# Handle last_review: Convert to days_since_last_review (using Oct 01, 2025 as current date)
current_date = pd.to_datetime('2025-10-01')  # Updated to match query date
df['days_since_last_review'] = (current_date - df['last_review']).dt.days
max_days = df['days_since_last_review'].max() + 1  # For no reviews
df['days_since_last_review'].fillna(max_days, inplace=True)
print("Days since last review stats:", df['days_since_last_review'].describe())

# Cap days_since_last_review at 95th percentile
days_95 = df['days_since_last_review'].quantile(0.95)
df['days_since_last_review'] = df['days_since_last_review'].clip(upper=days_95)

# New feature: recent_reviews_ratio (for forecasting and demand)
df['recent_reviews_ratio'] = df['number_of_reviews_ltm'] / \
    df['number_of_reviews'].replace(0, 1)  # Avoid divide by zero
df['recent_reviews_ratio'].fillna(0, inplace=True)  # If no reviews, ratio=0

# Create is_inactive flag
df['is_inactive'] = (df['availability_365'] == 0).astype(int)

# Outlier handling: Cap price at 95th percentile
price_95 = df['price'].quantile(0.95)
df['price'] = df['price'].clip(upper=price_95)

# Cap minimum_nights at 95th percentile
min_nights_95 = df['minimum_nights'].quantile(0.95)
df['minimum_nights_cleaned'] = df['minimum_nights'].clip(upper=min_nights_95)

# UPDATE: Cap number_of_reviews_ltm at 95th percentile (for clustering stability)
reviews_ltm_95 = df['number_of_reviews_ltm'].quantile(0.95)
df['number_of_reviews_ltm_cleaned'] = df['number_of_reviews_ltm'].clip(
    upper=reviews_ltm_95)

# Impute remaining numerical NaNs with median
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Handle rare categories in neighbourhood_group
threshold = 100
value_counts = df['neighbourhood_group'].value_counts()
rare_neigh = value_counts[value_counts < threshold].index
df['neighbourhood_group_cleaned'] = df['neighbourhood_group'].replace(
    rare_neigh, 'Other')

# Merge rare room types
df['room_type_cleaned'] = df['room_type'].replace(
    {'Shared room': 'Other', 'Hotel room': 'Other'})

# Missing after cleaning
print("Missing after cleaning:\n", df.isnull().sum())

# Plot missing heatmap after
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap (After Cleaning)')
plt.show()

# Plot price distribution after
plt.figure(figsize=(8, 4))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Price Distribution After Cleaning')
plt.show()

# Additional plot for new feature
plt.figure(figsize=(8, 4))
sns.histplot(df['recent_reviews_ratio'], bins=50, kde=True)
plt.title('Recent Reviews Ratio Distribution')
plt.show()

df.to_csv(CLEANING_OUTPUT_CSV, index=False)
