# Code for Data Cleaning
# Explanation: Handle missing values, remove duplicates, correct data types.
# For reviews_per_month: Set to 0 when number_of_reviews == 0 (meaningful, as no reviews mean 0 per month).
# For last_review: Set to a placeholder like 'No Reviews' or drop if not useful for regression (we'll drop it later if irrelevant).
# License is often missing – drop if >90% missing.
# Remove duplicates based on 'id'.
# Handle outliers: Cap price at 99th percentile if needed.
# Output: Show before/after shape, missing values.
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from common.config import INPUT_CSV

df = pd.read_csv(INPUT_CSV)
# Remove duplicates based on 'id'
df = df.drop_duplicates(subset=['id'])
print("Shape after removing duplicates:", df.shape)

# Handle missing reviews_per_month: Set to 0 where number_of_reviews == 0
df.loc[df['number_of_reviews'] == 0, 'reviews_per_month'] = 0

# Handle last_review: Set to 'No Reviews' for categorical handling (or drop later)
df['last_review'].fillna('No Reviews', inplace=True)

# Handle other missing: Drop rows with missing price (target), or impute means for numerical.
df.dropna(subset=['price'], inplace=True)  # Critical target

# For license: Mostly missing, drop column if >80% missing
if df['license'].isnull().sum() / len(df) > 0.8:
    df.drop('license', axis=1, inplace=True)
    print("Dropped 'license' column due to high missing values.")

# Convert data types: id/host_id to string, dates to datetime if needed.
df['id'] = df['id'].astype(str)
df['host_id'] = df['host_id'].astype(str)
# If needed, but since mixed, keep as string for now.
df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

# Outlier handling: Cap price at 99th percentile
price_99 = df['price'].quantile(0.99)
df['price'] = df['price'].clip(upper=price_99)

# Missing after cleaning
print("Missing after cleaning:\n", df.isnull().sum())

# Plot missing heatmap after
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap (After Cleaning)')
plt.show()

# Plot price distribution after outlier capping
plt.figure(figsize=(8, 4))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Price Distribution After Cleaning')
plt.show()
