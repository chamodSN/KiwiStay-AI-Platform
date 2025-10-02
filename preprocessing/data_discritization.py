import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from common.config import TRANSFORM_OUTPUT_CSV, DISCRETIZATION_OUTPUT_CSV

np.random.seed(42)  # Seed for reproducibility

df = pd.read_csv(TRANSFORM_OUTPUT_CSV)

# Code for Data Discretization
# Explanation: Bin continuous features (e.g., price into low/medium/high).
# For other ML: Bins for association rules.
# Output: Show binned columns.

# Bin price into 3 categories
df['price_bin'] = pd.cut(df['price'], bins=3, labels=['low', 'medium', 'high'])

# Bin availability_365 into quarters
df['availability_bin'] = pd.qcut(df['availability_365'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# UPDATE: Bin reviews_per_month for Apriori (low/medium/high reviews)
df['reviews_per_month_bin'] = pd.cut(df['reviews_per_month'], bins=3, labels=['low', 'medium', 'high'])

# UPDATE: Bin recent_reviews_ratio for demand forecasting
df['recent_reviews_ratio_bin'] = pd.cut(df['recent_reviews_ratio'], bins=3, labels=['low', 'medium', 'high'])

# UPDATE: New 'demand_indicator' bin (high/low based on reviews_per_month > median for association rules like "low price + high reviews → high demand")
median_reviews = df['reviews_per_month'].median()
df['demand_indicator'] = np.where(df['reviews_per_month'] > median_reviews, 'high', 'low')
# WHY: Directly supports Apriori for price fairness; binary for classification target variant.

print(df[['price', 'price_bin', 'availability_365', 'availability_bin', 'reviews_per_month', 'reviews_per_month_bin', 'demand_indicator']].head())

# Plots
fig, ax = plt.subplots(1, 3, figsize=(18, 4))
sns.countplot(x='price_bin', data=df, ax=ax[0]).set_title('Price Bins')
sns.countplot(x='reviews_per_month_bin', data=df, ax=ax[1]).set_title('Reviews Per Month Bins')
sns.countplot(x='demand_indicator', data=df, ax=ax[2]).set_title('Demand Indicator')
plt.show()

df.to_csv(DISCRETIZATION_OUTPUT_CSV, index=False)