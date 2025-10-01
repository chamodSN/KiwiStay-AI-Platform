import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common.config import TRANSFORM_OUTPUT_CSV, DISCRETIZATION_OUTPUT_CSV

df = pd.read_csv(TRANSFORM_OUTPUT_CSV)

# Code for Data Discretization
# Explanation: Bin continuous features (e.g., price into low/medium/high).
# For other ML: Bins for association rules.
# Output: Show binned columns.

# Bin price into 3 categories
df['price_bin'] = pd.cut(df['price'], bins=3, labels=['low', 'medium', 'high'])

# Bin availability_365 into quarters
df['availability_bin'] = pd.qcut(
    df['availability_365'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

print(df[['price', 'price_bin', 'availability_365', 'availability_bin']].head())

# Plot binned price count
plt.figure(figsize=(6, 4))
sns.countplot(x='price_bin', data=df)
plt.title('Price Bins Distribution')
plt.show()

df.to_csv(DISCRETIZATION_OUTPUT_CSV, index=False)
