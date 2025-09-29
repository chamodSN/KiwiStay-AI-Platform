# Code for Data Integration
# Explanation: Since there's only one dataset, integration might not apply. But if needed, we can assume merging with external data (e.g., economic indicators).
# Here, we skip or simulate by adding a dummy column (e.g., average price per neighbourhood from the data itself).
# For other ML tasks: Create 'high_price' for classification (price > median).
# Output: Show integrated DataFrame.
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from common.config import INPUT_CSV

df = pd.read_csv(INPUT_CSV)
# Simulate integration: Compute average price per neighbourhood_group and merge back.
avg_price_group = df.groupby('neighbourhood_group')['price'].mean(
).reset_index(name='avg_price_neighbourhood_group')
df = df.merge(avg_price_group, on='neighbourhood_group', how='left')

print("Shape after integration:", df.shape)
print(df.head())

# For classification: Add binary 'high_price' (1 if price > median)
median_price = df['price'].median()
df['high_price'] = (df['price'] > median_price).astype(int)

# Plot average price per group
plt.figure(figsize=(10, 6))
sns.barplot(x='neighbourhood_group',
            y='avg_price_neighbourhood_group', data=avg_price_group)
plt.xticks(rotation=90)
plt.title('Average Price per Neighbourhood Group (Integrated Feature)')
plt.show()
