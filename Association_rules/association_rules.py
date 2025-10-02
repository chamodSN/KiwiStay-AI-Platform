from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

df = pd.read_csv('data/listings_discretized_enhanced.csv')
# Use binned data
print(df.columns)
df_rules = df[['price_bin', 'reviews_per_month_bin', 'demand_indicator', 'room_type_cleaned', 'availability_bin']]

# One-hot for Apriori
df_onehot = pd.get_dummies(df_rules)

# Apriori
freq_items = apriori(df_onehot, min_support=0.1, use_colnames=True)
rules = association_rules(freq_items, metric='confidence', min_threshold=0.7)
print("Association Rules:\n", rules.sort_values('lift', ascending=False).head(10))  # E.g., low_price + high_reviews -> high_demand

# Plot support vs confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.title('Association Rules Scatter')
plt.show()
