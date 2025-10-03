from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

np.random.seed(42)

# Load enhanced preprocessed data
df = pd.read_csv('data/listings_discretized_enhanced.csv')

# Debug: Print columns to verify
print("Available columns:\n", df.columns.tolist())

# Select binned/categorical features for Apriori (use 'room_type' since 'room_type_cleaned' is dropped after one-hot)
# FIX: Switched to 'room_type' (original categorical) for interpretable rules
df_rules = df[['price_bin', 'reviews_per_month_bin',
               'demand_indicator', 'room_type', 'availability_bin']]

# One-hot encode for Apriori (converts categoricals to binary columns)
df_onehot = pd.get_dummies(df_rules)

# Apriori: Find frequent itemsets (min_support=0.1: at least 10% of listings)
freq_items = apriori(df_onehot, min_support=0.1, use_colnames=True)

# Generate rules (confidence >= 0.7: strong associations)
rules = association_rules(freq_items, metric='confidence', min_threshold=0.7)

# Sort by lift (how much better than random) and display top 10
if not rules.empty:
    print("Association Rules (Top 10 by Lift):\n",
          rules.sort_values('lift', ascending=False).head(10))
else:
    print("No rules found with min_support=0.1 and min_confidence=0.7. Lowering thresholds...")
    # Fallback: Lower thresholds for more rules
    freq_items_low = apriori(df_onehot, min_support=0.05, use_colnames=True)
    rules = association_rules(
        freq_items_low, metric='confidence', min_threshold=0.5)
    print("Fallback Rules (Top 10 by Lift):\n",
          rules.sort_values('lift', ascending=False).head(10))

# Example interpretation: Rules like {('price_bin', 'low'), ('room_type', 'Private room')} → {('demand_indicator', 'high')} indicate fair pricing.

# Save rules for frontend (Streamlit)
os.makedirs('data/processed', exist_ok=True)
rules.to_csv('data/processed/association_rules.csv', index=False)
print("Saved association rules to data/processed/association_rules.csv")

# Plot support vs confidence (support: frequency, confidence: strength; size by lift)
if not rules.empty:
    plt.figure(figsize=(8, 6))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.5,
                s=rules['lift']*100, c=rules['lift'], cmap='viridis')
    plt.colorbar(label='Lift')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Association Rules: Support vs Confidence (Size/Color = Lift)')
    plt.show()
else:
    print("No rules to plot.")
