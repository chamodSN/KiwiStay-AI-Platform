# Enhanced Association_rules/association_rules.py
# Changes:
# - Dynamically adjust min_support based on dataset size to ensure rule generation.
# - Explicitly validate binned columns and handle missing preprocessing.
# - Add detailed logging to diagnose why rules aren't generated.
# - Ensure required columns are present and saved with fallback if empty.
# - Improved error handling for debugging.

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import os

np.random.seed(42)  # For reproducibility

# Load discretized data
input_file = 'data/listings_discretized_enhanced.csv'
try:
    df = pd.read_csv(input_file)
    print(f"Loaded {input_file}, shape: {df.shape}, columns: {df.columns.tolist()}")
except Exception as e:
    print(f"Error loading {input_file}: {e}")
    exit(1)

# Select binned/categorical columns for association rules
rule_cols = ['price_bin', 'availability_bin_high_low', 'reviews_per_month_bin', 'recent_reviews_ratio_bin', 'demand_indicator']
missing_cols = [col for col in rule_cols if col not in df.columns]
if missing_cols:
    print(f"Error: Missing columns {missing_cols} in {input_file}. Ensure data_discritization.py and feature_importance.py are run successfully.")
    print("Available columns: ", df.columns.tolist())
    exit(1)

# Prepare data for Apriori (one-hot encode categorical columns)
df_rules = df[rule_cols].copy()
if df_rules.empty or df_rules.isnull().all().any():
    print(f"Error: No valid data in {rule_cols} after selection. Check preprocessing output.")
    exit(1)
df_rules = pd.get_dummies(df_rules, columns=rule_cols, dtype=bool)
print(f"Prepared data shape for Apriori: {df_rules.shape}")

# Dynamically set min_support based on dataset size
n_transactions = len(df_rules)
min_support = max(0.01, 100 / n_transactions)  # Minimum 0.01 or 100 transactions, whichever is higher
print(f"Setting min_support to {min_support} for {n_transactions} transactions")

# Run Apriori
try:
    frequent_itemsets = apriori(df_rules, min_support=min_support, use_colnames=True, low_memory=True)
    print(f"Frequent itemsets generated: {len(frequent_itemsets)} itemsets")
    if frequent_itemsets.empty:
        print("Warning: No frequent itemsets found. Lowering min_support to 0.005 for retry.")
        frequent_itemsets = apriori(df_rules, min_support=0.005, use_colnames=True, low_memory=True)
        print(f"Retry frequent itemsets: {len(frequent_itemsets)} itemsets")
except Exception as e:
    print(f"Error running Apriori: {e}")
    exit(1)

# Generate association rules
if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.8)  # Lowered min_threshold to 0.8
    print(f"Generated {len(rules)} association rules")
else:
    print("Error: No frequent itemsets to generate rules. Check data or adjust min_support.")
    exit(1)

# Filter and ensure required columns
required_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
if not all(col in rules.columns for col in required_cols):
    print(f"Error: Missing required columns in rules. Available columns: {rules.columns.tolist()}")
    exit(1)
rules = rules[required_cols].copy()

# Convert frozensets to strings for readability
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)) if isinstance(x, frozenset) else str(x))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)) if isinstance(x, frozenset) else str(x))

# Check if rules are empty and provide fallback
if rules.empty:
    print("Warning: No rules generated with current settings. Creating minimal rule set as fallback.")
    # Fallback: Generate rules with very low thresholds
    frequent_itemsets_fallback = apriori(df_rules, min_support=0.001, use_colnames=True, low_memory=True)
    rules_fallback = association_rules(frequent_itemsets_fallback, metric="lift", min_threshold=0.5)
    rules = rules_fallback[required_cols].copy()
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)) if isinstance(x, frozenset) else str(x))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)) if isinstance(x, frozenset) else str(x))
    print(f"Fallback generated {len(rules)} rules")

# Save rules
output_file = 'data/processed/association_rules.csv'
os.makedirs('data/processed', exist_ok=True)
rules.to_csv(output_file, index=False)
print(f"Saved association rules to {output_file}, shape: {rules.shape}")
print(f"Rules columns: {rules.columns.tolist()}")

# Debugging: Display sample rules
print("\nSample rules:")
print(rules.head())