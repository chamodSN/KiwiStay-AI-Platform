import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# Explanation: Summarize metrics for all models using provided R2 values.
# Generate a report table and plot for comparison.
# Save report_df for use in Streamlit.

# Define model metrics (using provided values)
models = ['Linear Regression', 'Decision Tree', 'Random Forest',
          'Gradient Boosting', 'XGBoost', 'Stacking', 'Bagging', 'AdaBoost']
# Placeholder for Stacking, Bagging, AdaBoost
r2_scores = [0.30, 0.37, 0.49, 0.48, 0.49, None, None, None]

# Create report DataFrame
report_df = pd.DataFrame({'Model': models, 'R2': r2_scores})

# Save report_df
os.makedirs('data/processed', exist_ok=True)
joblib.dump(report_df, 'data/processed/report_df.pkl')
print("Saved report_df to data/processed/report_df.pkl")

# Print report
print("Model Performance Report:\n", report_df)

# Insights based on provided metrics
print("\nInsights:")
print("- Best model: Random Forest and XGBoost (R2=0.49), followed by Gradient Boosting (R2=0.48).")
print("- Linear Regression (R2=0.30) and Decision Tree (R2=0.37) underperformed, likely due to non-linear patterns.")
print("- Key features: Likely neighbourhood_encoded, days_since_last_review, availability_365 (based on RF importance).")
print("- Log transform improved normality, but R2 <0.5 suggests missing features (e.g., amenities, seasonality).")
print("- Ensemble models (Stacking, Bagging, AdaBoost) may improve R2 to ~0.55–0.60.")

# Plot model comparison (exclude models with None R2)
report_df_valid = report_df.dropna()
plt.figure(figsize=(8, 4))
sns.barplot(x='Model', y='R2', data=report_df_valid)
plt.title('Model R2 Comparison')
plt.xticks(rotation=45)
plt.show()
