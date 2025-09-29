import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# Explanation: Streamlit app for Airbnb price prediction.
# Load all optimized models and report_df for R2 scores.
# Input fields match X_train features from data_transform.py and feature_importance.py.
# Apply MinMaxScaler to numerical features to match preprocessing.

st.title('Airbnb Price Predictor NZ')

# Load data and report_df
try:
    X_train = joblib.load('data/processed/X_train.pkl')
    report_df = joblib.load('data/processed/report_df.pkl')
except FileNotFoundError as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Print X_train columns for debugging
st.write("X_train columns:", X_train.columns.tolist())

# Load models (handle missing files)
models = {
    'Linear Regression': joblib.load('data/models/linear_regression_optimized.pkl') if 'linear_regression_optimized.pkl' in os.listdir('data/models') else None,
    'Decision Tree': joblib.load('data/models/decision_tree_optimized.pkl') if 'decision_tree_optimized.pkl' in os.listdir('data/models') else None,
    'Random Forest': joblib.load('data/models/random_forest_optimized.pkl') if 'random_forest_optimized.pkl' in os.listdir('data/models') else None,
    'Gradient Boosting': joblib.load('data/models/gradient_boosting_optimized.pkl') if 'gradient_boosting_optimized.pkl' in os.listdir('data/models') else None,
    'XGBoost': joblib.load('data/models/xgboost_optimized.pkl') if 'xgboost_optimized.pkl' in os.listdir('data/models') else None,
    'Stacking': joblib.load('data/models/stacking_ensemble.pkl') if 'stacking_ensemble.pkl' in os.listdir('data/models') else None,
    'Bagging': joblib.load('data/models/bagging_ensemble.pkl') if 'bagging_ensemble.pkl' in os.listdir('data/models') else None,
    'AdaBoost': joblib.load('data/models/adaboost_ensemble.pkl') if 'adaboost_ensemble.pkl' in os.listdir('data/models') else None
}
model_names = [name for name, model in models.items() if model is not None]

# Select model
selected_model = st.selectbox('Select Model', model_names, index=model_names.index('Random Forest') if 'Random Forest' in model_names else 0)

# Get and format R2 score
r2_value = report_df[report_df['Model'] == selected_model]['R2'].values
r2_display = f"{r2_value[0]:.2f}" if len(r2_value) > 0 and pd.notnull(r2_value[0]) else 'Not Available'
st.write(f"Selected Model: {selected_model} (R2: {r2_display})")

# Input fields for key features
st.header("Enter Listing Details")
latitude = st.number_input('Latitude', -48.0, -34.0, -36.8485)
longitude = st.number_input('Longitude', 166.0, 179.0, 174.7633)
availability_365 = st.slider('Availability (days)', 0, 365, 180)
minimum_nights = st.slider('Minimum Nights', 1, 90, 1)
number_of_reviews = st.slider('Number of Reviews', 0, 1000, 50)
reviews_per_month = st.number_input('Reviews per Month', 0.0, 50.0, 2.0)
calculated_host_listings_count = st.slider('Host Listings Count', 1, 100, 1)
number_of_reviews_ltm = st.slider('Reviews Last 12 Months', 0, 500, 20)
days_since_last_review = st.slider('Days Since Last Review', 0, 1000, 100)
is_inactive = st.checkbox('Is Inactive (no reviews in 12 months)')
room_type = st.selectbox('Room Type', ['Entire home/apt', 'Private room', 'Other'])
neighbourhood_group = st.selectbox('Neighbourhood Group', ['Auckland', 'Wellington', 'Christchurch', 'Other'])
neighbourhood = st.text_input('Neighbourhood', 'Unknown')  # Placeholder, uses mean encoding

# Prepare input data (match X_train columns)
input_data = {col: 0 for col in X_train.columns}

# Update numerical features
input_data.update({
    'latitude': latitude,
    'longitude': longitude,
    'availability_365': availability_365,
    'minimum_nights_cleaned': minimum_nights,
    'number_of_reviews': number_of_reviews,
    'reviews_per_month': reviews_per_month,
    'calculated_host_listings_count': calculated_host_listings_count,
    'number_of_reviews_ltm': number_of_reviews_ltm,
    'days_since_last_review': days_since_last_review,
    'is_inactive': int(is_inactive),
    'neighbourhood_encoded': X_train['neighbourhood_encoded'].mean() if 'neighbourhood_encoded' in X_train.columns else 0
})

# Update one-hot encoded columns dynamically
for col in X_train.columns:
    if col.startswith('room_type_cleaned_'):
        input_data[col] = 1 if col == f'room_type_cleaned_{room_type}' else 0
    if col.startswith('neighbourhood_group_cleaned_'):
        input_data[col] = 1 if col == f'neighbourhood_group_cleaned_{neighbourhood_group}' else 0

# Update polynomial features
if 'availability_365_minimum_nights_cleaned' in X_train.columns:
    input_data['availability_365_minimum_nights_cleaned'] = availability_365 * minimum_nights
if 'availability_365_calculated_host_listings_count' in X_train.columns:
    input_data['availability_365_calculated_host_listings_count'] = availability_365 * calculated_host_listings_count
if 'minimum_nights_cleaned_calculated_host_listings_count' in X_train.columns:
    input_data['minimum_nights_cleaned_calculated_host_listings_count'] = minimum_nights * calculated_host_listings_count

# Update PCA features (if present)
for col in ['pca1', 'pca2', 'pca3']:
    if col in X_train.columns:
        input_data[col] = X_train[col].mean() if col in X_train else 0

# Create input DataFrame
input_df = pd.DataFrame([input_data])

# Apply MinMaxScaler to numerical and polynomial features
numerical_cols = ['latitude', 'longitude', 'minimum_nights_cleaned', 'number_of_reviews', 'reviews_per_month',
                 'calculated_host_listings_count', 'availability_365', 'number_of_reviews_ltm',
                 'days_since_last_review', 'is_inactive', 'neighbourhood_encoded']
poly_cols = ['availability_365_minimum_nights_cleaned', 'availability_365_calculated_host_listings_count',
             'minimum_nights_cleaned_calculated_host_listings_count']
scaler_cols = [col for col in numerical_cols + poly_cols if col in X_train.columns]
if scaler_cols:
    scaler = MinMaxScaler()
    # Fit scaler on X_train to ensure same scaling
    scaler.fit(X_train[scaler_cols])
    input_df[scaler_cols] = scaler.transform(input_df[scaler_cols])

# Predict
if st.button('Predict Price'):
    try:
        pred = models[selected_model].predict(input_df)
        st.write(f'Predicted Price: ${np.expm1(pred[0]):.2f} NZD')
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Optional: Display feature importance (for tree-based models)
if selected_model in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    st.header("Feature Importance")
    model = models[selected_model]
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        st.bar_chart(imp.head(10))