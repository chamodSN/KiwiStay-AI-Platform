import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Explanation: Streamlit app for Airbnb price prediction.
# Load all optimized models and allow user to select one.
# Prepare input data matching X_train features.

st.title('Airbnb Price Predictor NZ')

# Load data and models
X_train = joblib.load('data/processed/X_train.pkl')
models = {
    'Linear Regression': joblib.load('data/models/linear_regression_optimized.pkl'),
    'Decision Tree': joblib.load('data/models/decision_tree_optimized.pkl'),
    'Random Forest': joblib.load('data/models/random_forest_optimized.pkl'),
    'Gradient Boosting': joblib.load('data/models/gradient_boosting_optimized.pkl'),
    'XGBoost': joblib.load('data/models/xgboost_optimized.pkl'),
    'Stacking': joblib.load('data/models/stacking_ensemble.pkl') if 'stacking_ensemble.pkl' in os.listdir('data/models') else None,
    'Bagging': joblib.load('data/models/bagging_ensemble.pkl') if 'bagging_ensemble.pkl' in os.listdir('data/models') else None,
    'AdaBoost': joblib.load('data/models/adaboost_ensemble.pkl') if 'adaboost_ensemble.pkl' in os.listdir('data/models') else None
}
model_names = [name for name, model in models.items() if model is not None]

# Select model
selected_model = st.selectbox('Select Model', model_names, index=model_names.index('Random Forest') if 'Random Forest' in model_names else 0)
st.write(f"Selected Model: {selected_model} (R2: {report_df[report_df['Model'] == selected_model]['R2'].values[0]:.2f})")

# Input fields for key features
st.header("Enter Listing Details")
latitude = st.number_input('Latitude', -48.0, -34.0, -36.8485)
longitude = st.number_input('Longitude', 166.0, 179.0, 174.7633)
availability_365 = st.slider('Availability (days)', 0, 365, 180)
minimum_nights = st.slider('Minimum Nights', 1, 90, 1)
room_type = st.selectbox('Room Type', ['Entire home/apt', 'Private room', 'Other'])
neighbourhood_group = st.selectbox('Neighbourhood Group', ['Auckland', 'Wellington', 'Christchurch', 'Other'])
has_luxury = st.checkbox('Luxury (e.g., Premium, Deluxe)')
has_view = st.checkbox('View (e.g., Ocean, Mountain)')

# Prepare input data (match X_train columns)
input_data = {col: 0 for col in X_train.columns}
input_data.update({
    'latitude': latitude,
    'longitude': longitude,
    'availability_365': availability_365,
    'minimum_nights_cleaned': minimum_nights,
    'has_luxury': int(has_luxury),
    'has_view': int(has_view),
    f'room_type_cleaned_{room_type}': 1 if room_type in ['Entire home/apt', 'Private room'] else 0,
    f'room_type_cleaned_Other': 1 if room_type == 'Other' else 0,
    f'neighbourhood_group_cleaned_{neighbourhood_group}': 1,
    'neighbourhood_encoded': X_train['neighbourhood_encoded'].mean()  # Use mean as placeholder
})

# Ensure all X_train columns are present
input_df = pd.DataFrame([input_data])

# Predict
if st.button('Predict Price'):
    pred = models[selected_model].predict(input_df)
    st.write(f'Predicted Price: ${np.expm1(pred[0]):.2f} NZD')

# Optional: Display feature importance (for RF/XGBoost/GB)
if selected_model in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    st.header("Feature Importance")
    model = models[selected_model]
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        st.bar_chart(imp.head(10))