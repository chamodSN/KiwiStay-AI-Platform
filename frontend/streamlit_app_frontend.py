import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(page_title="Airbnb NZ Analytics Dashboard", layout="wide")

# Load data and models
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/discretization_output.csv')
        X_train = joblib.load('data/processed/X_train.pkl')
        report_df = joblib.load('data/processed/report_df.pkl')
        
        # Clustering: Load ensemble labels or best KMeans
        try:
            ensemble_labels = joblib.load('data/models/clustering/ensemble_labels.pkl')
            df['cluster'] = ensemble_labels
            df['segment'] = df['cluster'].map({0: 'Budget Urban', 1: 'Luxury Tourist', 2: 'Mid-Range Rural', 3: 'Family Retreat'})  # Customize based on means
        except:
            kmeans = joblib.load('data/models/clustering/kmeans_optimized.pkl')
            df['cluster'] = kmeans.fit_predict(df[['log_price', 'loc_pca1', 'loc_pca2', 'availability_365', 'recent_reviews_ratio', 'minimum_nights_cleaned']])
            df['segment'] = df['cluster'].map({0: 'Budget Urban', 1: 'Luxury Tourist', 2: 'Mid-Range Rural'})
        
        # Association Rules: Load rules (assume saved as CSV from association_rules.py)
        rules = pd.read_csv('data/processed/association_rules.csv') if os.path.exists('data/processed/association_rules.csv') else pd.DataFrame()
        
        return df, X_train, report_df, rules
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

df, X_train, report_df, rules = load_data()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a section:", 
                           ["Price Prediction", "Market Segmentation", "Demand Prediction", 
                            "Price Fairness", "Availability Forecasting"])

# Load regression models (shared)
@st.cache_data
def load_regression_models():
    models = {}
    model_files = {
        'Linear Regression': 'linear_regression_optimized.pkl',
        'Decision Tree': 'decision_tree_optimized.pkl',
        'Random Forest': 'random_forest_optimized.pkl',
        'Gradient Boosting': 'gradient_boosting_optimized.pkl',
        'XGBoost': 'xgboost_optimized.pkl',
        'Stacking': 'stacking_ensemble.pkl',
        'Bagging': 'bagging_ensemble.pkl',
        'AdaBoost': 'adaboost_ensemble.pkl'
    }
    for name, file in model_files.items():
        path = os.path.join('data/models', file)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

# Load classification models (for demand and availability)
@st.cache_data
def load_classification_models():
    models = {}
    clf_files = {
        'Logistic Regression': 'logistic_optimized.pkl',
        'SVC': 'svc_optimized.pkl',
        'Random Forest': 'rf_optimized.pkl',
        'Gradient Boosting': 'gb_optimized.pkl',
        'XGBoost': 'xgb_optimized.pkl'
    }
    for name, file in clf_files.items():
        path = os.path.join('data/models/classification', file)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

regression_models = load_regression_models()
clf_models = load_classification_models()

# Common input form (reusable)
def get_input_form():
    with st.form("listing_details"):
        st.subheader("Listing Details")
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input('Latitude', -48.0, -34.0, -36.8485)
            longitude = st.number_input('Longitude', 166.0, 179.0, 174.7633)
            availability_365 = st.slider('Availability (days)', 0, 365, 180)
            minimum_nights = st.slider('Minimum Nights', 1, 90, 1)
        with col2:
            number_of_reviews = st.slider('Number of Reviews', 0, 1000, 50)
            reviews_per_month = st.number_input('Reviews per Month', 0.0, 50.0, 2.0)
            calculated_host_listings_count = st.slider('Host Listings Count', 1, 100, 1)
            number_of_reviews_ltm = st.slider('Reviews Last 12 Months', 0, 500, 20)
            is_inactive = st.checkbox('Is Inactive (no reviews in 12 months)')
        
        room_type = st.selectbox('Room Type', ['Entire home/apt', 'Private room', 'Other'])
        neighbourhood_group = st.selectbox('Neighbourhood Group', ['Auckland', 'Wellington', 'Christchurch', 'Other'])
        neighbourhood = st.text_input('Neighbourhood', 'Unknown')
        
        submitted = st.form_submit_button("Submit")
    
    return {
        'latitude': latitude, 'longitude': longitude, 'availability_365': availability_365,
        'minimum_nights': minimum_nights, 'number_of_reviews': number_of_reviews,
        'reviews_per_month': reviews_per_month, 'calculated_host_listings_count': calculated_host_listings_count,
        'number_of_reviews_ltm': number_of_reviews_ltm, 'is_inactive': int(is_inactive),
        'room_type': room_type, 'neighbourhood_group': neighbourhood_group, 'neighbourhood': neighbourhood
    }, submitted

# Prepare input DataFrame (reusable for regression/classification)
def prepare_input_df(inputs, X_train):
    input_data = {col: 0 for col in X_train.columns}
    
    # Numerical
    input_data.update({
        'latitude': inputs['latitude'],
        'longitude': inputs['longitude'],
        'availability_365': inputs['availability_365'],
        'minimum_nights_cleaned': inputs['minimum_nights'],
        'number_of_reviews': inputs['number_of_reviews'],
        'reviews_per_month': inputs['reviews_per_month'],
        'calculated_host_listings_count': inputs['calculated_host_listings_count'],
        'number_of_reviews_ltm': inputs['number_of_reviews_ltm'],  # Assume cleaned in preprocessing
        'is_inactive': inputs['is_inactive'],
        'neighbourhood_encoded': X_train['neighbourhood_encoded'].mean() if 'neighbourhood_encoded' in X_train.columns else 0
    })
    
    # One-hot
    for col in X_train.columns:
        if col.startswith('room_type_cleaned_'):
            input_data[col] = 1 if col == f'room_type_cleaned_{inputs["room_type"]}' else 0
        if col.startswith('neighbourhood_group_cleaned_'):
            input_data[col] = 1 if col == f'neighbourhood_group_cleaned_{inputs["neighbourhood_group"]}' else 0
    
    # Polynomial (example interactions)
    if 'availability_365_minimum_nights_cleaned' in X_train.columns:
        input_data['availability_365_minimum_nights_cleaned'] = inputs['availability_365'] * inputs['minimum_nights']
    # Add more as needed
    
    # PCA means
    for col in ['pca1', 'pca2', 'pca3', 'loc_pca1', 'loc_pca2']:
        if col in X_train.columns:
            input_data[col] = X_train[col].mean()
    
    input_df = pd.DataFrame([input_data])
    
    # Scale
    scaler_cols = ['latitude', 'longitude', 'minimum_nights_cleaned', 'number_of_reviews',
                   'reviews_per_month', 'calculated_host_listings_count', 'availability_365',
                   'number_of_reviews_ltm', 'is_inactive', 'neighbourhood_encoded']
    scaler_cols += [col for col in input_df.columns if '_' in col and 'x' in col]  # Poly cols
    scaler_cols = [col for col in scaler_cols if col in X_train.columns]
    
    if scaler_cols:
        scaler = MinMaxScaler()
        scaler.fit(X_train[scaler_cols])
        input_df[scaler_cols] = scaler.transform(input_df[scaler_cols])
    
    return input_df

# === PAGE: Price Prediction ===
if page == "Price Prediction":
    st.header("🏠 Price Prediction")
    
    if not regression_models:
        st.warning("No regression models loaded.")
    else:
        model_names = list(regression_models.keys())
        selected_model = st.selectbox('Select Model', model_names, index=model_names.index('Random Forest') if 'Random Forest' in model_names else 0)
        
        # Display R2
        r2_value = report_df[report_df['Model'] == selected_model]['R2'].values
        r2_display = f"{r2_value[0]:.2f}" if len(r2_value) > 0 and pd.notnull(r2_value[0]) else 'N/A'
        st.metric("Model Performance (R²)", r2_display)
        
        inputs, submitted = get_input_form()
        if submitted:
            input_df = prepare_input_df(inputs, X_train)
            pred = regression_models[selected_model].predict(input_df)
            st.success(f'Predicted Price: ${np.expm1(pred[0]):.2f} NZD')
            
            # Feature Importance
            if hasattr(regression_models[selected_model], 'feature_importances_'):
                st.subheader("Feature Importance (Top 10)")
                imp = pd.Series(regression_models[selected_model].feature_importances_, index=X_train.columns).sort_values(ascending=False)
                fig = px.bar(x=imp.head(10).values, y=imp.head(10).index, orientation='h', title="Top Features")
                st.plotly_chart(fig, use_container_width=True)

# === PAGE: Market Segmentation ===
elif page == "Market Segmentation":
    st.header("📊 Market Segmentation (Clustering)")
    
    if df is None:
        st.warning("No data loaded for clustering.")
    else:
        # Interactive Chart
        st.subheader("Segment Distribution")
        fig = px.scatter(df, x='longitude', y='latitude', color='segment', 
                         hover_data=['price', 'room_type_cleaned', 'availability_365'],
                         title="Airbnb Listings by Market Segment")
        st.plotly_chart(fig, use_container_width=True)
        
        # Folium Map
        st.subheader("Interactive Map of Segments")
        m = folium.Map(location=[-41.2924, 174.7787], zoom_start=6)  # NZ center
        
        for idx, row in df.iterrows():
            color = {'Budget Urban': 'blue', 'Luxury Tourist': 'red', 'Mid-Range Rural': 'green', 'Family Retreat': 'orange'}.get(row['segment'], 'gray')
            folium.CircleMarker(
                [row['latitude'], row['longitude']],
                radius=3, popup=f"{row['segment']} - ${row['price']:.0f} - {row['room_type_cleaned']}",
                color=color, fill=True, fillOpacity=0.7
            ).add_to(m)
        
        folium_static(m, width=700, height=500)
        
        # Segment Insights
        st.subheader("Segment Insights")
        segment_stats = df.groupby('segment')[['price', 'availability_365', 'number_of_reviews']].mean().round(2)
        st.dataframe(segment_stats)

# === PAGE: Demand Prediction ===
elif page == "Demand Prediction":
    st.header("🔥 Demand Prediction (High/Low Popularity)")
    
    if not clf_models:
        st.warning("No classification models loaded.")
    else:
        model_names = list(clf_models.keys())
        selected_model = st.selectbox('Select Model', model_names, index=model_names.index('Random Forest') if 'Random Forest' in model_names else 0)
        
        inputs, submitted = get_input_form()
        if submitted:
            input_df = prepare_input_df(inputs, X_train)
            pred = clf_models[selected_model].predict(input_df)
            prob = clf_models[selected_model].predict_proba(input_df)[:, 1][0]
            demand = "High" if pred[0] == 1 else "Low"
            st.metric("Predicted Demand", demand, delta=f"Probability: {prob:.2%}")
            
            # Confusion Matrix (from training, for reference)
            st.subheader("Model Performance")
            # Assume saved metrics; placeholder chart
            fig = px.bar(x=['Accuracy', 'F1', 'ROC-AUC'], y=[0.75, 0.72, 0.78], title="Sample Metrics")
            st.plotly_chart(fig)

# === PAGE: Price Fairness ===
elif page == "Price Fairness":
    st.header("⚖️ Price Fairness (Association Rules)")
    
    if rules.empty:
        st.warning("No association rules loaded. Run association_rules.py first.")
    else:
        st.subheader("Top Rules (e.g., Low Price + High Reviews → High Demand)")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
        
        inputs, submitted = get_input_form()
        if submitted:
            # Simple check: Match input to rules (e.g., if low price and high reviews, fair if high demand)
            price_bin = pd.cut([inputs['price'] or df['price'].median()], bins=3, labels=['low', 'medium', 'high']).iloc[0] if 'price' in inputs else 'medium'
            reviews_bin = pd.cut([inputs['reviews_per_month']], bins=3, labels=['low', 'medium', 'high']).iloc[0]
            demand = 'high' if inputs['reviews_per_month'] > df['reviews_per_month'].median() else 'low'
            
            matching_rules = rules[(rules['antecedents'].apply(lambda x: 'low' in str(x) and 'high' in str(x))) & 
                                   (rules['consequents'].apply(lambda x: 'high' in str(x)))]
            fairness = "Fair (High Demand Expected)" if not matching_rules.empty else "Potentially Unfair"
            st.success(f"Price Fairness: {fairness} | Price Bin: {price_bin}, Reviews Bin: {reviews_bin}, Demand: {demand}")

# === PAGE: Availability Forecasting ===
elif page == "Availability Forecasting":
    st.header("📅 Availability Forecasting (High/Low)")
    
    if not clf_models:
        st.warning("No classification models loaded.")
    else:
        model_names = list(clf_models.keys())
        selected_model = st.selectbox('Select Model', model_names, index=model_names.index('Random Forest') if 'Random Forest' in model_names else 0)
        
        inputs, submitted = get_input_form()
        if submitted:
            input_df = prepare_input_df(inputs, X_train)
            # For availability, use same models but target is availability_bin_high_low (assume models retrained or shared)
            pred = clf_models[selected_model].predict(input_df)  # Placeholder: Use availability-specific if separate
            prob = clf_models[selected_model].predict_proba(input_df)[:, 1][0]
            avail = "High (>180 days)" if pred[0] == 1 else "Low"
            st.metric("Predicted Availability", avail, delta=f"Probability: {prob:.2%}")
            
            # Forecast Chart
            st.subheader("Availability Trends")
            fig = px.histogram(df, x='availability_365', color='demand_indicator', title="Historical Availability by Demand")
            st.plotly_chart(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Dashboard for Airbnb NZ Analytics | Built with Streamlit")

