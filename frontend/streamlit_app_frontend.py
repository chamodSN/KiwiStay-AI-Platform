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

# Load data and models (FIX: Correct file path to match generated names like 'listings_discretized_enhanced.csv')


@st.cache_data
def load_data():
    df_path = 'data/listings_discretized_enhanced.csv' 
    rules_path = 'data/processed/association_rules.csv'

    try:
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
        else:
            df = pd.DataFrame()

        if os.path.exists('data/processed/X_train.pkl'):
            X_train = joblib.load('data/processed/X_train.pkl')
        else:
            X_train = pd.DataFrame()

        if os.path.exists('data/processed/report_df.pkl'):
            report_df = joblib.load('data/processed/report_df.pkl')
        else:
            report_df = pd.DataFrame({'Model': [], 'R2': []})

        if os.path.exists(rules_path):
            rules = pd.read_csv(rules_path)
            # FIX: Bust cache by resetting index
            rules = rules.reset_index(drop=True)
        else:
            rules = pd.DataFrame()

        # Clustering: Try to load or compute on-the-fly if df exists and not empty
        if not df.empty:
            try:
                ensemble_labels = joblib.load(
                    'data/models/clustering/ensemble_labels.pkl')
                # Ensure length match
                df['cluster'] = ensemble_labels[:len(df)]
                df['segment'] = df['cluster'].map(
                    {0: 'Budget Urban', 1: 'Luxury Tourist', 2: 'Mid-Range Rural', 3: 'Family Retreat'})
            except:
                try:
                    kmeans = joblib.load(
                        'data/models/clustering/kmeans_optimized.pkl')
                    cluster_features = df[['log_price', 'loc_pca1', 'loc_pca2', 'availability_365',
                                           'recent_reviews_ratio', 'minimum_nights_cleaned']].fillna(0)
                    df['cluster'] = kmeans.fit_predict(cluster_features)
                    df['segment'] = df['cluster'].map(
                        {0: 'Budget Urban', 1: 'Luxury Tourist', 2: 'Mid-Range Rural'})
                except:
                    df['segment'] = 'Unknown'  # Fallback
        else:
            df = pd.DataFrame()  # Empty fallback

        return df, X_train, report_df, rules
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


df, X_train, report_df, rules = load_data()

# Sidebar for navigation and setup guide
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a section:",
                            ["Price Prediction", "Market Segmentation", "Demand Prediction",
                             "Price Fairness", "Availability Forecasting"])

if st.sidebar.button("📋 Setup Guide"):
    st.sidebar.markdown("""
    ### Quick Setup (Run These Scripts):
    1. **Preprocessing**: `python data_quality.py` → `data_cleaning.py` → `data_reduction.py` → `data_transform.py` → `data_discretization.py` → `feature_importance.py`
    2. **Regression**: Run scripts in `regression/` (e.g., `python regression/XGBoost_regressor/basic_parameter.py`)
    3. **Classification**: Run scripts in `classification/` (e.g., `python classification/rf_basic.py`)
    4. **Clustering**: Run scripts in `clustering/` (e.g., `python clustering/kmeans_basic.py`)
    5. **Association Rules**: `python Association_rules/association_rules.py`
    
    Check `data/` for generated files. Refresh app after.
    """)

# FIX: Add this line for cache bust (forces reload after saving CSV)
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()
# Load regression models (shared) - Enhanced error handling


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

# Load classification models - Enhanced error handling


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

# Common input form (reusable) - FIX: Added 'price' input for Price Fairness assessment


def get_input_form():
    with st.form("listing_details"):
        st.subheader("Listing Details")
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input('Latitude', -48.0, -34.0, -36.8485)
            longitude = st.number_input('Longitude', 166.0, 179.0, 174.7633)
            availability_365 = st.slider('Availability (days)', 0, 365, 180)
            minimum_nights = st.slider('Minimum Nights', 1, 90, 1)
            # FIX: Added for fairness check
            price = st.number_input('Price (NZD)', 50.0, 2000.0, 150.0)
        with col2:
            number_of_reviews = st.slider('Number of Reviews', 0, 1000, 50)
            reviews_per_month = st.number_input(
                'Reviews per Month', 0.0, 50.0, 2.0)
            calculated_host_listings_count = st.slider(
                'Host Listings Count', 1, 100, 1)
            number_of_reviews_ltm = st.slider(
                'Reviews Last 12 Months', 0, 500, 20)
            is_inactive = st.checkbox('Is Inactive (no reviews in 12 months)')

        room_type = st.selectbox(
            'Room Type', ['Entire home/apt', 'Private room', 'Other'])
        neighbourhood_group = st.selectbox(
            'Neighbourhood Group', ['Auckland', 'Wellington', 'Christchurch', 'Other'])
        neighbourhood = st.text_input('Neighbourhood', 'Unknown')

        submitted = st.form_submit_button("Submit")

    return {
        'latitude': latitude, 'longitude': longitude, 'availability_365': availability_365,
        'minimum_nights': minimum_nights, 'price': price,  # FIX: Include price
        'number_of_reviews': number_of_reviews,
        'reviews_per_month': reviews_per_month, 'calculated_host_listings_count': calculated_host_listings_count,
        'number_of_reviews_ltm': number_of_reviews_ltm, 'is_inactive': int(is_inactive),
        'room_type': room_type, 'neighbourhood_group': neighbourhood_group, 'neighbourhood': neighbourhood
    }, submitted

# Prepare input DataFrame (reusable for regression/classification) - Handle empty X_train


def prepare_input_df(inputs, X_train):
    if X_train.empty:
        st.warning("X_train not loaded—using dummy data for prediction.")
        return pd.DataFrame({'dummy': [0]})

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
        'number_of_reviews_ltm': inputs['number_of_reviews_ltm'],
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
        input_data['availability_365_minimum_nights_cleaned'] = inputs['availability_365'] * \
            inputs['minimum_nights']

    # PCA means
    for col in ['pca1', 'pca2', 'pca3', 'loc_pca1', 'loc_pca2']:
        if col in X_train.columns:
            input_data[col] = X_train[col].mean()

    input_df = pd.DataFrame([input_data])

    # Scale
    scaler_cols = ['latitude', 'longitude', 'minimum_nights_cleaned', 'number_of_reviews',
                   'reviews_per_month', 'calculated_host_listings_count', 'availability_365',
                   'number_of_reviews_ltm', 'is_inactive', 'neighbourhood_encoded']
    scaler_cols += [col for col in input_df.columns if '_' in col and any(
        k in col for k in ['x', ' * '])]  # Poly cols
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
        st.warning(
            "No regression models loaded. Run regression scripts first (e.g., python regression/XGBoost_regressor/basic_parameter.py).")
    else:
        model_names = list(regression_models.keys())
        selected_model = st.selectbox('Select Model', model_names, index=model_names.index(
            'Random Forest') if 'Random Forest' in model_names else 0)

        # Display R2
        r2_value = report_df[report_df['Model'] == selected_model]['R2'].values
        r2_display = f"{r2_value[0]:.2f}" if len(
            r2_value) > 0 and pd.notnull(r2_value[0]) else 'N/A'
        st.metric("Model Performance (R²)", r2_display)

        inputs, submitted = get_input_form()
        if submitted:
            input_df = prepare_input_df(inputs, X_train)
            pred = regression_models[selected_model].predict(input_df)
            st.success(f'Predicted Price: ${np.expm1(pred[0]):.2f} NZD')

            # Feature Importance
            if hasattr(regression_models[selected_model], 'feature_importances_'):
                st.subheader("Feature Importance (Top 10)")
                imp = pd.Series(regression_models[selected_model].feature_importances_,
                                index=X_train.columns).sort_values(ascending=False)
                fig = px.bar(x=imp.head(10).values, y=imp.head(
                    10).index, orientation='h', title="Top Features")
                st.plotly_chart(fig, use_container_width=True)

# === PAGE: Market Segmentation ===
elif page == "Market Segmentation":
    st.header("📊 Market Segmentation (Clustering)")

    if df.empty:
        st.warning(
            "No data loaded for clustering. Run preprocessing and feature_importance.py first.")
    else:
        # Interactive Chart
        st.subheader("Segment Distribution")
        if 'segment' not in df.columns:
            st.info(
                "No clustering results loaded. Run clustering scripts (e.g., python clustering/kmeans_basic.py).")
        else:
            fig = px.scatter(df, x='longitude', y='latitude', color='segment',
                             hover_data=['price', 'room_type',
                                         'availability_365'],
                             title="Airbnb Listings by Market Segment")
            st.plotly_chart(fig, use_container_width=True)

            # Folium Map
            st.subheader("Interactive Map of Segments")
            m = folium.Map(location=[-41.2924, 174.7787],
                           zoom_start=6)  # NZ center

            color_map = {'Budget Urban': 'blue', 'Luxury Tourist': 'red',
                         'Mid-Range Rural': 'green', 'Family Retreat': 'orange', 'Unknown': 'gray'}
            for idx, row in df.iterrows():
                color = color_map.get(row.get('segment', 'Unknown'), 'gray')
                folium.CircleMarker(
                    [row['latitude'], row['longitude']],
                    radius=3, popup=f"{row.get('segment', 'Unknown')} - ${row['price']:.0f} - {row['room_type']}",
                    color=color, fill=True, fillOpacity=0.7
                ).add_to(m)

            folium_static(m, width=700, height=500)

            # Segment Insights
            st.subheader("Segment Insights")
            if 'segment' in df.columns:
                segment_stats = df.groupby(
                    'segment')[['price', 'availability_365', 'number_of_reviews']].mean().round(2)
                st.dataframe(segment_stats)
            else:
                st.info("Run clustering to generate segments.")

# === PAGE: Demand Prediction ===
elif page == "Demand Prediction":
    st.header("🔥 Demand Prediction (High/Low Popularity)")

    if not clf_models:
        st.warning(
            "No classification models loaded. Run classification scripts first (e.g., python classification/rf_basic.py).")
    elif X_train.empty:
        st.warning("X_train not loaded—run feature_importance.py first.")
    else:
        model_names = list(clf_models.keys())
        selected_model = st.selectbox('Select Model', model_names, index=model_names.index(
            'Random Forest') if 'Random Forest' in model_names else 0)

        inputs, submitted = get_input_form()
        if submitted:
            input_df = prepare_input_df(inputs, X_train)
            pred = clf_models[selected_model].predict(input_df)
            prob = clf_models[selected_model].predict_proba(input_df)[:, 1][0]
            demand = "High" if pred[0] == 1 else "Low"
            st.metric("Predicted Demand", demand,
                      delta=f"Probability: {prob:.2%}")

            # Model Performance (placeholder; add saved metrics later)
            st.subheader("Model Performance")
            fig = px.bar(x=['Accuracy', 'F1', 'ROC-AUC'], y=[0.75, 0.72,
                         0.78], title="Sample Metrics (Run model for real values)")
            st.plotly_chart(fig)

# === PAGE: Price Fairness ===
elif page == "Price Fairness":
    st.header("⚖️ Price Fairness (Association Rules)")

    # FIX: Check shape instead of empty for robustness
    if rules.shape[0] == 0:
        st.warning(
            "No association rules loaded. Run Association_rules/association_rules.py first.")
    else:
        st.subheader(
            "Top Rules (e.g., Low Price + High Reviews → High Demand)")
        display_cols = ['antecedents', 'consequents',
                        'support', 'confidence', 'lift']
        if all(col in rules.columns for col in display_cols):
            # FIX: Convert frozensets to readable strings for display
            rules_display = rules.copy()
            rules_display['antecedents'] = rules_display['antecedents'].apply(
                lambda x: ', '.join(list(x)))
            rules_display['consequents'] = rules_display['consequents'].apply(
                lambda x: ', '.join(list(x)))
            st.dataframe(rules_display[display_cols].head(10))
        else:
            st.info("Rules columns incomplete—re-run association_rules.py.")

        inputs, submitted = get_input_form()
        if submitted and not df.empty:
            # Simple check using input price
            median_price = df['price'].median()
            price_bin = pd.cut([inputs.get('price', median_price)], bins=3, labels=[
                               'low', 'medium', 'high']).iloc[0]
            reviews_bin = pd.cut([inputs['reviews_per_month']], bins=3, labels=[
                                 'low', 'medium', 'high']).iloc[0]
            expected_demand = 'high' if inputs['reviews_per_month'] > df['reviews_per_month'].median(
            ) else 'low'

            # FIX: Better matching (handle frozenset and string conversion)
            def has_low_antecedent(antecedents):
                return any('low' in str(item) for item in antecedents)

            def has_high_consequent(consequents):
                return any('high' in str(item) for item in consequents)

            matching_rules = rules[
                rules['antecedents'].apply(has_low_antecedent) &
                rules['consequents'].apply(has_high_consequent)
            ]
            fairness = "Fair (High Demand Expected)" if len(
                matching_rules) > 0 else "Potentially Unfair (Review Pricing)"
            st.success(
                f"Price Fairness Assessment: {fairness}\nPrice Bin: {price_bin} | Reviews Bin: {reviews_bin} | Expected Demand: {expected_demand}")
# === PAGE: Availability Forecasting ===
elif page == "Availability Forecasting":
    st.header("📅 Availability Forecasting (High/Low)")

    if not clf_models:
        st.warning(
            "No classification models loaded. Run classification scripts first (e.g., python classification/gb_basic.py).")
    elif X_train.empty:
        st.warning("X_train not loaded—run feature_importance.py first.")
    else:
        model_names = list(clf_models.keys())
        selected_model = st.selectbox('Select Model', model_names, index=model_names.index(
            'Random Forest') if 'Random Forest' in model_names else 0)

        inputs, submitted = get_input_form()
        if submitted:
            input_df = prepare_input_df(inputs, X_train)
            pred = clf_models[selected_model].predict(input_df)
            prob = clf_models[selected_model].predict_proba(input_df)[:, 1][0]
            avail = "High (>180 days)" if pred[0] == 1 else "Low"
            st.metric("Predicted Availability", avail,
                      delta=f"Probability: {prob:.2%}")

            # Forecast Chart (if df loaded)
            if not df.empty:
                st.subheader("Historical Availability Trends")
                fig = px.histogram(
                    df, x='availability_365', color='demand_indicator', title="Availability by Demand Level")
                st.plotly_chart(fig)
            else:
                st.info("Load data for trends.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "Dashboard for Airbnb NZ Analytics | Built with Streamlit | v1.0")
