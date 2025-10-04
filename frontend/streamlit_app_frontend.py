from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px

# Page config for better UX
st.set_page_config(
    page_title="KiwiStay AI - Airbnb NZ Insights",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load shared data/models (cached for performance)


@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv('data/listings_discretized_enhanced.csv')
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df_viz = df.sample(
            min(500, len(df))) if not df.empty else pd.DataFrame()
        return df, df_viz
    except Exception:
        st.warning(
            "Data not loaded. Ensure 'data/listings_discretized_enhanced.csv' exists and run preprocessing scripts.")
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=3600)
def load_processed():
    try:
        X_reg_train = joblib.load('data/processed/X_reg_train.pkl')
        X_reg_train = X_reg_train.loc[:, ~
                                      X_reg_train.columns.duplicated()].copy()
    except:
        X_reg_train = pd.DataFrame()
    try:
        X_clf_train = joblib.load('data/processed/X_clf_train.pkl')
        X_clf_train = X_clf_train.loc[:, ~
                                      X_clf_train.columns.duplicated()].copy()
    except:
        X_clf_train = pd.DataFrame()
    try:
        X_avail_train = joblib.load('data/processed/X_avail_train.pkl')
        X_avail_train = X_avail_train.loc[:, ~
                                          X_avail_train.columns.duplicated()].copy()
    except:
        X_avail_train = pd.DataFrame()
    try:
        rules = pd.read_csv(
            'data/processed/association_rules.csv').reset_index(drop=True)
    except:
        rules = pd.DataFrame()
    return X_reg_train, X_clf_train, X_avail_train, rules


@st.cache_resource
def load_models():
    try:
        xg_reg = joblib.load('data/models/xgboost_optimized.pkl')
    except:
        xg_reg = None
    try:
        xg_clf = joblib.load('data/models/classification/xgb_optimized.pkl')
    except:
        xg_clf = None
    try:
        xg_avail = joblib.load(
            'data/models/classification/xgb_availability.pkl')
    except:
        xg_avail = None
    try:
        scaler = joblib.load('data/processed/scaler.pkl')
        if hasattr(scaler, 'feature_names_in_'):
            scaler.feature_names_in_ = pd.Index(
                scaler.feature_names_in_).unique().tolist()
    except:
        scaler = None
    return xg_reg, xg_clf, xg_avail, scaler


df, df_viz = load_data()
X_reg_train, X_clf_train, X_avail_train, rules = load_processed()
xg_reg, xg_clf, xg_avail, scaler = load_models()

# Sidebar Navigation
st.sidebar.title("🏠 KiwiStay AI")
st.sidebar.markdown("Airbnb Insights for New Zealand")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["Welcome", "Price Prediction", "Demand Prediction",
        "Price Fairness", "Availability Forecasting"]
)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# Page Content
if page == "Welcome":
    st.title("🏡 KiwiStay AI - Your Airbnb NZ Companion")
    st.markdown("""
        <style>
            .welcome {
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                color: #1a2b49;
                font-family: 'Arial', sans-serif;
            }
            .highlight {
                color: #2e7d32;
                font-weight: bold;
            }
        </style>
        <div class="welcome">
            <h3>Welcome to KiwiStay AI!</h3>
            <p>Empower your Airbnb journey in New Zealand with AI-driven insights:</p>
            <ul>
                <li><span class="highlight">Hosts:</span> Optimize pricing, predict demand, and ensure fairness.</li>
                <li><span class="highlight">Guests:</span> Discover market trends for smarter bookings.</li>
            </ul>
            <p>Explore the sidebar to get started. Ensure data and models are prepared by running backend scripts in order.</p>
        </div>
    """, unsafe_allow_html=True)

elif page == "Price Prediction":
    st.header("🏠 Price Prediction")
    st.markdown("Enter listing details to get an AI-suggested price.")
    if xg_reg is None or X_reg_train.empty or scaler is None:
        st.warning(
            "Model, data, or scaler not loaded. Run preprocessing and regression scripts.")
    else:
        with st.form("price_form"):
            st.subheader("Your Listing Details")
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input(
                    'Latitude', -48.0, -34.0, value=-36.8485)
                longitude = st.number_input(
                    'Longitude', 166.0, 179.0, value=174.7633)
                availability_365 = st.slider(
                    'Availability (days/year)', 0, 365, 180)
                minimum_nights = st.slider('Minimum Nights', 1, 90, 1)
            with col2:
                number_of_reviews = st.slider('Number of Reviews', 0, 1000, 50)
                reviews_per_month = st.number_input(
                    'Reviews per Month', 0.0, 50.0, 2.0)
                calculated_host_listings_count = st.slider(
                    'Host Listings Count', 1, 100, 1)
                number_of_reviews_ltm = st.slider(
                    'Reviews Last 12 Months', 0, 500, 20)
                is_inactive = st.checkbox('Is Inactive', value=False)

            room_type = st.selectbox(
                'Room Type', ['Entire home/apt', 'Private room', 'Other'])
            neighbourhood = st.text_input('Neighbourhood', 'Unknown')

            submitted = st.form_submit_button("Predict Price")

        if submitted:
            input_data = {col: 0.0 for col in X_reg_train.columns}
            num_map = {
                'latitude': latitude,
                'longitude': longitude,
                'availability_365': availability_365,
                'minimum_nights_cleaned': minimum_nights,
                'number_of_reviews': number_of_reviews,
                'reviews_per_month': reviews_per_month,
                'calculated_host_listings_count': calculated_host_listings_count,
                'number_of_reviews_ltm_cleaned': number_of_reviews_ltm,
                'is_inactive': int(is_inactive),
                'recent_reviews_ratio': number_of_reviews_ltm / max(number_of_reviews, 1)
            }
            for key, val in num_map.items():
                if key in input_data:
                    input_data[key] = val

            if 'neighbourhood_encoded' in input_data:
                input_data['neighbourhood_encoded'] = df['neighbourhood_encoded'].mean(
                ) if 'neighbourhood_encoded' in df.columns else 0.0

            for col in X_reg_train.columns:
                if col.startswith('room_type_cleaned_'):
                    rt_cleaned = room_type if room_type != 'Other' else 'Other'
                    input_data[col] = 1 if col == f'room_type_cleaned_{rt_cleaned}' else 0

            base_inter = ['availability_365', 'minimum_nights_cleaned',
                          'calculated_host_listings_count', 'recent_reviews_ratio']
            for col in X_reg_train.columns:
                if ' ' in col and len(col.split(' ')) == 2:
                    parts = col.split(' ')
                    if all(p in num_map for p in parts):
                        input_data[col] = num_map[parts[0]] * num_map[parts[1]]

            for col in ['pca1', 'pca2', 'pca3', 'loc_pca1', 'loc_pca2']:
                if col in input_data:
                    input_data[col] = X_reg_train[col].mean(
                    ) if not X_reg_train.empty else 0.0

            input_df = pd.DataFrame([input_data])
            input_df = input_df.loc[:, ~input_df.columns.duplicated()].copy()
            input_df = input_df.reindex(
                columns=X_reg_train.columns, fill_value=0)

            if hasattr(scaler, 'feature_names_in_'):
                scaler_features = pd.Index(
                    scaler.feature_names_in_).unique().tolist()
                scaler_input = pd.DataFrame(columns=scaler_features, index=[0])
                scaler_input[:] = 0
                for col in scaler_input.columns:
                    if col in input_df.columns:
                        scaler_input.loc[0, col] = input_df.loc[0, col]
                scaled_values = scaler.transform(scaler_input)
                scaled_df = pd.DataFrame(
                    scaled_values, columns=scaler_features, index=[0])
                for col in input_df.columns:
                    if col in scaled_df.columns:
                        input_df.loc[0, col] = scaled_df.loc[0, col]

            pred = xg_reg.predict(input_df)
            st.success(
                f'Predicted Optimal Price: ${np.expm1(pred[0]):.2f} NZD')
            st.markdown("*Tip*: Adjust based on seasonal demand.")

elif page == "Demand Prediction":
    st.header("🔥 Demand Prediction")
    st.markdown("Predict demand levels for your listing to optimize bookings.")
    if xg_clf is None or X_clf_train.empty or scaler is None:
        st.warning(
            "Model, data, or scaler not loaded. Run preprocessing and classification scripts.")
    else:
        with st.form("demand_form"):
            st.subheader("Your Listing Details")
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input(
                    'Latitude', -48.0, -34.0, value=-36.8485)
                longitude = st.number_input(
                    'Longitude', 166.0, 179.0, value=174.7633)
                availability_365 = st.slider(
                    'Availability (days/year)', 0, 365, 180)
                minimum_nights = st.slider('Minimum Nights', 1, 90, 1)
            with col2:
                number_of_reviews = st.slider('Number of Reviews', 0, 1000, 50)
                reviews_per_month = st.number_input(
                    'Reviews per Month', 0.0, 50.0, 2.0)
                calculated_host_listings_count = st.slider(
                    'Host Listings Count', 1, 100, 1)
                number_of_reviews_ltm = st.slider(
                    'Reviews Last 12 Months', 0, 500, 20)
                is_inactive = st.checkbox('Is Inactive', value=False)

            room_type = st.selectbox(
                'Room Type', ['Entire home/apt', 'Private room', 'Other'])
            neighbourhood = st.text_input('Neighbourhood', 'Unknown')

            submitted = st.form_submit_button("Predict Demand")

        if submitted:
            input_data = {col: 0.0 for col in X_clf_train.columns}
            num_map = {
                'latitude': latitude,
                'longitude': longitude,
                'availability_365': availability_365,
                'minimum_nights_cleaned': minimum_nights,
                'number_of_reviews': number_of_reviews,
                'reviews_per_month': reviews_per_month,
                'calculated_host_listings_count': calculated_host_listings_count,
                'number_of_reviews_ltm_cleaned': number_of_reviews_ltm,
                'is_inactive': int(is_inactive),
                'recent_reviews_ratio': number_of_reviews_ltm / max(number_of_reviews, 1)
            }
            for key, val in num_map.items():
                if key in input_data:
                    input_data[key] = val

            if 'neighbourhood_encoded' in input_data:
                input_data['neighbourhood_encoded'] = df['neighbourhood_encoded'].mean(
                ) if 'neighbourhood_encoded' in df.columns else 0.0

            for col in X_clf_train.columns:
                if col.startswith('room_type_cleaned_'):
                    rt_cleaned = room_type if room_type != 'Other' else 'Other'
                    input_data[col] = 1 if col == f'room_type_cleaned_{rt_cleaned}' else 0

            base_inter = ['availability_365', 'minimum_nights_cleaned',
                          'calculated_host_listings_count', 'recent_reviews_ratio']
            for col in X_clf_train.columns:
                if ' ' in col and len(col.split(' ')) == 2:
                    parts = col.split(' ')
                    if all(p in num_map for p in parts):
                        input_data[col] = num_map[parts[0]] * num_map[parts[1]]

            for col in ['loc_pca1', 'loc_pca2']:
                if col in input_data:
                    input_data[col] = X_clf_train[col].mean(
                    ) if not X_clf_train.empty else 0.0

            input_df = pd.DataFrame([input_data])
            input_df = input_df.loc[:, ~input_df.columns.duplicated()].copy()
            input_df = input_df.reindex(
                columns=X_clf_train.columns, fill_value=0)

            if hasattr(scaler, 'feature_names_in_'):
                scaler_features = pd.Index(
                    scaler.feature_names_in_).unique().tolist()
                scaler_input = pd.DataFrame(columns=scaler_features, index=[0])
                scaler_input[:] = 0
                for col in scaler_input.columns:
                    if col in input_df.columns:
                        scaler_input.loc[0, col] = input_df.loc[0, col]
                scaled_values = scaler.transform(scaler_input)
                scaled_df = pd.DataFrame(
                    scaled_values, columns=scaler_features, index=[0])
                for col in input_df.columns:
                    if col in scaled_df.columns:
                        input_df.loc[0, col] = scaled_df.loc[0, col]

            pred = xg_clf.predict(input_df)
            prob = xg_clf.predict_proba(input_df)[:, 1][0]
            demand = "High" if pred[0] == 1 else "Low"
            st.metric("Predicted Demand Level", demand,
                      delta=f"Confidence: {prob:.2%}")
            st.markdown(
                "*Tip*: High demand? Raise prices. Low? Improve reviews.")

elif page == "Price Fairness":
    st.header("⚖ Price Fairness Check")
    st.markdown("Assess if your pricing aligns with market trends.")
    if rules.empty:
        st.warning(
            "Rules not loaded. Run preprocessing and association_rules.py.")
    else:
        st.subheader("Top Market Rules")
        display_cols = ['antecedents', 'consequents',
                        'support', 'confidence', 'lift']
        rules_display = rules.copy()
        rules_display['antecedents'] = rules_display['antecedents'].apply(
            lambda x: ', '.join(x) if isinstance(x, (list, tuple, set)) else str(x))
        rules_display['consequents'] = rules_display['consequents'].apply(
            lambda x: ', '.join(x) if isinstance(x, (list, tuple, set)) else str(x))
        st.dataframe(rules_display[display_cols].head(
            10).sort_values('lift', ascending=False))

        with st.form("fairness_form"):
            st.subheader("Your Listing Details")
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input(
                    'Latitude', -48.0, -34.0, value=-36.8485)
                longitude = st.number_input(
                    'Longitude', 166.0, 179.0, value=174.7633)
                availability_365 = st.slider(
                    'Availability (days/year)', 0, 365, 180)
                minimum_nights = st.slider('Minimum Nights', 1, 90, 1)
                price = st.number_input('Price (NZD)', 50.0, 2000.0, 150.0)
            with col2:
                number_of_reviews = st.slider('Number of Reviews', 0, 1000, 50)
                reviews_per_month = st.number_input(
                    'Reviews per Month', 0.0, 50.0, 2.0)
                calculated_host_listings_count = st.slider(
                    'Host Listings Count', 1, 100, 1)
                number_of_reviews_ltm = st.slider(
                    'Reviews Last 12 Months', 0, 500, 20)
                is_inactive = st.checkbox('Is Inactive', value=False)

            room_type = st.selectbox(
                'Room Type', ['Entire home/apt', 'Private room', 'Other'])
            neighbourhood = st.text_input('Neighbourhood', 'Unknown')

            submitted = st.form_submit_button("Assess Fairness")

        if submitted:
            price_bin = pd.cut([price], bins=3, labels=[
                               'low', 'medium', 'high'])[0]
            reviews_bin = pd.cut([reviews_per_month], bins=3, labels=[
                                 'low', 'medium', 'high'])[0]
            median_reviews = df['reviews_per_month'].median(
            ) if not df.empty else 0.5
            expected_demand = 'high' if reviews_per_month > median_reviews else 'low'

            def has_low_antecedent(antecedents):
                return any('low' in str(item) for item in antecedents)

            def has_high_consequent(consequents):
                return any('high' in str(item) for item in consequents)

            matching_rules = rules[rules['antecedents'].apply(
                has_low_antecedent) & rules['consequents'].apply(has_high_consequent)]
            fairness = "Fair" if len(matching_rules) > 0 else "Review Pricing"
            st.success(
                f"Price Fairness: {fairness}\nPrice: {price_bin} | Reviews: {reviews_bin} | Demand: {expected_demand}")
            st.markdown(
                "*Tip*: Align prices with market rules for better bookings.")

elif page == "Availability Forecasting":
    st.header("📅 Availability Forecasting")
    st.markdown("Predict future availability to plan effectively.")
    if xg_avail is None or X_avail_train.empty or scaler is None:
        st.warning(
            "Model, data, or scaler not loaded. Run preprocessing and availability script.")
    else:
        with st.form("avail_form"):
            st.subheader("Your Listing Details")
            col1, col2 = st.columns(2)
            with col1:
                latitude = st.number_input(
                    'Latitude', -48.0, -34.0, value=-36.8485)
                longitude = st.number_input(
                    'Longitude', 166.0, 179.0, value=174.7633)
                availability_365 = st.slider(
                    'Availability (days/year)', 0, 365, 180)
                minimum_nights = st.slider('Minimum Nights', 1, 90, 1)
            with col2:
                number_of_reviews = st.slider('Number of Reviews', 0, 1000, 50)
                reviews_per_month = st.number_input(
                    'Reviews per Month', 0.0, 50.0, 2.0)
                calculated_host_listings_count = st.slider(
                    'Host Listings Count', 1, 100, 1)
                number_of_reviews_ltm = st.slider(
                    'Reviews Last 12 Months', 0, 500, 20)
                is_inactive = st.checkbox('Is Inactive', value=False)

            room_type = st.selectbox(
                'Room Type', ['Entire home/apt', 'Private room', 'Other'])
            neighbourhood = st.text_input('Neighbourhood', 'Unknown')

            submitted = st.form_submit_button("Forecast Availability")

        if submitted:
            input_data = {col: 0.0 for col in X_avail_train.columns}
            num_map = {
                'latitude': latitude,
                'longitude': longitude,
                'availability_365': availability_365,
                'minimum_nights_cleaned': minimum_nights,
                'number_of_reviews': number_of_reviews,
                'reviews_per_month': reviews_per_month,
                'calculated_host_listings_count': calculated_host_listings_count,
                'number_of_reviews_ltm_cleaned': number_of_reviews_ltm,
                'is_inactive': int(is_inactive),
                'recent_reviews_ratio': number_of_reviews_ltm / max(number_of_reviews, 1)
            }
            for key, val in num_map.items():
                if key in input_data:
                    input_data[key] = val

            if 'neighbourhood_encoded' in input_data:
                input_data['neighbourhood_encoded'] = df['neighbourhood_encoded'].mean(
                ) if 'neighbourhood_encoded' in df.columns else 0.0

            for col in X_avail_train.columns:
                if col.startswith('room_type_cleaned_'):
                    rt_cleaned = room_type if room_type != 'Other' else 'Other'
                    input_data[col] = 1 if col == f'room_type_cleaned_{rt_cleaned}' else 0

            base_inter = ['availability_365', 'minimum_nights_cleaned',
                          'calculated_host_listings_count', 'recent_reviews_ratio']
            for col in X_avail_train.columns:
                if ' ' in col and len(col.split(' ')) == 2:
                    parts = col.split(' ')
                    if all(p in num_map for p in parts):
                        input_data[col] = num_map[parts[0]] * num_map[parts[1]]

            for col in ['loc_pca1', 'loc_pca2']:
                if col in input_data:
                    input_data[col] = X_avail_train[col].mean(
                    ) if not X_avail_train.empty else 0.0

            input_df = pd.DataFrame([input_data])
            input_df = input_df.loc[:, ~input_df.columns.duplicated()].copy()
            input_df = input_df.reindex(
                columns=X_avail_train.columns, fill_value=0)

            if hasattr(scaler, 'feature_names_in_'):
                scaler_features = pd.Index(
                    scaler.feature_names_in_).unique().tolist()
                scaler_input = pd.DataFrame(columns=scaler_features, index=[0])
                scaler_input[:] = 0
                for col in scaler_input.columns:
                    if col in input_df.columns:
                        scaler_input.loc[0, col] = input_df.loc[0, col]
                scaled_values = scaler.transform(scaler_input)
                scaled_df = pd.DataFrame(
                    scaled_values, columns=scaler_features, index=[0])
                for col in input_df.columns:
                    if col in scaled_df.columns:
                        input_df.loc[0, col] = scaled_df.loc[0, col]

            pred = xg_avail.predict(input_df)
            prob = xg_avail.predict_proba(input_df)[:, 1][0]
            avail = "High (>180 days)" if pred[0] == 1 else "Low"
            st.metric("Forecasted Availability Level",
                      avail, delta=f"Confidence: {prob:.2%}")
            st.markdown(
                "*Tip*: High availability? Promote long stays. Low? Schedule updates.")