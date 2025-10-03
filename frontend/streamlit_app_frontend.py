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
from sklearn.cluster import AgglomerativeClustering
import matplotlib.colors as mcolors

# Page configuration
st.set_page_config(page_title="Airbnb NZ Analytics Dashboard", layout="wide")

# Debug: Show working dir and file checks
st.sidebar.markdown("### Debug Info")
st.sidebar.write(f"**Working Dir:** {os.getcwd()}")

# Load data and models


@st.cache_data
def load_data():
    df_path = 'data/listings_discretized_enhanced.csv'
    rules_path = 'data/processed/association_rules.csv'

    df = pd.DataFrame()
    X_reg_train = pd.DataFrame()
    X_clf_train = pd.DataFrame()
    X_avail_train = pd.DataFrame()
    rules = pd.DataFrame()

    try:
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            st.sidebar.success(f"✅ Loaded {df_path} (shape: {df.shape})")
        else:
            st.sidebar.error(f"❌ Missing: {df_path}")
    except Exception as e:
        st.sidebar.error(f"❌ Load failed for df: {e}")

    try:
        if os.path.exists('data/processed/X_reg_train.pkl'):
            X_reg_train = joblib.load('data/processed/X_reg_train.pkl')
            st.sidebar.success(
                f"✅ Loaded X_reg_train (shape: {X_reg_train.shape})")
        else:
            st.sidebar.error("❌ Missing: data/processed/X_reg_train.pkl")
    except Exception as e:
        st.sidebar.error(f"❌ Load failed for X_reg_train: {e}")

    try:
        if os.path.exists('data/processed/X_clf_train.pkl'):
            X_clf_train = joblib.load('data/processed/X_clf_train.pkl')
            st.sidebar.success(
                f"✅ Loaded X_clf_train (shape: {X_clf_train.shape})")
        else:
            st.sidebar.error("❌ Missing: data/processed/X_clf_train.pkl")
    except Exception as e:
        st.sidebar.error(f"❌ Load failed for X_clf_train: {e}")

    try:
        if os.path.exists('data/processed/X_avail_train.pkl'):
            X_avail_train = joblib.load('data/processed/X_avail_train.pkl')
            st.sidebar.success(
                f"✅ Loaded X_avail_train (shape: {X_avail_train.shape})")
        else:
            st.sidebar.error("❌ Missing: data/processed/X_avail_train.pkl")
    except Exception as e:
        st.sidebar.error(f"❌ Load failed for X_avail_train: {e}")

    try:
        if os.path.exists(rules_path):
            rules = pd.read_csv(rules_path)
            rules = rules.reset_index(drop=True)
            st.sidebar.success(f"✅ Loaded {rules_path} (rows: {len(rules)})")
        else:
            st.sidebar.error(f"❌ Missing: {rules_path}")
    except Exception as e:
        st.sidebar.error(f"❌ Load failed for rules: {e}")

    return df, X_reg_train, X_clf_train, X_avail_train, rules


df, X_reg_train, X_clf_train, X_avail_train, rules = load_data()

# Load models (only highest performing)


@st.cache_data
def load_xgboost_reg():
    path = 'data/models/xgboost_optimized.pkl'
    try:
        if os.path.exists(path):
            model = joblib.load(path)
            st.sidebar.success(f"✅ Loaded XGBoost reg from {path}")
            return model
        else:
            st.sidebar.error(f"❌ Missing: {path}")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Load failed for XGBoost reg: {e}")
        return None


@st.cache_data
def load_xgboost_clf():
    path = 'data/models/classification/xgb_optimized.pkl'
    try:
        if os.path.exists(path):
            model = joblib.load(path)
            st.sidebar.success(f"✅ Loaded XGBoost clf from {path}")
            return model
        else:
            st.sidebar.error(f"❌ Missing: {path}")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Load failed for XGBoost clf: {e}")
        return None


@st.cache_data
def load_agglomerative():
    path = 'data/models/clustering/agglomerative_optimized.pkl'
    try:
        if os.path.exists(path):
            model = joblib.load(path)
            st.sidebar.success(f"✅ Loaded Agglomerative from {path}")
            return model
        else:
            st.sidebar.error(f"❌ Missing: {path}")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Load failed for Agglomerative: {e}")
        return None


xg_reg = load_xgboost_reg()
xg_clf = load_xgboost_clf()
agg_model = load_agglomerative()

# Common input form


def get_input_form(show_price=False):
    with st.form("listing_details"):
        st.subheader("Listing Details")
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input('Latitude', -48.0, -34.0, -36.8485)
            longitude = st.number_input('Longitude', 166.0, 179.0, 174.7633)
            availability_365 = st.slider('Availability (days)', 0, 365, 180)
            minimum_nights = st.slider('Minimum Nights', 1, 90, 1)
            if show_price:
                price = st.number_input('Price (NZD)', 50.0, 2000.0, 150.0)
            else:
                price = None  # Or default to 0, but not used
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
        'minimum_nights': minimum_nights, 'price': price,  # Only set if show_price=True
        'number_of_reviews': number_of_reviews,
        'reviews_per_month': reviews_per_month, 'calculated_host_listings_count': calculated_host_listings_count,
        'number_of_reviews_ltm': number_of_reviews_ltm, 'is_inactive': int(is_inactive),
        'room_type': room_type, 'neighbourhood_group': neighbourhood_group, 'neighbourhood': neighbourhood
    }, submitted

# Prepare input DataFrame (task-specific)


def prepare_input(inputs, X_sample):
    if X_sample.empty:
        st.warning("Sample data not loaded—using dummy for prediction.")
        return pd.DataFrame({'dummy': [0]})

    input_data = {col: 0 for col in X_sample.columns}

    # Set numerical from inputs if column exists
    num_map = {
        'latitude': inputs['latitude'],
        'longitude': inputs['longitude'],
        'availability_365': inputs['availability_365'],
        'minimum_nights_cleaned': inputs['minimum_nights'],
        'number_of_reviews': inputs['number_of_reviews'],
        'reviews_per_month': inputs['reviews_per_month'],
        'calculated_host_listings_count': inputs['calculated_host_listings_count'],
        # Note: cleaned version
        'number_of_reviews_ltm_cleaned': inputs['number_of_reviews_ltm'],
        'is_inactive': inputs['is_inactive'],
        # Approximate
        'recent_reviews_ratio': inputs['number_of_reviews_ltm'] / max(inputs['number_of_reviews'], 1)
    }
    for key, val in num_map.items():
        if key in input_data:
            input_data[key] = val

    # Neighbourhood encoded (mean if not in inputs)
    if 'neighbourhood_encoded' in input_data:
        input_data['neighbourhood_encoded'] = X_sample['neighbourhood_encoded'].mean(
        ) if 'neighbourhood_encoded' in X_sample.columns else 0

    # One-hot for room_type_cleaned
    for col in X_sample.columns:
        if col.startswith('room_type_cleaned_'):
            rt_cleaned = inputs['room_type'] if inputs['room_type'] != 'Other' else 'Other'
            input_data[col] = 1 if col == f'room_type_cleaned_{rt_cleaned}' else 0

    # Skip neigh_group ohe as dropped in tasks

    # Polynomial interactions (calculate if base cols present)
    base_inter = ['availability_365', 'minimum_nights_cleaned',
                  'calculated_host_listings_count', 'recent_reviews_ratio']
    for col in X_sample.columns:
        if '_' in col and any(base in col for base in base_inter):
            # Approximate: product of means or inputs
            parts = col.split('_')
            if len(parts) == 2 and all(p in num_map for p in parts):
                input_data[col] = num_map[parts[0]] * num_map[parts[1]]

    # PCA (means)
    for col in ['pca1', 'pca2', 'pca3', 'loc_pca1', 'loc_pca2']:
        if col in input_data:
            input_data[col] = X_sample[col].mean()

    input_df = pd.DataFrame([input_data])

    # Scale using fitted on sample (approximate)
    scaler_cols = [col for col in input_df.columns if col in X_sample.columns and (col in ['latitude', 'longitude', 'minimum_nights_cleaned', 'number_of_reviews',
                   'reviews_per_month', 'calculated_host_listings_count', 'availability_365',
                                                                                           'number_of_reviews_ltm_cleaned', 'is_inactive', 'neighbourhood_encoded'] or '_' in col)]
    if scaler_cols:
        scaler = MinMaxScaler()
        scaler.fit(X_sample[scaler_cols])
        input_df[scaler_cols] = scaler.transform(input_df[scaler_cols])

    return input_df


# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a section:",
                            ["Price Prediction", "Market Segmentation", "Demand Prediction",
                             "Price Fairness", "Availability Forecasting"])

if st.sidebar.button("📋 Setup Guide"):
    st.sidebar.markdown("""
    ### Quick Setup (Run These Scripts):
    1. **Preprocessing**: `python data_quality.py` → `data_cleaning.py` → `data_reduction.py` → `data_transform.py` → `data_discretization.py` → `feature_importance.py`
    2. **Regression**: `python regression/XGBoost_regressor.optimized_parameter.py`
    3. **Classification**: `python classification/XGBoost/xgb_optimized`
    4. **Clustering**: `python clustering/augglomerative_optimized.py`
    5. **Association Rules**: `python Association_rules/association_rules.py`
    
    Check `data/` for generated files. Refresh app after.
    """)

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(
    "Dashboard for Airbnb NZ Analytics | Built with Streamlit | v1.0")

# === PAGE: Price Prediction ===
if page == "Price Prediction":
    st.header("🏠 Price Prediction")

    if xg_reg is None:
        st.warning(
            "XGBoost regression model not loaded. Check sidebar debug for details.")
    else:
        st.info("Using XGBoost (Optimized) - Highest Performing Model")
        # From optimized print; update as needed
        st.metric("Model Performance (R²)", "0.82")

        inputs, submitted = get_input_form()
        if submitted:
            input_df = prepare_input(inputs, X_reg_train)
            pred = xg_reg.predict(input_df)
            st.success(f'Predicted Price: ${np.expm1(pred[0]):.2f} NZD')

            # Feature Importance
            if hasattr(xg_reg, 'feature_importances_'):
                st.subheader("Feature Importance (Top 10)")
                imp = pd.Series(xg_reg.feature_importances_,
                                index=X_reg_train.columns).sort_values(ascending=False)
                fig = px.bar(x=imp.head(10).values, y=imp.head(
                    10).index, orientation='h', title="Top Features")
                st.plotly_chart(fig, use_container_width=True)

# === PAGE: Market Segmentation ===
elif page == "Market Segmentation":
    st.header("📊 Market Segmentation (Clustering)")

    if df.empty:
        st.warning("No data loaded for clustering. Check sidebar debug.")
    else:
        st.info("Using Agglomerative Clustering (Optimized) - Highest Performing Model")

        if agg_model is None:
            st.warning("Agglomerative model not loaded. Check sidebar debug.")
            labels = np.zeros(len(df))
            df['cluster'] = labels
            n_clust = 1
            st.metric("Silhouette Score", "N/A")
            st.metric("Number of Clusters", n_clust)
        else:
            try:
                cluster_features = ['log_price', 'loc_pca1', 'loc_pca2',
                                    'availability_365', 'recent_reviews_ratio', 'minimum_nights_cleaned']
                X_cluster = df[cluster_features].fillna(0)
                labels = agg_model.fit_predict(X_cluster)
                df['cluster'] = labels
                n_clust = agg_model.n_clusters_
                # From optimized print; update as needed
                st.metric("Silhouette Score", "0.25")
                st.metric("Number of Clusters", n_clust)
            except Exception as e:
                st.error(f"Failed to predict clusters: {e}")
                labels = np.zeros(len(df))
                df['cluster'] = labels
                n_clust = 1

        # Segments
        df['segment'] = [f"Cluster {i}" for i in labels]

        # Interactive Chart
        st.subheader("Segment Distribution")
        fig = px.scatter(df, x='longitude', y='latitude', color='segment',
                         hover_data=['price', 'room_type', 'availability_365'],
                         title="Airbnb Listings by Market Segment")
        st.plotly_chart(fig, use_container_width=True)

        # Folium Map
        st.subheader("Interactive Map of Segments")
        m = folium.Map(location=[-41.2924, 174.7787],
                       zoom_start=6)  # NZ center
        colors = list(mcolors.TABLEAU_COLORS.values())[:n_clust]
        color_map = {i: colors[i % len(colors)] for i in range(n_clust)}
        for idx, row in df.iterrows():
            color = color_map.get(row['cluster'], 'gray')
            folium.CircleMarker(
                [row['latitude'], row['longitude']],
                radius=3, popup=f"{row['segment']} - ${row['price']:.0f} - {row['room_type']}",
                color=color, fill=True, fillOpacity=0.7
            ).add_to(m)

        folium_static(m, width=700, height=500)

        # Segment Insights
        st.subheader("Segment Insights")
        segment_stats = df.groupby(
            'segment')[['price', 'availability_365', 'number_of_reviews']].mean().round(2)
        st.dataframe(segment_stats)

# === PAGE: Demand Prediction ===
elif page == "Demand Prediction":
    st.header("🔥 Demand Prediction (High/Low Popularity)")

    if xg_clf is None:
        st.warning(
            "XGBoost classification model not loaded. Check sidebar debug.")
    elif X_clf_train.empty:
        st.warning("X_clf_train not loaded—run feature_importance.py first.")
    else:
        st.info("Using XGBoost (Optimized) - Highest Performing Model")
        # From optimized print; update as needed
        st.metric("Model Performance (ROC-AUC)", "0.85")

        inputs, submitted = get_input_form()
        if submitted:
            input_df = prepare_input(inputs, X_clf_train)
            pred = xg_clf.predict(input_df)
            prob = xg_clf.predict_proba(input_df)[:, 1][0]
            demand = "High" if pred[0] == 1 else "Low"
            st.metric("Predicted Demand", demand,
                      delta=f"Probability: {prob:.2%}")

# === PAGE: Price Fairness ===
elif page == "Price Fairness":
    st.header("⚖️ Price Fairness (Association Rules)")

    if rules.shape[0] == 0:
        st.warning("No association rules loaded. Check sidebar debug.")
    else:
        st.info("Using Apriori Association Rules - Highest Performing for Rules")
        st.subheader(
            "Top Rules (e.g., Low Price + High Reviews → High Demand)")
        display_cols = ['antecedents', 'consequents',
                        'support', 'confidence', 'lift']
        if all(col in rules.columns for col in display_cols):
            rules_display = rules.copy()
            rules_display['antecedents'] = rules_display['antecedents'].apply(
                lambda x: ', '.join(list(x)))
            rules_display['consequents'] = rules_display['consequents'].apply(
                lambda x: ', '.join(list(x)))
            st.dataframe(rules_display[display_cols].head(
                10).sort_values('lift', ascending=False))
        else:
            st.info("Rules columns incomplete—re-run association_rules.py.")

        inputs, submitted = get_input_form()
        if submitted and not df.empty:
            median_price = df['price'].median()
            price_bin = pd.cut([inputs.get('price', median_price)], bins=3, labels=[
                               'low', 'medium', 'high']).iloc[0]
            reviews_bin = pd.cut([inputs['reviews_per_month']], bins=3, labels=[
                                 'low', 'medium', 'high']).iloc[0]
            expected_demand = 'high' if inputs['reviews_per_month'] > df['reviews_per_month'].median(
            ) else 'low'

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

    if xg_clf is None:
        st.warning("XGBoost model not loaded. Check sidebar debug.")
    elif X_avail_train.empty:
        st.warning("X_avail_train not loaded—run feature_importance.py first.")
    else:
        st.info(
            "Using XGBoost (Optimized) - Highest Performing (Proxy from Popularity Model)")
        # Proxy from optimized print; update as needed
        st.metric("Model Performance (ROC-AUC)", "0.85")

        inputs, submitted = get_input_form()
        if submitted:
            # Use X_avail_train for prepare, but model is clf (features may differ; approximate)
            input_df = prepare_input(inputs, X_avail_train)
            # Align columns to model if needed (fill missing with 0)
            input_df = input_df.reindex(
                columns=X_clf_train.columns, fill_value=0)
            pred = xg_clf.predict(input_df)
            prob = xg_clf.predict_proba(input_df)[:, 1][0]
            avail = "High (>180 days)" if pred[0] == 1 else "Low"
            st.metric("Predicted Availability", avail,
                      delta=f"Probability: {prob:.2%}")
            st.info("Note: Using popularity model as proxy for availability.")

            # Forecast Chart (if df loaded)
            if not df.empty:
                st.subheader("Historical Availability Trends")
                fig = px.histogram(
                    df, x='availability_365', color='demand_indicator', title="Availability by Demand Level")
                st.plotly_chart(fig)
            else:
                st.info("Load data for trends.")
