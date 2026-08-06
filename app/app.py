import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(
    page_title="KiwiStay - Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# Load model (cached so it only loads once)


@st.cache_resource
def load_model():
    # Works whether you run from the project root or the app/ folder
    for path in ["models/price_model.pkl", "../models/price_model.pkl"]:
        if os.path.exists(path):
            return joblib.load(path)
    st.error("Model file not found. Run notebook 04 first.")
    st.stop()


model = load_model()

# Approximate lat/lon for major NZ regions
REGION_COORDS = {
    "Auckland":                 (-36.86, 174.76),
    "Queenstown-Lakes District": (-45.03, 168.66),
    "Christchurch City":        (-43.53, 172.63),
    "Wellington City":          (-41.28, 174.77),
    "Rotorua District":         (-38.14, 176.25),
    "Dunedin City":             (-45.87, 170.50),
    "Taupo District":           (-38.68, 176.07),
    "Whangarei District":       (-35.72, 174.32),
    "Hamilton City":            (-37.78, 175.28),
    "Other":                    (-40.90, 172.68),
}

# UI
st.title("🏠 KiwiStay — Airbnb Price Predictor")
st.markdown("Enter your listing details to estimate a competitive nightly rate.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Property")
    property_type = st.selectbox(
        "Property type",
        ["Entire home/apt", "Private room",
            "Hotel/Boutique", "Unique/Other", "Shared room"],
    )
    room_type = st.selectbox(
        "Room type",
        ["Entire home/apt", "Private room", "Hotel room", "Shared room"],
    )
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=2)
    bathrooms = st.number_input(
        "Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5,
    )
    accommodates = st.number_input(
        "Guests (accommodates)", min_value=1, max_value=20, value=4)
    amenities = st.slider("Number of amenities", 0, 100, 30)

with col2:
    st.subheader("Location & host")
    district = st.selectbox("District (region)", sorted(REGION_COORDS.keys()))
    ward = st.text_input("Ward / suburb name", "Auckland City Ward")
    min_nights = st.number_input(
        "Minimum nights", min_value=1, max_value=365, value=2)
    instant = st.checkbox("Instant bookable", value=True)
    host_response = st.selectbox(
        "Host response time",
        ["within an hour", "within a few hours", "within a day",
         "a few days or more", "unknown"],
    )

st.divider()

# Build prediction row


def build_row():
    lat, lon = REGION_COORDS.get(district, (-40.90, 172.68))
    beds_val = max(float(bedrooms), 1.0)  # beds ≥ 1 as a safe default

    return pd.DataFrame([{
        # Categorical (must be in CATEGORICAL_COLS order)
        "room_type":            room_type,
        "region_name":          ward,
        "region_parent_name":   district,
        "property_type_grouped": property_type,
        "host_response_time":   host_response,
        # Numeric: user-provided
        "bedrooms":             float(bedrooms),
        "bathrooms_num":        float(bathrooms),
        "accommodates":         int(accommodates),
        "amenities_count":      int(amenities),
        "minimum_nights":       int(min_nights),
        "instant_bookable":     int(instant),
        "latitude":             lat,
        "longitude":            lon,
        # Numeric: sensible defaults
        "host_response_rate":   100.0,
        "host_acceptance_rate": 95.0,
        "host_is_superhost":    0,
        "host_listings_count":  1.0,
        "host_total_listings_count": 1.0,
        "host_has_profile_pic": 1,
        "host_identity_verified": 1,
        "beds":                 beds_val,
        "maximum_nights":       365,
        "availability_365":     180,
        "availability_eoy":     90,
        "number_of_reviews":    10,
        "review_scores_rating": 4.5,
        "review_scores_accuracy": 4.5,
        "review_scores_cleanliness": 4.5,
        "review_scores_checkin": 4.5,
        "review_scores_communication": 4.5,
        "review_scores_location": 4.5,
        "review_scores_value":  4.3,
        "reviews_per_month":    1.0,
        "calculated_host_listings_count": 1,
        "host_verifications_count": 3,
        "has_reviews":          1,
        "host_tenure_days":     1000,
        "review_recency_days":  60,
    }])


# Predict button
if st.button("Estimate price", type="primary", use_container_width=True):
    row = build_row()
    log_pred = model.predict(row)[0]
    price = np.expm1(log_pred)

    st.success(f"### Estimated nightly price: **NZD {price:,.0f}**")
    st.caption(
        "Based on Inside Airbnb NZ data scraped July 2025. "
        "Model: XGBoost (R² = 0.77 on hold-out set)."
    )

    # Show what drove the prediction
    with st.expander("Key inputs used"):
        st.table(
            pd.DataFrame({
                "Feature": ["Bedrooms", "Accommodates", "Bathrooms",
                            "Property type", "District"],
                "Value":   [bedrooms, accommodates, bathrooms,
                            property_type, district],
            })
        )
