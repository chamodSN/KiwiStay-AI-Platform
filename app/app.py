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

# Load model


@st.cache_resource
def load_model():
    for path in ["models/price_model.pkl", "../models/price_model.pkl"]:
        if os.path.exists(path):
            return joblib.load(path)
    st.error("Model file not found. Run notebook 04 first.")
    st.stop()


model = load_model()

# Approximate lat/lon for major NZ regions
REGION_COORDS = {
    "Auckland":                  (-36.86, 174.76),
    "Queenstown-Lakes District": (-45.03, 168.66),
    "Christchurch City":         (-43.53, 172.63),
    "Wellington City":           (-41.28, 174.77),
    "Rotorua District":          (-38.14, 176.25),
    "Dunedin City":              (-45.87, 170.50),
    "Taupo District":            (-38.68, 176.07),
    "Whangarei District":        (-35.72, 174.32),
    "Hamilton City":             (-37.78, 175.28),
    "Other":                     (-40.90, 172.68),
}

# UI
st.title("🏠 KiwiStay - Airbnb Price Predictor")
st.markdown(
    "Enter your listing details below to get an estimated nightly price "
    "based on real New Zealand Airbnb data (July 2025). "
    "Hover the **?** next to any field for guidance."
)
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏡 Property")

    property_type = st.selectbox(
        "Property type",
        ["Entire home/apt", "Private room",
            "Hotel/Boutique", "Unique/Other", "Shared room"],
        help=(
            "**Entire home/apt** - guests have the whole property to themselves "
            "(house, apartment, flat, cottage, etc.).\n\n"
            "**Private room** - guests have their own bedroom but share common areas "
            "(kitchen, lounge) with the host or other guests.\n\n"
            "**Hotel/Boutique** - professionally run accommodation such as a hotel room, "
            "hostel bed, or bed & breakfast.\n\n"
            "**Unique/Other** - unusual properties like tiny homes, treehouses, farm stays, "
            "boats, or anything that doesn't fit the categories above.\n\n"
            "**Shared room** - guests sleep in a shared dormitory or shared bedroom."
        ),
    )

    room_type = st.selectbox(
        "Room type",
        ["Entire home/apt", "Private room", "Hotel room", "Shared room"],
        help=(
            "Airbnb's standard room-type classification — similar to property type "
            "but used differently in search filters.\n\n"
            "**Entire home/apt** - the whole space is yours.\n\n"
            "**Private room** - your own room, shared common areas.\n\n"
            "**Hotel room** - a room in a professionally run property.\n\n"
            "**Shared room** - shared sleeping space."
        ),
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=20,
        value=2,
        help=(
            "Number of dedicated bedrooms in the listing. "
            "Enter **0** for a studio apartment (open-plan with no separate bedroom). "
            "This is the single strongest predictor of price."
        ),
    )

    bath_options = {
        "0.5 - half-bath (toilet & sink only)": 0.5,
        "1 bath": 1.0,
        "1.5 baths (1 full + 1 half)": 1.5,
        "2 baths": 2.0,
        "2.5 baths (2 full + 1 half)": 2.5,
        "3 baths": 3.0,
        "3.5 baths": 3.5,
        "4+ baths": 4.0,
    }
    bath_label = st.selectbox(
        "Bathrooms",
        list(bath_options.keys()),
        index=1,
        help=(
            "A **full bath** has a shower or bathtub, toilet, and sink.\n\n"
            "A **half-bath** (powder room) has only a toilet and sink — no shower. "
            "Common in multi-bedroom homes.\n\n"
            "Example: ensuite + main bathroom + downstairs toilet = **2.5 baths**."
        ),
    )
    bathrooms = bath_options[bath_label]

    accommodates = st.number_input(
        "Guests (accommodates)",
        min_value=1,
        max_value=20,
        value=4,
        help=(
            "Maximum number of guests your listing can sleep — "
            "includes all beds, sofa beds, and any sleeping space you advertise. "
            "More capacity generally means a higher price."
        ),
    )

    amenities = st.slider(
        "Number of amenities",
        0,
        100,
        30,
        help=(
            "Count of amenities listed on your Airbnb page. "
            "Common examples: WiFi, kitchen, parking, washing machine, dryer, "
            "air conditioning, heating, TV, pool, gym.\n\n"
            "Typical listings have **20–50** amenities."
        ),
    )

with col2:
    st.subheader("📍 Location & host")

    district = st.selectbox(
        "District",
        sorted(REGION_COORDS.keys()),
        help=(
            "The New Zealand district (territorial authority) your listing is in. "
            "Queenstown and Auckland command the highest rates nationally."
        ),
    )

    ward = st.text_input(
        "Ward / suburb",
        "Auckland City Ward",
        help=(
            "The specific ward or suburb within the district. "
            "Examples: *Ponsonby*, *Queenstown-Wakatipu Ward*, *Riccarton Ward*.\n\n"
            "If unsure, leave the district-level name — "
            "the model will still use the district information."
        ),
    )

    min_nights = st.number_input(
        "Minimum nights",
        min_value=1,
        max_value=365,
        value=2,
        help=(
            "Minimum number of nights a guest must book. "
            "Most NZ listings use **1–3 nights**. "
            "A higher minimum filters out short-stay guests."
        ),
    )

    instant = st.checkbox(
        "Instant bookable",
        value=True,
        help=(
            "If ticked, guests can book immediately without host approval. "
            "Instant bookable listings appear higher in Airbnb search results "
            "and typically earn **10–20% more** than non-instant listings."
        ),
    )

    host_response = st.selectbox(
        "Host response time",
        ["within an hour", "within a few hours", "within a day",
         "a few days or more", "unknown"],
        help=(
            "How quickly you typically respond to new enquiries. "
            "Displayed on your Airbnb profile and affects Superhost status.\n\n"
            "**within an hour** - best for your listing's search ranking.\n\n"
            "**unknown** - used for brand-new listings with no response history."
        ),
    )

st.divider()

# Build prediction row


def build_row():
    lat, lon = REGION_COORDS.get(district, (-40.90, 172.68))
    beds_val = max(float(bedrooms), 1.0)

    return pd.DataFrame([{
        # Categorical
        "room_type":             room_type,
        "region_name":           ward,
        "region_parent_name":    district,
        "property_type_grouped": property_type,
        "host_response_time":    host_response,
        # Numeric — user-provided
        "bedrooms":              float(bedrooms),
        "bathrooms_num":         float(bathrooms),
        "accommodates":          int(accommodates),
        "amenities_count":       int(amenities),
        "minimum_nights":        int(min_nights),
        "instant_bookable":      int(instant),
        "latitude":              lat,
        "longitude":             lon,
        # Numeric — sensible defaults
        "host_response_rate":    100.0,
        "host_acceptance_rate":  95.0,
        "host_is_superhost":     0,
        "host_listings_count":   1.0,
        "host_total_listings_count": 1.0,
        "host_has_profile_pic":  1,
        "host_identity_verified": 1,
        "beds":                  beds_val,
        "maximum_nights":        365,
        "availability_365":      180,
        "availability_eoy":      90,
        "number_of_reviews":     10,
        "review_scores_rating":  4.5,
        "review_scores_accuracy": 4.5,
        "review_scores_cleanliness": 4.5,
        "review_scores_checkin": 4.5,
        "review_scores_communication": 4.5,
        "review_scores_location": 4.5,
        "review_scores_value":   4.3,
        "reviews_per_month":     1.0,
        "calculated_host_listings_count": 1,
        "host_verifications_count": 3,
        "has_reviews":           1,
        "host_tenure_days":      1000,
        "review_recency_days":   60,
    }])


# Predict button
if st.button("Estimate nightly price", type="primary", use_container_width=True):
    row = build_row()
    log_pred = model.predict(row)[0]
    price = np.expm1(log_pred)

    st.success(f"### Estimated nightly price: **NZD {price:,.0f}**")

    st.caption(
        "Based on Inside Airbnb NZ data (July 2025). "
        "Model: XGBoost · R² = 0.77 · MAE = NZD 70 on hold-out data. "
        "Estimate is most reliable for listings priced NZD 50–600/night."
    )

    # FIX: cast all values to str so Arrow can serialize the column
    with st.expander("📋 Inputs used for this estimate"):
        display_df = pd.DataFrame({
            "Field": [
                "Bedrooms", "Bathrooms", "Accommodates",
                "Property type", "District", "Amenities",
                "Min nights", "Instant bookable",
            ],
            "Value": [
                str(bedrooms), bath_label, str(accommodates),
                property_type, district, str(amenities),
                str(min_nights), "Yes" if instant else "No",
            ],
        })
        st.table(display_df)
