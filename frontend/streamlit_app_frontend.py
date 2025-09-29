import streamlit as st
import joblib
import pandas as pd

# Assume model saved as 'best_model.pkl'
model = joblib.load('best_model.pkl')  # Save your best model first

st.title('Airbnb Price Predictor NZ')

# Input fields (example, adjust toyour features)
room_type = st.selectbox('Room Type', ['Entire home/apt', 'Private room', 'Shared room', 'Hotel room'])
latitude = st.number_input('Latitude', -48.0, -34.0)
longitude = st.number_input('Longitude', 166.0, 179.0)
# Add more inputs...

# Prepare input df
input_data = pd.DataFrame({
    'room_type_Entire home/apt': [1 if room_type == 'Entire home/apt' else 0],
    # Encode similarly...
    'latitude': [latitude],
    'longitude': [longitude],
    # Add all features...
})

if st.button('Predict Price'):
    pred = model.predict(input_data)
    st.write(f'Predicted Price: ${np.expm1(pred[0]):.2f}')  # If log_price used