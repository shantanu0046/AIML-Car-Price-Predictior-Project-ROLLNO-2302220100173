import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the saved model and feature names (Backend Logic)
model = joblib.load('model.pkl')
features = joblib.load('features.pkl')

# 2. Build the UI (Frontend)
st.title("🚗 Car Price Prediction App")
st.write("Welcome! Enter the car's details below to estimate its selling price.")

# We will store the user's answers in this dictionary
user_input = {}

# 3. Dynamically create an input box for every single feature the model needs
st.write("### Car Features")
for feature in features:
    # This creates a number box for every feature (Year, Power, Brands, etc.)
    user_input[feature] = st.number_input(f"{feature}", value=0.0)

# 4. The Prediction Button
if st.button("Predict Price"):
    # Convert the user's answers into a Pandas DataFrame (just like our training data)
    input_df = pd.DataFrame([user_input])
    
    # Make the prediction
    log_predicted_price = model.predict(input_df)
    
    # Reverse the log transformation using np.exp to get real Rupees!
    actual_price = np.exp(log_predicted_price[0])
    
    # Show the final result on the screen in a nice green box
    st.success(f"Estimated Selling Price: ₹ {actual_price:,.2f}")