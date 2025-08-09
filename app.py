import pandas as pd
import pickle as pk
import streamlit as st

# Load the saved model
with open(r'C:\Users\Dhairya\Desktop\ML_01\house-prices-advanced-regression-techniques\House_prediction_model.pkl', 'rb') as file:
    model = pk.load(file)

# Streamlit UI
st.header('House Price Predictor Using Linear Regression')

# User Inputs
sqft = st.number_input('Enter Total Sqft (GrLivArea)', min_value=0)
beds = st.number_input('Enter No of Bedrooms', min_value=0)
full_bath = st.number_input('Enter No of Full Bathrooms', min_value=0)
half_bath = st.number_input('Enter No of Half Bathrooms', min_value=0)

# Create DataFrame in same order as training
input_df = pd.DataFrame(
    [[sqft, beds, full_bath, half_bath]],
    columns=['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']
)

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Price: {prediction[0]:,.2f}")
