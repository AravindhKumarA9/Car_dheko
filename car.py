import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the pickle file containing recommendations
model_location = "D:/I-P/Data_Science/Guvi/Project/car/Car/car_price_model.pkl"
with open(model_location, 'rb') as file:
    model = pickle.load(file)

fl = pd.read_csv('car_dheko_filled.csv')

# Streamlit app
st.title("Car Price Prediction App")
st.write("Enter the details of the car to predict its price.")

# Input fields for the user
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2024, step=1, value=2010)
mileage = st.number_input("Mileage (in km/l or mi/gal)", min_value=0.0, step=0.1, value=15.0)
engine = st.number_input("Engine Size (in CC)", min_value=500, step=50, value=1500)
power = st.number_input("Power (in HP)", min_value=20, step=5, value=100)
seats = st.number_input("Number of Seats", min_value=2, max_value=8, step=1, value=5)

# A predict button
if st.button("Predict Price"):
    # Prepare the input data for the model
    input_data = np.array([[year, mileage, engine, power, seats]])
    
    # Predict the price using the loaded model
    predicted_price = model.predict(input_data)
    
    # Display the predicted price
    st.success(f"The predicted price of the car is: ₹{predicted_price[0]:,.2f}")

st.write("Note: This is a demo app. The predictions depend on the model's training data.")
