import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the pickle file containing the model
model_location = "D:/I-P/Data_Science/Guvi/Project/car/Car/car_price_model.pkl"
with open(model_location, 'rb') as file:
    model = pickle.load(file)

# Load the dataset for dropdown options
fl = pd.read_csv('car_dheko_filled.csv')

# Streamlit app
st.title("Car Price Prediction App")
st.write("Enter the details of the car to predict its price.")

# Sidebar for filtering options
st.sidebar.header('Filter Options')

# Input fields for the user
body_type = st.sidebar.selectbox("Select body type:", options=fl['bt'].unique())
city = st.sidebar.selectbox("Select the city:", options=fl['city'].unique())
year = st.sidebar.selectbox("Select the year:", options=sorted(fl['modelYear'].unique()))
mileage = st.sidebar.slider("Select mileage (in km/l):", float(fl['Mileage'].min()), float(fl['Mileage'].max()))
engine_type = st.sidebar.selectbox("Select the engine type:", options=sorted(fl['Engine Type'].unique()))
engine_displacement = st.sidebar.slider("Select engine displacement (in CC):", 
                                         float(fl['Engine_Displacement'].min()), 
                                         float(fl['Engine_Displacement'].max()))
seating_capacity = st.sidebar.selectbox("Select seating capacity:", options=sorted(fl['Seating_Capacity'].unique()))
kms_driven = st.sidebar.number_input("Enter kilometers driven:", min_value=0, value=5000)

fuel_type = st.sidebar.selectbox("Select fuel type:", options=fl['Fuel Type'].unique())
ownership = st.sidebar.selectbox("Select ownership type:", options=fl['Ownership'].unique())
transmission = st.sidebar.selectbox("Select transmission type:", options=fl['Transmission'].unique())
# Add additional inputs based on the missing features
oem = st.sidebar.selectbox("Select OEM (Manufacturer):", options=fl['oem'].unique())
model_name = st.sidebar.selectbox("Select Car Model:", options=fl['model'].unique())
max_power = st.sidebar.slider("Enter Max Power (in BHP):", float(fl['Max Power'].min()), float(fl['Max Power'].max()))

# Update input preparation
if st.button("Predict Price"):
    try:
        # Encode categorical variables
        body_type_encoded = fl['bt'].astype('category').cat.categories.get_loc(body_type)
        city_encoded = fl['city'].astype('category').cat.categories.get_loc(city)
        engine_type_encoded = fl['Engine Type'].astype('category').cat.categories.get_loc(engine_type)
        fuel_type_encoded = fl['Fuel Type'].astype('category').cat.categories.get_loc(fuel_type)
        ownership_encoded = fl['Ownership'].astype('category').cat.categories.get_loc(ownership)
        transmission_encoded = fl['Transmission'].astype('category').cat.categories.get_loc(transmission)
        oem_encoded = fl['oem'].astype('category').cat.categories.get_loc(oem)
        model_name_encoded = fl['model'].astype('category').cat.categories.get_loc(model_name)

        # Prepare input data in the correct order
        input_data = np.array([[city_encoded, body_type_encoded, oem_encoded, model_name_encoded, year, 
                                fuel_type_encoded, ownership_encoded, transmission_encoded, engine_type_encoded, 
                                seating_capacity, mileage, max_power, engine_displacement, kms_driven]])

        # Predict price
        predicted_price = model.predict(input_data)[0]
        st.success(f"The predicted price of the car is: ₹{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")


st.write("Note: This is a demo app. The predictions depend on the model's training data.")
