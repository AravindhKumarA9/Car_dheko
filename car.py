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

# Clean and preprocess dataset (remove commas from `Kms_Driven` and convert to integer if needed)
fl['Kms_Driven'] = fl['Kms_Driven'].fillna(0).astype(str).str.replace(',', '').astype(int)

# Streamlit app
st.title("Car Price Prediction App")
st.write("Enter the details of the car to predict its price.")

# Sidebar for filtering options
st.sidebar.header('Filter Options')

# Filter data dynamically based on user selections
city = st.sidebar.selectbox("Select the city:", options=fl['city'].unique())
filtered_data = fl[fl['city'] == city]

body_type = st.sidebar.selectbox("Select body type:", options=filtered_data['bt'].unique())
filtered_data = filtered_data[filtered_data['bt'] == body_type]

year = st.sidebar.selectbox("Select the year:", options=sorted(filtered_data['modelYear'].unique()))
filtered_data = filtered_data[filtered_data['modelYear'] == year]

oem = st.sidebar.selectbox("Select OEM (Manufacturer):", options=filtered_data['oem'].unique())
filtered_data = filtered_data[filtered_data['oem'] == oem]

model_name = st.sidebar.selectbox("Select Car Model:", options=filtered_data['model'].unique())
filtered_data = filtered_data[filtered_data['model'] == model_name]

Mileage = st.sidebar.selectbox("Select Mileage (in km/l):", options=sorted(filtered_data['Mileage'].unique()))
filtered_data = filtered_data[filtered_data['Mileage'] == Mileage]

engine_type = st.sidebar.selectbox("Select the engine type:", options=sorted(filtered_data['Engine Type'].unique()))
filtered_data = filtered_data[filtered_data['Engine Type'] == engine_type]

engine_displacement = st.sidebar.selectbox("Select Engine Displacement (in CC):", options=sorted(filtered_data['Engine_Displacement'].unique()))
filtered_data = filtered_data[filtered_data['Engine_Displacement'] == engine_displacement]

seating_capacity = st.sidebar.selectbox("Select seating capacity:", options=sorted(filtered_data['Seating_Capacity'].unique()))
filtered_data = filtered_data[filtered_data['Seating_Capacity'] == seating_capacity]

ownership = st.sidebar.selectbox("Select ownership type:", options=filtered_data['Ownership'].unique())
filtered_data = filtered_data[filtered_data['Ownership'] == ownership]

kms_driven = st.sidebar.number_input("Enter kilometers driven:", min_value=0, value=10000)

fuel_type = st.sidebar.selectbox("Select fuel type:", options=filtered_data['Fuel Type'].unique())
filtered_data = filtered_data[filtered_data['Fuel Type'] == fuel_type]

transmission = st.sidebar.selectbox("Select transmission type:", options=filtered_data['Transmission'].unique())
filtered_data = filtered_data[filtered_data['Transmission'] == transmission]

max_power = st.sidebar.selectbox("Select Max Power (in BHP):", options=filtered_data['Max Power'].unique())
filtered_data = filtered_data[filtered_data['Max Power'] == max_power]

# Predict button
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
                                seating_capacity, Mileage, max_power, engine_displacement, kms_driven]])

        # Predict price
        predicted_price = model.predict(input_data)[0]
        st.success(f"The predicted price of the car is: ₹{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

st.write("Note: This is a demo app. The predictions depend on the model's training data.")