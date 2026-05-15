import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.title("Housing Price Predictor")


longitude = st.slider("Longitude", min_value=-124.3500, max_value=-114.3100)
latitude = st.slider("Latitude", min_value=32.5400, max_value=41.9500)
housing_age = st.slider("Housing Median Age", min_value=1.0, max_value=52.0)
total_rooms = st.slider("Total Rooms", min_value=2.0, max_value=39320.0)
total_bedrooms = st.slider("Total Bedrooms",min_value=1.0, max_value=6445.0)
population = st.slider("Population", min_value=3.0, max_value=35682.0)
households = st.slider("Households", min_value=1.0, max_value=6082.0)
median_income = st.slider("Median Income", min_value=0.4999, max_value=15.0)
median_house_value = st.slider("Median House Value", min_value=14999.0, max_value=500001.0)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"]
)

if st.button("Predict Price"):

    payload = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
        "Median House Value": median_house_value
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)


        if response.status_code == 422:
            errors = response.json().get("detail", [])
            for err in errors:
                field = err.get("loc", [""])[-1]
                msg = err.get("msg", "")
                st.error(f" {field}: {msg}")


        elif response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Price: #{result['prediction']:,.2f}")


        else:
            st.error(f"API error: {response.status_code} - {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Is FastAPI running?")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
