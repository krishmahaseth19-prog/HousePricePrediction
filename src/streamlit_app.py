import streamlit as st
import requests

# PAGE CONFIGURATION

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# TITLE

st.title("🏠 House Price Prediction System")
st.markdown(
    "Enter the house details below and click **Predict Price**."
)

# INPUT FORM

col1, col2 = st.columns(2)

with col1:

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=1,
        value=3
    )

    bathrooms = st.number_input(
        "Bathrooms",
        min_value=1.0,
        value=2.0
    )

    sqft_living = st.number_input(
        "Sqft Living",
        min_value=100,
        value=1800
    )

    sqft_lot = st.number_input(
        "Sqft Lot",
        min_value=500,
        value=5000
    )

    floors = st.number_input(
        "Floors",
        min_value=1.0,
        value=1.0
    )

    waterfront = st.selectbox(
        "Waterfront",
        [0, 1]
    )

    view = st.selectbox(
        "View",
        [0, 1, 2, 3, 4]
    )

with col2:

    condition = st.selectbox(
        "Condition",
        [1, 2, 3, 4, 5]
    )

    sqft_above = st.number_input(
        "Sqft Above",
        min_value=100,
        value=1800
    )

    sqft_basement = st.number_input(
        "Sqft Basement",
        min_value=0,
        value=0
    )

    yr_built = st.number_input(
        "Year Built",
        min_value=1900,
        value=2000
    )

    yr_renovated = st.number_input(
        "Year Renovated",
        min_value=0,
        value=0
    )

    year = st.number_input(
        "Year",
        value=2014
    )

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=5
    )

# LOCATION DETAILS

st.subheader("Location Information")

city = st.text_input(
    "City",
    value="Seattle"
)

statezip = st.text_input(
    "State ZIP",
    value="WA 98178"
)

country = st.text_input(
    "Country",
    value="USA"
)

# PREDICTION BUTTON

if st.button("Predict Price"):

    payload = {
        "bedrooms": int(bedrooms),
        "bathrooms": float(bathrooms),
        "sqft_living": int(sqft_living),
        "sqft_lot": int(sqft_lot),
        "floors": float(floors),
        "waterfront": int(waterfront),
        "view": int(view),
        "condition": int(condition),
        "sqft_above": int(sqft_above),
        "sqft_basement": int(sqft_basement),
        "yr_built": int(yr_built),
        "yr_renovated": int(yr_renovated),
        "city": city,
        "statezip": statezip,
        "country": country,
        "year": int(year),
        "month": int(month)
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        if response.status_code == 200:

            result = response.json()

            st.success(
                f"💰 Predicted House Price: ${result['predicted_price']:,.2f}"
            )

        else:

            st.error(
                f"API Error: {response.status_code}"
            )

            st.write(response.text)

    except Exception as e:

        st.error(
            "Could not connect to FastAPI server."
        )

        st.write(str(e))