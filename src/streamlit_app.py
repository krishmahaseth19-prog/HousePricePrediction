
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

# LIVE RENDER FASTAPI URL
API_URL = "https://house-price-prediction-api-2q84.onrender.com/predict"

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .price-card {
        background: linear-gradient(120deg, #142d4e, #176b87);
        padding: 30px;
        border-radius: 16px;
        margin: 15px 0 25px 0;
    }

    .price-label {
        color: #d6eaf8;
        font-size: 16px;
    }

    .price-value {
        color: white;
        font-size: 40px;
        font-weight: 800;
        margin: 8px 0;
    }

    .price-description {
        color: #d6eaf8;
        font-size: 13px;
    }

    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        padding: 15px;
        border-radius: 12px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🏠 House Price Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Machine Learning Powered House Valuation Dashboard</div>',
    unsafe_allow_html=True
)


# ==========================================
# TABS
# ==========================================

tab1, tab2 = st.tabs([
    "🏠 House Price Prediction",
    "📊 Model Performance"
])


# ==========================================
# TAB 1: HOUSE PRICE PREDICTION
# ==========================================

with tab1:

    st.header("Enter House Details")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        # ----------------------------------
        # PROPERTY DETAILS
        # ----------------------------------

        with col1:

            st.subheader("🏡 Property Details")

            bedrooms = st.number_input(
                "Bedrooms", 0, 20, 3
            )

            bathrooms = st.number_input(
                "Bathrooms", 0.0, 20.0, 2.0,
                step=0.5
            )

            sqft_living = st.number_input(
                "Living Area (sqft)", 1,
                value=1800
            )

            sqft_lot = st.number_input(
                "Lot Area (sqft)", 1,
                value=5000
            )

            floors = st.number_input(
                "Floors", 1.0, 10.0, 1.0,
                step=0.5
            )

            waterfront = st.selectbox(
                "Waterfront", [0, 1]
            )

            view = st.number_input(
                "View", 0, 4, 0
            )

            condition = st.number_input(
                "Condition", 1, 5, 3
            )

        # ----------------------------------
        # BUILDING DETAILS
        # ----------------------------------

        with col2:

            st.subheader("🏢 Building Details")

            sqft_above = st.number_input(
                "Area Above Ground (sqft)",
                0, value=1500
            )

            sqft_basement = st.number_input(
                "Basement Area (sqft)",
                0, value=300
            )

            yr_built = st.number_input(
                "Year Built", 1800, 2026, 2000
            )

            yr_renovated = st.number_input(
                "Year Renovated (0 if not renovated)",
                0, 2026, 0
            )

            year = st.number_input(
                "Sale Year", 1900, 2026, 2014
            )

            month = st.number_input(
                "Sale Month", 1, 12, 6
            )

        # ----------------------------------
        # LOCATION DETAILS
        # ----------------------------------

        st.divider()
        st.subheader("📍 Location Details")

        loc1, loc2, loc3 = st.columns(3)

        with loc1:
            city = st.text_input(
                "City", "Seattle"
            )

        with loc2:
            statezip = st.text_input(
                "State ZIP", "WA 98103"
            )

        with loc3:
            country = st.text_input(
                "Country", "USA"
            )

        submitted = st.form_submit_button(
            "🔍 Predict House Price",
            use_container_width=True
        )

    # ======================================
    # PREDICTION REQUEST
    # ======================================

    if submitted:

        input_data = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "waterfront": waterfront,
            "view": view,
            "condition": condition,
            "sqft_above": sqft_above,
            "sqft_basement": sqft_basement,
            "yr_built": yr_built,
            "yr_renovated": yr_renovated,
            "city": city,
            "statezip": statezip,
            "country": country,
            "year": year,
            "month": month
        }

        try:

            with st.spinner("Analyzing property and predicting price..."):

                response = requests.post(
                    API_URL,
                    json=input_data,
                    timeout=60
                )

            response.raise_for_status()

            result = response.json()

            prediction = result.get(
                "predicted_price",
                result.get(
                    "prediction",
                    result.get("price")
                )
            )

            if prediction is None:
                st.error("Prediction value not found in API response.")

            else:

                st.session_state["latest_prediction"] = float(prediction)
                st.session_state["latest_input"] = input_data

                st.success("House price predicted successfully!")

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to the live prediction API. "
                "Please check whether the Render service is running."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The prediction API took too long to respond. "
                "Please try again."
            )

        except requests.exceptions.HTTPError as e:

            st.error(f"API returned an error: {e}")

            try:
                st.json(response.json())
            except ValueError:
                st.write(response.text)

        except requests.exceptions.RequestException as e:

            st.error(f"API request failed: {e}")

        except (ValueError, TypeError) as e:

            st.error(f"Prediction error: {e}")

    # ======================================
    # LIVE PREDICTION DASHBOARD
    # ======================================

    if "latest_prediction" in st.session_state:

        prediction = st.session_state["latest_prediction"]
        house = st.session_state["latest_input"]

        st.divider()

        st.subheader("🏡 House Valuation Result")

        st.markdown(
            f"""
            <div class="price-card">
                <div class="price-label">Estimated Market Price</div>
                <div class="price-value">${prediction:,.2f}</div>
                <div class="price-description">
                    Extra Trees Regressor · Machine Learning Prediction
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------
        # PROPERTY SUMMARY
        # ----------------------------------

        st.subheader("Property Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("🛏️ Bedrooms", house["bedrooms"])
        c2.metric("🛁 Bathrooms", house["bathrooms"])
        c3.metric("📐 Living Area", f'{house["sqft_living"]:,} sqft')
        c4.metric("🏢 Floors", house["floors"])

        st.divider()

        # ----------------------------------
        # PROPERTY AREA CHART
        # ----------------------------------

        st.subheader("📐 Property Area Analysis")

        area_names = [
            "Living Area",
            "Above Ground",
            "Basement",
            "Lot Area"
        ]

        area_values = [
            house["sqft_living"],
            house["sqft_above"],
            house["sqft_basement"],
            house["sqft_lot"]
        ]

        fig_area = go.Figure()

        fig_area.add_trace(go.Bar(
            x=area_names,
            y=area_values,
            text=[f"{v:,} sqft" for v in area_values],
            textposition="outside",
            marker_color=[
                "#38bdf8",
                "#818cf8",
                "#a78bfa",
                "#34d399"
            ],
            hovertemplate="%{x}<br>%{y:,} sqft<extra></extra>"
        ))

        fig_area.update_layout(
            title="Distribution of Property Areas",
            xaxis_title="Area Type",
            yaxis_title="Area (sqft)",
            template="plotly_dark",
            height=420,
            showlegend=False
        )

        st.plotly_chart(
            fig_area,
            use_container_width=True
        )

        # ----------------------------------
        # HOUSE FEATURE CHART
        # ----------------------------------

        st.subheader("🏠 House Feature Analysis")

        feature_names = [
            "Bedrooms",
            "Bathrooms",
            "Floors",
            "View",
            "Condition"
        ]

        feature_values = [
            house["bedrooms"],
            house["bathrooms"],
            house["floors"],
            house["view"],
            house["condition"]
        ]

        fig_features = go.Figure()

        fig_features.add_trace(go.Bar(
            x=feature_names,
            y=feature_values,
            text=feature_values,
            textposition="outside",
            marker_color="#38bdf8",
            hovertemplate="%{x}: %{y}<extra></extra>"
        ))

        fig_features.update_layout(
            title="Property Characteristics",
            yaxis_title="Value",
            template="plotly_dark",
            height=400,
            showlegend=False
        )

        st.plotly_chart(
            fig_features,
            use_container_width=True
        )

        # ----------------------------------
        # LOCATION AND BUILDING INFORMATION
        # ----------------------------------

        st.divider()
        st.subheader("📍 Property Information")

        info1, info2, info3 = st.columns(3)

        with info1:
            st.markdown("**Location**")
            st.write(f"🏙️ {house['city']}")
            st.write(f"📮 {house['statezip']}")
            st.write(f"🌎 {house['country']}")

        with info2:
            st.markdown("**Building Information**")
            st.write(f"🏗️ Built: {house['yr_built']}")
            st.write(
                f"🔨 Renovated: {house['yr_renovated'] or 'No'}"
            )
            st.write(
                f"📐 Above ground: {house['sqft_above']:,} sqft"
            )

        with info3:
            st.markdown("**Sale Information**")
            st.write(f"📅 Year: {house['year']}")
            st.write(f"📆 Month: {house['month']}")
            st.write(
                f"🌊 Waterfront: {'Yes' if house['waterfront'] else 'No'}"
            )

        st.caption(
            "This is a machine-learning estimate, not a formal property valuation."
        )


# ==========================================
# TAB 2: MODEL PERFORMANCE
# ==========================================

with tab2:

    st.header("📊 Model Performance Dashboard")

    st.write(
        "Evaluation of the trained Extra Trees Regression model."
    )

    metrics_file = BASE_DIR / "prediction_errors.csv"

    if metrics_file.exists():

        try:

            df = pd.read_csv(metrics_file)

            if {"actual", "predicted"}.issubset(df.columns):

                actual = df["actual"]
                predicted = df["predicted"]

                errors = actual - predicted

                # ----------------------------------
                # METRICS
                # ----------------------------------

                mae = mean_absolute_error(actual, predicted)
                mse = mean_squared_error(actual, predicted)
                rmse = np.sqrt(mse)
                r2 = r2_score(actual, predicted)

                st.subheader("Model Evaluation Metrics")

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("MAE", f"${mae:,.2f}")
                c2.metric("MSE", f"${mse:,.2f}")
                c3.metric("RMSE", f"${rmse:,.2f}")
                c4.metric("R² Score", f"{r2:.4f}")

                st.divider()

                # ----------------------------------
                # ERROR DISTRIBUTION
                # ----------------------------------

                st.subheader("1. Prediction Error Distribution")

                fig1 = go.Figure()

                fig1.add_trace(go.Histogram(
                    x=errors,
                    nbinsx=30,
                    name="Prediction Errors"
                ))

                fig1.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color="red"
                )

                fig1.update_layout(
                    title="Distribution of Prediction Errors",
                    xaxis_title="Actual - Predicted ($)",
                    yaxis_title="Frequency",
                    template="plotly_dark",
                    height=420
                )

                st.plotly_chart(
                    fig1,
                    use_container_width=True
                )

                st.divider()

                # ----------------------------------
                # RESIDUAL PLOT
                # ----------------------------------

                st.subheader("2. Residual Error Plot")

                fig2 = go.Figure()

                fig2.add_trace(go.Scatter(
                    x=predicted,
                    y=errors,
                    mode="markers",
                    name="Residuals",
                    marker=dict(
                        size=7,
                        opacity=0.65,
                        color="#38bdf8"
                    )
                ))

                fig2.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="red"
                )

                fig2.update_layout(
                    title="Predicted Price vs Residual Error",
                    xaxis_title="Predicted Price ($)",
                    yaxis_title="Actual - Predicted ($)",
                    template="plotly_dark",
                    height=450
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

                st.divider()

                # ----------------------------------
                # ACTUAL VS PREDICTED
                # ----------------------------------

                st.subheader("3. Actual vs Predicted Prices")

                fig3 = go.Figure()

                fig3.add_trace(go.Scatter(
                    x=actual,
                    y=predicted,
                    mode="markers",
                    name="Predictions",
                    marker=dict(
                        size=7,
                        opacity=0.65,
                        color="#38bdf8"
                    )
                ))

                minimum = min(actual.min(), predicted.min())
                maximum = max(actual.max(), predicted.max())

                fig3.add_trace(go.Scatter(
                    x=[minimum, maximum],
                    y=[minimum, maximum],
                    mode="lines",
                    name="Perfect Prediction",
                    line=dict(
                        color="red",
                        width=2,
                        dash="dash"
                    )
                ))

                fig3.update_layout(
                    title="Actual vs Predicted House Prices",
                    xaxis_title="Actual Price ($)",
                    yaxis_title="Predicted Price ($)",
                    template="plotly_dark",
                    height=550
                )

                st.plotly_chart(
                    fig3,
                    use_container_width=True
                )

                st.divider()

                # ----------------------------------
                # ERROR TABLE
                # ----------------------------------

                st.subheader("4. Prediction Error Details")

                st.dataframe(
                    df.head(50),
                    use_container_width=True
                )

            else:

                st.error(
                    "CSV must contain 'actual' and 'predicted' columns."
                )

        except Exception as e:

            st.error(f"Could not load model performance: {e}")

    else:

        st.warning(
            "Prediction results not found. Run python main.py first."
        )

    st.caption(
        "House Price Prediction System | Extra Trees Regression"
    )
