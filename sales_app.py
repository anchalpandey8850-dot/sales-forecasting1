
import streamlit as st
import pandas as pd
import requests

# ==================================================
# STEP 1: PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Daily Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# STEP 2: FASTAPI BACKEND URL
# ==================================================
API_BASE_URL = "https://sales-forecasting1-5.onrender.com"

# ==================================================
# STEP 3: DASHBOARD HEADER
# ==================================================
st.title("📊 Daily Sales Forecasting Dashboard")

st.write(
    "Generate daily sales predictions and review monthly totals "
    "using the deployed FastAPI forecasting service."
)

st.divider()

# ==================================================
# ==================================================
# STEP 4: CHECK BACKEND STATUS
# ==================================================
try:
    health_response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=60
    )

    st.write("HTTP Status Code:", health_response.status_code)
    st.write(
        "Content-Type:",
        health_response.headers.get("content-type")
    )

    st.write("Raw API Response:", health_response.text[:500])

    health_response.raise_for_status()

    try:
        health_data = health_response.json()

        if health_data.get("status") == "healthy":
            st.success("FastAPI backend is connected and healthy.")
        else:
            st.warning(f"Backend response: {health_data}")

    except ValueError:
        st.error("Backend did not return valid JSON.")

except requests.exceptions.RequestException as error:
    st.error("Unable to connect to FastAPI backend.")
    st.exception(error)
# ==================================================
# STEP 5: FORECAST SETTINGS
# ==================================================
st.subheader("Forecast Settings")

forecast_days = st.selectbox(
    "Forecast Duration",
    options=[30, 90, 180, 365],
    index=2,
    format_func=lambda days: f"{days} days"
)

st.info(
    f"**Forecast duration:** {forecast_days} days\n\n"
    "The API automatically starts from the day after "
    "the last date in its historical sales dataset."
)

st.caption(
    "Forecast dates are taken directly from the API response."
)

# ==================================================
# STEP 6: GENERATE FORECAST
# ==================================================
if st.button(
    "Generate Sales Forecast",
    type="primary",
    use_container_width=True
):

    try:
        with st.spinner(
            "Connecting to FastAPI and generating forecast..."
        ):

            response = requests.get(
                f"{API_BASE_URL}/forecast",
                params={"days": forecast_days},
                timeout=180
            )

            response.raise_for_status()
            result = response.json()

        # Validate API response
        required_keys = {
            "forecast_days",
            "forecast_start_date",
            "forecast_end_date",
            "forecast"
        }

        missing_keys = required_keys - set(result.keys())

        if missing_keys:
            raise ValueError(
                f"Missing API fields: {missing_keys}"
            )

        if not result["forecast"]:
            raise ValueError("API returned an empty forecast.")

        # Convert response into DataFrame
        forecast_df = pd.DataFrame(result["forecast"])

        expected_columns = {"date", "predicted_sales"}
        missing_columns = expected_columns - set(forecast_df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing forecast columns: {missing_columns}"
            )

        forecast_df["ForecastDate"] = pd.to_datetime(
            forecast_df["date"],
            errors="raise"
        )

        forecast_df["PredictedSales"] = pd.to_numeric(
            forecast_df["predicted_sales"],
            errors="raise"
        )

        if forecast_df["PredictedSales"].isna().any():
            raise ValueError("Invalid sales predictions received.")

        forecast_df = forecast_df.sort_values(
            "ForecastDate"
        ).reset_index(drop=True)

        # Month-wise aggregation
        forecast_df["Month"] = (
            forecast_df["ForecastDate"]
            .dt.to_period("M")
            .astype(str)
        )

        monthly_sales = (
            forecast_df.groupby("Month", as_index=False)
            ["PredictedSales"].sum()
        )

        # Summary metrics
        total_sales = forecast_df["PredictedSales"].sum()
        average_sales = forecast_df["PredictedSales"].mean()

        # ==================================================
        # STEP 7: FORECAST SUMMARY
        # ==================================================
        st.divider()
        st.subheader("Forecast Summary")

        st.caption(
            f"Forecast period: {result['forecast_start_date']} "
            f"to {result['forecast_end_date']}"
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Forecast Days",
            f"{int(result['forecast_days']):,}"
        )

        metric2.metric(
            "Total Predicted Sales",
            f"{total_sales:,.2f}"
        )

        metric3.metric(
            "Average Daily Sales",
            f"{average_sales:,.2f}"
        )

        # ==================================================
        # STEP 8: MONTHLY BAR CHART
        # ==================================================
        st.divider()
        st.subheader("Monthly Predicted Sales")

        monthly_chart = monthly_sales.set_index("Month")

        st.bar_chart(
            monthly_chart[["PredictedSales"]],
            x_label="Month",
            y_label="Predicted Sales"
        )

        st.dataframe(
            monthly_sales.rename(
                columns={
                    "Month": "Forecast Month",
                    "PredictedSales": "Monthly Predicted Sales"
                }
            ),
            use_container_width=True,
            hide_index=True
        )

        # ==================================================
        # STEP 9: DAILY LINE CHART
        # ==================================================
        st.divider()
        st.subheader("Daily Sales Forecast")

        daily_chart = forecast_df.set_index("ForecastDate")

        st.line_chart(
            daily_chart[["PredictedSales"]],
            x_label="Forecast Date",
            y_label="Predicted Sales"
        )

        # ==================================================
        # STEP 10: DAILY FORECAST TABLE
        # ==================================================
        st.divider()
        st.subheader("Daily Forecast Details")

        display_df = forecast_df[
            ["ForecastDate", "PredictedSales"]
        ].rename(
            columns={
                "ForecastDate": "Date",
                "PredictedSales": "Predicted Daily Sales"
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ==================================================
        # STEP 11: DOWNLOAD FORECAST
        # ==================================================
        st.download_button(
            label="Download Forecast CSV",
            data=display_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="sales_forecast.csv",
            mime="text/csv",
            use_container_width=True
        )

    # ==================================================
    # ERROR HANDLING
    # ==================================================
    except requests.exceptions.HTTPError as error:
        try:
            details = response.json().get(
                "detail",
                response.text
            )
        except Exception:
            details = response.text

        st.error(f"FastAPI returned an error: {details}")

    except requests.exceptions.Timeout:
        st.error(
            "Request timed out. Render may be waking up. "
            "Please try again."
        )

    except requests.exceptions.RequestException as error:
        st.error(f"Unable to reach FastAPI: {error}")

    except (ValueError, KeyError, TypeError) as error:
        st.error(f"Unable to process forecast response: {error}")

    except Exception as error:
        st.error("An unexpected error occurred.")
        st.exception(error)
