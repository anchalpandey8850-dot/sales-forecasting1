import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ==================================================
# STEP 1: PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Daily Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# STEP 2: FILE PATHS
# GitHub/Streamlit Cloud:
# Keep all required files beside this script.
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "linear_regression_sales_model.pkl"
FEATURES_PATH = BASE_DIR / "linear_regression_sales_features.pkl"
DATA_PATH = BASE_DIR / "New_Query_2026_10_22_10_01_37.csv"


# ==================================================
# STEP 3: LOAD MODEL AND FEATURES
# Removed cache_resource to avoid reusing an old model.
# ==================================================
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(FEATURES_PATH, "rb") as file:
        features = pickle.load(file)

    return model, list(features)


# ==================================================
# STEP 4: LOAD HISTORICAL SALES DATA
# ==================================================
@st.cache_data
def load_historical_data():
    df = pd.read_csv(DATA_PATH)

    required_columns = {"OrderDate", "Sales"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV file must contain OrderDate and Sales columns."
        )

    # Convert dates to timezone-naive timestamps
    df["OrderDate"] = (
        pd.to_datetime(
            df["OrderDate"],
            errors="coerce",
            utc=True
        )
        .dt.tz_convert(None)
        .dt.normalize()
    )

    df["Sales"] = pd.to_numeric(
        df["Sales"],
        errors="coerce"
    )

    df = df.dropna(subset=["OrderDate", "Sales"])

    if df.empty:
        raise ValueError(
            "No valid historical sales data found."
        )

    # Aggregate sales for each date
    daily_sales = (
        df.groupby("OrderDate")["Sales"]
        .sum()
        .sort_index()
    )

    # Fill missing calendar dates with zero.
    # Use this only if missing dates represent zero sales.
    full_dates = pd.date_range(
        start=daily_sales.index.min(),
        end=daily_sales.index.max(),
        freq="D"
    )

    daily_sales = daily_sales.reindex(
        full_dates,
        fill_value=0
    )

    daily_sales.index.name = "OrderDate"
    daily_sales.name = "Sales"

    return daily_sales


# ==================================================
# STEP 5: CREATE MODEL FEATURES
# These must match the training features.
# ==================================================
def create_features(history):
    return {
        "Lag_1": float(history.iloc[-1]),
        "Lag_7": float(history.iloc[-7]),
        "Lag_30": float(history.iloc[-30]),
        "Rolling_Mean_7": float(
            history.iloc[-7:].mean()
        ),
        "Rolling_Mean_30": float(
            history.iloc[-30:].mean()
        )
    }


# ==================================================
# STEP 6: GENERATE RECURSIVE FORECAST
# ==================================================
def generate_forecast(
    model,
    feature_names,
    historical_sales,
    start_date,
    forecast_days
):
    start_timestamp = (
        pd.Timestamp(start_date)
        .normalize()
    )

    # Use sales before the selected forecast date
    history = historical_sales.loc[
        historical_sales.index < start_timestamp
    ].copy()

    if len(history) < 30:
        raise ValueError(
            "At least 30 days of historical sales are required "
            "before the selected forecast date."
        )

    last_historical_date = (
        historical_sales.index.max().normalize()
    )

    # Prevent forecasting distant future dates using stale history
    if start_timestamp > (
        last_historical_date + pd.Timedelta(days=1)
    ):
        raise ValueError(
            f"Historical data ends on {last_historical_date.date()}. "
            "For future forecasting, select the day immediately "
            "after the last historical date or update the dataset. "
            "Historical dates can be selected for backtesting."
        )

    forecast_dates = pd.date_range(
        start=start_timestamp,
        periods=forecast_days,
        freq="D"
    )

    predictions = []

    for forecast_date in forecast_dates:

        # Generate the five input features
        feature_values = create_features(history)

        input_df = pd.DataFrame(
            [[feature_values[name] for name in feature_names]],
            columns=feature_names
        )

        # Predict sales with Linear Regression
        prediction = float(
            np.asarray(
                model.predict(input_df)
            ).reshape(-1)[0]
        )

        if not np.isfinite(prediction):
            raise ValueError(
                "The model generated an invalid prediction."
            )

        # Sales cannot be negative
        prediction = max(0.0, prediction)

        predictions.append(prediction)

        # Use the prediction to forecast the next day
        history.loc[forecast_date] = prediction

    forecast_df = pd.DataFrame({
        "ForecastDate": forecast_dates,
        "PredictedSales": predictions
    })

    return forecast_df


# ==================================================
# STEP 7: DASHBOARD HEADER
# ==================================================
st.title("📊 Daily Sales Forecasting Dashboard")

st.write(
    "Generate daily sales predictions and review monthly totals."
)

st.divider()


# ==================================================
# STEP 8: LOAD MODEL AND DATA
# ==================================================
try:
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    if not FEATURES_PATH.exists():
        st.error(f"Feature file not found: {FEATURES_PATH}")
        st.stop()

    if not DATA_PATH.exists():
        st.error(f"Historical CSV file not found: {DATA_PATH}")
        st.stop()

    model, feature_names = load_model()

    # Verify the saved model type
    if type(model).__name__ != "LinearRegression":
        st.error(
            f"Expected LinearRegression, but loaded "
            f"{type(model).__name__}. Please check the model file "
            "uploaded to GitHub."
        )
        st.stop()

    historical_sales = load_historical_data()

    expected_features = {
        "Lag_1",
        "Lag_7",
        "Lag_30",
        "Rolling_Mean_7",
        "Rolling_Mean_30"
    }

    if set(feature_names) != expected_features:
        st.error(
            "Saved feature names do not match the forecasting code."
        )
        st.write("Loaded features:", feature_names)
        st.stop()

    st.success(
        "Linear Regression model and historical data loaded successfully!"
    )

    # Show which model file and model type are in use
    st.caption(f"Model file: {MODEL_PATH.name}")
    st.caption(f"Loaded model: {type(model).__name__}")

except Exception as error:
    st.error("Unable to load the forecasting resources.")
    st.exception(error)
    st.stop()


# ==================================================
# STEP 9: HISTORICAL DATA INFORMATION
# ==================================================
historical_end_date = historical_sales.index.max().date()

st.caption(
    f"Historical data: "
    f"{historical_sales.index.min():%d %b %Y} "
    f"to {historical_sales.index.max():%d %b %Y}"
)

st.caption(
    f"Historical daily records: {len(historical_sales):,}"
)


# ==================================================
# STEP 10: FORECAST SETTINGS
# ==================================================
st.subheader("Forecast Settings")

col1, col2 = st.columns(2)

with col1:
    default_forecast_date = (
        pd.Timestamp(historical_end_date)
        + pd.Timedelta(days=1)
    ).date()

    start_date = st.date_input(
        "Select Forecast Start Date",
        value=default_forecast_date,
        min_value=pd.Timestamp("2000-01-01").date(),
        max_value=pd.Timestamp("2100-12-31").date(),
        format="YYYY-MM-DD"
    )

with col2:
    forecast_days = st.selectbox(
        "Forecast Duration",
        options=[30, 90, 180, 365],
        index=2,
        format_func=lambda days: f"{days} days"
    )

start_timestamp = pd.Timestamp(start_date)

end_timestamp = (
    start_timestamp
    + pd.Timedelta(days=forecast_days - 1)
)

st.info(
    f"**Forecast period:** {start_timestamp:%Y-%m-%d} "
    f"to {end_timestamp:%Y-%m-%d}\n\n"
    "The selected duration includes both the start and end dates."
)

st.caption(
    "For historical backtesting, the model uses sales before "
    "the selected start date. For future forecasting, recent "
    "historical data should be available."
)


# ==================================================
# STEP 11: GENERATE FORECAST
# ==================================================
if st.button(
    "Generate Sales Forecast",
    type="primary",
    use_container_width=True
):
    try:
        with st.spinner("Generating sales forecast..."):

            forecast_df = generate_forecast(
                model=model,
                feature_names=feature_names,
                historical_sales=historical_sales,
                start_date=start_date,
                forecast_days=forecast_days
            )

            forecast_df["Month"] = (
                forecast_df["ForecastDate"]
                .dt.to_period("M")
                .astype(str)
            )

            monthly_sales = (
                forecast_df.groupby(
                    "Month",
                    as_index=False
                )["PredictedSales"]
                .sum()
            )

            total_sales = forecast_df[
                "PredictedSales"
            ].sum()

            average_sales = forecast_df[
                "PredictedSales"
            ].mean()

        # ==========================================
        # STEP 12: FORECAST SUMMARY
        # ==========================================
        st.divider()
        st.subheader("Forecast Summary")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Forecast Days",
            f"{len(forecast_df):,}"
        )

        metric2.metric(
            "Total Predicted Sales",
            f"{total_sales:,.2f}"
        )

        metric3.metric(
            "Average Daily Sales",
            f"{average_sales:,.2f}"
        )

        # ==========================================
        # STEP 13: MONTHLY BAR CHART
        # ==========================================
        st.divider()
        st.subheader("Monthly Predicted Sales")

        monthly_chart = monthly_sales.set_index("Month")

        st.bar_chart(
            monthly_chart[["PredictedSales"]],
            x_label="Month",
            y_label="Predicted Sales"
        )

        st.dataframe(
            monthly_sales.rename(columns={
                "Month": "Forecast Month",
                "PredictedSales": "Monthly Predicted Sales"
            }),
            use_container_width=True,
            hide_index=True
        )

        # ==========================================
        # STEP 14: DAILY LINE CHART
        # ==========================================
        st.divider()
        st.subheader("Daily Sales Forecast")

        daily_chart = forecast_df.set_index("ForecastDate")

        st.line_chart(
            daily_chart[["PredictedSales"]],
            x_label="Forecast Date",
            y_label="Predicted Sales"
        )

        # ==========================================
        # STEP 15: DAILY FORECAST TABLE
        # ==========================================
        st.subheader("Daily Forecast Details")

        display_df = forecast_df[
            ["ForecastDate", "PredictedSales"]
        ].copy()

        display_df["ForecastDate"] = (
            display_df["ForecastDate"]
            .dt.strftime("%Y-%m-%d")
        )

        display_df = display_df.rename(columns={
            "ForecastDate": "Date",
            "PredictedSales": "Predicted Sales"
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ==========================================
        # STEP 16: DOWNLOAD FORECAST CSV
        # ==========================================
        csv_data = forecast_df[
            ["ForecastDate", "PredictedSales"]
        ].to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Forecast as CSV",
            data=csv_data,
            file_name="daily_sales_forecast.csv",
            mime="text/csv",
            use_container_width=True
        )

        # ==========================================
        # STEP 17: FORECAST LIMITATIONS
        # ==========================================
        st.warning(
            "This is an experimental recursive forecast. "
            f"The historical dataset ends on "
            f"{historical_end_date:%Y-%m-%d}. "
            "Predictions use historical sales patterns and "
            "should not be treated as guaranteed actual sales. "
            "Errors can accumulate over longer forecast horizons. "
            "The model's test R2 was approximately 0.003."
        )

    except Exception as error:
        st.error("Forecast generation failed.")
        st.exception(error)


# ==================================================
# STEP 18: FOOTER
# ==================================================
st.divider()

st.caption(
    "Daily Sales Forecasting Dashboard | "
    "Experimental Machine Learning Forecast"
)
