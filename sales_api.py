
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

with open("linear_regression_sales_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

with open("linear_regression_sales_features.pkl", "rb") as file:
    feature_names = pickle.load(file)
    
# Load historical sales data
history_df = pd.read_csv("daily_sales_history.csv")

date_col = next(
    c for c in history_df.columns if c.lower() == "date"
)

sales_col = next(
    c for c in history_df.columns
    if c.lower() in ["daily_sales", "dailysales"]
)

history_df[date_col] = pd.to_datetime(history_df[date_col])
history_df = history_df.sort_values(date_col)

sales_history = history_df[sales_col].astype(float).tolist()
last_date = history_df[date_col].max()

# Create FastAPI application
app = FastAPI(title="Sales Forecasting API")


@app.get("/")
def home():
    return {"message": "Sales Forecasting API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/forecast")
def forecast_sales(days: int = Query(default=91, ge=1, le=365)):

    if len(sales_history) < 30:
        raise HTTPException(
            status_code=400,
            detail="At least 30 days of historical sales are required"
        )

    history = sales_history.copy()
    predictions = []

    for i in range(1, days + 1):

        input_features = {
            "Lag_1": history[-1],
            "Lag_7": history[-7],
            "Lag_30": history[-30],
            "Rolling_Mean_7": sum(history[-7:]) / 7,
            "Rolling_Mean_30": sum(history[-30:]) / 30
        }

        input_df = pd.DataFrame(
            [[input_features[f] for f in feature_names]],
            columns=feature_names
        )

        prediction = float(loaded_model.predict(input_df)[0])
        history.append(prediction)

        forecast_date = last_date + pd.Timedelta(days=i)

        predictions.append({
            "date": forecast_date.strftime("%Y-%m-%d"),
            "predicted_sales": round(prediction, 2)
        })

    return {
        "forecast_days": days,
        "forecast_start_date": predictions[0]["date"],
        "forecast_end_date": predictions[-1]["date"],
        "forecast": predictions
    }
