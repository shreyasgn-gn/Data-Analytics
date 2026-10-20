import pandas as pd

# Load dataset
df = pd.read_csv(
    r"E:\data analytics project\Sales_Analytics_Project\sales_customer_analytics_final.csv"
)

# Convert Order_Date to datetime
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Create monthly sales
monthly_sales = (
    df.set_index("Order_Date")
      .resample("ME")["Sales"]
      .sum()
      .reset_index()
)

print(monthly_sales)


from sklearn.linear_model import LinearRegression

# Create a time index
monthly_sales["Month_Number"] = range(1, len(monthly_sales) + 1)

# Features and target
X = monthly_sales[["Month_Number"]]
y = monthly_sales["Sales"]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict existing months
monthly_sales["Predicted_Sales"] = model.predict(X)

print(monthly_sales)

# Forecast next 3 months
future_months = pd.DataFrame({
    "Month_Number": [13, 14, 15]
})

future_months["Forecasted_Sales"] = model.predict(
    future_months[["Month_Number"]]
)

print("\nFuture Sales Forecast:")
print(future_months)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Actual and predicted sales
actual = monthly_sales["Sales"]
predicted = monthly_sales["Predicted_Sales"]

# Evaluation metrics
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print("\nModel Evaluation:")
print(f"MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R² Score: {r2:.4f}")


from sklearn.ensemble import RandomForestRegressor

# Create forecasting features
monthly_sales["Month"] = monthly_sales["Order_Date"].dt.month
monthly_sales["Quarter"] = monthly_sales["Order_Date"].dt.quarter

# Features
X = monthly_sales[["Month_Number", "Month", "Quarter"]]
y = monthly_sales["Sales"]

# Train Random Forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

rf_model.fit(X, y)

# Historical predictions
monthly_sales["RF_Predicted_Sales"] = rf_model.predict(X)

print("\nRandom Forest Predictions:")
print(
    monthly_sales[
        ["Order_Date", "Sales", "RF_Predicted_Sales"]
    ]
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Features
X = monthly_sales[["Month_Number", "Month", "Quarter"]]
y = monthly_sales["Sales"]

# First 9 months = training
X_train = X.iloc[:9]
y_train = y.iloc[:9]

# Last 3 months = testing
X_test = X.iloc[9:]
y_test = y.iloc[9:]

# Create model
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train
rf_model.fit(X_train, y_train)

# Predict test period
test_predictions = rf_model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, test_predictions)
rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
r2 = r2_score(y_test, test_predictions)

print("\nTest Results:")
print(f"MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R² Score: {r2:.4f}")

print("\nActual vs Predicted:")
for date, actual_value, predicted_value in zip(
    monthly_sales["Order_Date"].iloc[9:],
    y_test,
    test_predictions
):
    print(
        f"{date.strftime('%B %Y')}: "
        f"Actual = {actual_value:,.2f}, "
        f"Predicted = {predicted_value:,.2f}"
    )
    
    # 3-month moving average
monthly_sales["Moving_Average_3"] = (
    monthly_sales["Sales"]
    .rolling(window=3)
    .mean()
)

print("\nMonthly Sales with 3-Month Moving Average:")
print(
    monthly_sales[
        ["Order_Date", "Sales", "Moving_Average_3"]
    ]
)

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Use moving average to predict months 4-12
actual = monthly_sales["Sales"].iloc[3:]
predicted = monthly_sales["Moving_Average_3"].iloc[3:]

mae_ma = mean_absolute_error(actual, predicted)
rmse_ma = np.sqrt(mean_squared_error(actual, predicted))

print("\n3-Month Moving Average Evaluation:")
print(f"MAE: {mae_ma:,.2f}")
print(f"RMSE: {rmse_ma:,.2f}")


# Generate future forecasts using recursive 3-month moving average

sales_history = monthly_sales["Sales"].tolist()

future_forecasts = []

for i in range(3):
    forecast = sum(sales_history[-3:]) / 3
    future_forecasts.append(forecast)
    sales_history.append(forecast)

future_forecast = pd.DataFrame({
    "Forecast_Month": [
        "January 2026",
        "February 2026",
        "March 2026"
    ],
    "Forecasted_Sales": future_forecasts
})

print("\nFuture 3-Month Sales Forecast:")
print(future_forecast)

import matplotlib.pyplot as plt

# Historical sales
plt.figure(figsize=(12, 6))

plt.plot(
    monthly_sales["Order_Date"],
    monthly_sales["Sales"],
    marker="o",
    label="Actual Sales"
)

# Forecast dates
forecast_dates = pd.date_range(
    start="2026-01-31",
    periods=3,
    freq="ME"
)

plt.plot(
    forecast_dates,
    future_forecasts,
    marker="o",
    linestyle="--",
    label="Forecast Sales"
)

plt.title("Actual vs Forecast Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# Create forecast output table
forecast_output = pd.DataFrame({
    "Forecast_Month": [
        "January 2026",
        "February 2026",
        "March 2026"
    ],
    "Forecasted_Sales": future_forecasts
})

# Correct project folder
output_path = r"E:\data analytics project\Sales_Analytics_Project\sales_forecast_2026.csv"

forecast_output.to_csv(output_path, index=False)

print("\nForecast saved successfully!")
print(f"File location: {output_path}")
print(forecast_output)