# Week 2 Mini-Project
# Predicting the Daily High
# City chosen: Spartanburg, South Carolina

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np


OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# --- Task 1: Fetch the Data ---
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 34.95,
    "longitude": -81.93,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/New_York",
}

response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("--- Task 1: Fetch the Data ---")
print("City chosen: Spartanburg, South Carolina")
print("Dataset shape:", df.shape)
print("First five rows:")
print(df.head())

# I chose Spartanburg, South Carolina because it is close to where I live,
# so the weather data feels relevant and easier to understand.


# --- Task 2: Look at the Features ---
print("\n--- Task 2: Look at the Features ---")
print("Dataset description:")
print(df.describe())

print("\nMissing values:")
print(df.isna().sum())

# There are no missing values in this dataset, so I do not need to drop or fill
# any rows before modeling.

print("\nCorrelations with temperature_2m_max:")
correlations = df.select_dtypes(include="number").corr()["temperature_2m_max"].sort_values()
print(correlations)

strongest_feature = correlations.drop("temperature_2m_max").abs().idxmax()
print("Strongest related feature:", strongest_feature)

plt.figure()
plt.scatter(df["temperature_2m_min"], df["temperature_2m_max"])
plt.title("Daily High vs Daily Low")
plt.xlabel("Daily Low Temperature")
plt.ylabel("Daily High Temperature")
plt.savefig(OUTPUT_DIR / "low_vs_high_project.png")
plt.close()
print("Saved plot: outputs/low_vs_high_project.png")

# The scatter plot shows that days with higher low temperatures usually also
# have higher high temperatures. This suggests temperature_2m_min should be a
# useful feature for predicting temperature_2m_max.


# --- Task 3: Engineer a Feature ---
print("\n--- Task 3: Engineer a Feature ---")
df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)

print("Summer counts:")
print(df["is_summer"].value_counts())

# I expect is_summer to help predict the daily high because summer days are
# usually warmer than non-summer days, especially in South Carolina.


# --- Task 4: Baseline Model ---
print("\n--- Task 4: Baseline Model ---")
X_baseline = df[["temperature_2m_min"]].values
y = df["temperature_2m_max"].values

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_baseline, y, test_size=0.2, random_state=42
)

baseline_model = LinearRegression()
baseline_model.fit(X_train_base, y_train_base)

baseline_pred = baseline_model.predict(X_test_base)
baseline_rmse = np.sqrt(mean_squared_error(y_test_base, baseline_pred))
baseline_r2 = baseline_model.score(X_test_base, y_test_base)

print("Baseline slope:", baseline_model.coef_[0])
print("Baseline RMSE:", baseline_rmse)
print("Baseline R²:", baseline_r2)

# The RMSE is in degrees Celsius. In plain English, this means a typical
# prediction is off by about the RMSE number of Celsius degrees.


# --- Task 5: Full Model ---
print("\n--- Task 5: Full Model ---")
feature_cols = [
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "is_summer",
]

X = df[feature_cols].values
y = df["temperature_2m_max"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
full_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)

print("Full model train R²:", train_r2)
print("Full model test R²:", test_r2)
print("Full model RMSE:", full_rmse)
print("Baseline test R²:", baseline_r2)
print("Improvement over baseline R²:", test_r2 - baseline_r2)

print("\nFeature coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:22s}: {coef:+.3f}")

largest_index = np.argmax(np.abs(model.coef_))
largest_feature = feature_cols[largest_index]
largest_coef = model.coef_[largest_index]
print("Largest effect feature:", largest_feature)
print("Largest effect coefficient:", largest_coef)

# Positive coefficients raise the predicted high, while negative coefficients
# lower the predicted high. I expect temperature_2m_min and is_summer to raise
# the predicted high. Precipitation and wind may lower the predicted high if
# rainy or windy days are cooler.
# Comparing train R² and test R² helps show whether the model generalizes well.
# If they are close, the model is likely performing similarly on new data.


# --- Task 6: Evaluate and Summarize ---
print("\n--- Task 6: Evaluate and Summarize ---")
plt.figure()
plt.scatter(y_pred, y_test)
min_value = min(y_pred.min(), y_test.min())
max_value = max(y_pred.max(), y_test.max())
plt.plot([min_value, max_value], [min_value, max_value])
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Daily High")
plt.ylabel("Actual Daily High")
plt.savefig(OUTPUT_DIR / "predicted_vs_actual_high.png")
plt.close()
print("Saved plot: outputs/predicted_vs_actual_high.png")

residuals = y_test - y_pred
low_end_error = np.mean(np.abs(residuals[y_test < np.median(y_test)]))
high_end_error = np.mean(np.abs(residuals[y_test >= np.median(y_test)]))

print("Average low-end error:", low_end_error)
print("Average high-end error:", high_end_error)

# The predicted vs actual plot shows how close the model's predictions are to
# the real daily highs. Points close to the diagonal are better predictions.
# Based on the printed low-end and high-end errors, I can compare whether the
# model struggles more on colder days, warmer days, or if the error is roughly even.

# Plain-language summary:
# The dataset covers one year of daily weather for Spartanburg, South Carolina.
# With an 80/20 split, the test set is about 20 percent of the total dataset.
# The full model RMSE and R² are printed above. The RMSE means the typical
# prediction is off by that many degrees Celsius.
# The coefficient list shows which feature has the largest effect and whether
# it raises or lowers the predicted high temperature.
# One result that surprised me was how much adding seasonal and weather features
# can change the model compared with using the daily low temperature by itself.
