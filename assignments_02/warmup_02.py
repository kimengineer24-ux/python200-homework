# Week 2 Warmup Exercises
# --- scikit-learn API ---

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# --- scikit-learn API ---
# Q1
temp_min_api = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max_api = np.array([8, 10, 15, 18, 23, 27])

model_api = LinearRegression()
model_api.fit(temp_min_api, temp_max_api)

api_predictions = model_api.predict(np.array([[6], [18]]))

print("--- scikit-learn API Q1 ---")
print("Slope:", model_api.coef_[0])
print("Intercept:", model_api.intercept_)
print("Predicted high for low of 6:", api_predictions[0])
print("Predicted high for low of 18:", api_predictions[1])


# Q2
x = np.array([10, 20, 30, 40, 50])
print("\n--- scikit-learn API Q2 ---")
print("Original shape:", x.shape)

x_reshaped = x.reshape(-1, 1)
print("Reshaped shape:", x_reshaped.shape)

# scikit-learn needs X to be 2D because it expects rows to be samples
# and columns to be features, even when there is only one feature.


# --- Linear Regression ---
# Shared synthetic weather dataset
rng = np.random.default_rng(42)
num_days = 120
temp_min = rng.uniform(-5, 22, num_days)
is_rainy = rng.integers(0, 2, num_days).astype(float)
temp_max = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)


# Q1
print("\n--- Linear Regression Q1 ---")
plt.figure()
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")
plt.title("Daily High vs Low")
plt.xlabel("Daily Low Temperature")
plt.ylabel("Daily High Temperature")
plt.savefig(OUTPUT_DIR / "high_vs_low.png")
plt.close()
print("Saved plot: outputs/high_vs_low.png")

# The plot shows an upward relationship between low and high temperature.
# There are also two visible groups because rainy days tend to have lower highs,
# which suggests is_rainy may be useful for predicting temp_max.


# Q2
print("\n--- Linear Regression Q2 ---")
X = temp_min.reshape(-1, 1)
y = temp_max

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# Q3
print("\n--- Linear Regression Q3 ---")
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("RMSE:", rmse)
print("R² on test set:", r2)

# The slope means that for each 1 degree increase in the daily low,
# the predicted daily high increases by about this many degrees.


# Q4
print("\n--- Linear Regression Q4 ---")
X_full = np.column_stack([temp_min, is_rainy])

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

y_pred_full = model_full.predict(X_test_full)
r2_full = model_full.score(X_test_full, y_test_full)

print("R² with temp_min only:", r2)
print("R² with temp_min and is_rainy:", r2_full)
print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# Adding the rain flag helps because the two-feature model has a better test R².
# The is_rainy coefficient represents how much the predicted high changes
# when a day is rainy instead of dry, while holding temp_min the same.


# Q5
print("\n--- Linear Regression Q5 ---")
plt.figure()
plt.scatter(y_pred_full, y_test_full)
min_value = min(y_pred_full.min(), y_test_full.min())
max_value = max(y_pred_full.max(), y_test_full.max())
plt.plot([min_value, max_value], [min_value, max_value])
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Daily High")
plt.ylabel("Actual Daily High")
plt.savefig(OUTPUT_DIR / "predicted_vs_actual_warmup.png")
plt.close()
print("Saved plot: outputs/predicted_vs_actual_warmup.png")

# A point above the diagonal means the actual high was higher than predicted.
# A point below the diagonal means the actual high was lower than predicted.
