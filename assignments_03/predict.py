"""Week 3: Load a saved weather model and make predictions."""

import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

model_path = MODEL_DIR / "weather_classifier.pkl"
metadata_path = MODEL_DIR / "weather_classifier_metadata.json"

# A saved model is required. Do not retrain in this script.
if not model_path.exists():
    raise FileNotFoundError(
        "Model missing. Run train_classifier.py first."
    )

if not metadata_path.exists():
    raise FileNotFoundError(
        "Metadata missing. Run train_classifier.py first."
    )

model = joblib.load(model_path)

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)

features = metadata["features"]

print("\nWEEK 3: WEATHER PREDICTIONS")
print("City:", metadata["city"])
print("Model:", metadata["model"])
print("Features:", features)
print("Test accuracy:", round(metadata["test_accuracy"], 4))
print("Test F1:", round(metadata["test_f1"], 4))

# Four hypothetical days: good, bad, borderline, and windy.
days = [
    {
        "temperature_2m_max": 20,
        "temperature_2m_min": 10,
        "precipitation_sum": 0,
        "wind_speed_10m_max": 10,
    },
    {
        "temperature_2m_max": 35,
        "temperature_2m_min": 25,
        "precipitation_sum": 12,
        "wind_speed_10m_max": 40,
    },
    {
        "temperature_2m_max": 26,
        "temperature_2m_min": 0,
        "precipitation_sum": 2.9,
        "wind_speed_10m_max": 29,
    },
    {
        "temperature_2m_max": 18,
        "temperature_2m_min": 8,
        "precipitation_sum": 1,
        "wind_speed_10m_max": 45,
    },
]

day_names = [
    "Good weather day",
    "Bad weather day",
    "Borderline weather day",
    "Windy weather day",
]

X_new = pd.DataFrame(days, columns=features)

predictions = model.predict(X_new)

# Find the probability corresponding to class 1.
good_index = list(model.classes_).index(1)
probabilities = model.predict_proba(X_new)[:, good_index]

for name, day, prediction, probability in zip(
    day_names,
    days,
    predictions,
    probabilities,
):
    print(f"\n{name}")

    for feature in features:
        print(f"  {feature}: {day[feature]}")

    recommendation = "GOOD FOR RUNNING" if prediction == 1 else "SKIP"

    print("  Prediction:", recommendation)
    print(f"  Probability of good weather: {probability:.2%}")


# REFLECTION
#
# A missing model file prevents prediction because this script
# loads an existing model rather than training a new one.
#
# A probability of 0.52 would be uncertain. I would review the
# conditions before recommending a run.
#
# KNN with k=1 produces probabilities of 0 or 1 because it uses
# only one neighbor. Those values do not guarantee certainty.
#
# For future automation, a scheduled process could collect
# weather data, validate the inputs, load the saved pipeline,
# produce predictions, and log the results without retraining.