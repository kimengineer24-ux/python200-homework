"""Use the saved weather model to predict running conditions."""

from pathlib import Path
from weather_model import WeatherClassifier


def main() -> None:
    model_path = (
        Path(__file__).resolve().parent
        / "models"
        / "weather_classifier.pkl"
    )

    classifier = WeatherClassifier(model_path)

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

    predictions = classifier.predict(days)

    for index, (day, result) in enumerate(
        zip(days, predictions), start=1
    ):
        print(f"\nDay {index}")
        print("Weather:", day)
        print("Prediction:", result.label)
        print(f"Probability of good weather: {result.probability:.1%}")


# The guard prevents predictions from running automatically
# when another script imports this file.
if __name__ == "__main__":
    main()


# REFLECTION
# 1. Fixed feature order prevents wind speed from accidentally
#    being interpreted as temperature when dictionary keys
#    are supplied in a different order.
#
# 2. Tests use their own small model so they do not depend
#    on the real saved model file. This makes them repeatable.
#
# 3. The class loads the model once and handles batches,
#    making it useful for a future cloud pipeline.
#    I would add logging and model version tracking
#    before using it in production.