"""Reusable weather classification component."""

from dataclasses import dataclass
from pathlib import Path
import math

import joblib
import pandas as pd


@dataclass
class Prediction:
    """A running recommendation and probability of good weather."""

    label: str
    probability: float


class WeatherClassifier:
    """Load a saved weather model and predict running conditions."""

    FEATURES = [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ]

    def __init__(self, model_path: str | Path) -> None:
        """Load the trained model once.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        path = Path(model_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"Weather model file not found: {path}"
            )

        self.model = joblib.load(path)

    def predict(
        self, days: list[dict[str, float]]
    ) -> list[Prediction]:
        """Predict weather for a list of days.

        Raises:
            ValueError: If required features are missing or invalid.
        """
        if not days:
            return []

        rows = []

        for index, day in enumerate(days, start=1):
            if not isinstance(day, dict):
                raise ValueError(
                    f"Day {index} must be a dictionary"
                )

            missing = [
                feature for feature in self.FEATURES
                if feature not in day
            ]

            if missing:
                raise ValueError(
                    f"Day {index} missing features: {', '.join(missing)}"
                )

            row = {}

            for feature in self.FEATURES:
                try:
                    value = float(day[feature])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid value for {feature}"
                    ) from error

                if not math.isfinite(value):
                    raise ValueError(
                        f"Invalid value for {feature}"
                    )

                row[feature] = value

            rows.append(row)

        frame = pd.DataFrame(rows, columns=self.FEATURES)

        labels = self.model.predict(frame)
        probabilities = self.model.predict_proba(frame)[:, 1]

        return [
            Prediction(
                label="good" if int(label) == 1 else "skip",
                probability=float(probability),
            )
            for label, probability in zip(labels, probabilities)
        ]