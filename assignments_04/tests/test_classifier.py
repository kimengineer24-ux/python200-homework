"""Tests for the WeatherClassifier component."""

import joblib
import pandas as pd
import pytest

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from weather_model import WeatherClassifier, Prediction


GOOD = {
    "temperature_2m_max": 20,
    "temperature_2m_min": 10,
    "precipitation_sum": 0,
    "wind_speed_10m_max": 10,
}

BAD = {
    "temperature_2m_max": -8,
    "temperature_2m_min": -13,
    "precipitation_sum": 14,
    "wind_speed_10m_max": 52,
}


@pytest.fixture
def classifier(tmp_path):
    """Train a small test model instead of using the saved model."""

    examples = [
        [20, 10, 0, 10],
        [22, 12, 1, 15],
        [16, 7, 0.2, 10],
        [24, 15, 1, 12],
        [19, 9, 0, 8],
        [21, 11, 0.4, 12],
        [-5, -12, 15, 60],
        [-8, -13, 14, 52],
        [-6, -10, 9, 50],
        [-10, -16, 10, 65],
        [-3, -8, 11, 55],
        [-2, -7, 12, 58],
    ]

    frame = pd.DataFrame(
        examples,
        columns=WeatherClassifier.FEATURES,
    )

    labels = [1] * 6 + [0] * 6

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=1)),
    ])

    model.fit(frame, labels)

    path = tmp_path / "test_model.pkl"
    joblib.dump(model, path)

    return WeatherClassifier(path)


# This test catches a bug that always returns "skip".
@pytest.mark.parametrize(
    "day,expected",
    [(GOOD, "good"), (BAD, "skip")],
)
def test_correct_labels(classifier, day, expected):
    assert classifier.predict([day])[0].label == expected


def test_probability_range(classifier):
    result = classifier.predict([GOOD])[0]
    assert 0 <= result.probability <= 1


def test_three_predictions_in_order(classifier):
    results = classifier.predict([GOOD, BAD, GOOD])

    assert len(results) == 3
    assert all(isinstance(p, Prediction) for p in results)
    assert [p.label for p in results] == [
        "good", "skip", "good"
    ]


def test_empty_list(classifier):
    assert classifier.predict([]) == []


def test_missing_feature(classifier):
    incomplete = GOOD.copy()
    del incomplete["wind_speed_10m_max"]

    with pytest.raises(ValueError, match="wind_speed_10m_max"):
        classifier.predict([incomplete])


def test_missing_model(tmp_path):
    with pytest.raises(
        FileNotFoundError,
        match="Weather model file not found",
    ):
        WeatherClassifier(tmp_path / "missing.pkl")


def test_feature_order(classifier):
    reversed_day = dict(reversed(list(GOOD.items())))

    assert classifier.predict([reversed_day]) == (
        classifier.predict([GOOD])
    )