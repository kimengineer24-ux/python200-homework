import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit import WeatherResponse


def test_valid_response_has_168_hours():
    # A plain relative path is unreliable because pytest may be run from different folders.
    path = Path(__file__).parent.parent / "weather_raw.json"

    with path.open() as file:
        raw_data = json.load(file)

    response = WeatherResponse.model_validate(raw_data)

    assert len(response.hourly.time) == 168


def test_invalid_latitude_raises():
    bad_data = {
        "latitude": 200.0,
        "longitude": -80.8431,
        "timezone": "America/New_York",
        "elevation": 232.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [7.8],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(bad_data)


def test_mismatched_hourly_lengths_raise():
    bad_data = {
        "latitude": 35.2271,
        "longitude": -80.8431,
        "timezone": "America/New_York",
        "elevation": 232.0,
        "hourly": {
            "time": ["a", "b", "c"],
            "temperature_2m": [1.0, 2.0],
            "precipitation": [0.0, 0.1, 0.2],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(bad_data)


def test_null_temperature_raises():
    bad_data = {
        "latitude": 35.2271,
        "longitude": -80.8431,
        "timezone": "America/New_York",
        "elevation": 232.0,
        "hourly": {
            "time": ["2026-04-08T00:00"],
            "temperature_2m": [None],
            "precipitation": [0.0],
        },
    }

    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(bad_data)
