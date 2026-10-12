from weatherkit import HourlyReading, WeatherResponse, to_readings


def sample_response():
    data = {
        "latitude": 35.2271,
        "longitude": -80.8431,
        "timezone": "America/New_York",
        "elevation": 232.0,
        "hourly": {
            "time": ["2026-04-08T00:00", "2026-04-08T01:00", "2026-04-08T02:00"],
            "temperature_2m": [7.8, 8.1, 8.5],
            "precipitation": [0.0, 0.1, 0.2],
        },
    }

    return WeatherResponse.model_validate(data)


def test_to_readings_returns_one_per_hour_in_order():
    readings = to_readings(sample_response())

    assert len(readings) == 3
    assert readings[0].timestamp == "2026-04-08T00:00"
    assert readings[-1].timestamp == "2026-04-08T02:00"


def test_reading_values_match_index():
    readings = to_readings(sample_response())

    assert readings[1].timestamp == "2026-04-08T01:00"
    assert readings[1].temperature_c == 8.1
    assert readings[1].precipitation_mm == 0.1


def test_hourly_reading_objects_compare_equal():
    first = HourlyReading("2026-04-08T00:00", 7.8, 0.0)
    second = HourlyReading("2026-04-08T00:00", 7.8, 0.0)

    assert first == second
