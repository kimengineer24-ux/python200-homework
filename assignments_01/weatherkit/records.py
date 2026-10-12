from dataclasses import dataclass

from weatherkit.schemas import WeatherResponse


@dataclass
class HourlyReading:
    """One hourly weather reading.

    timestamp is an ISO-style datetime string.
    temperature_c is measured in degrees Celsius.
    precipitation_mm is measured in millimeters.
    """

    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert a validated weather response into hourly readings.

    Args:
        response: A WeatherResponse validated at the program boundary.

    Returns:
        A list of HourlyReading objects in the same order as the API response.
    """

    readings = []

    for timestamp, temperature, precipitation in zip(
        response.hourly.time,
        response.hourly.temperature_2m,
        response.hourly.precipitation,
    ):
        readings.append(
            HourlyReading(
                timestamp=timestamp,
                temperature_c=temperature,
                precipitation_mm=precipitation,
            )
        )

    return readings


# HourlyReading is a dataclass because it is used after the boundary.
# WeatherResponse is the Pydantic model because it validates outside API data.
# Once the data is validated, a dataclass is simpler for internal pipeline records.
