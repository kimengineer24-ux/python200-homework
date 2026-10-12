from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, field

import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator


# --- Classes ---
# Q1 and Q2
class Thermometer:
    def __init__(self, location: str, readings: list[float] | None = None):
        self.location = location
        self.readings = readings if readings is not None else []

    def add(self, reading: float) -> None:
        self.readings.append(reading)

    def average(self) -> float | None:
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self) -> float | None:
        if not self.readings:
            return None
        return max(self.readings)

    def __repr__(self) -> str:
        return (
            f"Thermometer(location={self.location!r}, "
            f"n_readings={len(self.readings)}, "
            f"average={self.average()})"
        )


# average() needs to handle the empty case because dividing by zero would crash the program.
# Without that check, an empty readings list would raise ZeroDivisionError.

# Without __repr__, Python displays the object's memory location, like <__main__.Thermometer object at ...>.
# That is not helpful when debugging because it does not show the useful data inside the object.


# Q3
class TemperatureAlert:
    def __init__(self, threshold: float = 30.0):
        self.threshold = threshold

    def breaches(self, thermometer: Thermometer) -> list[float]:
        return [reading for reading in thermometer.readings if reading > self.threshold]


# The threshold is stored on TemperatureAlert so each alert object has its own rule.
# That is useful with twenty thermometers because I can reuse the same alert against all of them
# instead of passing the threshold over and over again.


# --- Dataclasses, Type Hints, and Docstrings ---
# Dataclass Q1 and Q2
@dataclass(frozen=True)
class Station:
    """A weather station with identifying information and location details."""

    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float


# Because dataclasses compare field values by default, two Station objects with the same
# values compare equal. With the original hand-written class, they would not be equal
# unless __eq__ was written manually.


# Dataclass Q3
@dataclass
class StationBatch:
    """A group of weather stations for one region."""

    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Add one station to this batch."""

        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with the highest elevation, or None if empty."""

        if not self.stations:
            return None
        return max(self.stations, key=lambda station: station.elevation)


# If I tried stations: list[Station] = [], Python would raise:
# ValueError: mutable default <class 'list'> for field stations is not allowed: use default_factory
# Python refuses it because one shared list could be reused across objects by mistake.
# default_factory=list gives each StationBatch its own new list.


# --- Pydantic ---
# Pydantic Q1-Q4
class Reading(BaseModel):
    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def reject_failed_sensor(self):
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError("humidity of 0.0 with temperature below -40 indicates a failed sensor")
        return self


# Pydantic accepts "21.5" because it can safely convert that string to a float.
# It rejects "very humid" because that string cannot be converted into a number.
# The rule is that Pydantic can coerce compatible values, but not values that do not fit the field type.

# The failed sensor rule cannot be expressed with Field constraints alone because it depends on
# two fields together: humidity and temperature_c.


# --- pytest ---
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""

    return celsius * 9 / 5 + 32


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)


# pytest.approx is necessary because floating point math can create tiny rounding differences.


def mean(values: list[float]) -> float:
    """Return the mean of a list of numbers."""

    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        mean([])


# pytest.raises(ValueError) alone would catch the type of error, but match= also checks
# that the message is the one I expected.


@pytest.mark.parametrize(
    "values, expected",
    [
        ([5.0], 5.0),
        ([1.0, 2.0, 3.0], 2.0),
        ([-3.0, -1.0, -2.0], -2.0),
        ([10.0, 20.0, 30.0, 40.0], 25.0),
    ],
)
def test_mean_values(values: list[float], expected: float):
    assert mean(values) == expected


# A parametrized test is better than four nearly identical test functions because it keeps
# the repeated test logic in one place while still showing each case separately in pytest output.

# Pytest summary line from a successful run:
# 6 passed

# Deliberate break note:
# I temporarily changed celsius_to_fahrenheit from 9 / 5 to 9 / 4.
# Pytest showed the actual calculated value and the expected value, which is more useful
# than a bare "assertion failed" because it points directly to what was wrong.


def run_demos() -> None:
    print("--- Classes ---")

    my_thermometer = Thermometer("Spartanburg")
    for reading in [24.0, 28.5, 31.2, 29.8]:
        my_thermometer.add(reading)

    print("Average:", my_thermometer.average())
    print("Hottest:", my_thermometer.hottest())
    print("Thermometer object:", my_thermometer)

    second_thermometer = Thermometer("Charlotte", [18.1, 19.2, 20.5, 21.0])
    print("Thermometer list:", [my_thermometer, second_thermometer])

    warm_alert = TemperatureAlert(29.0)
    hot_alert = TemperatureAlert(30.0)

    print("Warm alert breaches:", warm_alert.breaches(my_thermometer))
    print("Hot alert breaches:", hot_alert.breaches(my_thermometer))

    print("\n--- Dataclasses ---")

    station_a = Station("STN001", "Main Station", 35.22, -80.84, 232.0)
    station_b = Station("STN001", "Main Station", 35.22, -80.84, 232.0)
    station_c = Station("STN002", "Hill Station", 35.30, -80.70, 300.0)

    print("station_a == station_b:", station_a == station_b)

    try:
        station_a.name = "Changed Station"
    except FrozenInstanceError as error:
        print("FrozenInstanceError:", error)

    station_set = {station_a, station_b, station_c}
    print("Station set length:", len(station_set))

    batch = StationBatch("Upstate")
    batch.add(station_a)
    batch.add(station_c)

    print("Highest station:", batch.highest())

    print("\n--- Pydantic ---")

    valid_reading = Reading(
        station_id="STN001",
        timestamp="2026-04-08T00:00",
        temperature_c=21.5,
        humidity=40.0,
    )
    print("Valid reading:", valid_reading)

    try:
        Reading(station_id="STN001", temperature_c=21.5, humidity=40.0)
    except ValidationError as error:
        print("Missing field error:", error)

    try:
        Reading(
            station_id="STN001",
            timestamp="2026-04-08T00:00",
            temperature_c=150.0,
            humidity=40.0,
        )
    except ValidationError as error:
        print("Temperature error:", error)

    try:
        Reading(
            station_id="STN001",
            timestamp="2026-04-08T00:00",
            temperature_c=21.5,
            humidity="very humid",
        )
    except ValidationError as error:
        print("Humidity type error:", error)

    converted_reading = Reading(
        station_id="STN001",
        timestamp="2026-04-08T00:00",
        temperature_c="21.5",
        humidity=40,
    )
    print("Converted reading:", converted_reading)
    print("temperature_c type:", type(converted_reading.temperature_c))
    print("humidity type:", type(converted_reading.humidity))

    try:
        Reading(station_id="AB", temperature_c="cold", humidity=40.0)
    except ValidationError as error:
        for item in error.errors():
            print("loc:", item["loc"], "msg:", item["msg"])

    # This reported 3 errors: station_id was too short, timestamp was missing,
    # and temperature_c was not numeric. Reporting all errors at once is more
    # useful than stopping at the first because I can fix everything in one pass.

    try:
        Reading(
            station_id="STN001",
            timestamp="2026-04-08T00:00",
            temperature_c=-50.0,
            humidity=0.0,
        )
    except ValidationError as error:
        print("Failed sensor error:", error)


if __name__ == "__main__":
    run_demos()
