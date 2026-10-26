"""Week 4 warmup exercises."""

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


# QUESTION 1
class ModelInfo:
    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as file:
            self.data = json.load(file)

    def feature_count(self) -> int:
        return len(self.data["features"])


# Loading in __init__ avoids reading the JSON file repeatedly.


# QUESTION 2
@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        rain = "rain expected" if self.will_rain else "no rain expected"
        return f"{self.date}: high {self.high_c:.1f}C, {rain}"


# @dataclass automatically creates the __init__ method.


# QUESTION 3
def parse_temperature(text: str) -> float:
    try:
        return float(text)
    except ValueError as error:
        raise ValueError("Invalid temperature") from error


def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# Without match, an unrelated ValueError could pass the test.


if __name__ == "__main__":
    with TemporaryDirectory() as folder:
        path = Path(folder) / "sample.json"
        path.write_text(
            json.dumps({"features": ["high", "low", "wind"]})
        )

        info = ModelInfo(path)
        print("Feature count:", info.feature_count())
    day1 = DayForecast("2023-06-15", 24.1, True)
    day2 = DayForecast("2023-06-16", 20.0, False)

    print(day1.summary())
    print(day2.summary())