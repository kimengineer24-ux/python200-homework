from dataclasses import dataclass
from collections import defaultdict

from weatherkit.records import HourlyReading


@dataclass
class DailySummary:
    """Summary of one day of hourly weather readings."""

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Return the difference between the high and low temperature."""

        return self.temp_max - self.temp_min


class DailyAggregator:
    """Aggregate hourly readings into daily weather summaries."""

    def __init__(self, min_hours: int = 24):
        """Create an aggregator.

        Args:
            min_hours: Minimum number of hourly readings required to keep a day.
        """

        self.min_hours = min_hours

    def _group_by_date(self, readings: list[HourlyReading]) -> dict[str, list[HourlyReading]]:
        grouped = defaultdict(list)

        for reading in readings:
            date = reading.timestamp[:10]
            grouped[date].append(reading)

        return grouped

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """Create daily summaries from hourly readings."""

        grouped = self._group_by_date(readings)
        summaries = []

        for date in sorted(grouped):
            day_readings = grouped[date]

            if len(day_readings) < self.min_hours:
                continue

            temperatures = [reading.temperature_c for reading in day_readings]
            precipitation_values = [reading.precipitation_mm for reading in day_readings]

            summaries.append(
                DailySummary(
                    date=date,
                    temp_max=max(temperatures),
                    temp_min=min(temperatures),
                    precipitation_sum=sum(precipitation_values),
                    hours_observed=len(day_readings),
                )
            )

        return summaries

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """Return dates dropped because they had fewer than min_hours readings."""

        grouped = self._group_by_date(readings)

        return [
            date
            for date in sorted(grouped)
            if len(grouped[date]) < self.min_hours
        ]
