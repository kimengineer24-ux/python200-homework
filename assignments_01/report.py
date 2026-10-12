import json
from pathlib import Path

from weatherkit import DailyAggregator, WeatherResponse, to_readings


def main() -> None:
    """Load raw weather data, validate it, summarize it, and print a report."""

    path = Path(__file__).parent / "weather_raw.json"

    with path.open() as file:
        raw_data = json.load(file)

    response = WeatherResponse.model_validate(raw_data)
    readings = to_readings(response)

    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)
    incomplete = aggregator.incomplete_days(readings)

    print("Daily Weather Report")
    print("date        high   low    precip   range")

    for summary in summaries:
        print(
            f"{summary.date}  "
            f"{summary.temp_max:5.1f}  "
            f"{summary.temp_min:5.1f}  "
            f"{summary.precipitation_sum:6.1f}  "
            f"{summary.temp_range():5.1f}"
        )

    if incomplete:
        print("Warning: incomplete days dropped:", ", ".join(incomplete))
    else:
        print("Warning: no incomplete days were dropped.")


if __name__ == "__main__":
    main()


# Reflection:
# 1. Rejecting the whole file when a single temperature is null can be right if the
# pipeline needs complete, trusted data for official reports. If this were a live
# dashboard, I might rather tolerate the gap and keep the rest of the data. To
# tolerate it, I would change temperature_2m to allow float | None and then skip
# missing values during aggregation.
#
# 2. If a pipeline runs at noon, the current day only has half the hours. Without
# min_hours, the daily high and low could look complete when they are not.
# incomplete_days() helps because it tells me which dates were dropped instead of
# letting missing data disappear silently.
#
# 3. Writing weatherkit as a package makes it easier in Week 10 because the
# pipeline can import the validation, conversion, and summary code without copying
# one large script.
#
# If the guard were omitted, main() would run automatically when another file
# imported report.py, which could print output or run pipeline work unexpectedly.
