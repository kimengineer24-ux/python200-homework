import pytest

from weatherkit import DailyAggregator, HourlyReading


@pytest.fixture
def sample_readings():
    return [
        HourlyReading("2026-04-08T00:00", 10.0, 0.1),
        HourlyReading("2026-04-08T01:00", 12.0, 0.2),
        HourlyReading("2026-04-08T02:00", 8.0, 0.3),
        HourlyReading("2026-04-09T00:00", 20.0, 0.0),
        HourlyReading("2026-04-09T01:00", 22.0, 0.1),
    ]


def test_grouping_creates_two_summaries(sample_readings):
    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2
    assert summaries[0].date == "2026-04-08"
    assert summaries[1].date == "2026-04-09"


def test_temp_max_and_min_are_correct(sample_readings):
    aggregator = DailyAggregator(min_hours=2)

    first_summary = aggregator.summarize(sample_readings)[0]

    assert first_summary.temp_max == 12.0
    assert first_summary.temp_min == 8.0


def test_precipitation_sum_uses_approx(sample_readings):
    aggregator = DailyAggregator(min_hours=2)

    first_summary = aggregator.summarize(sample_readings)[0]

    assert first_summary.precipitation_sum == pytest.approx(0.6)


def test_incomplete_day_is_dropped(sample_readings):
    aggregator = DailyAggregator(min_hours=3)

    summaries = aggregator.summarize(sample_readings)
    incomplete = aggregator.incomplete_days(sample_readings)

    assert len(summaries) == 1
    assert "2026-04-09" in incomplete


def test_lowering_min_hours_keeps_day(sample_readings):
    aggregator = DailyAggregator(min_hours=2)

    summaries = aggregator.summarize(sample_readings)

    assert len(summaries) == 2


@pytest.mark.parametrize(
    "high, low, expected",
    [
        (12.0, 8.0, 4.0),
        (22.0, 20.0, 2.0),
    ],
)
def test_temp_range_parametrize(high, low, expected):
    from weatherkit import DailySummary

    summary = DailySummary(
        date="2026-04-08",
        temp_max=high,
        temp_min=low,
        precipitation_sum=0.0,
        hours_observed=24,
    )

    assert summary.temp_range() == expected


# Deliberate break note: I temporarily changed temp_range to return temp_max + temp_min.
# test_temp_range_parametrize caught the change because the expected ranges no longer matched.
