from pydantic import BaseModel, Field, model_validator


class HourlyBlock(BaseModel):
    """Columnar hourly weather data from the API response."""

    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def matching_lengths(self):
        lengths = {
            len(self.time),
            len(self.temperature_2m),
            len(self.precipitation),
        }
        if len(lengths) != 1:
            raise ValueError("time, temperature_2m, and precipitation must have the same length")
        return self


class WeatherResponse(BaseModel):
    """Top-level weather API response with location metadata and hourly data."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock
