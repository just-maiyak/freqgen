from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

data_base_path = Path("data")
analytics_db = Path("analytics")
assets_base_path = Path("assets")


class Settings(BaseSettings):
    PROMPTS_PATH: str | Path = data_base_path / "prompts"
    STATION_NAMES_PATH: str | Path = data_base_path / "names"
    TAGS_PATH: str | Path = data_base_path / "tags"
    TERMS_PATH: str | Path = data_base_path / "terms"
    ASSETS_PATH: str | Path = assets_base_path

    ANALYTICS_DB_LOCATION: str | Path = analytics_db / "analytics.sqlite"

    CURRENT_DEVICE: str = "mps"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings():
    print(settings := Settings())
    return settings


settings = get_settings()
