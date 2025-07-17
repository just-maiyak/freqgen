from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

data_base_path = Path("data")


class Settings(BaseSettings):
    PROMPTS_PATH: str | Path = data_base_path / "prompts.yaml"
    STATION_NAMES_PATH: str | Path = data_base_path / "names"
    TAGS_PATH: str | Path = data_base_path / "tags.yaml"

    CURRENT_DEVICE: str = "mps"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings():
    print(settings := Settings())
    return settings
