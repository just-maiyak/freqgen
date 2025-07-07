from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

data_base_path = Path("data")

class Settings(BaseSettings):
    PROMPTS_PATH: str | Path = data_base_path / "prompts.yaml"
    STATIONS_PATH: str | Path = data_base_path / "stations.yaml"
    TAGS_PATH: str | Path = data_base_path / "tags.yaml"

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

def get_settings():
    return Settings()
