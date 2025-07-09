from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_raw_as

from freqgen.config import get_settings

settings = get_settings()


class Station(StrEnum):
    slower = "slower"
    slow = "slow"
    fast = "fast"
    faster = "faster"


class Choice(BaseModel):
    answer: str
    station: Station


class Question(BaseModel):
    question: str
    choices: list[Choice]


class Prompts(BaseModel):
    questionnaire: list[Question]


def get_questionnaire() -> Prompts:
    yaml = Path(settings.PROMPTS_PATH).read_text()
    return parse_yaml_raw_as(Prompts, yaml)


def get_tags() -> set[str]:
    return set(Path(settings.TAGS_PATH).read_text().split())


def get_station_names() -> dict[str, set[str]]:
    return {
        "en": set((Path(settings.STATION_NAMES_PATH) / "en.yaml").read_text().split()),
        "de": set((Path(settings.STATION_NAMES_PATH) / "de.yaml").read_text().split()),
    }
