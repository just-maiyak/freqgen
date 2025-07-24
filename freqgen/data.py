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


class Questionnaire(BaseModel):
    place: Question
    spirit: Question
    outfit: Question
    aesthetic: Question
    fuel: Question


def get_questionnaire(language: str = "fr") -> Questionnaire:
    yaml = Path(settings.PROMPTS_PATH / f"{language}.yaml").read_text()
    return parse_yaml_raw_as(Questionnaire, yaml)


def get_tags(language: str = "fr") -> set[str]:
    return set(Path(settings.TAGS_PATH / f"{language}.yaml").read_text().split())


def get_station_names(language: str = "fr") -> set[str]:
    return set(
        (Path(settings.STATION_NAMES_PATH) / f"{language}.yaml").read_text().split()
    )


def get_radio_terms(language: str = "fr") -> set[str]:
    return set((Path(settings.TERMS_PATH) / f"{language}.yaml").read_text().split())
