from enum import StrEnum

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class Frequency(StrEnum):
    slower = "slower"
    slow = "slow"
    fast = "fast"
    faster = "faster"


class Answer(BaseModel):
    question: str
    answer: str

class PromptAnswers(BaseModel):
    answers: list[Answer]


class StationInformation(BaseModel):
    frequency: Frequency
    name: str
    verbatims: list[str]
    tags: list[str]
    artists: list[str]
    playlist: str


test_station = StationInformation(
    frequency=Frequency.slow,
    name="Hard Speed Radio",
    verbatims=["Me perdre dans la masse", "Il faut que je me dépense"],
    tags=["Sombre", "Moite", "Soutenu"],
    artists=["I Hate Models", "Clara Cuvé", "Rebekah"],
    playlist="https://link.deezer.com/s/30iKS8WFIDokwCdWfihFA",
)


@app.post("/predict")
def predict(prompt_answers: PromptAnswers) -> StationInformation:
    return test_station
