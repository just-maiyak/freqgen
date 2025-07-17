from enum import StrEnum

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from freqgen.model import get_model


origins = [
    "http://localhost",
    "http://localhost:1234",
    "https://just-maiyak.github.io",
    "https://station-r.club",
    "https://freqscan.yefimch.uk",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.MODEL = get_model()


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


class PlaylistLinks(BaseModel):
    deezer: str
    spotify: str


class StationInformation(BaseModel):
    frequency: Frequency
    name: str
    verbatims: list[str]
    tags: list[str]
    artists: list[str]
    playlist: PlaylistLinks


test_station = StationInformation(
    frequency=Frequency.slow,
    name="Hard Speed Radio",
    verbatims=["Me perdre dans la masse", "Il faut que je me dépense"],
    tags=["Sombre", "Moite", "Soutenu"],
    artists=["I Hate Models", "Clara Cuvé", "Rebekah"],
    playlist=PlaylistLinks(
        deezer="https://link.deezer.com/s/30iKS8WFIDokwCdWfihFA",
        spotify=""
    ),
)


@app.post("/predict")
def predict(prompt_answers: PromptAnswers) -> StationInformation:
    model = app.state.MODEL
    answers = {
        question.question: question.answer for question in prompt_answers.answers
    }

    return StationInformation(
        frequency=model.compute_user_station(answers),
        name=" ".join(model.generate_station_name(answers)),
        verbatims=model.get_best_verbatims(answers),
        tags=model.generate_best_tags(answers),
        artists=model.generate_best_artists(answers),
        playlist=PlaylistLinks(**model.get_best_playlist(answers)),
    )
