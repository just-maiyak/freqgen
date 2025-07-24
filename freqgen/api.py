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

app.state.MODELS = {lang: get_model(lang) for lang in ("fr", "en", "de")}


class Frequency(StrEnum):
    slower = "slower"
    slow = "slow"
    fast = "fast"
    faster = "faster"


class Answer(BaseModel):
    question_id: str
    answer: str


class PromptAnswers(BaseModel):
    answers: list[Answer]


class PlaylistLinks(BaseModel):
    deezer: str
    spotify: str
    apple: str
    youtube: str


class StationInformation(BaseModel):
    frequency: Frequency
    name: str
    verbatims: list[str]
    tags: list[str]
    artists: list[str]
    playlist: PlaylistLinks


@app.post("/predict")
def predict(prompt_answers: PromptAnswers, language: str = "fr") -> StationInformation:
    model = app.state.MODELS[language]
    answers = {
        question.question_id: question.answer for question in prompt_answers.answers
    }

    return StationInformation(
        frequency=(best_station := model.compute_user_station(answers)),
        name=" ".join(model.generate_station_name(answers)),
        verbatims=model.get_best_verbatims(answers),
        tags=model.generate_best_tags(answers),
        artists=model.generate_best_artists(best_station),
        playlist=PlaylistLinks(**model.get_best_playlist(best_station)),
    )
