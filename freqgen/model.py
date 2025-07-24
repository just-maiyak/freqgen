from collections import Counter
from functools import lru_cache
from operator import itemgetter
from random import choice, sample

import numpy as np
import numpy.typing as npt
import torch
import sentence_transformers as st

from freqgen import config
from freqgen.data import (
    get_questionnaire,
    get_tags,
    get_station_names,
    get_radio_terms,
    Station,
)

settings = config.get_settings()


class FreqGenModel:
    language: str = "fr"

    radio_terms: set[str] | None = None
    tag_embeddings: tuple[list[str], npt.NDArray[np.float32]] | None = None
    station_names_embeddings: tuple[list[str], npt.NDArray[np.float32]] | None = None
    questionnaire_embeddings: (
        dict[str, tuple[list[Station], npt.NDArray[np.float32]]] | None
    ) = None

    # Dunder
    # ======

    def __init__(self, language: str = "fr"):
        self.language = language
        self.radio_terms = get_radio_terms(self.language)
        self.tag_embeddings = self.get_tag_embeddings()
        self.station_names_embeddings = self.get_station_name_embeddings()
        self.questionnaire_embeddings = self.get_questionnaire_embeddings()

    def __repr__(self):
        len_tags = len(self.tag_embeddings) if self.tag_embeddings else "no"
        len_names = (
            len(self.station_names_embeddings)
            if self.station_names_embeddings
            else "no"
        )
        len_questionnaire = (
            len(self.questionnaire_embeddings)
            if self.questionnaire_embeddings
            else "no"
        )

        return f"FreqGenModel({len_tags} tags, {len_names} names, {len_questionnaire} choices)"

    # Model / Embeddings
    # ==================

    @lru_cache
    def get_model(
        self, language: str = "fr", device: str = settings.CURRENT_DEVICE
    ) -> st.SentenceTransformer:
        model_name = (
            "LaJavaness/sentence-camembert-base"
            if language == "fr"
            else "sentence-transformers/all-MiniLM-L6-v2"
        )
        return st.SentenceTransformer(model_name, device=device)

    def get_tag_embeddings(self) -> tuple[list[str], npt.NDArray[np.float32]]:
        model = self.get_model(self.language)
        tags = list(get_tags(self.language))

        return tags, model.encode(tags)

    def get_questionnaire_embeddings(
        self,
    ) -> dict[str, tuple[list[Station], npt.NDArray[np.float32]]]:
        model = self.get_model(self.language)
        questionnaire = get_questionnaire(self.language)

        return {
            question_id: (
                [choice.station for choice in question.choices],
                model.encode([choice.answer for choice in question.choices]),
            )
            for question_id, question in questionnaire
        }

    def get_station_name_embeddings(
        self,
    ) -> tuple[list[str], npt.NDArray[np.float32]]:
        return (
            ordered_names := list(get_station_names(self.language)),
            self.get_model(self.language).encode(ordered_names),
        )

    # Generation
    # ==========

    def generate_station_name(
        self, answers: dict[str, str], length: int = 1
    ) -> list[str]:
        if self.station_names_embeddings is None or self.radio_terms is None:
            raise ValueError("Model has not been initialized")

        names, embeddings = self.station_names_embeddings

        model = self.get_model(self.language)

        user_embeddings = model.encode(list(answers.values()))

        similarity = model.similarity(user_embeddings, embeddings)

        best_names_index = torch.argmax(similarity, dim=1)

        best_names = [names[index] for index in best_names_index]

        return (
            [choice(list(self.radio_terms)), *sample(best_names, length)]
            if self.language == "fr"
            else [*sample(best_names, length), choice(list(self.radio_terms))]
        )

    def get_best_station(
        self,
        answer: str,
        choice_stations: list[Station],
        choice_embeddings: npt.NDArray[np.float32],
    ) -> Station:
        model = self.get_model(self.language)

        answer_embedding = model.encode([answer])
        best_index = model.similarity(answer_embedding, choice_embeddings).argmax()

        return choice_stations[best_index]

    def compute_user_station(self, answers: dict[str, str]) -> Station:
        if self.questionnaire_embeddings is None:
            raise ValueError("Model has not been initialized")

        best_stations = [
            self.get_best_station(answer, *self.questionnaire_embeddings[question_id])
            for question_id, answer in answers.items()
        ]

        [(best_station, _), *_] = Counter(best_stations).most_common()

        return best_station

    def generate_best_tags(self, answers: dict[str, str], limit: int = 5) -> list[str]:
        if self.tag_embeddings is None:
            raise ValueError("Model has not been initialized")

        tags, tag_embeddings = self.tag_embeddings
        model = self.get_model(self.language)

        user_embeddings = model.encode(list(answers.values()))

        maxes, _ = model.similarity(user_embeddings, tag_embeddings).max(dim=0)
        tag_similarities = zip(tags, maxes.tolist())

        return [tag for tag, _ in sorted(tag_similarities, key=itemgetter(1))][:limit]

    def get_best_verbatims(self, answers: dict[str, str]) -> list[str]:
        user_input = list(answers.values())
        model = self.get_model()

        user_embeddings = model.encode(user_input)
        similarities = model.similarity(user_embeddings, user_embeddings).triu(
            diagonal=1
        )

        first, second, *_ = torch.unravel_index(
            similarities.argmax(), similarities.shape
        )
        return [user_input[first], user_input[second]]

    def generate_best_artists(self, station: Station, length: int = 3) -> list[str]:

        artists = ["DJ Mehdi", "Myd", "Sebastian"]

        match station:
            case Station.slower:
                artists = ["Folamour", "Polo&Pan", "Peggy Gou"]
            case Station.slow:
                artists = [
                    "DJ Heartstrings",
                    "Uper90",
                    "Paramida",
                    "Asphalt DJ",
                    "Bliss Inc.",
                ]
            case Station.fast:
                artists = ["Alarico", "Chlär", "Mac Declos"]
            case Station.faster:
                artists = ["Shlømo", "999999999", "Rebekah", "Clara Cuvé", "SPFDJ"]

        return sample(artists, min(length, len(artists)))

    def get_best_playlist(self, station: Station) -> dict[str, str]:
        playlists = {
            "deezer": "https://link.deezer.com/s/30zcKNfVHeCY1kVap7koa",
            "spotify": "https://open.spotify.com/playlist/2XiQ26aJA4eOkJAtVmfJzl",
            "apple": "https://music.apple.com/fr/playlist/uk-garage-hard-house-stationr/pl.u-JPoNFWWZjNq",
            "youtube": "",
        }
        match station:
            case Station.slower:
                playlists = {
                    "deezer": "https://link.deezer.com/s/30zcMKTY2yWNartKHdNsz",
                    "spotify": "https://open.spotify.com/playlist/7lQ1MWScSLk8AzB54qM9Bq",
                    "apple": "https://music.apple.com/fr/playlist/baile-funk-disco-nu-house-stationr/pl.u-aZK7FVVMvpW",
                    "youtube": "",
                }
            case Station.slow:
                playlists = {
                    "deezer": "https://link.deezer.com/s/30zcKNfVHeCY1kVap7koa",
                    "spotify": "https://open.spotify.com/playlist/2XiQ26aJA4eOkJAtVmfJzl",
                    "apple": "https://music.apple.com/fr/playlist/uk-garage-hard-house-stationr/pl.u-JPoNFWWZjNq",
                    "youtube": "",
                }
            case Station.fast:
                playlists = {
                    "deezer": "https://link.deezer.com/s/30yHSUSoWLo9rwTU4qpQs",
                    "spotify": "https://open.spotify.com/playlist/0L4xtzuxTmNMXFn0dDdu79",
                    "apple": "https://music.apple.com/fr/playlist/techno-hypno-mentale/pl.u-76E6uNNXJdg",
                    "youtube": "",
                }
            case Station.faster:
                playlists = {
                    "deezer": "https://link.deezer.com/s/30z33Wn4MDmCfW6Ab4GB3",
                    "spotify": "https://open.spotify.com/playlist/17zBdBpK1PrHEsWhcTYluS",
                    "apple": "https://music.apple.com/fr/playlist/raw-hard-techno/pl.u-11DBHZZEB6M",
                    "youtube": "",
                }
        return playlists


def coin_flip():
    return choice((True, False))


@lru_cache
def get_model(language: str = "fr") -> FreqGenModel:
    return FreqGenModel(language)
