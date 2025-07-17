from collections import Counter
from functools import lru_cache
from operator import itemgetter
from random import choice, sample

import numpy as np
import numpy.typing as npt
import torch
import sentence_transformers as st

from freqgen import config
from freqgen.data import get_questionnaire, get_tags, get_station_names, Station

settings = config.get_settings()


class FreqGenModel:
    tag_embeddings: tuple[list[str], npt.NDArray[np.float32]] | None = None
    station_names_embeddings: (
        dict[str, tuple[list[str], npt.NDArray[np.float32]]] | None
    ) = None
    questionnaire_embeddings: (
        dict[str, tuple[list[Station], npt.NDArray[np.float32]]] | None
    ) = None

    # Dunder
    # ======

    def __init__(self):
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
        model = self.get_model("fr")
        tags = list(get_tags())

        return tags, model.encode(tags)

    def get_questionnaire_embeddings(
        self,
    ) -> dict[str, tuple[list[Station], npt.NDArray[np.float32]]]:
        model = self.get_model("fr")
        questionnaire = get_questionnaire().questionnaire

        return {
            question.question: (
                [choice.station for choice in question.choices],
                model.encode([choice.answer for choice in question.choices]),
            )
            for question in questionnaire
        }

    def get_station_name_embeddings(
        self,
    ) -> dict[str, tuple[list[str], npt.NDArray[np.float32]]]:
        station_names = get_station_names()

        return {
            lang: (
                ordered_names := list(names),
                self.get_model(lang).encode(ordered_names),
            )
            for lang, names in station_names.items()
        }

    # Generation
    # ==========

    def generate_station_name(
        self, answers: dict[str, str], length: int = 2
    ) -> list[str]:
        if self.station_names_embeddings is None:
            raise ValueError("Model has not been initialized")

        en_names, en_embeddings = self.station_names_embeddings["en"]
        de_names, de_embeddings = self.station_names_embeddings["de"]

        model = self.get_model("multi")

        user_embeddings = model.encode(list(answers.values()))

        en_similarity = model.similarity(user_embeddings, en_embeddings)
        de_similarity = model.similarity(user_embeddings, de_embeddings)

        en_best_names_index = torch.argmax(en_similarity, dim=1)
        de_best_names_index = torch.argmax(de_similarity, dim=1)

        en_best_names = [en_names[index] for index in en_best_names_index]
        de_best_names = [de_names[index] for index in de_best_names_index]

        return (
            [*sample(en_best_names, length), "Frequency"]
            if coin_flip()
            else [*sample(de_best_names, length), "Radio"]
        )

    def get_best_station(
        self,
        answer: str,
        choice_stations: list[Station],
        choice_embeddings: npt.NDArray[np.float32],
    ) -> Station:
        model = self.get_model("fr")

        answer_embedding = model.encode([answer])
        best_index = model.similarity(answer_embedding, choice_embeddings).argmax()

        return choice_stations[best_index]

    def compute_user_station(self, answers: dict[str, str]) -> Station:
        if self.questionnaire_embeddings is None:
            raise ValueError("Model has not been initialized")

        best_stations = [
            self.get_best_station(answer, *self.questionnaire_embeddings[question])
            for question, answer in answers.items()
        ]

        [(best_station, _), *_] = Counter(best_stations).most_common()

        return best_station

    def generate_best_tags(self, answers: dict[str, str], limit: int = 5) -> list[str]:
        if self.tag_embeddings is None:
            raise ValueError("Model has not been initialized")

        tags, tag_embeddings = self.tag_embeddings
        model = self.get_model("fr")

        user_embeddings = model.encode(list(answers.values()))

        maxes, _ = model.similarity(user_embeddings, tag_embeddings).max(dim=0)
        tag_similarities = zip(tags, maxes.tolist())

        return [tag for tag, _ in sorted(tag_similarities, key=itemgetter(1))][:limit]

    def get_best_verbatims(self, answers: dict[str, str]) -> list[str]:
        user_input = list(answers.values())
        model = self.get_model()

        user_embeddings = model.encode(user_input)
        similarities = model.similarity(user_embeddings, user_embeddings).triu(diagonal=1)
        
        first, second, *_ = torch.unravel_index(similarities.argmax(), similarities.shape)
        return [user_input[first], user_input[second]]

    def generate_best_artists(self, answers: dict[str, str]) -> list[str]:
        return [""]

    def get_best_playlist(self, answers: dict[str, str]) -> dict[str, str]:
        return {"deezer": "https://link.deezer.com/s/30iKS8WFIDokwCdWfihFA", "spotify": ""}


def coin_flip():
    return choice((True, False))


@lru_cache
def get_model() -> FreqGenModel:
    return FreqGenModel()
