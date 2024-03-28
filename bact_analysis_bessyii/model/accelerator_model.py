import functools
from dataclasses import dataclass
from typing import TypeVar, Sequence

from bact_analysis_bessyii.model.analysis_model import index_for_datum_with_name
from bact_analysis_bessyii.model.twiss import Twiss

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


@dataclass
class Position:
    value: float
    name: str


@dataclass
class SurveyPositions:
    positions: Sequence[Position]

    def get(self, pos_name: str):
        @functools.lru_cache
        def indices():
            return index_for_datum_with_name(self.positions)

        return self.positions[indices()[pos_name]]


@dataclass
class AcceleratorDescription:
    twiss: Twiss
    survey: SurveyPositions
