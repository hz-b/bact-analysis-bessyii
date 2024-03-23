from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional


class GISInterface(metaclass=ABCMeta):
    @abstractmethod
    def get_coordinate(self, pos_name: str) -> Coordinate:
        raise NotImplementedError("xx")


@dataclass
class Coordinate:
    horizontal: Optional[float]
    vertical: Optional[float]
    longitudinal: float
