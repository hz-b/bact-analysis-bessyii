from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass
class ElementFamily:
    members : Sequence[str]


class ElementFamilies(metaclass=ABCMeta):
    @abstractmethod
    def get(self, family_name: str) -> ElementFamily:
        raise NotImplementedError("use concrete implementation instead")
