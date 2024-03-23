import functools
from typing import Sequence

from ..interfaces.name_index_mapping import NameIndexMappingInterface


class FitVariableNameIndexMapping(NameIndexMappingInterface):
    def __init__(self, names: Sequence[str]):
        self.names = names

    def get_name(self, index: int) -> str:
        return self.names[index]

    def get_index(self, name: str) -> int:
        @functools.lru_cache
        def indices():
            return {idx : name for idx, name in enumerate(self.names)}

        return indices()[name]


