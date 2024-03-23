"""Common naming for machine location names in an independent manner

Translation service interface from the common naming to the
local used identifier
"""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class StandardLocationName:
    """representing the name
    """
    family_name: str
    sector_index: int
    child_index: int


class StandardLocationNameInterface(metaclass=ABCMeta):
    """translation service
    """
    @abstractmethod
    def standard(self, pos_name_at_machine: str) -> StandardLocationName:
        """
        """

    @abstractmethod
    def local(self, standard_name: StandardLocationName) -> str:
        """
        """
