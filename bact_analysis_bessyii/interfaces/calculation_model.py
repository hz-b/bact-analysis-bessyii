from abc import ABCMeta, abstractmethod
from typing import Union, Sequence

from bact_analysis_bessyii.steerer_response.model import StateSpace


class CalculationModelInterface(metaclass=ABCMeta):
    @abstractmethod
    def update(self,  device_name: str, property: str, value: object):
        raise NotImplementedError("implement in derived class")

    @abstractmethod
    def compute_values_at_positions(self, device_names: Union[Sequence[str], None]) -> None:
        """Should it return a Future?
        """
        raise NotImplementedError("Implement in derived class")

    def get_calculated_positions(self)-> Sequence[StateSpace]:
        """
        """
        raise NotImplementedError("implement in derived class")
