import functools
from dataclasses import dataclass
from typing import Sequence

from bact_analysis_bessyii.model.planes import Planes


@dataclass
class TwissForPlane:
    """
    Todo:
        refactor it to TwissAtPosition
        add name to it
    """
    alpha: Sequence[float]
    beta: Sequence[float]
    nu: Sequence[float]

    def at_index(self, index):
        return TwissForPlane(alpha=self.alpha[index], beta=self.beta[index], nu=self.nu[index])


@dataclass
class Twiss:
    """
    Todo:
        Refactor x, y to Sequence[TwissAtPosition]
    """
    x: TwissForPlane
    y: TwissForPlane
    names: Sequence

    def at_position(self, name: str):
        @functools.lru_cache
        def indices():
            return {name: cnt for cnt, name in enumerate(self.names)}
        idx = indices()[name]
        return Twiss(x=self.x.at_index(idx), y=self.y.at_index(idx), names=[self.names[idx]])

    def get_plane(self, plane: Planes):
        plane = Planes(plane)
        if plane == Planes.x:
            return self.x
        elif plane == Planes.y:
            return self.y
        else:
            raise AssertionError("How could I end up here")