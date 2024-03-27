from enum import Enum
from typing import Sequence

from ..interfaces.device_location import DeviceLocationServiceInterface
from bact_analysis_bessyii.interfaces.space_mapping import (
    SpaceMappingInterface,
    SpaceMappingCollectionInterface,
)
from ..interfaces.element_families import ElementFamilies, ElementFamily


class SpaceMappingIdentity(SpaceMappingInterface):
    def forward(self, v) -> object:
        return v

    def inverse(self, v: object) -> object:
        return v


class SpaceMappingLinearWithoutBias(SpaceMappingInterface):
    def __init__(self, factor: float, device_name: str):
        self.factor = factor
        self.device_name = device_name

    def forward(self, v) -> object:
        return self.factor * v

    def inverse(self, v: object) -> object:
        return v / self.factor


class SpaceMappingCollectionBESSYII(SpaceMappingCollectionInterface):
    def get(self, device_name) -> SpaceMappingInterface:
        return SpaceMappingIdentity()


class BESSYIIFamilyNames(Enum):
    horizontal_steerers = "horizontal_steerers"
    vertical_steerers = "vertical_steerers"


class BessyIIELementFamilies(ElementFamilies):
    def get(self, family_name: str) -> ElementFamily:
        family_name = BESSYIIFamilyNames(family_name)
        if family_name == BESSYIIFamilyNames.horizontal_steerers:
            return self.get_horizontal_steerer_names()
        elif family_name == BESSYIIFamilyNames.vertical_steerers:
            return self.get_vertical_steerer_names()
        else:
            raise NotImplementedError(f"Don't know family name {family_name}")

    def get_horizontal_steerer_names(self) -> ElementFamily:
        return self.horizontal_steerer_names

    def get_vertical_steerer_names(self) -> ElementFamily:
        return self.vertical_steerer_names

    def add_element_names(self, element_names: Sequence[str]):
        horizontal_steerer_names = list(self.horizontal_steerer_names.members)
        vertical_steerer_names = list(self.vertical_steerer_names.members)
        bpm_names = list(self.bpm_names.members)

        for name in element_names:
            if not check_in_bessyii_ring(name):
                continue
            if check_bessyii_steerer_name(name):
                if name[0] == "H":
                    horizontal_steerer_names.append(name)
                elif name[0] == "V":
                    vertical_steerer_names.append(name)
                else:
                    raise AssertionError("can'T hanndle checked steerer {steerer}")
            elif name[:3] == "BPM":
                bpm_names.append(name)

        self.bpm_names = ElementFamily(members=bpm_names)
        self.horizontal_steerer_names = ElementFamily(members=horizontal_steerer_names)
        self.vertical_steerer_names = ElementFamily(members=vertical_steerer_names)

    def __init__(self, element_names: Sequence[str]):
        """
        Args:
            element_names:
        """
        self.horizontal_steerer_names = ElementFamily(members=[])
        self.vertical_steerer_names = ElementFamily(members=[])
        self.bpm_names = ElementFamily(members=[])
        self.add_element_names(element_names=element_names)


def check_in_bessyii_ring(device_name: str):
    try:
        sector = int(device_name[-2])
    except ValueError:
        return False

    return (
        device_name[-1] == "R"
        and sector in range(1, 9)
        and device_name[-3] in ["D", "T"]
    )


def check_bessyii_steerer_name(device_name: str) -> bool:
    """

    Args:
        device_name:

    Returns:

    """
    expected = ["P", "M"]

    if device_name[0] not in ["H", "V"] or device_name[1] != "S":
        return False

    # check that the one before is the child number
    if device_name[-4] in ["1", "2", "3", "4"] and device_name[-5] in expected:
        return True
    elif device_name[-4] in expected:
        return True
    else:
        return False


class DeviceLocationServiceBESSYII(DeviceLocationServiceInterface):
    """ """

    def steerer_location(self, device_name: str):
        assert check_in_bessyii_ring(device_name)
        assert check_bessyii_steerer_name(device_name)

        # accepting power converter here too
        expected = ["P", "M"]

        housing_device_name = device_name[1:]
        # check that the one before is the child number
        if device_name[-5] in expected:
            return convert_pc_to_magnet_name_if_needed(housing_device_name, index=-5)
        elif device_name[-4] in expected:
            return convert_pc_to_magnet_name_if_needed(housing_device_name, index=-4)
        else:
            raise AssertionError(f"Can not handle sextupole device name {device_name}")

    def get_location_name(self, device_name: str) -> str:
        """a poor man's implementation based on Heuristics"""
        if check_bessyii_steerer_name(device_name):
            return self.steerer_location(device_name)
        # BPM device and position names are identical
        if device_name[:3] == "BPM":
            return device_name
        # need to implement further
        assert 0


def convert_pc_to_magnet_name_if_needed(device_name: str, *, index: int) -> str:
    if device_name[index] == "P":
        return device_name[:index] + "M" + device_name[index + 1 :]
    assert device_name[index] == "M"
    return device_name
