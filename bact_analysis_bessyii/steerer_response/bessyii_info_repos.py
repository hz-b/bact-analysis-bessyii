from ..interfaces.device_location import DeviceLocationServiceInterface
from bact_analysis_bessyii.interfaces.space_mapping import SpaceMappingInterface, SpaceMappingCollectionInterface


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


class DeviceLocationServiceBESSYII(DeviceLocationServiceInterface):
    """
    """
    def check_in_ring(self, device_name: str):
        assert device_name[-1] == "R"
        assert int(device_name[-2]) in range(1, 9)
        assert device_name[-3] in ["D", "T"]

    def steerer_location(self, device_name: str):
        self.check_in_ring(device_name)

        # accepting power converter here too
        expected = ["P", "M"]

        # check that the one before is the child number
        if device_name[-4] in ["1", "2", "3", "4"] and device_name[-5] in expected:
            return convert_pc_to_magnet_name_if_needed(device_name, index=-5)
        elif device_name[-4] in expected:
            return convert_pc_to_magnet_name_if_needed(device_name, index=-4)
        else:
            raise AssertionError(f"Can not handle sextupole device name {device_name}")

    def get_location_name(self, device_name: str) -> str:
        """a poor man's implementation based on Heuristics
        """
        if device_name[0] in ["H", "V"] and device_name[1] == "S":
            return self.steerer_location(device_name[1:])
        # need to implement further
        assert(0)


def convert_pc_to_magnet_name_if_needed(device_name: str, *, index: int) -> str:
    if device_name[index] == "P":
        return device_name[:index] + "M" + device_name[index+1:]
    assert device_name[index] == "M"
    return device_name
