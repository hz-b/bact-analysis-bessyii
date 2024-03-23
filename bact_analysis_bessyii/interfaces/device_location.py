from abc import ABCMeta, abstractmethod


class DeviceLocationServiceInterface(metaclass=ABCMeta):
    @abstractmethod
    def get_location_name(self, device_name: str) -> str:
        """
        Args:
            device_name: name of the device

        Returns:
            position name

        Raises exception if no position name is applicable
        """
