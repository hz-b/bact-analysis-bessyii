from abc import ABCMeta, abstractmethod


class SpaceMappingInterface(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, v) -> object:
        """from user side to instrument side
        """
        raise NotImplementedError("implement in derived class")

    def inverse(self, v: object) -> object:
        """from instrument side to user side
        """
        raise NotImplementedError("implement in derived class")


class SpaceMappingCollectionInterface(metaclass=ABCMeta):
    @abstractmethod
    def get(self, device_name) -> SpaceMappingInterface:
        """space mapping for this particular device, channel or instrument
        """
        raise NotImplementedError("implement in derived class")
