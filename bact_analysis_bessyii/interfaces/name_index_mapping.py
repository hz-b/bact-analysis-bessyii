from abc import ABCMeta, abstractmethod


class NameIndexMappingInterface(metaclass=ABCMeta):
    @abstractmethod
    def get_name(self, index: int) -> str:
        raise NotImplementedError("implement in derived class")

    @abstractmethod
    def get_index(self, name: str) -> int:
        raise NotImplementedError("implement in derived class")
