from abc import ABC, abstractmethod

class FrameSource(ABC):
    @abstractmethod
    def __iter__(self):
        pass


    @abstractmethod
    def __next__(self):
        pass

    @property
    def get_fps(self):
        pass

    @property
    def get_src_name(self):
        pass