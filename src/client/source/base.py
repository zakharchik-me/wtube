from abc import ABC, abstractmethod

class FrameSource(ABC):
    @abstractmethod
    def __iter__(self):
        pass


    @abstractmethod
    def __next__(self):
        pass

    @property
    def total_frames(self):
        pass
