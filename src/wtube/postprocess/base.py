from abc import ABC, abstractmethod

class BasePostprocessor(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass