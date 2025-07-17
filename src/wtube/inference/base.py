from abc import ABC, abstractmethod
from wtube.utils.instance import PreprocDetData, DetResults
from utils.registry import register


@register("inference")
class DetInference(ABC):

    @abstractmethod
    def __call__(self, preprocessed_data: "PreprocDetData") -> "DetResults":
        raise NotImplementedError
