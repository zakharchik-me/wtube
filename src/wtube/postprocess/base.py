from abc import ABC, abstractmethod
from wtube.utils.instance import DetResults
from utils.registry import register


@register("postprocess")
class DetPostprocessor(ABC):

    @abstractmethod
    def __call__(self, raw_det_data: DetResults) -> "DetResults":
        raise NotImplementedError