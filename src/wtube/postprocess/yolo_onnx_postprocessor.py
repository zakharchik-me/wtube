from utils.registry import register
from .base import DetPostprocessor, DetResults
from wtube.utils.ops import nms


_match_metrics = ["IOU", "IOS"]

@register("postprocess")
class YoloOnnxPostprocessor(DetPostprocessor):
    def __init__(self, *args, **kwargs):
        self.match_metric = kwargs.get('match_metric') or 'IOS'
        self.match_threshold = kwargs.get('match_threshold') or 0.5

    def __call__(
            self,
            raw_det_data: "DetResults",
    ) -> "DetResults":

        assert self.match_metric in _match_metrics, f"Invalid match metric: {self.match_metric}, format must be one of {_match_metrics}"

        indices = nms(raw_det_data.data[:, 2:], raw_det_data.data[:, 0], self.match_metric, self.match_threshold)

        raw_det_data.convert('xyxy')
        return raw_det_data[indices]
