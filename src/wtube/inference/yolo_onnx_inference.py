import onnxruntime as ort
from .base import DetInference
from utils.registry import register
import numpy as np
from wtube.utils.instance import PreprocDetData, DetResults


@register("inference")
class YoloOnnxInference(DetInference):
    model: ort.InferenceSession
    model_width: int
    model_height: int
    model_input: str

    def __init__(self, *args, **kwargs):
        self.model_weights = kwargs.get('model_weights') or None
        self.load_model(self.model_weights)
        self.get_model_inputs()


    def __call__(self, preprocessed_data: "PreprocDetData") -> "DetResults":
        outputs = []
        if preprocessed_data.tensor_format == "nhwc":
            preprocessed_data.convert("nchw")

        for patch, cord, scale in preprocessed_data:
            output = self.model.run(None, {self.model_input: np.expand_dims(patch, 0)})
            outputs.append((output, cord, scale))

        return self.convert(outputs)

    def load_model(self, path: str):
        self.model = ort.InferenceSession(path, providers=["CUDAExecutionProvider"])

    def get_model_inputs(self):
        model_inputs = self.model.get_inputs()
        input_shape = model_inputs[0].shape
        self.model_width = input_shape[3]
        self.model_height = input_shape[2]
        self.model_input = model_inputs[0].name


    def convert(self, outputs: list) -> "DetResults":
        boxes = []
        scores = []
        class_ids = []

        for output, cords, scale in outputs:
            output = np.transpose(np.squeeze(output[0]))
            output = output[np.any(output[:, 4:] >= 0.05, axis=1)]
            rows = output.shape[0]

            bboxes = np.zeros((rows, 4), dtype=np.int32)
            classes_scores = output[:, 4:]

            max_score = np.amax(classes_scores, axis=1)

            class_id = np.argmax(classes_scores, axis=1)

            bboxes[:, 0] = ((output[:, 0] - output[:, 2] / 2 + cords[1]) / scale)
            bboxes[:, 1] = ((output[:, 1] - output[:, 3] / 2 + cords[0]) / scale)
            bboxes[:, 2] = output[:, 2] / scale
            bboxes[:, 3] = output[:, 3] / scale

            boxes.append(bboxes)
            scores.append(max_score)
            class_ids.append(class_id)

        boxes = np.vstack(boxes)
        scores = np.hstack(scores)
        class_ids = np.hstack(class_ids)
        results = np.column_stack((scores, class_ids, boxes))

        return DetResults(results=results, bbox_format="ltwh")