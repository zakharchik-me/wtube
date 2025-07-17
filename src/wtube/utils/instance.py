from .ops import ltwh2xywh, ltwh2xyxy, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh
from typing import Union, List, Tuple
import numpy as np
import torch
from collections import namedtuple


DetectionResult = namedtuple("DetectionResult", ["boxes", "scores", "labels"])

_formats = ["xyxy", "xywh", "ltwh"]
_tensor_formats = ["nhwc", "nchw"]
_bbox_formats = ["xyxy", "xywh", "ltwh"]


__all__ = ("Bboxes", "PreprocDetData", "DetResults")


class PreprocDetData:
    def __init__(self, batch: tuple, tensor_format: str):
        assert tensor_format in _tensor_formats, f"Invalid bounding box format: {tensor_format}, format must be one of {_tensor_formats}"
        self.batch, self.cords, self.scales = batch
        self.tensor_format = tensor_format

    def __len__(self):
        """Return batch size"""
        return len(self.batch)

    def convert(self, tensor_format: str) -> None:
        assert tensor_format in _tensor_formats, f"Invalid tensor format: {tensor_format}, format must be one of {_tensor_formats}"
        if self.tensor_format == tensor_format:
            return
        elif self.tensor_format == "nchw":
            self.batch = np.transpose(self.batch, (0, 2, 3, 1))
        elif self.tensor_format == "nhwc":
            self.batch = np.transpose(self.batch, (0, 3, 1, 2))

        self.tensor_format = tensor_format

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.batch.shape

    def __getitem__(self, index) -> tuple:
        return self.batch[index], self.cords[index], self.scales[index]

    def __iter__(self):
        for frame, cord, scale in zip(self.batch, self.cords, self.scales):
            yield frame, cord, scale


class DetResults:
    def __init__(self, results: Union[np.ndarray, torch.tensor], bbox_format: str):
        assert bbox_format in _bbox_formats, f"Invalid bounding box format: {bbox_format}, format must be one of {_bbox_formats}"
        assert results.shape[1] == 6

        self.results = results
        self.bbox_format = bbox_format

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.results.shape


    def __len__(self):
        return len(self.results)


    @property
    def data(self) -> Union[np.ndarray, torch.tensor]:
        return self.results

    def convert(self, bbox_format: str) -> None:
        assert bbox_format in _bbox_formats, f"Invalid bounding box format: {bbox_format}, format must be one of {_bbox_formats}"
        if self.bbox_format == bbox_format:
            return
        elif self.bbox_format == "xyxy":
            func = xyxy2xywh if bbox_format == "xywh" else xyxy2ltwh
        elif self.bbox_format == "xywh":
            func = xywh2xyxy if bbox_format == "xyxy" else xywh2ltwh
        else:
            func = ltwh2xyxy if bbox_format == "xyxy" else ltwh2xywh

        self.results[:, 2:] = func(self.results[:, 2:])
        self.bbox_format = bbox_format

    def __getitem__(self, index) -> "DetResults":
        if isinstance(index, int):
            return DetResults(self.results[index], bbox_format=self.bbox_format)
        b = self.results[index]
        assert b.ndim == 2, f"Indexing on batch with {index} failed to return a matrix!"
        return DetResults(b, bbox_format=self.bbox_format)

    @property
    def bls(self) -> namedtuple:
        return DetectionResult(
            boxes=self.results[:, 2:],
            scores=self.results[:, 0],
            labels=self.results[:, 1]
        )


class Bboxes:
    """
    A class for handling bounding boxes in multiple formats.

    The class supports various bounding box formats like 'xyxy', 'xywh', and 'ltwh' and provides methods for format
    conversion, scaling, and area calculation. Bounding box data should be provided as numpy arrays.

    Attributes:
        bboxes (np.ndarray): The bounding boxes stored in a 2D numpy array with shape (N, 4).
        format (str): The format of the bounding boxes ('xyxy', 'xywh', or 'ltwh').

    Methods:
        convert: Convert bounding box format from one type to another.
        areas: Calculate the area of bounding boxes.
        mul: Multiply bounding box coordinates by scale factor(s).
        add: Add offset to bounding box coordinates.
        concatenate: Concatenate multiple Bboxes objects.

    Examples:
        Create bounding boxes in YOLO format
        >>> bboxes = Bboxes(np.array([[100, 50, 150, 100]]), format="xywh")
        >>> bboxes.convert("xyxy")
        >>> print(bboxes)

    Notes:
        This class does not handle normalization or denormalization of bounding boxes.
    """

    def __init__(self, bboxes: Union[np.ndarray, torch.Tensor], format: str = "xyxy", norm: bool = False) -> None:
        """
        Initialize the Bboxes class with bounding box data in a specified format.

        Args:
            bboxes (np.ndarray): Array of bounding boxes with shape (N, 4) or (4,).
            format (str): Format of the bounding boxes, one of 'xyxy', 'xywh', or 'ltwh'.
            norm (bool): Flag of normalized bboxes or not
        """
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format
        self.normalized = norm

    def convert(self, format: str) -> None:
        """
        Convert bounding box format from one type to another.

        Args:
            format (str): Target format for conversion, one of 'xyxy', 'xywh', or 'ltwh'.
        """
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        if self.format == format:
            return
        elif self.format == "xyxy":
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh
        elif self.format == "xywh":
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh
        else:
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
        self.bboxes = func(self.bboxes)
        self.format = format


    def __len__(self):
        """Return the number of bounding boxes."""
        return len(self.bboxes)


    def denormalize(self, w: int, h: int) -> None:
        if not self.normalized:
            return

        self.scale(scale=(w, h, w, h))
        self.normalized = False


    def normalize(self, w: int, h: int) -> None:
        if self.normalized:
            return
        self.scale(scale=(1/w, 1/h, 1/w, 1/h))
        self.normalized = True


    def scale(self, scale: Union[list, tuple, int, float]):
        """
        Multiply bounding box coordinates by scale factor(s).

        Args:
            scale (int | tuple | list): Scale factor(s) for four coordinates. If int, the same scale is applied to
                all coordinates.
        """
        if isinstance(scale, (int, float)):
            scale = (scale, scale, scale, scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 4
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    @classmethod
    def concatenate(cls, boxes_list: List["Bboxes"], axis=0) -> "Bboxes":
        """
        Concatenate a list of Bboxes objects into a single Bboxes object.

        Args:
            boxes_list (List[Bboxes]): A list of Bboxes objects to concatenate.
            axis (int, optional): The axis along which to concatenate the bounding boxes.

        Returns:
            (Bboxes): A new Bboxes object containing the concatenated bounding boxes.

        Notes:
            The input should be a list or tuple of Bboxes objects.
        """
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty(0))
        assert all(isinstance(box, Bboxes) for box in boxes_list)

        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))

    def __getitem__(self, index) -> "Bboxes":
        """
        Retrieve a specific bounding box or a set of bounding boxes using indexing.

        Args:
            index (int | slice | np.ndarray): The index, slice, or boolean array to select the desired bounding boxes.

        Returns:
            (Bboxes): A new Bboxes object containing the selected bounding boxes.

        Notes:
            When using boolean indexing, make sure to provide a boolean array with the same length as the number of
            bounding boxes.
        """
        if isinstance(index, int):
            return Bboxes(self.bboxes[index].reshape(1, -1))
        b = self.bboxes[index]
        assert b.ndim == 2, f"Indexing on Bboxes with {index} failed to return a matrix!"
        return Bboxes(b)


