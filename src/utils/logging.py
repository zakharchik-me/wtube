import json
import logging
import os
from collections import namedtuple
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Union, Tuple

import cv2
from numpy import ndarray

data_log: Dict[str, List] = {
    "images": [],
    "annotations": []
}


def log_frame(frame: Any, frame_num: int, frame_ts: float) -> int:
    """
    Adds frame logs to data_log['images'], and writes out the JPEG
    to a path relative to this script: logs/test/images/000001.jpg, etc.
    """
    h, w = frame.shape[:2]

    data_log["images"].append({
        "id": frame_num,
        "width": w,
        "height": h,
        "frame_num": frame_num,
        "frame_ts": frame_ts,
    })

    return frame_num

def log_annotation(
        image_id: int,
        bls: namedtuple,
        image_ts: float,
) -> None:
    """
    Adds annotations to data_log['annotations'], then writes the entire
    data_log dict to a JSON file under logs/test/annotations/000001.json, etc.
    """
    annotation_id_counter = 0

    # for each detected box, assign a new annotation ID
    for bbox, score, label in zip(bls.boxes, bls.scores, bls.labels):
        # convert bbox coords to ints
        bbox_int = [int(coord) for coord in bbox]

        annotation_id_counter += 1

        data_log["annotations"].append({
            "id": annotation_id_counter,
            "image_id": image_id,
            "frame_ts": image_ts,
            "bbox": bbox_int,
            "score": float(score),
            "label": int(label),
        })

def get_log() -> Dict[str, List]:
    """Return the full in-memory data_log dict."""
    return data_log

def clear_log() -> None:
    """Clear the in-memory data_log (but do not reset the counters)."""
    data_log["images"].clear()
    data_log["annotations"].clear()

def save_img(frame: ndarray, frame_num: int, images_dir: str) -> None:
    """

    :param img_path:
    :param frame:
    :return:
    """

    filename = f"{frame_num:09d}.jpg"
    img_path = os.path.join(images_dir, filename)

    cv2.imwrite(str(img_path), frame)

    return

def save_log(data: Dict[str, List], frame_num: int, annons_dir: str) -> None:
    """
    Saves log ann to Disk
    """
    filename = f"{frame_num:09d}.json"
    ann_path = os.path.join(annons_dir, filename)

    with open(ann_path, "w") as f:
        json.dump(data, f, indent=2)

def load_log(path: str, log_num: int) -> Tuple[List[List[float]], List[float], List[int]]:
    """
    Loads annotations (bbox, score, label) from a JSON file.

    :param path: Path to the folder containing log files
    :param log_num: Log file number (e.g., 3 → 000000003.json)
    :return: Tuple of lists (boxes, scores, labels)
    """
    json_path = os.path.join(path, f"{log_num:09d}.json")

    with open(json_path, "r") as f:
        log = json.load(f)
        annotations = log["logs"]["annotations"]

    boxes = [ann["bbox"] for ann in annotations]
    scores = [ann["score"] for ann in annotations]
    labels = [ann["label"] for ann in annotations]

    return boxes, scores, labels