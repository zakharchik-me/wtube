# tests/test_metadata.py
import os
import json
import pytest

from utils.logging import get_log

# from src import main

def test_metadata_consistency(tmp_path):
    logs = get_log()
    images = logs.get("images", [])
    annotations = logs.get("annotations", [])

    assert images, "No images have been logged"
    img_record = images[-1]
    image_id = img_record["id"]

    assert annotations, "No annotations have been logged"

    for ann in annotations:
        assert ann["image_id"] == image_id, (
            f"Annotation {ann['id']} refers to image_id={ann['image_id']}, "
            f"expected {image_id}"
        )
        assert isinstance(ann["bbox"], (list, tuple)) and len(ann["bbox"]) == 4, (
            f"Annotation {ann['id']} bbox should be a list/tuple of length 4, got {ann['bbox']}"
        )

    img_path = img_record.get("file")
    assert img_path and os.path.exists(img_path), f"Image file not found: {img_path}"

    ann_path = os.path.join("logs", "annotations", f"{image_id:06d}_ann.json")
    assert os.path.exists(ann_path), f"Annotations JSON file not found: {ann_path}"

    with open(ann_path, "r") as f:
        ann_data = json.load(f)
    expected = [a for a in annotations if a["image_id"] == image_id]
    assert ann_data == expected, (
        f"Annotations JSON content {ann_data} does not match expected {expected}"
    )

if __name__ == "__main__":
    main()
    test_metadata_consistency(None)