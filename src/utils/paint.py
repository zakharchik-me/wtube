from typing import List, Tuple, Optional, Union
import yaml
import numpy as np
import cv2


def load_class_config(path: str = "client/configs/class_config.yml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return {int(k): v for k, v in config["id_to_label"].items()}


def hex2rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert hex string (#RRGGBB) to RGB tuple."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def build_palette() -> List[Tuple[int, int, int]]:
    hexs = (
        "042AFF", "0BDBEB", "F3F3F3", "00DFB7", "111F68",
        "FF6FDD", "FF444F", "CCED00", "00F344", "BD00FF",
        "00B4FF", "DD00BA", "00FFFF", "26C000", "01FFB3",
        "7D24FF", "7B0068", "FF1B6C", "FC6D2F", "A2FF0B",
    )
    return [hex2rgb(f"#{c}") for c in hexs]

